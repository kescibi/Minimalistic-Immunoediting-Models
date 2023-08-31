clear all; close all;
rng(123);
%%%%% OPTIONS %%%%%
approach = 0 %choose between approaches 0,1,2,3

predict = 0 %choose whether to predict 1/3 (or less) of the points
%%%%%%%%%%%%%%%%%%%

%Using the conversion rule 10^-3 ml = 1 mm^3 = 10^6 cells
%we get the maximum capacity K = 1kg = 10^12 [cells] = 10^12 [cells] * 10^-6
%[mm^3 / cells] * 10^-3 [cm^3 / mm^3] = 1000 cm^3 (=1L)
%K = 1000;
T = readtable("./data/online_data/pcbi.1009822.s006.xlsx", "Sheet", "Study3");

patient_ids = unique(table2array(T(:, 1)));

no_eligible = -1;

max_manual_volume = 0; %in cm^3
max_total_volume = [];
min_manual_volume = 1000; %in cm^3
for id = 1 : length(patient_ids)
    patient_data = T((T.Patient_Anonmyized == patient_ids(id)), :);
    y = table2array(patient_data(:,3));
    current_manual_volume = ((4/3)*pi*(y(1)/2).^3) / 1000;  %mm^3 in cm^3
    max_manual_volume = max(current_manual_volume, max_manual_volume);

    if current_manual_volume > 0
        min_manual_volume = min(current_manual_volume, min_manual_volume);
    end
end
max_manual_volume
min_manual_volume %cm^3

patient_list_id = [];
%patient_count = 0;
for id = 1 : length(patient_ids)
    patient_data = T((T.Patient_Anonmyized == patient_ids(id)), :);

    t = table2array(patient_data(:,2));
    y = table2array(patient_data(:,3));

    if any(isnan(y)) || length(y) < 6 || y(1) > 97 || not(issorted(t))
       %last condition filters out patients where 10^12 / cells would be exceeded 
       %also filters out patients who do not have sorted time entries

       %skip this patient
    else
        patient_list_id(end+1) = id;
        %patient_count = patient_count + 1;
    end
end

if approach == 3
    fit_info = zeros(8, length(patient_list_id)); %approach 3 has one more parameter
else
    fit_info = zeros(7, length(patient_list_id));
end

figure_counter = 0;
f = -1;
for i = 1 : length(patient_list_id)
    patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);

    t = table2array(patient_data(:,2));
    y = table2array(patient_data(:,3));

    no_eligible = mod(no_eligible+1,4);
    t = table2array(patient_data(:,2)); %days
    y = table2array(patient_data(:,3)); %mm

    manual_volume = ((4/3)*pi*(y/2).^3) / 1000 %mm^3 in cm^3

    if no_eligible == 0
        if f ~= -1
            if predict == 0
                saveas(f, "figures/fits/approach_"+string(approach)+"_"+string(figure_counter)+".png");
            else
                saveas(f, "figures/fits/prediction_approach_"+string(approach)+"_"+string(figure_counter)+".png");
            end
        end
        figure_counter = figure_counter + 1;
        f = figure();
    end

    subplot(2,2,no_eligible+1,'Parent',f);

    if predict == 0
        interval_fit = 1 : length(t);
        %t_fit = t;
    else
        interval_fit = 1 : (length(t) - floor(length(t) / 3));
        %t_fit = t(1 : (length(t) - floor(length(t) / 3)));
    end

    
    scatter(t(interval_fit), manual_volume(interval_fit), [], 'Color', '#0072BD');
    hold on;
    if predict == 1
        interval_predict = (interval_fit(end) + 1) : length(t);
        scatter(t(interval_predict), manual_volume(interval_predict), [], 'Color', '#D95319');
    end
    hold on;

    options = optimset();%'Display','iter');

    r_min_found = r_min(convert2cells(manual_volume(1)));
    r_max_found = r_max(convert2cells(manual_volume(1)));

    T_0 = 10^9;
    K_0 = 10^3;
    T_Td = convert2cells(manual_volume(1)) / T_0;

    gs = GlobalSearch('Display','iter');

    if approach == 0
        toMin = @(parameters) eval_approach0(t(interval_fit), parameters, manual_volume(interval_fit));
    elseif approach == 1
        toMin = @(parameters) eval_approach1(t(interval_fit), parameters, manual_volume(interval_fit));
    elseif approach == 2
        toMin = @(parameters) eval_approach2(t(interval_fit), parameters, manual_volume(interval_fit));
    elseif approach == 3
        toMin = @(parameters) eval_approach3(t(interval_fit), parameters, manual_volume(interval_fit));
    end

    if approach == 0
        %Values should all be normalized, as clearly the search space is
        %not searched well, due to large orders of magnitude! Even though
        %many points are tested, the large orders of magnitude are
        %difficult to test. This is why normalization would help
        %Very similar values to Kuznetsov 1994
        problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, 10^-2], 'objective', toMin, ...
            'lb', [r_min_found,0,0], 'ub', [r_max_found, r_max_found, 1], 'options', options);
        
        params = run(gs, problem)
        %params = fmincon(problem);

        r = params(1);
        mu_0 = params(2);
        mu_star = params(3);
        t_sim = linspace(t(1),t(end),1000);
        y_sim = run_approach0(t_sim, params, T_Td);

        solpts = run_approach0(t, params, T_Td);
        L2_error = ((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));
        L1_error = ((1 / length(manual_volume))*sum(abs(((convert2ml((solpts(1,:))*T_0) - manual_volume')))));
        R2_error = calculate_R2(manual_volume', (convert2ml((solpts(1,:))*T_0)));
        
%         if predict == 1
%             predicted = solpts(1,:);
%             %MAPE_error = mape(convert2ml((predicted(interval_predict))*T_0),manual_volume(interval_predict)');
%             %R2_prediction = calculate_R2(manual_volume(interval_predict)', convert2ml((predicted(interval_predict))*T_0));
%         end

        fit_info(:,i) = [i, r, mu_0, mu_star, L2_error, L1_error, R2_error]';

        plot(t_sim,convert2ml(y_sim(1,:)*T_0), 'Color', '#0072BD');
    elseif approach == 1
        %Values should all be normalized, as clearly the search space is
        %not searched well, due to large orders of magnitude! Even though
        %many points are tested, the large orders of magnitude are
        %difficult to test. This is why normalization would help
        %Very similar values to Kuznetsov 1994
        problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, 10^-3], 'objective', toMin, ...
            'lb', [r_min_found,0,0], 'ub', [r_max_found, r_max_found, 1], 'options', options);
        
        params = run(gs, problem)
        %params = fmincon(problem);


        r = params(1);
        y_0 = params(2);
        mu_2_tilde = params(3);
        t_sim = linspace(t(1),t(end),1000);
        y_sim = run_approach1(t_sim, params, T_Td);

        solpts = run_approach1(t, params, T_Td);
        L2_error = ((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2))
        L1_error = ((1 / length(manual_volume))*sum(abs(((convert2ml((solpts(1,:))*T_0) - manual_volume')))))
        R2_error = calculate_R2(manual_volume', (convert2ml((solpts(1,:))*T_0)))


        fit_info(:,i) = [i, r, y_0, mu_2_tilde, L2_error, L1_error, R2_error]';

        plot(t_sim,convert2ml(y_sim(1,:)*T_0), 'Color', '#0072BD');
    elseif approach == 2
         Max_T_res_n = (convert2cells(manual_volume(1)) / T_0);
         problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, Max_T_res_n / 2], 'objective', toMin, ...
            'lb', [r_min_found,0,0], 'ub', [r_max_found, r_max_found, Max_T_res_n], 'options', options);

         params = run(gs, problem)

         r = params(1);
         mu1 = params(2);
         T_res_n_estimated = params(3);

         t_sim = linspace(t(1),t(end),1000);
         y_sim = run_approach2(t_sim, [r, mu1, T_res_n_estimated], convert2cells(manual_volume(1)) / T_0);


         solpts = run_approach2(t, [r, mu1, T_res_n_estimated], convert2cells(manual_volume(1)) / T_0);

         L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2))
         L1_error = ((1 / length(manual_volume))*sum(abs(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')))))
         R2_error = calculate_R2(manual_volume', (convert2ml((solpts(1,:) + solpts(2,:))*T_0)))

         fit_info(:,i) = [i, r, mu1, T_res_n_estimated, L2_error, L1_error, R2_error]';

         plot(t_sim,convert2ml((y_sim(1,:)+y_sim(2,:))*T_0), 'Color', '#0072BD');
         plot(t_sim,convert2ml(y_sim(1,:)*T_0), "Color", 'green');
         plot(t_sim,convert2ml(y_sim(2,:)*T_0), "Color", 'magenta');
    elseif approach == 3
         Max_T_s_n = (convert2cells(manual_volume(1)) / T_0);
         problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2, 0.9999*Max_T_s_n, r_max_found / 10, 0], 'objective', toMin, ...
            'lb', [r_min_found,0,0,0], 'ub', [r_max_found, Max_T_s_n, r_max_found, 1], 'options', options, 'nonlcon', @constraint_approach_3);

         params = run(gs, problem)

         r = params(1);
         T_s_n_estimated = params(2);
         E_n_estimated = params(3);
         beta_tilde = params(4);

         t_sim = linspace(t(1),t(end),1000);
         y_sim = run_approach3(t_sim, [r, T_s_n_estimated, E_n_estimated, beta_tilde], convert2cells(manual_volume(1)) / T_0);


         solpts = run_approach3(t, [r, T_s_n_estimated, E_n_estimated, beta_tilde], convert2cells(manual_volume(1)) / T_0);

         L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2))
         L1_error = ((1 / length(manual_volume))*sum(abs(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')))))
         R2_error = calculate_R2(manual_volume', (convert2ml((solpts(1,:) + solpts(2,:))*T_0)))

         fit_info(:,i) = [i, r, T_s_n_estimated, E_n_estimated, beta_tilde, L2_error, L1_error, R2_error]';

         plot(t_sim,convert2ml((y_sim(1,:)+y_sim(2,:))*T_0), 'Color', '#0072BD');
         plot(t_sim,convert2ml(y_sim(1,:)*T_0), "Color", 'green');
         plot(t_sim,convert2ml(y_sim(2,:)*T_0), "Color", 'magenta');
         %plot(t_sim,convert2ml(y_sim(3,:)*T_0*K_0),'--');
    else
        error("Approach does not exist")
    end

    xlabel("Time in days");
    ylabel("Tumor size [cm^3]");
    xlim([t(1), t(end)])
    title(string(i) + ", R_2 = " + string(round(R2_error,2)) + ", MAE: " + string(round(L1_error, 3)));
    
    hold on;

    xline(0);
    shg;
end

if predict == 0
    saveas(f, "figures/fits/approach_"+string(approach)+"_"+string(figure_counter)+".png");
    save("fit_info_approach_"+string(approach)+".mat","fit_info");
else
    saveas(f, "figures/fits/prediction_approach_"+string(approach)+"_"+string(figure_counter)+".png");
    save("prediction_fit_info_approach_"+string(approach)+".mat","fit_info");
end

function r = r_min(T) %input: tumor cells at diagnosis, output: smallest r [1/d] based on TVDT <= 300d
    r = log((1 - (T/10^12)) / (0.5 - (T/10^12))) / 400;
end

function r = r_max(T) %input: tumor cells at diagnosis, output: largest r [1/d] based on TVDT >= 25
    r = log((1 - (T/10^12)) / (0.5 - (T/10^12))) / 25;
end

% function [c,ceq] = constraint_approach_1(x)
%     T_0 = 10^12;
%     K_0 = 1;
%     E_0 = 10^12;
%     r = x(1);
%     mu_1 = x(2);
%     mu_2 = x(3);
%     E_Td = x(4);
%     c(1) = mu_1 * (E_Td * E_0) - r;
%     %c(2) = mu_2 - mu_1;
%     ceq = [];
% end

function [c,ceq] = constraint_approach_3(parameters)
    T_0 = 10^9;
    r = parameters(1);
    T_s_n_estimated = parameters(2);
    z_0 = parameters(3);
    beta_tilde = parameters(4);
    c(1) = z_0 - r;
    c(2) = beta_tilde - (z_0 / (T_s_n_estimated/100));
    ceq = [];
end

function dydt = approach0(y,t,r,mu_star) %using cells
    T_0 = 10^9;
    K_0 = 10^3; %10^12 / T_0 = 10^3
    T = y(1);
    mu = y(2);
    if t < 0
        mu = 0;
        mu_star = 0;
    end
    dydt = zeros(2,1);
    dydt(1) = r * T * (1 - (T / K_0)) - mu * T;
    dydt(2) = -mu_star * mu;
end

function solpts = run_approach0(days,parameters,initial_tumor_size) %accepts as parameters only normalized values
T_0 = 10^9;
K_0 = 10^3;
r = parameters(1);
mu_0 = parameters(2);
mu_star = parameters(3);
tspan = [days(1), days(end)];
y0 = [initial_tumor_size, mu_0];
sol = ode23s(@(t,y) approach0(y,t,r,mu_star),tspan,y0);
solpts = deval(sol, days);
end

function L2_error = eval_approach0(days,parameters,manual_volume) %parameters: normalized, manual_volume: not normalized
T_0 = 10^9;
initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
solpts = run_approach0(days, parameters, initial_tumor_size);
L2_error = ((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));
end

function L2_error = eval_approach1(days,parameters,manual_volume) %parameters: normalized, manual_volume: not normalized
T_0 = 10^9;
K_0 = 10^3;
initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
solpts = run_approach1(days, parameters, initial_tumor_size);
L2_error = ((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));
end

function solpts = run_approach1(days,parameters,initial_tumor_size) %accepts as parameters only normalized values
T_0 = 10^9;
K_0 = 10^3;
r = parameters(1);
y_0 = parameters(2);
mu_2_tilde = parameters(3);
tspan = [days(1), days(end)];
y0 = [initial_tumor_size, y_0];
sol = ode23s(@(t,y) approach1(y,t,r,mu_2_tilde),tspan,y0);
solpts = deval(sol, days);
end

function dydt = approach1(y,t,r,mu_2_tilde) %using normalized values
    T_0 = 10^9;
    K_0 = 10^3;
    T_n = y(1);
    E_n = y(2);
    if t < 0
        E_n = 0;
        mu_2_tilde = 0;
    end
    dydt = zeros(2,1);
    dydt(1) = r * T_n * (1 - (T_n / K_0)) - T_n * E_n;
    dydt(2) = -mu_2_tilde * T_n * E_n;
end

function dydt = approach2(y,t,r,mu1) %using cells
    T_0 = 10^9;
    K_0 = 10^3; %10^12 / T_0 = 10^3
    if t < 0
        mu1 = 0;
    end
    dydt = zeros(2,1);
    T_p_n = y(1);
    T_res_n = y(2);
    dydt(1) = r * T_p_n * (1 - ((T_p_n + T_res_n) / K_0)) - mu1 * T_p_n;
    dydt(2) = r * T_res_n * (1 - ((T_p_n + T_res_n) / K_0));
end

function solpts = run_approach2(days,parameters,initial_tumor_size) %accepts as parameters only normalized values
T_0 = 10^9;
K_0 = 10^3;
r = parameters(1);
mu1 = parameters(2);
T_res_n_estimated = parameters(3);
tspan = [days(1), days(end)];
y0 = [initial_tumor_size - T_res_n_estimated, T_res_n_estimated];
sol = ode23s(@(t,y) approach2(y,t,r,mu1),tspan,y0);
solpts = deval(sol, days);
end

function L2_error = eval_approach2(days,parameters,manual_volume) %parameters: normalized, manual_volume: not normalized
T_0 = 10^9;
initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
solpts = run_approach2(days, parameters, initial_tumor_size);
L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));
end

function R2_error = calculate_R2(y_true, y_fit)
    y_mean = sum(y_true) / length(y_true);
    ss_res = sum((y_true - y_fit).^2);
    ss_tot = sum((y_true - y_mean).^2);
    R2_error = 1 - (ss_res / ss_tot);
end

function dydt = approach3(y,t,r,beta_tilde) %using cells
    T_0 = 10^9;
    K_0 = 10^3; %10^12 / T_0 = 10^3
    dydt = zeros(3,1);
    T_p_n = y(1);
    T_res_n = y(2);
    E_n = y(3);

    if t < 0
        E_n = 0;
        beta_tilde = 0;
    end

    dydt(1) = r * T_p_n * (1 - ((T_p_n + T_res_n) / K_0)) - E_n * T_p_n;
    dydt(2) = r * T_res_n * (1 - ((T_p_n + T_res_n) / K_0));
    dydt(3) = beta_tilde * E_n * T_p_n;
end

function solpts = run_approach3(days,parameters,initial_tumor_size) %accepts as parameters only normalized values
T_0 = 10^9;
K_0 = 10^3;
r = parameters(1);
T_s_n_estimated = parameters(2);
z_0 = parameters(3);
beta_tilde = parameters(4);
tspan = [days(1), days(end)];
y0 = [T_s_n_estimated, initial_tumor_size - T_s_n_estimated, z_0];
sol = ode23s(@(t,y) approach3(y,t,r,beta_tilde),tspan,y0);
solpts = deval(sol, days);
end

function L2_error = eval_approach3(days,parameters,manual_volume) %parameters: normalized, manual_volume: not normalized
T_0 = 10^9;
initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
solpts = run_approach3(days, parameters, initial_tumor_size);
L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));
end

