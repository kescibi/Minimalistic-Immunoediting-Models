clear all; close all;
rng(123);
%%%%% OPTIONS %%%%%
numStartPoints = 200;
%randIters = 5;

approaches = [0,1,2,3] %choose between approaches 0,1,2,3
%approaches = [0] %choose between approaches 0,1,2,3
%datasets = ["BIRCH", "OAK", "FIR", "POPLAR"]
datasets = ["BIRCH", "OAK", "FIR", "POPLAR"]
%datasets = ["OAK", "BIRCH", "FIR", "POPLAR"]
%predicts = [0,1] %run code both for prediction and full fit
predicts = [0,1] %run code both for prediction and full fit
use_gompertz = 0
%%%%%%%%%%%%%%%%%%%
%predict = 1
tic
for dataset = datasets
for predict = predicts
    for approach = approaches
        close all;
        
        %Using the conversion rule 10^-3 ml = 1 mm^3 = 10^6 cells
        %we get the maximum capacity K = 1kg = 10^12 [cells] = 10^12 [cells] * 10^-6
        %[mm^3 / cells] * 10^-3 [cm^3 / mm^3] = 1000 cm^3 (=1L)
        %K = 1000;
        if dataset == "BIRCH"
            T = readtable("./data/online_data/pcbi.1009822.s006.xlsx", "Sheet", "Study3");
            patient_ids = unique(table2array(T(:, 1)));
        elseif dataset == "OAK"
            T = readtable("./data/online_data/pcbi.1009822.s006.xlsx", "Sheet", "Study4");
            patient_ids = unique(T(string(cell2mat(T.Study_Arm)) == 'Study_4_Arm_2',:).Patient_Anonmyized); 
        elseif dataset == "FIR"
            T = readtable("./data/online_data/pcbi.1009822.s006.xlsx", "Sheet", "Study1");
            patient_ids = unique(table2array(T(:, 1)));
        elseif dataset == "POPLAR"
            T = readtable("./data/online_data/pcbi.1009822.s006.xlsx", "Sheet", "Study2");
            patient_ids = unique(T(string(cell2mat(T.Study_Arm)) == 'Study_2_Arm_2',:).Patient_Anonmyized);     
        else
            error("Dataset does not exist");
        end

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
        
        if predict == 1
            losses = ["MAE", "MSE"];
        else
            losses = ["MSE"];
        end
        
        progression_probability = zeros(1,length(patient_list_id));
        progression_GT = zeros(1,length(patient_list_id));
        
        figure_counter = 0;
        f = -1;
        
        for i = 1 : length(patient_list_id)
        %for i = 10: length(patient_list_id)
        %for i = 26:26
        %for i = 5:6

            fit_info_dic = containers.Map('KeyType', 'char', 'ValueType', 'any');

            % if approach == 3
            %     fit_info_dic("MAE") = zeros(11, 1); %approach 3 has one more parameter
            %     fit_info_dic("MSE") = zeros(11, 1); %approach 3 has one more parameter
            % else
            %     fit_info_dic("MAE") = zeros(10, 1);
            %     fit_info_dic("MSE") = zeros(10, 1);
            % end
            fit_info_dic("MAE") = {};
            fit_info_dic("MSE") = {};

            patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
        
            no_eligible = mod(no_eligible+1,4);
            t = table2array(patient_data(:,2)); %days
            y = table2array(patient_data(:,3)); %mm
        
            manual_volume = ((4/3)*pi*(y/2).^3) / 1000 %mm^3 in cm^3
        
            if no_eligible == 0
                if f ~= -1
                    if predict == 0
                        dirPath = dataset + "/figures/fits";
                        if ~exist(dirPath, 'dir')
                            mkdir(dirPath);
                        end
                        saveas(f, dirPath + "/full_approach_" + string(approach) + "_" + string(figure_counter) + ".png");
                    else
                        dirPath = dataset + "/figures/fits";
                        if ~exist(dirPath, 'dir')
                            mkdir(dirPath);
                        end
                        saveas(f, dirPath + "/prediction_approach_" + string(approach) + "_" + string(figure_counter) + ".png");
                    end
                end
                figure_counter = figure_counter + 1;
                close all;
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
        
            
            % scatter(t(interval_fit), y(interval_fit), 35, 'MarkerEdgeColor', '#0072BD', 'MarkerFaceColor', '#0072BD');
            % hold on;
            % if predict == 1
            %     interval_predict = (interval_fit(end) + 1) : length(t);
            %     scatter(t(interval_predict), y(interval_predict), 35, 'MarkerEdgeColor', '#D95319', 'MarkerFaceColor', '#D95319');
            % end
            hold on;
        
            options = optimset();%'Display','iter');
            %options = optimoptions('fmincon', 'Algorithm','sqp')
            %tol = 1e-10;
            %options = optimoptions('fmincon', 'OptimalityTolerance', tol, 'ConstraintTolerance', tol,  'StepTolerance', tol, 'FunctionTolerance', tol, 'MaxFunctionEvaluations', 3000, 'MaxIterations', 3000)
            %options = optimoptions('fmincon', 'MaxFunctionEvaluations', 1000)

            r_min_found = r_min(convert2cells(manual_volume(1)));
            r_max_found = r_max(convert2cells(manual_volume(1)));

            r_min_gompertz_found = r_min_gompertz(convert2cells(manual_volume(1)));
            r_max_gompertz_found = r_max_gompertz(convert2cells(manual_volume(1)));
        
            T_0 = 10^9;
            K_0 = 10^3;
            T_Td = convert2cells(manual_volume(1)) / T_0;
            
            ttp_index = calcTTP(y);
            if ttp_index ~= -1
                progression_GT(1,i) = 1;
                xline(t(ttp_index), "--b", "PD",'Color','red');
                hold on;
            else
                progression_GT(1,i) = 0;
            end

            progression_probability_dic = containers.Map('KeyType', 'char', 'ValueType', 'any');
            progression_probability_dic("MAE") = 0;
            progression_probability_dic("MSE") = 0;

            for loss = losses
            %gs = GlobalSearch('Display','iter');
            gs = MultiStart('UseParallel', true, 'Display', 'iter');
            startPoints = RandomStartPointSet('NumStartPoints', numStartPoints);
        
            if approach == 0
                toMin = @(parameters) eval_approach0(t(interval_fit), parameters, manual_volume(interval_fit), loss, use_gompertz);
            elseif approach == 1
                toMin = @(parameters) eval_approach1(t(interval_fit), parameters, manual_volume(interval_fit), loss, use_gompertz);
            elseif approach == 2
                toMin = @(parameters) eval_approach2(t(interval_fit), parameters, manual_volume(interval_fit), loss, use_gompertz);
            elseif approach == 3
                toMin = @(parameters) eval_approach3(t(interval_fit), parameters, manual_volume(interval_fit), loss, use_gompertz);
            end
        
            weight_sum = 0;
            progression_probability_best_fit = -1;

            if approach == 0
                %Values should all be normalized, as clearly the search space is
                %not searched well, due to large orders of magnitude! Even though
                %many points are tested, the large orders of magnitude are
                %difficult to test. This is why normalization would help
                %Very similar values to Kuznetsov 1994
                
                if use_gompertz
                    problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, 10^-2], 'objective', toMin, ...
                        'lb', [r_min_gompertz_found,0,0], 'ub', [r_max_gompertz_found, (60*r_max_found + log(T_Td * T_0))/60, (60*r_max_found + log(T_Td * T_0))/60], 'options', options);
                else
                    problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, 10^-2], 'objective', toMin, ...
                        'lb', [r_min_found,0,0], 'ub', [r_max_found, r_max_found - (log(T_Td * T_0) / (t(1) - 60)), r_max_found - (log(T_Td * T_0) / (t(1) - 60))], 'options', options);
                end


                %params = run(gs, problem, startPoints)
                [params, fval, exitflag, output, manymins] = run(gs, problem, startPoints);
                %[params, fval, exitflag, output, manymins] = run(gs, problem);
        
                if predict == 1
                    num_runs = min(length(manymins), floor(0.1*numStartPoints));
                else
                    num_runs = 1;
                end
                %num_runs = length(manymins);
        
                max_weight = 0;
                for min_index = 1 : num_runs
                    Fvalue = manymins(min_index).Fval;
                    weighting = length(manymins(min_index).X0) / Fvalue;
                    max_weight = max(max_weight, weighting);
                end
                max_weight = max_weight;
        
                for min_index = 1 : num_runs
                    params = manymins(min_index).X;
                    Fvalue = manymins(min_index).Fval;
                    weighting = length(manymins(min_index).X0) / Fvalue;
                    r = params(1);
                    mu_0 = params(2);
                    mu_star = params(3);
                    t_sim = linspace(t(1),t(end),1000);
                    y_sim = run_approach0(t_sim, params, T_Td, use_gompertz);
            
                    solpts = run_approach0(t, params, T_Td, use_gompertz);
            
                    L2_error = ((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));
                    L1_error = ((1 / length(manual_volume))*sum(abs(((convert2ml((solpts(1,:))*T_0) - manual_volume')))));
                    R2_error = calculate_R2(manual_volume', (convert2ml((solpts(1,:))*T_0)));
                    %R2_error = calculate_R2(y', LD_calc(solpts(1,:)));
            
                    %if min_index == 1 %picks top solution
                    fit_info = fit_info_dic(loss);
                    %fit_info(:,min_index) = [i, r, mu_0, mu_star, L2_error, L1_error, R2_error, Fvalue, weighting, max_weight]';
                    fit_info{min_index} = {i, params, L2_error, L1_error, R2_error, length(interval_fit), Fvalue, weighting, max_weight}';
                    fit_info_dic(loss) = fit_info;
                    %end

                    ttp_index_predicted = calcTTP_with_data(t_sim, LD_calc(y_sim(1,:)), t(interval_fit), y(interval_fit));
        
                    if min_index == 1 && loss == "MSE"
                        if ttp_index_predicted ~= -1
                            progression_probability_best_fit = 1;
                        else
                            progression_probability_best_fit = 0;
                        end
                    end

                    if ttp_index_predicted ~= -1
                        %xline(t_sim(ttp_index_predicted), ":",'Color', 'magenta');
                        %hold on;
                        %progression_probability(1,i) = progression_probability(1,i) + weighting;
                        progression_probability_dic(loss) = progression_probability_dic(loss) + weighting;
                    end
                    if max_weight > 0
                        plot(t_sim,LD_calc(y_sim(1,:)), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                        % if loss == "MAE"
                        %     plot(t_sim,LD_calc(y_sim(1,:)), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                        % elseif loss == "MSE"
                        %     plot(t_sim,LD_calc(y_sim(1,:)), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                        % else
                        %     error("Loss not implemented")
                        % end
                        %plot(t_sim,LD_calc(y_sim(2,:)), 'Color', 'green');
                        %plot(t_sim,LD_calc(y_sim(1,:)), 'Color', [0.55, 0.55, 0.55, 1]);
                        hold on;
                    end
                    weight_sum = weight_sum + weighting;
                end
            elseif approach == 1
                %Values should all be normalized, as clearly the search space is
                %not searched well, due to large orders of magnitude! Even though
                %many points are tested, the large orders of magnitude are
                %difficult to test. This is why normalization would help
                %Very similar values to Kuznetsov 1994
                if use_gompertz
                    problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, 10^-3], 'objective', toMin, ...
                        'lb', [r_min_gompertz_found,0,0], 'ub', [r_max_gompertz_found, (60*r_max_found + log(T_Td * T_0))/60, (60*r_max_found + log(T_Td * T_0))/60], 'options', options);
                else
                    problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, 10^-3], 'objective', toMin, ...
                        'lb', [r_min_found,0,0], 'ub', [r_max_found, r_max_found - (log(T_Td * T_0) / (t(1) - 60)), 1], 'options', options);
                end

                [params, fval, exitflag, output, manymins] = run(gs, problem, startPoints);
        
                if predict == 1
                    num_runs = min(length(manymins), floor(0.1*numStartPoints));
                else
                    num_runs = 1;
                end
                %num_runs = length(manymins);
        
                %computes weighting sum so we can plot directly with the weighting
                %weight_sum_pre = 0;
                max_weight = 0;
                for min_index = 1 : num_runs
                    Fvalue = manymins(min_index).Fval;
                    weighting = length(manymins(min_index).X0) / Fvalue;
                    max_weight = max(max_weight, weighting);
                end
        
                for min_index = 1 : num_runs
                    params = manymins(min_index).X;
                    Fvalue = manymins(min_index).Fval;
                    weighting = length(manymins(min_index).X0) / Fvalue;
                    r = params(1);
                    y_0 = params(2);
                    mu_2_tilde = params(3);
                    t_sim = linspace(t(1),t(end),1000);
                    y_sim = run_approach1(t_sim, params, T_Td, use_gompertz);
            
                    solpts = run_approach1(t, params, T_Td, use_gompertz);

                    L2_error = ((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));
                    L1_error = ((1 / length(manual_volume))*sum(abs(((convert2ml((solpts(1,:))*T_0) - manual_volume')))));
                    R2_error = calculate_R2(manual_volume', (convert2ml((solpts(1,:))*T_0)))
                    %R2_error = calculate_R2(y', LD_calc(solpts(1,:)));
            
                    fit_info = fit_info_dic(loss);
                    fit_info{min_index} = {i, params, L2_error, L1_error, R2_error, length(interval_fit), Fvalue, weighting, max_weight}';
                    fit_info_dic(loss) = fit_info;

                    %ttp_index_predicted = calcTTP(LD_calc(y_sim(1,:)));
                    ttp_index_predicted = calcTTP_with_data(t_sim, LD_calc(y_sim(1,:)), t(interval_fit), y(interval_fit));
        
                    if min_index == 1 && loss == "MSE"
                        if ttp_index_predicted ~= -1
                            progression_probability_best_fit = 1;
                        else
                            progression_probability_best_fit = 0;
                        end
                    end

                    if ttp_index_predicted ~= -1
                        %xline(t_sim(ttp_index_predicted), ":",'Color', 'magenta');
                        %hold on;
                        %progression_probability(1,i) = progression_probability(1,i) + (1*weighting);
                        progression_probability_dic(loss) = progression_probability_dic(loss) + weighting;
                    end
        
                    if max_weight > 0
                        plot(t_sim,LD_calc(y_sim(1,:)), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                        hold on;
                    end
        
                    weight_sum = weight_sum + weighting;
                end
            elseif approach == 2
                 Max_T_res_n = (convert2cells(manual_volume(1)) / T_0);
                 %problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, Max_T_res_n / 2], 'objective', toMin, ...
                 %   'lb', [r_min_gompertz_found,0,0], 'ub', [r_max_gompertz_found, (60*r_max_gompertz_found + log(T_Td * T_0))/60, Max_T_res_n], 'options', options);
                 if use_gompertz
                     problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, Max_T_res_n / 2], 'objective', toMin, ...
                        'lb', [r_min_gompertz_found,0,0], 'ub', [r_max_gompertz_found, r_max_gompertz_found - (log(T_Td * T_0) / (t(1) - 60)), Max_T_res_n], 'options', options);        
                     %params = run(gs, problem)
                 else
                     problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2,r_max_found / 2, Max_T_res_n / 2], 'objective', toMin, ...
                        'lb', [r_min_found,0,0], 'ub', [r_max_found, r_max_found - (log(T_Td * T_0) / (t(1) - 60)), Max_T_res_n], 'options', options);  
                 end
                 [params, fval, exitflag, output, manymins] = run(gs, problem, startPoints);
        
                if predict == 1
                    num_runs = min(length(manymins), floor(0.1*numStartPoints));
                else
                    num_runs = 1;
                end
                 %num_runs = length(manymins);
        
                 max_weight = 0;
                 for min_index = 1 : num_runs
                     Fvalue = manymins(min_index).Fval;
                     weighting = length(manymins(min_index).X0) / Fvalue;
                     max_weight = max(max_weight, weighting);
                 end
        
                 for min_index = 1 : num_runs
                     params = manymins(min_index).X;
                     Fvalue = manymins(min_index).Fval;
                     weighting = length(manymins(min_index).X0) / Fvalue;
                     r = params(1);
                     mu1 = params(2);
                     T_res_n_estimated = params(3);
            
                     t_sim = linspace(t(1),t(end),1000);
                     y_sim = run_approach2(t_sim, [r, mu1, T_res_n_estimated], convert2cells(manual_volume(1)) / T_0, use_gompertz);
            
            
                     solpts = run_approach2(t, [r, mu1, T_res_n_estimated], convert2cells(manual_volume(1)) / T_0, use_gompertz);
            
                     L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));
                     L1_error = ((1 / length(manual_volume))*sum(abs(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')))));
                     R2_error = calculate_R2(manual_volume', (convert2ml((solpts(1,:) + solpts(2,:))*T_0)))
                     %R2_error = calculate_R2(y', LD_calc(solpts(1,:) + solpts(2,:)));
            
                    fit_info = fit_info_dic(loss);
                    fit_info{min_index} = {i, params, L2_error, L1_error, R2_error, length(interval_fit), Fvalue, weighting, max_weight}';
                    fit_info_dic(loss) = fit_info;
                     %ttp_index_predicted = calcTTP(LD_calc(y_sim(1,:) + y_sim(2,:)));
                    ttp_index_predicted = calcTTP_with_data(t_sim, LD_calc(y_sim(1,:) + y_sim(2,:)), t(interval_fit), y(interval_fit));

                    if min_index == 1 && loss == "MSE"
                        if ttp_index_predicted ~= -1
                            progression_probability_best_fit = 1;
                        else
                            progression_probability_best_fit = 0;
                        end
                    end

                     if ttp_index_predicted ~= -1
                        %xline(t_sim(ttp_index_predicted), ":",'Color', 'magenta');
                        %hold on;
                        %progression_probability(1,i) = progression_probability(1,i) + (1*weighting);
                        progression_probability_dic(loss) = progression_probability_dic(loss) + weighting;
                     end
        
                     if max_weight > 0
                        plot(t_sim,LD_calc((y_sim(1,:)+y_sim(2,:))), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                        hold on;
                     end
        
                     weight_sum = weight_sum + weighting;
                 end
            elseif approach == 3
                 % combined_results = [];
                 % for rand_iter = 1:randIters
                 %     noise_level = 0.01;
                 %     errors = -(noise_level / 2) + (noise_level * rand(size(manual_volume)));
                 %     manual_volume_noisy = manual_volume .* (1 + errors);
                 %     manual_volume_noisy = max(0,manual_volume_noisy); %minimum value is 0
                 % 
                 %     toMin = @(parameters) eval_approach3(t(interval_fit), parameters, manual_volume_noisy(interval_fit), loss);
                 %     Max_T_s_n = (convert2cells(manual_volume(1)) / T_0);
                 %     problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2, 0.9999*Max_T_s_n, (60*r_max_found + log(T_Td * T_0))/30, 0], 'objective', toMin, ...
                 %       'lb', [r_min_gompertz_found,0,0,0], 'ub', [r_max_gompertz_found, Max_T_s_n, (30*r_max_found + log(T_Td * T_0))/30, Inf], 'options', options, 'nonlcon', @constraint_approach_3);
                 %     problem = createOptimProblem('fmincon', 'x0', [0, 0, 0, 0], 'objective', toMin, ...
                 %        'lb', [r_min_found,0,0,0], 'ub', [r_max_found, Max_T_s_n, r_min_found, Inf], 'options', options, 'nonlcon', @constraint_approach_3);
                 % 
                 %     params = run(gs, problem)
                 %     [params, fval, exitflag, output, manymins] = run(gs, problem);
                 %     combined_results = [combined_results, manymins];
                 % end
                 % [~, sortIdx] = sort([combined_results.Fval], 'ascend');
                 % manymins = combined_results(sortIdx);
                 % num_runs = min(length(manymins), 50);
                 % num_runs = min(length(manymins), floor(0.05*numStartPoints*randIters));
                 % num_runs = min(length(manymins), 3);
                 % num_runs = length(manymins);

                 Max_T_s_n = (convert2cells(manual_volume(1)) / T_0);
                 %problem = createOptimProblem('fmincon', 'x0', [(r_min_gompertz_found + r_max_gompertz_found)/2, 0.9999*Max_T_s_n, (60*r_max_found + log(T_Td * T_0))/30, 0], 'objective', toMin, ...
                 %  'lb', [r_min_gompertz_found,0,0,0], 'ub', [r_max_gompertz_found, Max_T_s_n, (60*r_max_found + log(T_Td * T_0))/60, 1], 'options', options, 'nonlcon', @constraint_approach_3);
                 % problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2, 0.9999*Max_T_s_n, r_max_found / 10, 0], 'objective', toMin, ...
                 %    'lb', [r_min_found,0,0,0], 'ub', [r_max_found, Max_T_s_n, r_max_found, Inf], 'options', options, 'nonlcon', @constraint_approach_3);
                 if use_gompertz
                    %problem = createOptimProblem('fmincon', 'x0', [(r_min_gompertz_found + r_max_gompertz_found)/2, 0.9999*Max_T_s_n, (60*r_max_found + log(T_Td * T_0))/30, 0], 'objective', toMin, ...
                    %    'lb', [r_min_gompertz_found,0,0,0], 'ub', [r_max_gompertz_found, Max_T_s_n, r_max_gompertz_found - (log(T_Td * T_0) / (t(1) - 60)), Inf], 'options', options, 'nonlcon', @constraint_approach_3);
                    problem = createOptimProblem('fmincon', 'x0', [(r_min_gompertz_found + r_max_gompertz_found)/2, 0.9999*Max_T_s_n, (60*r_max_found + log(T_Td * T_0))/30, 0], 'objective', toMin, ...
                        'lb', [r_min_gompertz_found,0,0,0], 'ub', [r_max_gompertz_found, Max_T_s_n, Inf, Inf], 'options', options, 'nonlcon', @constraint_approach_3);     
                 else
                    %problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2, 0.9999*Max_T_s_n, r_max_found / 10, 0], 'objective', toMin, ...
                    %    'lb', [r_min_found,0,0,0], 'ub', [r_max_found, Max_T_s_n, r_max_found - (log(T_Td * T_0) / (t(1) - 60)), Inf], 'options', options, 'nonlcon', @constraint_approach_3);
                    problem = createOptimProblem('fmincon', 'x0', [(r_min_found + r_max_found)/2, 0.9999*Max_T_s_n, r_max_found / 10, 0], 'objective', toMin, ...
                        'lb', [r_min_found,0,0,0], 'ub', [r_max_found, Max_T_s_n, r_max_found, Inf], 'options', options, 'nonlcon', @constraint_approach_3);
                    %(r_max_found - (log(T_Td * T_0) / (t(1) - 60)))
                    %(60*r_max_found + log(T_Td * T_0))/60
                 end
                 %params = run(gs, problem)
                 %[params, fval, exitflag, output, manymins] = run(gs, problem);
                 [params, fval, exitflag, output, manymins] = run(gs, problem, startPoints);
                 %num_runs = min(length(manymins), 50);
                if predict == 1
                    num_runs = min(length(manymins), floor(0.1*numStartPoints));
                else
                    num_runs = 1;
                end
                 %num_runs = min(length(manymins), 3);
                 %num_runs = length(manymins);
        
                 max_weight = 0;
                 for min_index = 1 : num_runs
                     Fvalue = manymins(min_index).Fval;
                     weighting = length(manymins(min_index).X0) / Fvalue;
                     max_weight = max(max_weight, weighting);
                 end
        
                 for min_index = 1 : num_runs
                     params = manymins(min_index).X;
                     Fvalue = manymins(min_index).Fval;
                     weighting = length(manymins(min_index).X0) / Fvalue;
                     r = params(1);
                     T_s_n_estimated = params(2);
                     E_n_estimated = params(3);
                     beta_tilde = params(4);
            
                     t_sim = linspace(t(1),t(end),1000);
                     y_sim = run_approach3(t_sim, [r, T_s_n_estimated, E_n_estimated, beta_tilde], convert2cells(manual_volume(1)) / T_0, use_gompertz);
            
            
                     solpts = run_approach3(t, [r, T_s_n_estimated, E_n_estimated, beta_tilde], convert2cells(manual_volume(1)) / T_0, use_gompertz);
            
                     L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));
                     L1_error = ((1 / length(manual_volume))*sum(abs(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')))));
                     R2_error = calculate_R2(manual_volume', (convert2ml((solpts(1,:) + solpts(2,:))*T_0)));
                     %R2_error = calculate_R2(y', LD_calc(solpts(1,:) + solpts(2,:)));
            
                    fit_info = fit_info_dic(loss);
                    fit_info{min_index} = {i, params, L2_error, L1_error, R2_error, length(interval_fit), Fvalue, weighting, max_weight}';
                    fit_info_dic(loss) = fit_info;
                     %ttp_index_predicted = calcTTP(LD_calc(y_sim(1,:) + y_sim(2,:)));
                    ttp_index_predicted = calcTTP_with_data(t_sim, LD_calc(y_sim(1,:) + y_sim(2,:)), t(interval_fit), y(interval_fit));

                    if min_index == 1 && loss == "MSE"
                        if ttp_index_predicted ~= -1
                            progression_probability_best_fit = 1;
                        else
                            progression_probability_best_fit = 0;
                        end
                    end                     
        
                     if ttp_index_predicted ~= -1
                        %progression_probability(1,i) = progression_probability(1,i) + (1*weighting);
                        progression_probability_dic(loss) = progression_probability_dic(loss) + weighting;
                     end
        
                     if max_weight > 0
                        plot(t_sim,LD_calc((y_sim(1,:)+y_sim(2,:))), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                        hold on;
                     end
                     %plot(t_sim,LD_calc(y_sim(1,:)), "Color", 'green');
                     %hold on;
                     %plot(t_sim,LD_calc(y_sim(2,:)), "Color", 'magenta');
                     %hold on;
                     % plot(t_sim,LD_calc(y_sim(3,:)*K_0),'--');
                     % hold on;
                     weight_sum = weight_sum + weighting;
                 end
            else
                error("Approach does not exist")
            end

            progression_probability_dic(loss) = progression_probability_dic(loss) / weight_sum;
            end
            %progression_probability(1,i) = progression_probability(1,i) / weight_sum;

            if predict == 1
                progression_probability(4,i) = progression_probability_best_fit;
                progression_probability(3,i) = progression_probability_dic("MSE");
                progression_probability(2,i) = progression_probability_dic("MAE");
                progression_probability(1,i) = (progression_probability_dic("MAE") + progression_probability_dic("MSE"))/2;
                %progression_probability(1,i) = (progression_probability_dic("MAE") + progression_probability_dic("MSE"))/2;
            else
                progression_probability(1,i) = progression_probability_dic("MSE");
            end

            scatter(t(interval_fit), y(interval_fit), 40, 'MarkerEdgeColor',[0.26, 0.41, 0.88], 'MarkerFaceColor', [0.26, 0.41, 0.88]); %'#0072BD'
            hold on;
            if predict == 1
                interval_predict = (interval_fit(end) + 1) : length(t);
                scatter(t(interval_predict), y(interval_predict), 40, 'MarkerEdgeColor',[1,0,0], 'MarkerFaceColor', [1,0,0]); %'#D95319'
            end
            hold on;
        
            PS = calcPS(y(interval_fit));
            yline(PS,"--b",'Color', [0.273, 0.135, 0.037, 0.7]);
            hold on;

            xlabel("Time in days");
            ylabel("LD [mm]");
            xlim([t(1), t(end)]);

            fit_info_mae = fit_info_dic("MAE");
            fit_info_mse = fit_info_dic("MSE");
            title(string(i) + ", p(PD) = " + round(progression_probability(1,i),3) + ", R^2 = " + round(fit_info_mse{1}{5},2));
            
            hold on;
        
            xline(0);
            shg;
            
            if predict == 0
                save(dataset+"/full_fit_info_" + string(i) + "_approach_"+string(approach)+".mat","fit_info_dic");
                save(dataset + "/full_progression_probability_approach_" + string(approach) + ".mat", "progression_probability");
            else
                save(dataset+"/prediction_fit_info_" + string(i) + "_approach_"+string(approach)+".mat","fit_info_dic");
                save(dataset + "/prediction_progression_probability_approach_" + string(approach) + ".mat", "progression_probability");
            end
            save(dataset + "/prediction_progression_GT.mat", "progression_GT");
        end
        
        % if predict == 0
        %     saveas(f, dataset+"/figures/fits/approach_"+string(approach)+"_"+string(figure_counter)+".png");
        %     %save("fit_info_approach_"+string(approach)+".mat","fit_info");
        % else
        %     saveas(f, dataset+"/figures/fits/prediction_approach_"+string(approach)+"_"+string(figure_counter)+".png");
        %     %save("prediction_fit_info_approach_"+string(approach)+".mat","fit_info");
        %     save(dataset+"progression_GT.mat","progression_GT");
        %     save(dataset+"progression_probability_approach_"+string(approach)+".mat","progression_probability");
        % end
        
        if predict == 0
            dirPath = dataset + "/figures/fits";
            if ~exist(dirPath, 'dir')
                mkdir(dirPath);
            end
            saveas(f, dirPath + "/full_approach_" + string(approach) + "_" + string(figure_counter) + ".png");
            %save(dataset+"/full_fit_info_approach_"+string(approach)+".mat","fit_info_dic");
            save(dataset + "/full_progression_GT.mat", "progression_GT");
            save(dataset + "/full_progression_probability_approach_" + string(approach) + ".mat", "progression_probability");
        else
            dirPath = dataset + "/figures/fits";
            if ~exist(dirPath, 'dir')
                mkdir(dirPath);
            end
            saveas(f, dirPath + "/prediction_approach_" + string(approach) + "_" + string(figure_counter) + ".png");
            save(dataset + "/prediction_progression_GT.mat", "progression_GT");
            save(dataset + "/prediction_progression_probability_approach_" + string(approach) + ".mat", "progression_probability");
            %save(dataset+"/prediction_fit_info_approach_"+string(approach)+".mat","fit_info_dic");
        end
    end
end
end
toc

function r = r_min(T) %input: tumor cells at diagnosis, output: smallest r [1/d] based on TVDT <= 300d
    r = log((1 - (T/10^12)) / (0.5 - (T/10^12))) / 400;
    assert(r >= 0);
end

function r = r_max(T) %input: tumor cells at diagnosis, output: largest r [1/d] based on TVDT >= 25
    r = log((1 - (T/10^12)) / (0.5 - (T/10^12))) / 25;
    assert(r >= 0);
end

function r = r_min_gompertz(T) %input: tumor cells at diagnosis, output: smallest r [1/d] based on TVDT <= 300d
    %r = -(log((1 - (log(2*T)/log(10^12)))) - log(1 - (log(T)/log(10^12)))) / 400;
    r = -(log((1 - (log(2*T)/log(10^12)))) - log(1 - (log(T)/log(10^12)))) / 400;
    assert(r >= 0);
end

function r = r_max_gompertz(T) %input: tumor cells at diagnosis, output: smallest r [1/d] based on TVDT <= 300d
    r = -(log((1 - (log(2*T)/log(10^12)))) - log(1 - (log(T)/log(10^12)))) / 25;
    assert(r >= 0);
end

function [c,ceq] = constraint_approach_1(params)
    T_0 = 10^9;
    r = params(1);
    mu_0 = params(2);
    mu_star = params(3);
    c = [];
    c(1) = mu_star - mu_0;
    ceq = [];
end

function [c,ceq] = constraint_approach_3(parameters)
    T_0 = 10^9;
    r = parameters(1);
    T_s_n_estimated = parameters(2);
    z_0 = parameters(3);
    beta_tilde = parameters(4);
    c = [];
    %c(1) = z_0 - r;

    c(1) = beta_tilde - (z_0 / (5*T_s_n_estimated/100));


    ceq = [];
end

% function dydt = approach0(y,t,r,mu_star,use_gompertz) %using cells
%     T_0 = 10^9;
%     K_0 = 10^3; %10^12 / T_0 = 10^3
%     T = y(1);
%     mu = y(2);
%     if t < 0
%         mu = 0;
%         mu_star = 0;
%     end
%     dydt = zeros(2,1);
%     if use_gompertz
%         dydt(1) = r * T * (-log(T / K_0)) - mu * T;
%     else
%         dydt(1) = r * T * (1 - (T / K_0)) - mu * T;
%     end
%     dydt(2) = -mu_star * mu;
% end

% function solpts = run_approach0(days,parameters,initial_tumor_size,use_gompertz) %accepts as parameters only normalized values
% T_0 = 10^9;
% K_0 = 10^3;
% r = parameters(1);
% mu_0 = parameters(2);
% mu_star = parameters(3);
% tspan = [days(1), days(end)];
% y0 = [initial_tumor_size, mu_0];
% sol = ode23s(@(t,y) approach0(y,t,r,mu_star,use_gompertz),tspan,y0);
% solpts = deval(sol, days);
% end

% function L2_error = eval_approach0(days,parameters,manual_volume, loss, use_gompertz) %parameters: normalized, manual_volume: not normalized
% T_0 = 10^9;
% initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
% solpts = run_approach0(days, parameters, initial_tumor_size, use_gompertz);
% %L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml(solpts(1,:)*T_0) - manual_volume'))));
% %L2_error = sqrt((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));
% %L2_error = ((1 / length(manual_volume))*sum(((log(convert2ml(solpts(1,:)*T_0)+1) - log(manual_volume'+1))).^2));

% % noise_level = 0.08;
% % errors = -(noise_level / 2) + (noise_level * rand(size(manual_volume)));
% % manual_volume = manual_volume .* (1 + errors);
% % manual_volume = max(0,manual_volume); %minimum value is 0

% %y(interval_fit) = min(97,y(interval_fit)); %maximum value is 97mm
% if ~isreal(solpts)
%     L2_error = Inf;
% else
%     if loss == "MAE"
%         L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml(solpts(1,:)*T_0) - manual_volume'))));
%     elseif loss == "MSE"
%         L2_error = sqrt((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));   
%     end
% end

% end

% function L2_error = eval_approach1(days,parameters,manual_volume,loss,use_gompertz) %parameters: normalized, manual_volume: not normalized
% T_0 = 10^9;
% K_0 = 10^3;
% initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
% solpts = run_approach1(days, parameters, initial_tumor_size,use_gompertz);
% %L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml(solpts(1,:)*T_0) - manual_volume'))));
% %L2_error = ((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));
% %L2_error = ((1 / length(manual_volume))*sum(((log(convert2ml(solpts(1,:)*T_0)+1) - log(manual_volume'+1))).^2));
% if ~isreal(solpts)
%     L2_error = Inf;
% else
%     if loss == "MAE"
%         L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml(solpts(1,:)*T_0) - manual_volume'))));
%     elseif loss == "MSE"
%         L2_error = sqrt((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));   
%     end
% end
% end

% function solpts = run_approach1(days,parameters,initial_tumor_size,use_gompertz) %accepts as parameters only normalized values
% T_0 = 10^9;
% K_0 = 10^3;
% r = parameters(1);
% y_0 = parameters(2);
% mu_2_tilde = parameters(3);
% tspan = [days(1), days(end)];
% y0 = [initial_tumor_size, y_0];
% sol = ode23s(@(t,y) approach1(y,t,r,mu_2_tilde,use_gompertz),tspan,y0);
% solpts = deval(sol, days);
% end

% function dydt = approach1(y,t,r,mu_2_tilde,use_gompertz) %using normalized values
%     T_0 = 10^9;
%     K_0 = 10^3;
%     T_n = y(1);
%     E_n = y(2);
%     if t < 0
%         E_n = 0;
%         mu_2_tilde = 0;
%     end
%     dydt = zeros(2,1);
%     if use_gompertz
%         dydt(1) = r * T_n * (-log(T_n / K_0)) - T_n * E_n;
%     else
%         dydt(1) = r * T_n * (1 - (T_n / K_0)) - T_n * E_n;
%     end
%     dydt(2) = -mu_2_tilde * T_n * E_n;
% end

% function dydt = approach2(y,t,r,mu1,use_gompertz) %using cells
%     T_0 = 10^9;
%     K_0 = 10^3; %10^12 / T_0 = 10^3
%     if t < 0
%         mu1 = 0;
%     end
%     dydt = zeros(2,1);
%     T_p_n = y(1);
%     T_res_n = y(2);
%     %dydt(1) = r * T_p_n * (1 - ((T_p_n + T_res_n) / K_0)) - mu1 * T_p_n;
%     %dydt(2) = r * T_res_n * (1 - ((T_p_n + T_res_n) / K_0));
%     if use_gompertz
%         dydt(1) = r * T_p_n * (-log(((T_p_n + T_res_n) / K_0))) - mu1 * T_p_n;
%         dydt(2) = r * T_res_n * (-log(((T_p_n + T_res_n) / K_0)));
%     else
%         dydt(1) = r * T_p_n * (1 - ((T_p_n + T_res_n) / K_0)) - mu1 * T_p_n;
%         dydt(2) = r * T_res_n * (1 - ((T_p_n + T_res_n) / K_0));
%     end
% end

% function solpts = run_approach2(days,parameters,initial_tumor_size,use_gompertz) %accepts as parameters only normalized values
% T_0 = 10^9;
% K_0 = 10^3;
% r = parameters(1);
% mu1 = parameters(2);
% T_res_n_estimated = parameters(3);
% tspan = [days(1), days(end)];
% y0 = [initial_tumor_size - T_res_n_estimated, T_res_n_estimated];
% sol = ode23s(@(t,y) approach2(y,t,r,mu1,use_gompertz),tspan,y0);
% solpts = deval(sol, days);
% end

% function L2_error = eval_approach2(days,parameters,manual_volume,loss,use_gompertz) %parameters: normalized, manual_volume: not normalized
% T_0 = 10^9;
% initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
% solpts = run_approach2(days, parameters, initial_tumor_size,use_gompertz);

% if ~isreal(solpts)
%     L2_error = Inf;
% else
%     %L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume'))));
%     %L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));
%     %L2_error = ((1 / length(manual_volume))*sum(((log(convert2ml((solpts(1,:) + solpts(2,:))*T_0)+1) - log(manual_volume'+1))).^2));
    
%     % noise_level = 0.001;
%     % errors = -(noise_level / 2) + (noise_level * rand(size(manual_volume)));
%     % manual_volume = manual_volume .* (1 + errors);
%     % manual_volume = max(0,manual_volume); %minimum value is 0
    
%     if loss == "MAE"
%         L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume'))));
%     elseif loss == "MSE"
%         L2_error = sqrt((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));   
%     end
% end
% end

% function R2_error = calculate_R2(y_true, y_fit)
%     y_mean = sum(y_true) / length(y_true);
%     ss_res = sum((y_true - y_fit).^2);
%     ss_tot = sum((y_true - y_mean).^2);
%     R2_error = 1 - (ss_res / ss_tot);
% end

% function dydt = approach3(y,t,r,beta_tilde,use_gompertz) %using cells
%     T_0 = 10^9;
%     K_0 = 10^3; %10^12 / T_0 = 10^3
%     dydt = zeros(3,1);
%     T_p_n = y(1);

%     T_res_n = y(2);
%     %T_res_n = 0;
%     %beta_tilde = 0;

%     E_n = y(3);

%     if t < 0
%         E_n = 0;
%         beta_tilde = 0;
%     end

%     if use_gompertz
%         dydt(1) = r * T_p_n * (-log((T_p_n + T_res_n) / K_0)) - E_n * T_p_n;
%         dydt(2) = r * T_res_n * (-log((T_p_n + T_res_n) / K_0));
%     else
%         dydt(1) = r * T_p_n * (1 - ((T_p_n + T_res_n) / K_0)) - E_n * T_p_n;
%         dydt(2) = r * T_res_n * (1 - ((T_p_n + T_res_n) / K_0));
%     end

%     % if E_n >= r - (log((T_p_n + T_res_n) * T_0) / (t(1) - 60))
%     %     dydt(3) = 0;
%     % else
%     %     dydt(3) = beta_tilde * E_n * T_p_n;
%     % end
%     dydt(3) = beta_tilde * E_n * T_p_n;
% end

% function solpts = run_approach3(days,parameters,initial_tumor_size,use_gompertz) %accepts as parameters only normalized values
% r = parameters(1);
% T_s_n_estimated = parameters(2);
% z_0 = parameters(3);
% beta_tilde = parameters(4);
% tspan = [days(1), days(end)];
% y0 = [T_s_n_estimated, initial_tumor_size - T_s_n_estimated, z_0];
% sol = ode23s(@(t,y) approach3(y,t,r,beta_tilde,use_gompertz),tspan,y0);
% solpts = deval(sol, days);
% end

% function L2_error = eval_approach3(days,parameters,manual_volume,loss,use_gompertz) %parameters: normalized, manual_volume: not normalized
% T_0 = 10^9;
% initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
% solpts = run_approach3(days, parameters, initial_tumor_size,use_gompertz);
% %L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume'))));
% %L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));
% %L2_error = ((1 / length(manual_volume))*sum(((log(convert2ml((solpts(1,:) + solpts(2,:))*T_0)+1) - log(manual_volume'+1))).^2));

% % noise_level = 0.01;
% % errors = -(noise_level / 2) + (noise_level * rand(size(manual_volume)));
% % manual_volume = manual_volume .* (1 + errors);
% % manual_volume = max(0,manual_volume); %minimum value is 0
% %if ~isreal(solpts)

% %r = parameters(1);
% %if any(solpts(3,:) >= (r - (log(initial_tumor_size * T_0) / (days(1) - 60))))
% if false
%     L2_error = 100000;
% else
%     if loss == "MAE"
%         L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume'))));
%     elseif loss == "MSE"
%         L2_error = sqrt((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));   
%         %L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));   
%     end
% end

% end

% function LD = LD_calc(volume)
%     LD = 2*nthroot((3/(4*pi))*1000*convert2ml(volume*10^9),3);
%     %LD = 2*nthroot((3/(4*pi))*1000*convert2ml(volume*10^9),3);
% end

% function index = calcTTP(y) %returns -1 if no progression, otherwise index in y array of progression
%     %%%% Input y: tumor LD measurements in mm over time
%     %%%%%%%%%%%%%%%% TTP CALC (very strict, PD definition on one lesion instead of 5) %%%%%%%%%%%%%%%%
%     index = -1
%     nadir = min(y);
%     nadir_index = min(find(y == nadir));
%     criterium = 1.2*nadir;
%     manual_volume_after_response = y((nadir_index+1) : end);
%     condition = (manual_volume_after_response >= criterium) & (manual_volume_after_response - y(nadir_index)) >= 5;
%     if any(condition)
%         ttp_index = min(find(condition));
%         index = ttp_index + nadir_index;
%     end
% end

% function index = calcTTP_with_data_old(t,y,t_measured,y_measured) %returns -1 if no progression, otherwise index in y array of progression
%     %%%% Input y: tumor LD measurements in mm over time
%     %%%%%%%%%%%%%%%% TTP CALC (very strict, PD definition on one lesion instead of 5) %%%%%%%%%%%%%%%%
%     %first we truncate simulation data
%     [~, index] = min(abs(t - t_measured(end)));
%     t_sim_measured = t(1:index);
%     y_sim_measured = y(1:index);

%     index = -1;
%     nadir_y = min(y_sim_measured);
%     nadir_y_measured = min(y_measured);
%     if nadir_y <= nadir_y_measured
%         nadir = nadir_y;
%         nadir_index = min(find(y == nadir));
%     else
%         nadir = nadir_y_measured;
%         nadir_timepoint = min(t_measured(y_measured == nadir));
%         disp(nadir_timepoint)
%         % Assuming the timepoints in t and t_measured are comparable
%         % probably code below doesn't make sense
%         [~, nadir_index] = min(abs(t - nadir_timepoint));
%     end
%     criterium = 1.2*nadir;
%     manual_volume_after_response = y((nadir_index) : end);

%     condition = (manual_volume_after_response >= criterium) & (manual_volume_after_response - nadir) >= 5;
%     if any(condition)
%         ttp_index = min(find(condition));
%         index = ttp_index + nadir_index;
%     end
% end

% function index = calcTTP_with_data(t,y,t_measured,y_measured) %returns -1 if no progression, otherwise index in y array of progression
%     %%%% Input y: tumor LD measurements in mm over time
%     %%%%%%%%%%%%%%%% TTP CALC (very strict, PD definition on one lesion instead of 5) %%%%%%%%%%%%%%%%
%     %first we truncate simulation data
%     [~, index] = min(abs(t - t_measured(end)));
%     t_sim_measured = t(1:index);
%     y_sim_measured = y(1:index);

%     index = -1;
%     nadir = min(y_measured(y_measured > 0));

%     if ~any(nadir)
%         nadir = 0;
%     end

%     nadir_timepoint = min(t_measured(y_measured == nadir));
%     %disp(nadir_timepoint)
%     % Assuming the timepoints in t and t_measured are comparable
%     % probably code below doesn't make sense
%     [~, nadir_index] = min(abs(t - nadir_timepoint));
%     criterium = 1.2*nadir;
%     manual_volume_after_response = y((nadir_index) : end);

%     condition = (manual_volume_after_response >= criterium) & (manual_volume_after_response - nadir) >= 5;
%     if any(condition)
%         ttp_index = min(find(condition));
%         index = ttp_index + nadir_index;
%     end
% end

% function y = safe_log(x)
%     %y = log(((x>0)*x) + 10^-50);
%     y = log(x);
% end