close all;
clear all;

approaches = [0,1,2,3] %choose between approaches 0,1,2,3
datasets = ["BIRCH", "OAK", "FIR", "POPLAR"]
predicts = [0,1] %run code both for prediction and full fit
plot_PD = true
compute_ML = false
save_ML = true

chosen_patients = [1,2,3,4,5];
chosen_dataset = "POPLAR";
predict_all_patients = false;

tic
diary console.txt
%table creation
median_R2 = [];
median_MAE = [];
median_AIC = [];
median_AICc = [];

tables = {};
model_performance_dataset = containers.Map('KeyType', 'char', 'ValueType', 'any');
model_performance_dataset_fullcert = containers.Map('KeyType', 'char', 'ValueType', 'any');
progression_predictions_static_dataset = containers.Map('KeyType', 'char', 'ValueType', 'any');
progression_probability_dataset_approaches = {};

t_data_all = {};
y_data_all = {};
Y_data_all = [];
length_to_predict_all = [];
distance_to_pd_threshold_all = [];
t_data_dataset = containers.Map('KeyType', 'char', 'ValueType', 'any');
y_data_dataset = containers.Map('KeyType', 'char', 'ValueType', 'any');
Y_data_dataset = containers.Map('KeyType', 'char', 'ValueType', 'any');
pd_time_dataset = containers.Map('KeyType', 'char', 'ValueType', 'any');
for dataset = datasets
    t_data_dataset(dataset) = {};
    y_data_dataset(dataset) = {};
    Y_data_dataset(dataset) = [];
    pd_time_dataset(dataset) = [];
end

params_all = cell(4, 2);
MSEs_all = cell(4, 2);
MAEs_all = cell(4, 2);
R2s_all = cell(4, 2);
AICs_all = cell(4, 2);
AIC_cs_all = cell(4, 2);
for dataset = datasets
    %assert((length(dir(dataset + "/full_fit_info_*_approach_0.mat")) == length(dir(dataset + "/full_fit_info_*_approach_1.mat"))) && (length(dir(dataset + "/full_fit_info_*_approach_1.mat")) == length(dir(dataset + "/full_fit_info_*_approach_2.mat"))) && (length(dir(dataset + "/full_fit_info_*_approach_2.mat")) == length(dir(dataset + "/full_fit_info_*_approach_3.mat"))));    
    %for mode = ["full", "predict"]
    load(dataset + "/prediction_progression_GT.mat");
    for mode = ["predict"]
        assert((length(dir(dataset + "/" + mode + "_fit_info_*_approach_0.mat")) == length(dir(dataset + "/" + mode + "_fit_info_*_approach_1.mat"))) && (length(dir(dataset + "/" + mode + "_fit_info_*_approach_1.mat")) == length(dir(dataset + "/" + mode + "_fit_info_*_approach_2.mat"))) && (length(dir(dataset + "/" + mode + "_fit_info_*_approach_2.mat")) == length(dir(dataset + "/" + mode + "_fit_info_*_approach_3.mat"))));    
    end
    numOfPatients = length(dir(dataset + "/prediction_fit_info_*_approach_0.mat"));

    progression_predictions_static = zeros(1, numOfPatients);

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

    assert(numOfPatients == length(patient_list_id))

    if predict_all_patients
        numOfPatientsLoop = numOfPatients;
        chosen_dataset = dataset;
    else
        numOfPatientsLoop = 1;
    end

    increment=5;
    full_patient_list = linspace(1,numOfPatientsLoop,numOfPatientsLoop);
    %set(gcf, 'PaperPositionMode', 'auto');
    for chosen_patients_first = 1:increment:numOfPatientsLoop

    if predict_all_patients
        chosen_patients = full_patient_list(chosen_patients_first:min(chosen_patients_first + increment - 1, end));
    end

    for patient = 1 : numOfPatients
        patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(patient))), :);  
        t = table2array(patient_data(:,2)); %days
        y = table2array(patient_data(:,3)); %mm
        
        interval_fit = 1 : (length(t) - floor(length(t) / 3));
        interval_predict = (interval_fit(end) + 1) : length(t);

        t_data_all{end+1} = t(interval_fit);
        t_data_dataset(dataset) = [t_data_dataset(dataset), {t(interval_fit)}];
        y_data_all{end+1} = y(interval_fit);
        y_data_dataset(dataset) = [y_data_dataset(dataset), {y(interval_fit)}];
        
        length_to_predict_all(end+1) = t(interval_predict(end)) - t(interval_fit(end));
        

        TTP = calcTTP(y);

        distance_to_pd_threshold_all(end+1) = calcTTP_distance_old(y);

        if TTP ~= -1
            Y_data_all(end+1) = 1;
            Y_data_dataset(dataset) = [Y_data_dataset(dataset), 1];
            assert(progression_GT(patient) == 1);
            pd_time_dataset(dataset) = [pd_time_dataset(dataset), t(TTP)];
            %distance_to_pd_threshold_all(end+1) = calcTTP_distance(y);
        else
            Y_data_all(end+1) = 0;
            Y_data_dataset(dataset) = [Y_data_dataset(dataset), 0];
            assert(progression_GT(patient) == 0);
        end
    end

    % load(dataset + "/prediction_progression_GT.mat");
    for predict = predicts
        close all;
        subfigure_counter = 1;
        if chosen_dataset == dataset
            f = figure();
            %set(gcf, 'Position', [100, 100, 1000, 220 * length(chosen_patients)]);
            set(gcf, 'Position', [100, 100, 1000, 220 * length(chosen_patients)]);
        end
        % progression_probability_approaches = {};
        for approach = approaches
        %% init vars
        if approach == 3
            numParameters = 4;
        else
            numParameters = 3;
        end
        MSEs = [];
        MAEs = [];
        R2s = [];
        AICs = [];
        BICs = [];
        AIC_cs = [];

        for patient = 1 : numOfPatients
            if predict == 0
                load(dataset + "/full_fit_info_" + patient + "_approach_" + approach + ".mat");
                load(dataset + "/full_progression_probability_approach_" + approach + ".mat");
            else
                load(dataset + "/prediction_fit_info_" + patient + "_approach_" + approach + ".mat");
                load(dataset + "/prediction_progression_probability_approach_" + approach + ".mat");
            end

            patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(patient))), :);
            t = table2array(patient_data(:,2)); %days
            y = table2array(patient_data(:,3)); %mm

            if predict == 0
                interval_fit = 1 : length(t);
                %t_fit = t;
            else
                interval_fit = 1 : (length(t) - floor(length(t) / 3));
                %t_fit = t(1 : (length(t) - floor(length(t) / 3)));
            end

        
            ttp_index_static = calcTTP(y(interval_fit));
            if ttp_index_static ~= -1
                progression_predictions_static(1,patient) = 1;
            else
                progression_predictions_static(1,patient) = 0;
            end
            %manual_volume = ((4/3)*pi*(y/2).^3) / 1000 %mm^3 in cm^3
            manual_volume = ld2vl(y);
            %end

            if plot_PD
                solutions_to_consider = length(fit_info_dic("MSE")) + length(fit_info_dic("MAE"));
            else
                solutions_to_consider = 1;
            end


            fit_arrays = {fit_info_dic("MSE"), fit_info_dic("MAE")};

            if plot_PD
                fit_lengths = [length(fit_info_dic("MSE")), length(fit_info_dic("MAE"))];
            else
                fit_lengths = [1,0];
            end

            for fit_array_id = 1 : length(fit_arrays)
                fit_array = fit_arrays{fit_array_id};
                fit_length = fit_lengths(fit_array_id);
                for fit_id = 1 : fit_length
                    fit_info = fit_array{fit_id};
                    i = fit_info{1};
                    assert(i == patient);
                    params = fit_info{2};
                    L2 = fit_info{3};
                    L1_error = fit_info{4};
                    R2_error = fit_info{5};
                    n = fit_info{6};
                    weighting = fit_info{8};
                    max_weight = fit_info{9};

                    if fit_array_id == 1 && fit_id == 1
                        MSEs(end+1) = L2;
                        MAEs(end+1) = L1_error;
                        R2s(end+1) = R2_error;
                        AICs(end+1) = AIC(numParameters, n, L2);
                        BICs(end+1) = BIC(numParameters, n, L2);
                        AIC_cs(end+1) = AICc(numParameters, n, L2);

                        MSEs_all{approach+1,predict+1}(end+1) = L2;
                        MAEs_all{approach+1,predict+1}(end+1) = L1_error;
                        R2s_all{approach+1,predict+1}(end+1) = R2_error;
                        AICs_all{approach+1,predict+1}(end+1) = AIC(numParameters, n, L2);
                        AIC_cs_all{approach+1,predict+1}(end+1) = AICc(numParameters, n, L2);

                        % if patient == 1
                        %     params_all{approach+1, predict+1} = params';
                        % else
                        %     params_all{approach+1, predict+1}(:,end+1) = params';
                        % end
                        params_all{approach+1, predict+1}(:,end+1) = params';

                       
                        % if predict == 0
                        %     MSEs_all_full(end+1) = L2;
                        %     MAEs_all_full(end+1) = L1_error;
                        %     R2s_all_full(end+1) = R2_error;
                        %     AICs_all_full(end+1) = AIC(numParameters, n, L2);
                        %     AIC_cs_all_full(end+1) = AICc(numParameters, n, L2);
                        % else
                        %     MSEs_all_predict(end+1) = L2;
                        %     MAEs_all_predict(end+1) = L1_error;
                        %     R2s_all_predict(end+1) = R2_error;
                        %     AICs_all_predict(end+1) = AIC(numParameters, n, L2);
                        %     AIC_cs_all_predict(end+1) = AICc(numParameters, n, L2);
                        % end
                    end

                    if (dataset == chosen_dataset && ismember(patient, chosen_patients))
                        patient_index = find(chosen_patients == patient);
                        %subplot(length(approaches), length(chosen_patients),subfigure_counter,'Parent',f);
                        %subplot(length(chosen_patients), length(approaches), (approach+1) + (4 * subfigure_counter - 4),'Parent',f);
                        subplot(length(chosen_patients), length(approaches), (patient_index - 1) * length(approaches) + (approach+1),'Parent',f);
                        hold on;
                        % scatter(t(interval_fit), manual_volume(interval_fit), 40, 'MarkerEdgeColor',[0.26, 0.41, 0.88], 'MarkerFaceColor', [0.26, 0.41, 0.88]); %'#0072BD'
                        % hold on;
                        % if predict == 1
                        %     interval_predict = (interval_fit(end) + 1) : length(t);
                        %     %scatter(t(interval_predict), manual_volume(interval_predict), 40, 'MarkerEdgeColor',[1,0,0], 'MarkerFaceColor', [1,0,0]); %'#D95319'
                        %     scatter(t(interval_predict), manual_volume(interval_predict), 40, 'MarkerEdgeColor',[0.25, 0.88, 0.82], 'MarkerFaceColor', [0.25, 0.88, 0.82]); %'#D95319'
                        % end
                        hold on;
                    
                        PS = calcPS(y(interval_fit));
                        if predict == 1
                            yline(ld2vl(PS),"--b",'Color', [0.273, 0.135, 0.037, 0.7]);
                        end
                        hold on;
                        
                        xlabel("Time in days");
                        ylabel("Tumor size [cm^3]");
                        xlim([t(1), t(end)]);
    
                        T_0 = 10^9;
                        K_0 = 10^3;
                        T_Td = convert2cells(manual_volume(1)) / T_0;
                        y
                        if approach == 0
                            t_sim = linspace(t(1),t(end),1000);
                            y_sim = run_approach0(t_sim, params, T_Td, false);
                            solpts = run_approach0(t, params, T_Td, false);
                            plot(t_sim,convert2ml(y_sim(1,:)*T_0), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                            %plot(t_sim,convert2ml(y_sim(1,:)*T_0), 'Color', [0.55, 0.55, 0.55, 1]);
                        elseif approach == 1
                            t_sim = linspace(t(1),t(end),1000);
                            y_sim = run_approach1(t_sim, params, T_Td, false);
                            solpts = run_approach1(t, params, T_Td, false);
                            plot(t_sim,convert2ml(y_sim(1,:)*T_0), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                            %plot(t_sim,convert2ml(y_sim(1,:)*T_0), 'Color', [0.55, 0.55, 0.55, 1]);
                        elseif approach == 2
                            t_sim = linspace(t(1),t(end),1000);
                            y_sim = run_approach2(t_sim, params, T_Td, false);
                            solpts = run_approach2(t, params, T_Td, false);
                            plot(t_sim,convert2ml((y_sim(1,:) + y_sim(2,:))*T_0), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                            %plot(t_sim,convert2ml((y_sim(1,:) + y_sim(2,:))*T_0), 'Color', [0.55, 0.55, 0.55, 1]);
                        elseif approach == 3
                            t_sim = linspace(t(1),t(end),1000);
                            y_sim = run_approach3(t_sim, params, T_Td, false);
                            solpts = run_approach3(t, params, T_Td, false);
                            plot(t_sim,convert2ml((y_sim(1,:) + y_sim(2,:))*T_0), 'Color', [0.55, 0.55, 0.55, weighting / max_weight]);
                            %plot(t_sim,convert2ml((y_sim(1,:) + y_sim(2,:))*T_0), 'Color', [0.55, 0.55, 0.55, 1]);
                        end

                        scatter(t(interval_fit), manual_volume(interval_fit), 40, 'MarkerEdgeColor',[0.047, 0.482, 0.863], 'MarkerFaceColor', [0.047, 0.482, 0.863]); %'#0072BD'
                        hold on;
                        if predict == 1
                            interval_predict = (interval_fit(end) + 1) : length(t);
                            %scatter(t(interval_predict), manual_volume(interval_predict), 40, 'MarkerEdgeColor',[1,0,0], 'MarkerFaceColor', [1,0,0]); %'#D95319'
                            scatter(t(interval_predict), manual_volume(interval_predict), 40, 'MarkerEdgeColor',[1.0, 0.529, 0.039], 'MarkerFaceColor', [1.0, 0.529, 0.039]); %'#D95319'
                        end

                        if fit_array_id == 1 && fit_id == 1
                            %title(string(i) + ", R^2 = " + string(round(R2_error,2)) + ", MAE: " + string(round(L1_error, 3)));
                            title_str = string(i) + ", R^2 = " + string(round(R2_error,2)); 
                            if predict == 1
                                title_str = string(i) + ", p(PD) = " + string(round(progression_probability(1,patient),2));
                            end
                            title(title_str);
                            hold on;
                            xline(0);
                            shg;
                            subfigure_counter = subfigure_counter + 1;
                        end
                    end
                end
            end


        end
        median_R2(approach+1) = median(R2s);
        median_MAE(approach+1) = median(MAEs);
        median_AIC(approach+1) = median(AICs);
        median_AICc(approach+1) = median(AIC_cs);
    end
    table_approaches = table(approaches', median_R2', median_MAE', median_AIC', median_AICc', 'VariableNames', {'Approach', 'Median R^2', 'Median MAE', 'Median AIC', 'Median AICc'});
    table_with_description = struct('Description', 'Dataset: ' + dataset + ', Predict: ' + predict, 'Data', table_approaches);

    tables{end+1} = table_with_description;

    if predict == 1
        progression_predictions_static_dataset(dataset) = progression_predictions_static;
    end

    if chosen_dataset == dataset
        saveas(f, "all_plots/chosen_patient_" + chosen_patients(1) + "_" + chosen_dataset + "_predict_" + predict + ".png");
    end

    end
    %plot_certainty((progression_probability_approaches{2}(1,:) + progression_probability_approaches{3}(1,:) + progression_probability_approaches{4}(1,:))/3, progression_GT(1,:), progression_predictions_static);
end
end

f = figure();
h = histogram([pd_time_dataset("BIRCH"), pd_time_dataset("FIR"), pd_time_dataset("POPLAR"), pd_time_dataset("OAK")], 15, 'FaceColor', [0.8500 0.3250 0.0980], 'FaceAlpha', 0.8);
xlabel('Days to PD'); % X-axis label
ylabel('Amount of Patients'); % Y-axis label
title('Time to Progressive Disease Distribution'); % Title

f = figure();
% distance_x = linspace(0,6,10000);
% cumulative_num_patients = [];
% for x = distance_x
%     cumulative_num_patients(end+1) = sum(abs(distance_to_pd_threshold_all) <= x);
% end
% plot(distance_x, cumulative_num_patients)
h = histogram(abs(distance_to_pd_threshold_all), 20, "BinLimits", [0,10], "Normalization", "cumcount", "FaceColor", [0.4940 0.1840 0.5560], "FaceAlpha", 0.8);
xlabel('Absolute LD [mm] distance from PD threshold'); % X-axis label
ylabel('Cumulative Count of Patients'); % Y-axis label
title("Absolute Distance from PD Threshold (cumulative)"); % Title


for predict = predicts
    %median_MSE_all = [];
    median_R2_all = [];
    median_MAE_all = [];
    median_AIC_all = [];
    median_AICc_all = [];

    for approach = approaches
       %median_MSE_all(approach+1) = median(MSEs_all{approach+1,predict+1});
        median_MAE_all(approach+1) = median(MAEs_all{approach+1,predict+1});
        median_R2_all(approach+1) = median(R2s_all{approach+1,predict+1});
        median_AIC_all(approach+1) = median(AICs_all{approach+1,predict+1});
        median_AICc_all(approach+1) = median(AIC_cs_all{approach+1,predict+1});
    end

    table_approaches = table(approaches', median_R2_all', median_MAE_all', median_AIC_all', median_AICc_all', 'VariableNames', {'Approach', 'Median R^2', 'Median MAE', 'Median AIC', 'Median AICc'});
    table_with_description = struct('Description', 'Dataset: ALL, Predict: ' + string(predict), 'Data', table_approaches);
    tables{end+1} = table_with_description;
end


patients_with_better_AICc_model4_vs_3 = (AIC_cs_all{4,1} < AIC_cs_all{3,1}) .* (linspace(1,410,410));

%%% Compute where AICc is better than AIC

% if predict == 0
%     table_approaches = table(approaches', 
% else
% end


%%%%%% Compute ML predictions
if compute_ML
    [predictions_interpolated, models_interpolated, predictions_padded, models_padded, c] = compute_ML_predictions(t_data_all, y_data_all, Y_data_all);
    predictions_dataset = {predictions_interpolated, models_interpolated, predictions_padded, models_padded, c};
    save("ALL_ML_predictions_" + string(datetime("now")) + ".mat", "predictions_dataset");

    for dataset = datasets
        [predictions_interpolated, models_interpolated, predictions_padded, models_padded, c] = compute_ML_predictions(t_data_dataset(dataset), y_data_dataset(dataset), Y_data_dataset(dataset));
        predictions_dataset = {predictions_interpolated, models_interpolated, predictions_padded, models_padded, c};
        save(dataset + "/ML_predictions_" + string(datetime("now")) + ".mat", "predictions_dataset");
    end

    % for dataset = ["POPLAR"]
    %     [predictions_interpolated, models_interpolated, predictions_padded, models_padded, c] = compute_ML_predictions(t_data_dataset(dataset), y_data_dataset(dataset), Y_data_dataset(dataset));
    %     predictions_dataset = {predictions_interpolated, models_interpolated, predictions_padded, models_padded, c};
    %     save(dataset + "/ML_predictions.mat", "predictions_dataset");
    % end
end

progression_GT_all = [];
progression_probability_approaches_all = {};
progression_predictions_static_all = [];
for approach = approaches
    progression_probability_approaches_all{approach+1} = [];
end

for dataset = [datasets, "ALL"]
    model_performance = containers.Map('KeyType', 'char', 'ValueType', 'any');
    model_performance_all = containers.Map('KeyType', 'char', 'ValueType', 'any');
    model_performance_fullcert = containers.Map('KeyType', 'char', 'ValueType', 'any');

    if dataset == "ALL"
        progression_GT = progression_GT_all;
        progression_predictions_static_dataset("ALL") = progression_predictions_static_all;
    else
        load(dataset + "/prediction_progression_GT.mat");
        progression_GT_all = [progression_GT_all, progression_GT];
        progression_predictions_static_all = [progression_predictions_static_all, progression_predictions_static_dataset(dataset)];
    end

    progression_probability_approaches = {};

    if save_ML
        if dataset == "ALL"
            load("ALL_ML_predictions.mat");
            model_performance("ML model (interpolated)") = compute_pred_metrics(progression_GT(1,:), predictions_dataset{1});
            model_performance("ML model (padded)") = compute_pred_metrics(progression_GT(1,:), predictions_dataset{3});
        else
            load(dataset + "/ML_predictions.mat");
            model_performance("ML model (interpolated)") = compute_pred_metrics(progression_GT(1,:), predictions_dataset{1});
            %model_performance("ML model (padded)") = compute_pred_metrics(progression_GT(1,:), predictions_dataset{3});
        end
    end

    for approach = approaches
        if dataset == "ALL"
            progression_probability_approaches = progression_probability_approaches_all;
            progression_probability = progression_probability_approaches{approach+1};
        else
            load(dataset + "/prediction_progression_probability_approach_" + approach + ".mat");
            progression_probability_approaches_all{approach+1} = [progression_probability_approaches_all{approach+1}, progression_probability];
            progression_probability_approaches{approach+1} = progression_probability;
        end

        perf_fullcert_approach = compute_performance_certainty_full(1, progression_probability_approaches{approach+1}(1,:), progression_GT(1,:), progression_predictions_static_dataset(dataset), predictions_dataset{1});
        model_performance_fullcert("Approach " + string(approach)) = perf_fullcert_approach;

        model_performance("Approach " + string(approach) + " (using only best fit)") = compute_pred_metrics(progression_GT(1,:), progression_probability(4,:));
        model_performance("Approach " + string(approach) + " (MSE weighting)") = compute_pred_metrics(progression_GT(1,:), double(progression_probability(3,:) > 0.5));
        model_performance("Approach " + string(approach) + " (MAE weighting)") = compute_pred_metrics(progression_GT(1,:), double(progression_probability(2,:) > 0.5));
        model_performance("Approach " + string(approach) + " (MAE+MSE weighting)") = compute_pred_metrics(progression_GT(1,:), double(progression_probability(1,:) > 0.5));
    end
    model_performance_fullcert("Approach 0+1+2+3") = compute_performance_certainty_full(1, (progression_probability_approaches{1}(1,:) + progression_probability_approaches{2}(1,:) + progression_probability_approaches{3}(1,:) + progression_probability_approaches{4}(1,:))/4, progression_GT(1,:), progression_predictions_static_dataset(dataset), predictions_dataset{1});
    model_performance("1+2+3+4 (MAE+MSE weighting)") = compute_pred_metrics(progression_GT(1,:), double(((progression_probability_approaches{1}(1,:) + progression_probability_approaches{2}(1,:) + progression_probability_approaches{3}(1,:)+progression_probability_approaches{4}(1,:))/4)>0.5));
    model_performance("Majority Voting 1+2+3 (only best fit)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(progression_probability_approaches{1}(4,:), progression_probability_approaches{2}(4,:), progression_probability_approaches{3}(4,:)));
    model_performance("Majority Voting 2+3+4 (only best fit)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(progression_probability_approaches{2}(4,:), progression_probability_approaches{3}(4,:), progression_probability_approaches{4}(4,:)));
    model_performance("Majority Voting 1+2+4 (only best fit)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(progression_probability_approaches{1}(4,:), progression_probability_approaches{2}(4,:), progression_probability_approaches{4}(4,:)));
    model_performance("Majority Voting 1+3+4 (only best fit)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(progression_probability_approaches{1}(4,:), progression_probability_approaches{3}(4,:), progression_probability_approaches{4}(4,:)));
    model_performance("Majority Voting 1+2+3 (MAE+MSE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{1}(1,:)>0.5), double(progression_probability_approaches{2}(1,:)>0.5), double(progression_probability_approaches{3}(1,:)>0.5)));
    model_performance("Majority Voting 2+3+4 (MAE+MSE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{2}(1,:)>0.5), double(progression_probability_approaches{3}(1,:)>0.5), double(progression_probability_approaches{4}(1,:)>0.5)));
    model_performance("Majority Voting 1+2+4 (MAE+MSE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{1}(1,:)>0.5), double(progression_probability_approaches{2}(1,:)>0.5), double(progression_probability_approaches{4}(1,:)>0.5)));
    model_performance("Majority Voting 1+3+4 (MAE+MSE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{1}(1,:)>0.5), double(progression_probability_approaches{3}(1,:)>0.5), double(progression_probability_approaches{4}(1,:)>0.5)));
    model_performance("Majority Voting 1+2+3 (MAE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{1}(2,:)>0.5), double(progression_probability_approaches{2}(2,:)>0.5), double(progression_probability_approaches{3}(2,:)>0.5)));
    model_performance("Majority Voting 2+3+4 (MAE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{2}(2,:)>0.5), double(progression_probability_approaches{3}(2,:)>0.5), double(progression_probability_approaches{4}(2,:)>0.5)));
    model_performance("Majority Voting 1+2+4 (MAE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{1}(2,:)>0.5), double(progression_probability_approaches{2}(2,:)>0.5), double(progression_probability_approaches{4}(2,:)>0.5)));
    model_performance("Majority Voting 1+3+4 (MAE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{1}(2,:)>0.5), double(progression_probability_approaches{3}(2,:)>0.5), double(progression_probability_approaches{4}(2,:)>0.5)));
    model_performance("Majority Voting 1+2+3 (MSE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{1}(3,:)>0.5), double(progression_probability_approaches{2}(3,:)>0.5), double(progression_probability_approaches{3}(3,:)>0.5)));
    model_performance("Majority Voting 2+3+4 (MSE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{2}(3,:)>0.5), double(progression_probability_approaches{3}(3,:)>0.5), double(progression_probability_approaches{4}(3,:)>0.5)));
    model_performance("Majority Voting 1+2+4 (MSE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{1}(3,:)>0.5), double(progression_probability_approaches{2}(3,:)>0.5), double(progression_probability_approaches{4}(3,:)>0.5)));
    model_performance("Majority Voting 1+3+4 (MSE weighting)") = compute_pred_metrics(progression_GT(1,:), majority_voting_vectorized(double(progression_probability_approaches{1}(3,:)>0.5), double(progression_probability_approaches{3}(3,:)>0.5), double(progression_probability_approaches{4}(3,:)>0.5)));

    % if dataset == "ALL"
    %     progression_predictions_static_dataset("ALL") = progression_predictions_static_all;
    % else
    %     progression_predictions_static_all = [progression_predictions_static_all, progression_predictions_static_dataset(dataset)];
    % end
    model_performance_dataset("Early RECIST Predictor for " + dataset) = compute_pred_metrics(progression_GT(1,:), progression_predictions_static_dataset(dataset));
    model_performance_dataset("Model Performance Table for " + dataset) = container2table(model_performance, {"Sensitivity", "Specificity", "Balanced Accuracy", "Accuracy"});
    model_performance_dataset_fullcert("Model Performance Full Certainty for " + dataset) = container2table(model_performance_fullcert, {"Sensitivity", "Specificity", "Balanced Accuracy", "Accuracy", "Percentage Patients", "Sensitivity Early RECIST", "Specificity Early RECIST", "Accuracy Early RECIST", "Sensitivity ML", "Specificity ML", "Accuracy ML"});
    if dataset == "ALL"
        %plot_certainty(progression_probability_approaches{1}(1,:), progression_GT(1,:), progression_predictions_static_dataset(dataset), "1");
        %plot_certainty(progression_probability_approaches{2}(1,:), progression_GT(1,:), progression_predictions_static_dataset(dataset), "2");
        %plot_certainty(progression_probability_approaches{3}(1,:), progression_GT(1,:), progression_predictions_static_dataset(dataset), "3");
        %plot_certainty(progression_probability_approaches{4}(1,:), progression_GT(1,:), progression_predictions_static_dataset(dataset), "4");

        plot_certainty((progression_probability_approaches{1}(1,:) + progression_probability_approaches{2}(1,:) + progression_probability_approaches{3}(1,:) + progression_probability_approaches{4}(1,:))/4, progression_GT(1,:), progression_predictions_static_dataset(dataset), "1+2+3+4");

        prediction_performance_days({ ...
            double(progression_probability_approaches{1}(1,:)>0.5), ...
            double(progression_probability_approaches{2}(1,:)>0.5), ...
            double(progression_probability_approaches{3}(1,:)>0.5), ...
            double(progression_probability_approaches{4}(1,:)>0.5)}, ...
            progression_GT, length_to_predict_all);

    end
end

% for dataset = [datasets, "ALL"]
%     if dataset == "ALL"
%         progression_GT = progression_GT_all;
%         progression_probability_approaches = progression_probability_approaches_all;
%     else
%         load(dataset + "/prediction_progression_GT.mat");
%     end
%     model_performance_fullcert = containers.Map('KeyType', 'char', 'ValueType', 'any');
%     for approach = approaches
%         load(dataset + "/prediction_progression_probability_approach_" + approach + ".mat");
%         perf_fullcert_approach = compute_performance_certainty(1, progression_probability_approaches{approach+1}(1,:), progression_GT(1,:), progression_predictions_static_dataset(dataset));
%         model_performance_fullcert("Approach " + string(approach)) = perf_fullcert_approach;
%     end
%     model_performance_dataset_fullcert("Model Performance Full Certainty for " + dataset) = container2table(model_performance_fullcert, {"Sensitivity", "Specificity", "Balanced Accuracy", "Percentage Patients", "Sensitivity Early RECIST", "Specificity Early RECIST"});
% end

toc
diary off

function plot_certainty(progression_probability, progression_GT, progression_predictions_static, modelid)
    certainties = linspace(0.5, 1.0, 30000);
    % 
    % chosen_colormap = colormap("parula");  % Replace "hot" with your desired colormap
    % num_colors = 4;
    % indices = round(linspace(1, size(chosen_colormap, 1), num_colors));
    % cmap = chosen_colormap(indices, :);

    sensitivity = [];
    specificity = [];
    percentage_patients = [];
    sensitivity_static = [];
    specificity_static = [];
    for index_certainty = 1 : length(certainties)
        perf_result = compute_performance_certainty(certainties(index_certainty), progression_probability, progression_GT, progression_predictions_static);
        sensitivity(end+1) = perf_result(1);
        specificity(end+1) = perf_result(2);
        percentage_patients(end+1) = perf_result(4); 
        sensitivity_static(end+1) = perf_result(5);
        specificity_static(end+1) = perf_result(6);
    end

    f = figure();
    plot(certainties, sensitivity, 'LineWidth', 2.5);
    hold on;
    plot(certainties, specificity, 'LineWidth', 2.5);
    plot(certainties, (sensitivity + specificity)/2, 'LineWidth', 2.5);
    %plot(certainties, percentage_patients, 'LineWidth', 2);
    %plot(certainties, (sensitivity_static + specificity_static)/2, 'LineWidth', 2);
    %plot(certainties, sensitivity_static, 'LineWidth', 2);
    legend({"Sensitivity","Specificity","Balanced Accuracy","Percentage Patients","Balanced Accuracy RECIST Predictor", "Sensitivity RECIST Predictor"}, 'Location', 'Best');
    xlabel("Certainty");
    ylabel("Performance");
    axis([0.5 1.0 55 86]);
    fontsize(12,"points");
    title("Performance for Model " + modelid);
end

function final_vote = majority_voting(votes)
    assert(mod(length(votes),2) == 1);
    votes_1 = sum(votes);
    votes_0 = length(votes) - votes_1;

    if votes_1 > votes_0
        final_vote = 1;
    else
        final_vote = 0;
    end
end

function final_votes = majority_voting_vectorized(x, y, z)
    assert(length(x) == length(y) && length(y) == length(z), 'Arrays must be of the same length.');
    final_votes = zeros(1, length(x));
    for i = 1:length(x)
        final_votes(i) = majority_voting([x(i), y(i), z(i)]);
    end
end