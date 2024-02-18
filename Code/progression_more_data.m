close all;
%%%%% OPTIONS %%%%%
%approach = 1 %choose between approaches 0,1,2,3
run_opti = 1
if run_opti == 1
    clear all; 
    run_opti=1
    approaches = [0,1,2,3]
    model_performance = zeros(12,3);
    % if all_patients == 1
    %     model_performance_all = zeros()
    % else
    % end
    rng(123);
end
all_patients = 1
debug = 0
%%%%%%%%%%%%%%%%%%%

if all_patients == 1
    start = 1
else
    start = 151
end

%Using the conversion rule 10^-3 ml = 1 mm^3 = 10^6 cells
%we get the maximum capacity K = 1kg = 10^12 [cells] = 10^12 [cells] * 10^-6
%[mm^3 / cells] * 10^-3 [cm^3 / mm^3] = 1000 cm^3 (=1L)
%K = 1000;
T = readtable("./data/online_data/pcbi.1009822.s006.xlsx", "Sheet", "Study3");
patient_ids = unique(table2array(T(:, 1)));

%patient_ids = unique(table2array(T(:, 1)));
%oak_patient_ids = unique(T_OAK(string(cell2mat(T_OAK.Study_Arm)) == 'Study_4_Arm_2',:).Patient_Anonmyized); 
%arm 2 is atezolizumab after checking in the original paper figure 1D and
%counting number of patients which have more than 6 measurements

%markerStyles = {'o', 's', '*', 'x', 'd', '^', 'v', '>', '<', 'p', 'h', '.'};
cmap = colormap(turbo(11));

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

% oak_patient_list_id = [];
% oak_dataset = [];
% oak_progression_GTs = [];
% oak_max_length = 0;
% for id = 1 : length(oak_patient_ids)
%     patient_data = T_OAK((T_OAK.Patient_Anonmyized == oak_patient_ids(id)), :);
% 
%     t = table2array(patient_data(:,2));
%     y = table2array(patient_data(:,3));
% 
%     if any(isnan(y)) || length(y) < 6 || y(1) > 97 || not(issorted(t))
%     %if length(y) < 6 %|| not(issorted(t))
%        %skip this patient
%     else
%         oak_patient_list_id(end+1) = id;
%         interval_fit = 1 : (length(t) - floor(length(t) / 3));
%         interval_predict = (interval_fit(end) + 1) : length(t);
%         oak_max_length = max(oak_max_length, length(interval_fit))
%         y = y';
%         t = t';
%         %oak_dataset(end+1,1:11)=interp1(1:length(interval_fit), y(interval_fit), 1:((length(interval_fit)-1)/10):length(interval_fit), 'nearest');
%         %oak_dataset(end,12:22)=interp1(1:length(interval_fit), t(interval_fit), 1:((length(interval_fit)-1)/10):length(interval_fit), 'linear');
% 
%         datapoint = [y(interval_fit),t(interval_fit)];
%         datapoint(23)=0;
%         datapoint(end)=[];
%         oak_dataset(end+1,:)=datapoint;
% 
%         ttp_index = calcTTP(y')
%         if ttp_index ~= -1
%             oak_progression_GTs(1,end+1) = 1;
%         else
%             oak_progression_GTs(1,end+1) = 0;
%         end
%     end
% end


% load("detailed_fit_info_approach_3.mat");
% approach_3_AICc = fit_info(10,:);
% 
% load("detailed_fit_info_approach_" + string(0)+".mat");
% approach_non3_AICc = fit_info(9,:);
% condition = approach_3_AICc < approach_non3_AICc;
if run_opti == 1
    if debug == 0
        amount_of_patients = length(patient_list_id);
    else
        amount_of_patients = length(8);
    end
    progression_GTs = zeros(1, amount_of_patients) - 1;
    progression_predictions = zeros(6, amount_of_patients) - 1;
    progression_predictions_static = zeros(1, amount_of_patients) - 1;
    dataset_full = zeros(amount_of_patients,22);
    dataset_full_interpolated = zeros(amount_of_patients,14);
end

train_set_size = round(0.8*amount_of_patients);

for i = 1 : amount_of_patients
    %patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
    patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);

    t = table2array(patient_data(:,2));
    y = table2array(patient_data(:,3));
    manual_volume = ((4/3)*pi*(y/2).^3) / 1000 %mm^3 in cm^3

    interval_fit = 1 : (length(t) - floor(length(t) / 3));
    interval_predict = (interval_fit(end) + 1) : length(t);

    dataset_full_interpolated(i,1:7) = interp1(1:length(interval_fit), y(interval_fit), 1:((length(interval_fit)-1)/6):length(interval_fit), 'nearest');
    dataset_full_interpolated(i,8:14) = interp1(1:length(interval_fit), t(interval_fit), 1:((length(interval_fit)-1)/6):length(interval_fit), 'linear');

    %dataset_full(i,1:11) = interp1(1:length(interval_fit), y(interval_fit), 1:((length(interval_fit)-1)/10):length(interval_fit), 'nearest');
    %dataset_full(i,12:22) = interp1(1:length(interval_fit), t(interval_fit), 1:((length(interval_fit)-1)/10):length(interval_fit), 'linear');

    datapoint = [y(interval_fit)', t(interval_fit)'];
    datapoint(23)=0;
    datapoint(end)=[];
    dataset_full(i,:) = datapoint;

    %dataset_full(i,8) = t_fit(1);
    %dataset_full(i,9) = t_fit(end);
    ttp_index = calcTTP(y)
    if ttp_index ~= -1
        progression_GTs(1,i) = 1;
    else
        progression_GTs(1,i) = 0;
    end

    %%%%%%%%%%%% Computes predictions for static and CG predictor %%%%%%%%%%%%%%%%%%%
    ttp_index_static = calcTTP(y(interval_fit))
    if ttp_index_static ~= -1
        progression_predictions_static(1,i) = 1;
    else
        progression_predictions_static(1,i) = 0;
    end
end

% p_1 = 0
% non_progressors = 0
% p_2 = 0
% progressors = 0
% 
% for i = 1 : 150
%     %patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
%     patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
% 
%     t = table2array(patient_data(:,2));
%     y = table2array(patient_data(:,3));
% 
%     interval_fit = 1 : (length(t) - floor(length(t) / 3));
%     interval_predict = (interval_fit(end) + 1) : length(t);
% 
%     %%%%%%%%%%%% Computes predictions for CG predictor %%%%%%%%%%%%%%%%%%%
%     ttp_index_static = calcTTP(y(interval_fit))
%     if ttp_index_static ~= -1
%         progressors = progressors + 1
%         if progression_GTs(1,i) == 1
%             p_2 = p_2 + 1
%         end
%     else
%         non_progressors = non_progressors + 1
%         if progression_GTs(1,i) == 1
%             p_1 = p_1 + 1
%         end
%     end
% end
% p_1 = p_1 / non_progressors
% p_2 = p_2 / progressors
% 
% %%%%%%%% in case of static predictor
% % p_1 = 0
% % p_2 = 1
% %%%% Stimmt mit Ergebnis von static predictor überein!!
% 
num_runs = 100
progression_predictions_CG = zeros(num_runs, length((train_set_size+1):amount_of_patients));
% for j = 1 : num_runs
%     for i = 151 : amount_of_patients
%         %patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
%         patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
% 
%         t = table2array(patient_data(:,2));
%         y = table2array(patient_data(:,3));
% 
%         interval_fit = 1 : (length(t) - floor(length(t) / 3));
%         interval_predict = (interval_fit(end) + 1) : length(t);
% 
%         %%%%%%%%%%%% Computes predictions for CG predictor %%%%%%%%%%%%%%%%%%%
%         ttp_index_static = calcTTP(y(interval_fit))
%         guess = rand();
%         if ttp_index_static ~= -1
%             if guess < p_2
%                 progression_predictions_CG(j,i-150) = 1;
%             else
%                 progression_predictions_CG(j,i-150) = 0;
%             end
%         else
%             if guess < p_1
%                 progression_predictions_CG(j,i-150) = 1;
%             else
%                 progression_predictions_CG(j,i-150) = 0;
%             end
%         end
%     end
% end
% 
% p_1_tilde = 0;
% p_2_tilde = 0;
% observed_progressors = 0;
% observed_non_progressors = 0;
% for i = 151 : amount_of_patients
%     %patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
%     patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
% 
%     t = table2array(patient_data(:,2));
%     y = table2array(patient_data(:,3));
% 
%     interval_fit = 1 : (length(t) - floor(length(t) / 3));
%     interval_predict = (interval_fit(end) + 1) : length(t);
% 
%     %%%%%%%%%%%% Computes predictions for CG predictor %%%%%%%%%%%%%%%%%%%
%     ttp_index_static = calcTTP(y(interval_fit))
%     if ttp_index_static ~= -1
%         observed_progressors = observed_progressors + 1
%         if progression_GTs(1,i) == 1
%             p_2_tilde = p_2_tilde + 1
%         end
%     else
%         observed_non_progressors = observed_non_progressors + 1
%         if progression_GTs(1,i) == 1
%             p_1_tilde = p_1_tilde + 1
%         end
%     end
% end
% 
% p_1_tilde = p_1_tilde / observed_non_progressors;
% p_2_tilde = p_2_tilde / observed_progressors;
% Q_1 = observed_progressors / length(151 : amount_of_patients);
% 
% %p_1 = 0
% %p_2 = 1
% 
% TP_CG_expected = ((1-Q_1)*p_1*p_1_tilde) + Q_1*p_2*p_2_tilde;
% FP_CG_expected = ((1-Q_1)*p_1*(1-p_1_tilde)) + Q_1 * p_2 * (1 - p_2_tilde);
% FN_CG_expected = ((1-Q_1)*(1-p_1)*p_1_tilde) + (Q_1 * (1 - p_2) * p_2_tilde);
% TN_CG_expected = ((1-Q_1) * (1-p_1) * (1-p_1_tilde)) + (Q_1 * (1-p_2) * (1-p_2_tilde));
% 
% TPR_CG_expected = TP_CG_expected / (TP_CG_expected + FN_CG_expected); %Sensitivity / Recall
% PPV_CG_expected = TP_CG_expected / (TP_CG_expected + FP_CG_expected); %Precision
% TNR_CG_expected = TN_CG_expected / (TN_CG_expected + FP_CG_expected);
% 
% F1_CG_expected = (2 * TP_CG_expected) / ((2*TP_CG_expected) + FP_CG_expected + FN_CG_expected);

train_val_full = dataset_full(1:train_set_size,:);
test_full = dataset_full((train_set_size+1):end,:);

%train_val_full = vertcat(train_val_full, oak_dataset);
%train_val_full = oak_dataset;

train_val_labels = progression_GTs(1:train_set_size)';

%train_val_labels = vertcat(train_val_labels, oak_progression_GTs');
%train_val_labels = oak_progression_GTs';

test_labels = progression_GTs((train_set_size+1):end)';

train_mean = mean(train_val_full);
train_std = std(train_val_full);

%svmmodel = fitcsvm(train_val_full, train_val_labels);
%svmmodel = fitcauto((train_val_full - train_mean)./train_std, train_val_labels, 'HyperparameterOptimizationOptions', struct('UseParallel', true));
%svmmodel = fitcensemble((train_val_full - train_mean)./train_std, train_val_labels);
%svmmodel = fitcensemble((train_val_full), train_val_labels);
svmmodel = fitctree((train_val_full - train_mean)./train_std, train_val_labels);
%svm_prediction_kfold = kfoldPredict(crossval(svmmodel));
[test_set_predictions, test_set_probs] = predict(svmmodel, (((test_full - train_mean)./train_std)));

%view(svmmodel,'Mode','graph')

testSetPredictions6foldCrossVal = zeros(1,amount_of_patients) - 1;
%%%%%%%%%%%%%%% 6-fold Cross Validation for ML approach %%%%%%%%%%%%%%%%%%
% foldSize = floor(amount_of_patients / 6);
% %assert(foldSize == 31);
% 
% testSetPredictions6foldCrossVal = zeros(1,amount_of_patients) - 1;
% 
% for i = 1 : 6
%     testIndices = (foldSize * (i - 1) + 1):(foldSize * i);
%     trainIndices = setdiff(1:amount_of_patients, testIndices);
% 
%     trainData = dataset_full_interpolated(trainIndices, :);
%     trainLabels = progression_GTs(trainIndices);
% 
%     testData = dataset_full_interpolated(testIndices, :);
%     testLabels = progression_GTs(testIndices);
% 
%     trainMean = mean(trainData);
%     trainStd = std(trainData);
% 
%     MLmodel = fitctree((trainData - trainMean)./trainStd, trainLabels);
%     %view(MLmodel,'Mode','graph')
%     %MLmodel = fitcauto((trainData - trainMean)./trainStd, trainLabels, 'HyperparameterOptimizationOptions', struct('UseParallel', true));
% 
%     [testSetPredictionsFoldi, testSetProbsFoldi] = predict(MLmodel, (((testData - trainMean)./trainStd)));    
%     testSetPredictions6foldCrossVal(testIndices) = testSetPredictionsFoldi;
% end


% fitLogistic = @(Xtrain, ytrain, Xtest) predict(fitglm(Xtrain, ytrain), Xtest);
% crossval_lrmodel = crossval('mse', dataset, progression_GTs', 'Predfun', fitLogistic);
% progression_predictions(8,:) = kfoldPredict(crossval_lrmodel)';
% 
% meta_fit_info_min_lambda = load("meta_fit_info_min_lambda.mat").meta_fit_info_min_lambda;
% meta_fit_testset = []

if run_opti == 1   
    for approach = approaches
        load("prediction_fit_info_approach_" + string(approach) + ".mat");
    
        if approach == 3
            r2 = fit_info(8,1:end);
        else
            r2 = fit_info(7,1:end);
        end
    
        % condition = r2>0
        condition = r2 >= -inf; %Always true
        %amount_of_patients = sum(condition);
        condition_index = find(condition);
    
        %if length(progression_GTs) == 0 && length(progression_predictions) ==0
        %    progression_GTs = zeros(length(approaches), amount_of_patients) - 1;
        %    progression_predictions = zeros(length(approaches), amount_of_patients) - 1;
        %end
    
        no_eligible = -1;
        figure_counter = 0;
        f = -1;
        %progression_GT = zeros(1, amount_of_patients) - 1;
        %progression_predicted = zeros(1, amount_of_patients) - 1;
        %amount_of_patients = length(patient_list_id)
        for i = 1 : amount_of_patients
            patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);

            no_eligible = mod(no_eligible+1,4);
            t = table2array(patient_data(:,2));
            y = table2array(patient_data(:,3));
            manual_volume = ((4/3)*pi*(y/2).^3) / 1000 %mm^3 in cm^3
        
            if (no_eligible == 0)
                figure_counter = figure_counter + 1;
                if ~exist(fullfile(pwd, 'prediction_' + string(approach+1)), 'dir'), mkdir('prediction_'+string(approach+1)); end
                if f ~= -1
                    saveas(f, 'prediction_' + string(approach+1)+'/'+string(i)+'.png');
                    close all;
                end
                f = figure();
            end
            subplot(2,2,no_eligible+1,'Parent',f);

            interval_fit = 1 : (length(t) - floor(length(t) / 3));
            scatter(t(interval_fit), y(interval_fit), [], 'Color', '#0072BD');
            hold on;
            interval_predict = (interval_fit(end) + 1) : length(t);
            scatter(t(interval_predict), y(interval_predict), [], 'Color', '#D95319');
            hold on;
            

            t_interpolated = interp1(1:length(interval_fit), t(interval_fit), 1:((length(interval_fit)-1)/6):length(interval_fit));
            y_interpolated = interp1(1:length(interval_fit), y(interval_fit), 1:((length(interval_fit)-1)/6):length(interval_fit),'nearest');

            scatter(t_interpolated, y_interpolated, 'Marker', 'x')
            hold on;

            ttp_index = calcTTP(y)
            if ttp_index ~= -1
                xline(t(ttp_index), "--b", "TTP",'Color', 'red');
                hold on;
                %progression_GTs(1,i) = 1;
            else
                %progression_GTs(1,i) = 0;
            end
        
            T_0 = 10^9;
            K_0 = 10^3;
            T_Td = convert2cells(manual_volume(1)) / T_0;
            bonus = 1.1
            if approach == 0
                r = fit_info(2,condition_index(i));
                mu_0 = fit_info(3,condition_index(i));
                mu_star = fit_info(4,condition_index(i));
                params = [r,mu_0,mu_star];
                t_sim = linspace(t(1),t(end)*bonus,1000);
                y_sim = run_approach0(t_sim, params, T_Td);
                LD_sim = LD_calc(y_sim(1,:));
                plot(t_sim,LD_sim, 'Color', '#0072BD');
            elseif approach == 1
                r = fit_info(2,condition_index(i));
                y_0 = fit_info(3,condition_index(i));
                mu_2_tilde = fit_info(4,condition_index(i));
                params = [r, y_0, mu_2_tilde];
                t_sim = linspace(t(1),t(end)*bonus,1000);
                y_sim = run_approach1(t_sim, params, T_Td);
                LD_sim = LD_calc(y_sim(1,:));
                plot(t_sim,LD_sim, 'Color', '#0072BD');
            elseif approach == 2
                 r = fit_info(2,condition_index(i));
                 mu1 = fit_info(3,condition_index(i));
                 T_res_n_estimated = fit_info(4,condition_index(i));
                 params = [r, mu1, T_res_n_estimated];
        
                 t_sim = linspace(t(1),t(end)*bonus,1000);
                 y_sim = run_approach2(t_sim, params, convert2cells(manual_volume(1)) / T_0);
                 LD_sim = LD_calc(y_sim(1,:)+y_sim(2,:));
                 plot(t_sim,LD_sim, 'Color', '#0072BD');
            elseif approach == 3
                r = fit_info(2,condition_index(i));
                T_s_n_estimated = fit_info(3,condition_index(i));
                E_n_estimated = fit_info(4,condition_index(i));
                beta_tilde = fit_info(5,condition_index(i));
                params = [r, T_s_n_estimated, E_n_estimated, beta_tilde];
                t_sim = linspace(t(1),t(end)*bonus,1000);
                y_sim = run_approach3(t_sim, params, convert2cells(manual_volume(1)) / T_0);
                LD_sim = LD_calc(y_sim(1,:)+y_sim(2,:));
                plot(t_sim,LD_sim, 'Color', '#0072BD');
            end
        
            ttp_index_predicted = calcTTP(LD_sim);
            if ttp_index_predicted ~= -1
                xline(t_sim(ttp_index_predicted), ":", "TTP Prediction",'Color', 'magenta');
                hold on;
                %progression_predicted(i) = 1;
                progression_predictions(approach+1, i) = 1;
            else
                progression_predictions(approach+1, i) = 0;
            end
        
            xlabel("Time in days");
            %ylabel("Tumor size [cm^3]");
            ylabel("LD [mm]");
            xlim([t(1), t(end)*1.1]);
            xline(0);
            title("Patient " + string(i));
        end
        
        %assert(~any(progression_GTs == -1))
        assert(isempty(find(progression_GTs(1,:) == -1)));
        %assert(~any(progression_predictions == -1))
        assert(isempty(find(progression_predictions(1,:) == -1)));
        
        C = confusionmat(progression_GTs(1,start:end), progression_predictions(approach+1,start:end))
        
        %note that matrix is mirrored
        TN = C(1,1) %True Negative -> true no progression
        TP = C(2,2) %True Positive -> True Progressive
        FN = C(2,1) %False Negative -> Predicted no progression, but is progressive
        FP = C(1,2) %False Positive
        
        TPR = TP / (TP + FN); %Sensitivity / Recall
        TNR = TN / (TN + FP); %Specificity
        PPV = TP / (TP + FP); %Precision
        
        F1 = (2 * TP) / ((2*TP) + FP + FN);
        
        model_performance(approach+1, 1) = PPV;
        model_performance(approach+1, 2) = TPR;
        model_performance(approach+1, 3) = F1;
        model_performance(approach+1, 4) = TNR;

        accuracy = (TP + TN) / (TP+TN+FP+FN);
        model_performance(approach+1, 5) = accuracy;
        % if approach ~= 3
        %     close all;
        % end
    end
end

%%%%%% Meta Model Evaluation %%%%%

% no_eligible = -1;
% figure_counter = 0;
% f = -1;
% for i = 151 : amount_of_patients
%     patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
% 
%     no_eligible = mod(no_eligible+1,4);
%     t = table2array(patient_data(:,2));
%     y = table2array(patient_data(:,3));
%     manual_volume = ((4/3)*pi*(y/2).^3) / 1000 %mm^3 in cm^3
% 
%     if no_eligible == 0
%         figure_counter = figure_counter + 1;
%         if ~exist(fullfile(pwd, 'prediction_meta'), 'dir'), mkdir('prediction_meta'); end
%         if f ~= -1
%             saveas(f, 'prediction_meta/'+string(i)+'.png');
%             close;
%         end
%         f = figure();
%     end
%     subplot(2,2,no_eligible+1,'Parent',f);
% 
%     interval_fit = 1 : (length(t) - floor(length(t) / 3));
%     scatter(t(interval_fit), y(interval_fit), [], 'Color', '#0072BD');
%     hold on;
%     interval_predict = (interval_fit(end) + 1) : length(t);
%     scatter(t(interval_predict), y(interval_predict), [], 'Color', '#D95319');
%     hold on;
% 
% 
%     t_interpolated = interp1(1:length(interval_fit), t(interval_fit), 1:((length(interval_fit)-1)/6):length(interval_fit));
%     y_interpolated = interp1(1:length(interval_fit), y(interval_fit), 1:((length(interval_fit)-1)/6):length(interval_fit),'nearest');
% 
%     scatter(t_interpolated, y_interpolated, 'Marker', 'x')
%     hold on;
% 
%     ttp_index = calcTTP(y)
%     if ttp_index ~= -1
%         xline(t(ttp_index), "--b", "TTP",'Color', 'red');
%         hold on;
%         progression_GTs(1,i) = 1;
%     else
%         progression_GTs(1,i) = 0;
%     end
% 
%     T_0 = 10^9;
%     K_0 = 10^3;
%     T_Td = convert2cells(manual_volume(1)) / T_0;
%     bonus = 1.1
% 
%     assert(meta_fit_info_min_lambda(i-150,2) == i);
% 
%     r = meta_fit_info_min_lambda(i-150,3);
%     T_s_n_estimated = meta_fit_info_min_lambda(i-150,4);
%     E_n_estimated = meta_fit_info_min_lambda(i-150,5);
%     beta_tilde = meta_fit_info_min_lambda(i-150,6);
%     params = [r, T_s_n_estimated, E_n_estimated, beta_tilde];
%     t_sim = linspace(t(1),t(end)*bonus,1000);
%     y_sim = run_approach3(t_sim, params, convert2cells(manual_volume(1)) / T_0);
%     LD_sim = LD_calc(y_sim(1,:)+y_sim(2,:));
%     plot(t_sim,LD_sim, 'Color', '#0072BD');
% 
%     ttp_index_predicted = calcTTP(LD_sim);
%     if ttp_index_predicted ~= -1
%         xline(t_sim(ttp_index_predicted), ":", "TTP Prediction",'Color', 'magenta');
%         hold on;
%         progression_predicted(i) = 1;
%         meta_fit_testset(i-150) = 1;
%     else
%         meta_fit_testset(i-150) = 0;
%     end
% 
%     xlabel("Time in days");
%     ylabel("Tumor size [cm^3]");
%     ylabel("LD [mm]");
%     xlim([t(1), t(end)*1.1]);
%     xline(0);
%     title("Meta - Patient " + string(i));
% end

%%%%%%%% Ensemble models %%%%%%%%%%
% P+P->P, N+N->N, N+P->N, P+N -> both variants explored

majority_vote_123 = zeros(1, amount_of_patients) - 1;
majority_vote_234 = zeros(1, amount_of_patients) - 1;
majority_vote_124 = zeros(1, amount_of_patients) - 1;
majority_vote_134 = zeros(1, amount_of_patients) - 1;

for i = 1 : amount_of_patients
    model_2_prediction = progression_predictions(2,i);
    model_4_prediction = progression_predictions(4,i);
    model_1_prediction = progression_predictions(1,i);
    model_3_prediction = progression_predictions(3,i);

    majority_vote_123(1,i) = majority_voting([model_1_prediction, model_2_prediction, model_3_prediction]);
    majority_vote_234(1,i) = majority_voting([model_2_prediction, model_3_prediction, model_4_prediction]);
    majority_vote_124(1,i) = majority_voting([model_1_prediction, model_2_prediction, model_4_prediction]);
    majority_vote_134(1,i) = majority_voting([model_1_prediction, model_3_prediction, model_4_prediction]);
end

if all_patients == 0
    five_voting = [];
    for i = start : amount_of_patients
        model_1_prediction = progression_predictions(1,i);
        model_2_prediction = progression_predictions(2,i);
        model_3_prediction = progression_predictions(3,i);
        model_4_prediction = progression_predictions(4,i);
        ML_prediction = test_set_predictions(i-(train_set_size));
        five_voting(end+1) = majority_voting([model_1_prediction, model_2_prediction, model_3_prediction, model_4_prediction, ML_prediction]);
    end
end

differing_N_P = []
differing_P_N = []

for i = 1 : amount_of_patients
    model_2_prediction = progression_predictions(2,i);
    model_4_prediction = progression_predictions(4,i);
    model_1_prediction = progression_predictions(1,i);
    model_3_prediction = progression_predictions(3,i);

    if model_1_prediction == model_2_prediction
        progression_predictions(5,i) = model_2_prediction;
    else
        progression_predictions(5,i) = model_4_prediction;
    end

    if model_2_prediction == model_3_prediction
        %progression_predictions(5,i) = model_2_prediction;
        progression_predictions(6,i) = model_2_prediction;
    else
        %progression_predictions(5,i) = model_4_prediction;
        progression_predictions(6,i) = model_4_prediction;        
    end
    % elseif model_2_prediction == 0 && model_4_prediction == 1
    %     % progression_predictions(5,i) = model_2_prediction;
    %     % progression_predictions(6,i) = model_2_prediction;
    %     progression_predictions(5,i) = model_1_prediction;
    %     progression_predictions(6,i) = model_3_prediction;        
    %     differing_N_P(:, end+1) = [i;progression_GTs(1,i)];
    %     %differing_N_P(2, end+1) = progression_GTs(1,i);
    % elseif model_2_prediction == 1 && model_4_prediction == 0
    %     % progression_predictions(5,i) = model_4_prediction;
    %     % progression_predictions(6,i) = model_2_prediction;
    %     progression_predictions(5,i) = model_1_prediction;
    %     progression_predictions(6,i) = model_3_prediction;  
    %     differing_P_N(:, end+1) = [i;progression_GTs(1,i)];
    %     %differing_P_N(2, end+1) = progression_GTs(1,i);
    % else
    %     error("Unknown case")
    % end
end

load("progression_probability_approach_0.mat");
progression_probability_all = progression_probability;
load("progression_probability_approach_1.mat");
progression_probability_all(2,:) = progression_probability;
load("progression_probability_approach_2.mat");
progression_probability_all(3,:) = progression_probability;
load("progression_probability_approach_3.mat");
progression_probability_all(4,:) = progression_probability;

progression_probability_all(5,:) = (progression_probability_all(1,:) + progression_probability_all(2,:) + progression_probability_all(3,:))/3;
progression_probability_all(6,:) = (progression_probability_all(2,:) + progression_probability_all(3,:) + progression_probability_all(4,:))/3;
progression_probability_all(7,:) = (progression_probability_all(1,:) + progression_probability_all(2,:) + progression_probability_all(4,:))/3;
progression_probability_all(8,:) = (progression_probability_all(1,:) + progression_probability_all(3,:) + progression_probability_all(4,:))/3;
progression_probability_all(9,:) = (progression_probability_all(1,:) + progression_probability_all(2,:) + progression_probability_all(3,:) + progression_probability_all(4,:))/4;

progression_probability_all_binary = double(progression_probability_all > 0.5);

model_performance_multi = zeros(9,5)-1;

for approach = [1,2,3,4,5,6,7,8,9]
    C = confusionmat(progression_GTs(1,start:end), progression_probability_all_binary(approach,start:end))
    
    %note that matrix is mirrored
    TN = C(1,1) %True Negative -> true no progression
    TP = C(2,2) %True Positive -> True Progressive
    FN = C(2,1) %False Negative -> Predicted no progression, but is progressive
    FP = C(1,2) %False Positive
    
    TPR = TP / (TP + FN); %Sensitivity / Recall
    TNR = TN / (TN + FP); %Specificity
    PPV = TP / (TP + FP); %Precision
    
    F1 = (2 * TP) / ((2*TP) + FP + FN);
    
    model_performance_multi(approach, 1) = PPV;
    model_performance_multi(approach, 2) = TPR;
    model_performance_multi(approach, 3) = F1;
    model_performance_multi(approach, 4) = TNR;
    accuracy = (TP + TN) / (TP+TN+FP+FN);
    model_performance_multi(approach, 5) = accuracy;    
end

subset_performance = zeros(9,8);

for approach = [1,2,3,4,5,6,7,8,9]
    perf_result = compute_performance_certainty(90/100, progression_probability_all(approach, :), progression_GTs(1,:),progression_predictions_static);
    subset_performance(approach, :) = perf_result;
    % condition = progression_probability_all(approach, :) < 0.1 | progression_probability_all(approach, :) > 0.9
    % C = confusionmat(progression_GTs(1,condition), progression_probability_all_binary(approach,condition))
    % 
    % %note that matrix is mirrored
    % TN = C(1,1) %True Negative -> true no progression
    % TP = C(2,2) %True Positive -> True Progressive
    % FN = C(2,1) %False Negative -> Predicted no progression, but is progressive
    % FP = C(1,2) %False Positive
    % 
    % TPR = TP / (TP + FN); %Sensitivity / Recall
    % TNR = TN / (TN + FP); %Specificity
    % PPV = TP / (TP + FP); %Precision
    % 
    % F1 = (2 * TP) / ((2*TP) + FP + FN);
    % 
    % subset_performance(approach, 1) = PPV;
    % subset_performance(approach, 2) = TPR;
    % subset_performance(approach, 3) = F1;
    % subset_performance(approach, 4) = TNR;
    % accuracy = (TP + TN) / (TP+TN+FP+FN);
    % subset_performance(approach, 5) = accuracy;
    % subset_performance(approach, 6) = sum(condition) / amount_of_patients;
    % %subset_performance(approach, 7) = (TP + FN); %how many progressive
end

approach_text = ["1", "2", "3", "4", "1+2+3", "2+3+4", "1+2+4", "1+3+4", "1+2+3+4"]
for approach = [1,2,3,4,5,6,7,8,9]
    approach_probability = progression_probability_all(approach, :);
    certainties = linspace(0.5, 1.0, 5000);
    recall = [];
    specificity = [];
    acc = [];
    percentage_patients = [];
    acc_static = [];
    recall_static = [];

    for index_certainty = 1 : length(certainties)
        perf_result = compute_performance_certainty(certainties(index_certainty), approach_probability, progression_GTs(1,:), progression_predictions_static);
        recall(end+1) = perf_result(2);
        specificity(end+1) = perf_result(4);
        acc(end+1) = perf_result(5);
        percentage_patients(end+1) = perf_result(6); 
        acc_static(end+1) = perf_result(7);
        recall_static(end+1) = perf_result(8);
    end

    f = figure();
    plot(certainties, recall, 'LineWidth', 2);
    hold on;
    plot(certainties, specificity, 'LineWidth', 2);
    plot(certainties, acc, 'LineWidth', 2);
    plot(certainties, percentage_patients, 'LineWidth', 2);
    plot(certainties, acc_static, 'LineWidth', 2);
    plot(certainties, recall_static, 'LineWidth', 2);
    legend({"Recall","Specificity","Balanced Accuracy","Percentage Patients","Balanced Accuracy RECIST Predictor", "Recall RECIST Predictor"}, 'Location', 'Best');
    xlabel("Certainty");
    ylabel("Performance");
    title("Performance for Model " + approach_text(approach));

    saveas(f, 'approach_' + string(approach) +  '_certainty_perf.fig');
end

f = figure();
barh([0 1 2 3 4 5 6 7 8], model_performance_multi);
xlabel("Performance");
ylabel("Model");
legend({"Precision","Recall","F1","Specificity","Accuracy"});
yticklabels({"Model 1", "Model 2", "Model 3", "Model 4", "1+2+3", "2+3+4", "1+2+4", "1+3+4","1+2+3+4"});
title("Model Performance based on Probabilities");
saveas(f, 'approach_' + string(approach) +  '_probabilities_perf.fig');

f = figure();
barh([0 1 2 3 4 5 6 7 8], subset_performance);
xlabel("Performance");
ylabel("Model");
legend({"Precision","Recall","F1","Specificity","Balanced Accuracy","Percentage Patients","Balanced Accuracy RECIST Predictor","Recall RECIST Predictor"});
yticklabels({"Model 1", "Model 2", "Model 3", "Model 4", "1+2+3", "2+3+4", "1+2+4", "1+3+4","1+2+3+4"});
title("Model Performance using 90% Certainty");
saveas(f, 'approach_' + string(approach) +  '_90certainty_perf.fig');

% figure();
% legend_descr = []
% for approach = [1,2,3,4]
%     [X,Y,T,AUC,OPTOPOINT] = perfcurve(progression_GTs(1,start:end), progression_probability_all_binary(approach,start:end), 1)  
%     h = plot(X,Y);
%     hold on
%     set(h, 'DisplayName', "Model " + string(approach) + ": AUC " + string(round(AUC,5)));
% end
% legend();
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC')

%for approach = [4,5,6,7,11,12,13,14,15,17]
for approach = [4,5,6,7,12,13,14,15,17]
    if approach == 6
        C = confusionmat(test_labels, test_set_predictions)
    elseif approach == 7 %static predictor
        C = confusionmat(progression_GTs(1,start:end), progression_predictions_static(1,start:end))
    elseif approach == 11
        C = confusionmat(progression_GTs(1,(train_set_size+1):end), meta_fit_testset(1,:))
    elseif approach == 12
        C = confusionmat(progression_GTs(1,start:end), majority_vote_123(1,start:end))
    elseif approach == 13
        C = confusionmat(progression_GTs(1,start:end), majority_vote_234(1,start:end))
    elseif approach == 14
        C = confusionmat(progression_GTs(1,start:end), majority_vote_124(1,start:end))
    elseif approach == 15
        C = confusionmat(progression_GTs(1,start:end), majority_vote_134(1,start:end))
    elseif approach == 17 %6fold cross val
        C = confusionmat(progression_GTs, testSetPredictions6foldCrossVal)
    else
        C = confusionmat(progression_GTs(1,start:end), progression_predictions(approach+1,start:end))
    end
    
    %note that matrix is mirrored
    TN = C(1,1) %True Negative -> true no progression
    TP = C(2,2) %True Positive -> True Progressive
    FN = C(2,1) %False Negative -> Predicted no progression, but is progressive
    FP = C(1,2) %False Positive
    
    TPR = TP / (TP + FN); %Sensitivity / Recall
    TNR = TN / (TN + FP); %Specificity
    PPV = TP / (TP + FP); %Precision
    
    F1 = (2 * TP) / ((2*TP) + FP + FN);
    
    model_performance(approach+1, 1) = PPV;
    model_performance(approach+1, 2) = TPR;
    model_performance(approach+1, 3) = F1;
    model_performance(approach+1, 4) = TNR;
    accuracy = (TP + TN) / (TP+TN+FP+FN);
    model_performance(approach+1, 5) = accuracy;    

end

if all_patients == 0
    for approach = [16]
        if approach == 16
            assert(length(progression_GTs(1,start:end)) == length(five_voting))
            C = confusionmat(progression_GTs(1,start:end), five_voting)
        end
        
        %note that matrix is mirrored
        TN = C(1,1) %True Negative -> true no progression
        TP = C(2,2) %True Positive -> True Progressive
        FN = C(2,1) %False Negative -> Predicted no progression, but is progressive
        FP = C(1,2) %False Positive
        
        TPR = TP / (TP + FN); %Sensitivity / Recall
        TNR = TN / (TN + FP); %Specificity
        PPV = TP / (TP + FP); %Precision
        
        F1 = (2 * TP) / ((2*TP) + FP + FN);
        
        model_performance(approach+1, 1) = PPV;
        model_performance(approach+1, 2) = TPR;
        model_performance(approach+1, 3) = F1;
        model_performance(approach+1, 4) = TNR;
        accuracy = (TP + TN) / (TP+TN+FP+FN);
        model_performance(approach+1, 5) = accuracy;        
    end
end

model_performance_CG = zeros(num_runs,3);
for j = 1 : num_runs
    C = confusionmat(progression_GTs(1,(train_set_size+1):end), progression_predictions_CG(j,:));
        %note that matrix is mirrored
    TN = C(1,1) %True Negative -> true no progression
    TP = C(2,2) %True Positive -> True Progressive
    FN = C(2,1) %False Negative -> Predicted no progression, but is progressive
    FP = C(1,2) %False Positive
    
    TPR = TP / (TP + FN); %Sensitivity / Recall
    TNR = TN / (TN + FP); %Specificity
    PPV = TP / (TP + FP); %Precision
    
    F1 = (2 * TP) / ((2*TP) + FP + FN);
    model_performance_CG(j, 1) = PPV;
    model_performance_CG(j, 2) = TPR;
    model_performance_CG(j, 3) = F1;
    model_performance_CG(j, 4) = TNR;
    accuracy = (TP + TN) / (TP+TN+FP+FN);
    model_performance_CG(j, 5) = accuracy;
end

CG_PPV_mean = mean(model_performance_CG(:,1));
CG_PPV_std = std(model_performance_CG(:,1));
CG_TPR_mean = mean(model_performance_CG(:,2));
CG_TPR_std = std(model_performance_CG(:,2));
CG_F1_mean = mean(model_performance_CG(:,3));
CG_F1_std = std(model_performance_CG(:,3));
CG_TNR_mean = mean(model_performance_CG(:,4));
CG_TNR_std = std(model_performance_CG(:,4));
CG_accuracy_mean = mean(model_performance_CG(:,5));
CG_accuracy_std = std(model_performance_CG(:,5));

model_performance(9,1)=CG_PPV_mean;
model_performance(9,2)=CG_TPR_mean;
model_performance(9,3)=CG_F1_mean;
model_performance(9,4)=CG_TNR_mean;
model_performance(9,5)=CG_accuracy_mean;

%%%%%%%%%%%%% Now we add random guessing (uninformed and informed)

p_PD_overall = sum(progression_GTs) / length(progression_GTs)
TP_uninformed = 0.5 * p_PD_overall;
FP_uninformed = 0.5 * (1 - p_PD_overall);
FN_uninformed = 0.5 * p_PD_overall;
TN_uninformed = 0.5 * (1 - p_PD_overall); %TODO: double check
precision_uninformed = TP_uninformed / (TP_uninformed + FP_uninformed);
recall_uninformed = TP_uninformed / (TP_uninformed + FN_uninformed);
specificity_uninformed = TN_uninformed / (TN_uninformed + FP_uninformed);
F1_uninformed = calcF1(precision_uninformed, recall_uninformed);

model_performance(10,1) = precision_uninformed;
model_performance(10,2) = recall_uninformed;
model_performance(10,3) = F1_uninformed;
model_performance(10,4) = specificity_uninformed;

accuracy_uninformed = (TP_uninformed + TN_uninformed) / (TP_uninformed+TN_uninformed+FP_uninformed+FN_uninformed);
model_performance(10, 5) = accuracy_uninformed;   

TP_informed = (p_PD_overall^2);
FP_informed = p_PD_overall * (1 - p_PD_overall);
FN_informed = (1 - p_PD_overall) * p_PD_overall;
TN_informed = (1-p_PD_overall)^2;
precision_informed = calcPrecision(TP_informed, FP_informed);
recall_informed = calcRecall(TP_informed, FN_informed);
F1_informed = calcF1(precision_informed, recall_informed);
specificity_informed = TN_informed / (TN_informed + FP_informed);

model_performance(11,1) = precision_informed;
model_performance(11,2) = recall_informed;
model_performance(11,3) = F1_informed;
model_performance(11,4) = specificity_informed;

accuracy_informed = (TP_informed + TN_informed) / (TP_informed+TN_informed+FP_informed+FN_informed);
model_performance(11, 5) = accuracy_informed;   

if all_patients == 1
    figure()
    model_selection = [1,2,3,4,5,6,8,10,13,14,15,16,18];
    barh([0 1 2 3 4 5 6 7 8 9 10 11 12 ], model_performance(model_selection,:));
    xlabel("Performance");
    ylabel("Model");
    legend({"Precision","Recall","F1","Specificity","Accuracy"});
    yticklabels({"Model 1", "Model 2", "Model 3", "Model 4", "Model 1+2 (4)", "Model 2+3 (4)", "Static predictor", "Random (uninformed)", "1+2+3", "2+3+4", "1+2+4", "1+3+4", "ML Model (6-fold CrossVal)"});
    saveas(f, 'performance_full.fig');

    figure()
    all_model_indices = [1,2,3,4,13,14,15,16,10,8,18]
    all_labels = ["Model 1","Model 2","Model 3", "Model 4", "Model 1+2+3", "Model 2+3+4", "Model 1+2+4", "Model 1+3+4", "Random", "Static Predictor","ML Model (6-fold CrossVal)"];
    best_youden = -1;
    for i = 1 : length(all_model_indices)
        hold on;
        x_i = 1 - model_performance(all_model_indices(i),4);
        y_i = model_performance(all_model_indices(i),2);
        youden = y_i - x_i;
        best_youden = max([best_youden, youden]);
    end
    for i = 1 : length(all_model_indices)
        hold on;
        x_i = 1 - model_performance(all_model_indices(i),4);
        y_i = model_performance(all_model_indices(i),2);
        scatter(x_i, y_i, 60, cmap(i,:), 'filled');
        youden = y_i - x_i
        if best_youden == youden
            line([x_i, x_i], [x_i, y_i], 'Color', cmap(i,:), 'LineStyle', '--','HandleVisibility','off');
        end
    end
    legend(all_labels);
    for approach = [1,2,3,4,5,6,7,8,9]
        [X,Y,T,AUC,OPTOPOINT] = perfcurve(progression_GTs(1,start:end), progression_probability_all(approach,start:end), 1)  
        h = plot(X,Y);
        scatter(OPTOPOINT(1), OPTOPOINT(2), 60, 'filled', 'HandleVisibility', 'off');
        hold on

        if approach == 5
            set(h, 'DisplayName', "Model 1 + 2 + 3" + " AUC " + string(round(AUC,5)) + ", Specificity = " + string(round(1-OPTOPOINT(1),3) + ", Sensitivity = " + string(round(OPTOPOINT(2), 3))));
        elseif approach == 6
            set(h, 'DisplayName', "Model 2 + 3 + 4" + ": AUC " + string(round(AUC,5)) + ", Specificity = " + string(round(1-OPTOPOINT(1),3) + ", Sensitivity = " + string(round(OPTOPOINT(2), 3))));
        elseif approach == 7
            set(h, 'DisplayName', "Model 1 + 2 + 4" + ": AUC " + string(round(AUC,5)) + ", Specificity = " + string(round(1-OPTOPOINT(1),3) + ", Sensitivity = " + string(round(OPTOPOINT(2), 3))));
        elseif approach == 8
            set(h, 'DisplayName', "Model 1 + 3 + 4" + ": AUC " + string(round(AUC,5)) + ", Specificity = " + string(round(1-OPTOPOINT(1),3) + ", Sensitivity = " + string(round(OPTOPOINT(2), 3))));
        elseif approach == 9
            set(h, 'DisplayName', "Model 1 + 2 + 3 + 4" + ": AUC " + string(round(AUC,5)) + ", Specificity = " + string(round(1-OPTOPOINT(1),3) + ", Sensitivity = " + string(round(OPTOPOINT(2), 3))));
        else
            set(h, 'DisplayName', "Model " + string(approach) + ": AUC " + string(round(AUC,5)) + ", Specificity = " + string(round(1-OPTOPOINT(1),3) + ", Sensitivity = " + string(round(OPTOPOINT(2), 3))));
        end
    end

    plot([0, 1], [0, 1], 'k--', 'HandleVisibility','off');
    xlabel("1 - Specificity");
    ylabel("Sensitivity"); %equivalent to recall
    xlim([0,1]);
    ylim([0,1]);
    legend('show')
    saveas(f, 'roc.fig');
else
    figure()
    barh([0 1 2 3 4 5 6 7 8 9 10 11], model_performance(1:12,:))
    xlabel("Performance");
    ylabel("Model");
    legend({"Precision","Recall","F1","Specificity","Accuracy"});
    yticklabels({"Model 1", "Model 2", "Model 3", "Model 4", "Model 1+2 (4)", "Model 2+3 (4)", "ML model", "Static predictor", "CG predictor", "Random (uninformed)", "Random (informed)", "Meta Model"});
    
    figure()
    barh([0 1 2 3 4], model_performance(13:17,:));
    xlabel("Performance");
    ylabel("Model");
    legend({"Precision","Recall","F1","Specificity","Accuracy"});
    yticklabels({"1+2+3", "2+3+4", "1+2+4", "1+3+4", "1+2+3+4+ML"});
end

% proportion_positive = sum(progression_GTs(start:end)) / length(progression_GTs(start:end));
% random_F1 = 2*proportion_positive / (2*proportion_positive + (1-proportion_positive));
% xline(random_F1, '--', 'linewidth', 2, 'Color', "#EDB120");

total_agreement = 0
total_correct = 0
progression_agreement = 0
progression_correct = 0
noprogression_agreement = 0
noprogression_correct = 0
agreement_GT = []
agreement_prediction = []
agreement_NN_prediction = []
progression_num = 0
noprogression_num = 0

for i = 1 : amount_of_patients
    if progression_predictions(2,i) == progression_predictions(3,i)
        total_agreement = total_agreement + 1;
        agreement_GT = [agreement_GT progression_GTs(1,i)]
        agreement_prediction = [agreement_prediction progression_predictions(2,i)]
        %TODO: check model and ML model prediction agreement performance for test set
        %agreement_NN_prediction = [agreement_NN_prediction ]
        if progression_predictions(2,i) == progression_GTs(1,i)
            total_correct = total_correct + 1;
        end
    end

    if progression_GTs(1,i) == 1
        % if progression_predictions(2,i) == 1
        %     progression_agreement = progression_agreement + 1;
        %     if progression_GTs(1,i) == 1
        %         progression_correct = progression_correct + 1;
        %     end
        % else
        %     noprogression_agreement = noprogression_agreement + 1;
        %     if progression_GTs(1,i) == 0
        %         noprogression_correct = noprogression_correct + 1;
        %     end
        % end
        progression_num = progression_num + 1;
        if progression_predictions(2,i) == progression_predictions(3,i)
            progression_agreement = progression_agreement + 1;
            if progression_predictions(2,i) == 1
                progression_correct = progression_correct + 1;
            end
        end
    else
        noprogression_num = noprogression_num + 1;
         if progression_predictions(2,i) == progression_predictions(3,i)
            noprogression_agreement = noprogression_agreement + 1;
            if progression_predictions(2,i) == 0
                noprogression_correct = noprogression_correct + 1;
            end
        end
    end
end
%TODO: i think here it's better to divide into subset of progressors and
%non-progressors!!
agreement = [total_agreement / amount_of_patients, total_correct / total_agreement, progression_agreement / progression_num, noprogression_agreement / noprogression_num, progression_correct / progression_agreement, noprogression_correct / noprogression_agreement]

C = confusionmat(agreement_GT, agreement_prediction);
%note that matrix is mirrored
TN = C(1,1); %True Negative -> true no progression
TP = C(2,2); %True Positive -> True Progressive
FN = C(2,1); %False Negative -> Predicted no progression, but is progressive
FP = C(1,2); %False Positive
TPR = TP / (TP + FN) %Sensitivity / Recall
TNR = TN / (TN + FP) %Specificity
PPV = TP / (TP + FP) %Precision
Accuracy = (TP + TN) / (TP+FN+FP+TN)
    
F1 = (2 * TP) / ((2*TP) + FP + FN)
Balanced_Accuracy = (TPR + TNR)/2

function result = compute_performance_certainty(percentage, progression_probability_all, progression_GTs, progression_predictions_static)
    inv_percentage = 1 - percentage;
    result = zeros(1,6);
    progression_probability_all_binary = double(progression_probability_all > 0.5);
    condition = progression_probability_all <= inv_percentage | progression_probability_all >= (1-inv_percentage);

    C = confusionmat(progression_GTs(1,condition), progression_probability_all_binary(condition));
    TN = C(1,1) %True Negative -> true no progression
    TP = C(2,2) %True Positive -> True Progressive
    FN = C(2,1) %False Negative -> Predicted no progression, but is progressive
    FP = C(1,2) %False Positive
    
    TPR = TP / (TP + FN); %Sensitivity / Recall
    TNR = TN / (TN + FP); %Specificity
    PPV = TP / (TP + FP); %Precision
    
    F1 = (2 * TP) / ((2*TP) + FP + FN);
    
    result(1) = PPV;
    result(2) = TPR;
    result(3) = F1;
    result(4) = TNR;
    accuracy = (TP + TN) / (TP+TN+FP+FN);
    result(5) = (TPR + TNR)/2;
    result(6) = sum(condition) / length(progression_probability_all_binary);

    C = confusionmat(progression_GTs(1,condition), progression_predictions_static(1,condition));
    TN = C(1,1) %True Negative -> true no progression
    TP = C(2,2) %True Positive -> True Progressive
    FN = C(2,1) %False Negative -> Predicted no progression, but is progressive
    FP = C(1,2) %False Positive
    TPR_static = TP / (TP + FN); 
    TNR_static = TN / (TN + FP); 
    result(7) = (TPR_static + TNR_static)/2; %static
    result(8) = TPR_static; %static Recall
    %assert(length(progression_probability_all_binary) == 186);
end

function final_vote = majority_voting(votes)
    assert(mod(length(votes),2) == 1)
    votes_1 = sum(votes)
    votes_0 = length(votes) - votes_1

    if votes_1 > votes_0
        final_vote = 1
    else
        final_vote = 0
    end
end

function recall = calcRecall(TP,FN)
    recall = TP / (TP + FN);
end

function precision = calcPrecision(TP,FP)
    precision = TP / (TP+FP);
end

function F1 = calcF1(precision, recall)
    F1 = 2 / ((1 / precision) + (1 / recall));
end

function LD = LD_calc(volume)
    LD = 2*nthroot((3/(4*pi))*1000*convert2ml(volume*10^9),3);
end

function index = calcTTP(y) %returns -1 if no progression, otherwise index in y array of progression
    %%%% Input y: tumor LD measurements in mm over time
    %%%%%%%%%%%%%%%% TTP CALC (very strict, PD definition on one lesion instead of 5) %%%%%%%%%%%%%%%%
    index = -1
    nadir = min(y);
    nadir_index = min(find(y == nadir));
    criterium = 1.2*nadir;
    manual_volume_after_response = y((nadir_index+1) : end);
    condition = (manual_volume_after_response >= criterium) & (manual_volume_after_response - y(nadir_index)) >= 5;
    if any(condition)
        ttp_index = min(find(condition));
        index = ttp_index + nadir_index;
    end
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