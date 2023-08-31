close all;
clear all;

% %%%%%% OPTIONS %%%%%%
% approach = 0
% predict = 0
% %%%%%%%%%%%%%%%%%%%%%

%approaches = [0,1,2,3]
%predict_arr = [0,1]

approaches = [0,1,2,3]
predict_arr = [0,1]

for approach = approaches
    for predict = predict_arr
    if predict == 0
        load("fit_info_approach_" + string(approach) + ".mat");
    else
        load("prediction_fit_info_approach_" + string(approach) + ".mat");
    end
    
    T = readtable("./data/online_data/pcbi.1009822.s006.xlsx", "Sheet", "Study3");
    patient_ids = unique(table2array(T(:, 1)));
    patient_list_id = [];
    for id = 1 : length(patient_ids)
        patient_data = T((T.Patient_Anonmyized == patient_ids(id)), :);
    
        t = table2array(patient_data(:,2));
        y = table2array(patient_data(:,3));
    
        if any(isnan(y)) || length(y) < 6 || y(1) > 97 || not(issorted(t))
        else
            patient_list_id(end+1) = id;
        end
    end
    
    r2 = [];
    for i = 1 : length(patient_list_id)
    %for i = 20:20
        patient_data = T((T.Patient_Anonmyized == patient_ids(patient_list_id(i))), :);
        t = table2array(patient_data(:,2));
        y = table2array(patient_data(:,3));
    
        manual_volume = ((4/3)*pi*(y/2).^3) / 1000; %mm^3 in cm^3
    
        if predict == 0
            interval_fit = 1 : length(t);
        else
            interval_fit = 1 : (length(t) - floor(length(t) / 3));
        end

        if approach == 3
            L2 = fit_info(6,i);
        else
            L2 = fit_info(5,i);
        end

        n = length(interval_fit);
    
        if approach == 3
            fit_info(9,i) = n;
            fit_info(10,i) = AICc(4, n, L2);
            fit_info(11,i) = AIC(4, n, L2);
            fit_info(12,i) = BIC(4, n, L2);
        else
            fit_info(8,i) = n;
            %fit_info(9,i) = n*log(L2) + 2*3 %calculates AIC
            fit_info(9,i) = AICc(3, n, L2);
            fit_info(10,i) = AIC(3, n, L2);
            fit_info(11,i) = BIC(3, n, L2);
        end
    end
    
    if predict == 0
        if approach == 3
            approach_3_AICc = fit_info(10,:);
            approach_3_AIC = fit_info(11,:);
            approach_3_BIC = fit_info(12,:);
            disp("Median AIC for approach 3: " + string(median(approach_3_AIC)) + ", median AICc: " + string(median(approach_3_AICc)) + ", median BIC: " + string(median(approach_3_BIC))); %calculates median AIC
        else
            approach_non3_AICc = fit_info(9,:);
            approach_non3_AIC = fit_info(10,:);
            approach_non3_BIC = fit_info(11,:);
            disp("Median AIC for approach " + string(approach) + ": " + string(median(approach_non3_AIC)) + ", median AICc: " + string(median(approach_non3_AICc)) + ", median BIC: " + string(median(approach_non3_BIC))); %calculates median AIC
        end
    end

    if approach == 3
        table_n_r2 = [fit_info(1,:)', fit_info(9,:)', fit_info(8,:)'];
    else
        table_n_r2 = [fit_info(1,:)', fit_info(8,:)', fit_info(7,:)'];
    end

    if predict == 0
        save("table_n_r2_approach_"+string(approach)+".mat","table_n_r2");
        save("detailed_fit_info_approach_"+string(approach)+".mat", "fit_info");
    else
        save("prediction_table_n_r2_approach_"+string(approach)+".mat","table_n_r2");
        save("prediction_detailed_fit_info_approach_"+string(approach)+".mat", "fit_info");
    end

    %%%%%%%%%%% Completed: save the table with n and R^2 score %%%%%%%%%%%%
    
%     n_s = unique(table_n_r2(:,2))
%     median_r_2_for_n = []
%     median_r_2_filtered_for_n = []
%     patients_filtered = []
%     patients = []
%     for i = 1:length(n_s)
%         n = n_s(i)
%         mask = table_n_r2(:,2) == n
%         r2 = table_n_r2(mask,3)
%         r_2_for_n = [table_n_r2(mask,1), table_n_r2(mask,3)]
%     
%         %num_infs = sum(isinf(r2));
%         %num_infs = num_infs + sum(r2 < -10^3);
%     
%         median_r_2_for_n(end+1) = median(r2)
%     
%         r2_filtered = r2(r2 > 0)
%         %r2_filtered = r2(isfinite(r2))
%         %r2_filtered = r2_filtered(not(r2_filtered < -10^3))
%         median_r_2_filtered_for_n(end+1) = median(r2_filtered)
%         patients_filtered(end+1) = length(r2_filtered)
%         patients(end+1) = length(r2)
%         %TODO: remove infs
%     end
%     
%     metrics_to_plot = []
%     metrics_to_plot{1} = median_r_2_for_n
%     metrics_to_plot{2} = median_r_2_filtered_for_n
%     for i = 1 : length(metrics_to_plot)
%         f = figure();
%         f = bar(n_s, metrics_to_plot{i})
%         ylim([-0.1,1])
%         title_str = ""
%         
%         if i == 1
%             title_str = title_str + "Median R^2"
%         elseif i == 2
%             title_str = title_str + "Median R^2(R^2 > 0)"
%         else
%             error("Undefined metric")
%         end
%     
%         if predict == 1
%             title_str = title_str + " | Prediction | ";
%         else
%             title_str = title_str + " | Full Fit | ";
%         end
%         
%         title_str = title_str + "Approach " + string(approach) 
%         
%         if i == 1
%             title_str = title_str + " | Patients: " + mat2str(patients)
%         elseif i == 2
%             title_str = title_str + " | Patients (fltrd): " + mat2str(patients_filtered)
%         end
%     
%         title(title_str)
%         
%         if predict == 0
%             saveas(f, "figures/barplot_r_2_approach_" + string(approach) +"_metric_"+string(i)+ ".png")
%             save("table_n_r2_approach_"+string(approach)+".mat","table_n_r2");
%         else
%             saveas(f, "figures/barplot_prediction_r_2_approach_" + string(approach) +"_metric_"+string(i)+ ".png")
%             save("prediction_table_n_r2_approach_"+string(approach)+".mat","table_n_r2");
%         end
%     end
    end
end

%Compare AICc's, choose one of {0,1,2} to compare with 3
approach_to_compare_with = 2;
load("detailed_fit_info_approach_3.mat");
approach_3_AICc = fit_info(10,:);

load("detailed_fit_info_approach_" + string(approach_to_compare_with)+".mat");
approach_non3_AIC = fit_info(9,:);
approach_3_better = approach_3_AICc < approach_non3_AICc;
disp("Patient IDs for which approach 3 performed better than approach " + string(approach_to_compare_with) + ...
    " (total: " + string(sum(approach_3_better)) +  " / " + length(fit_info(1,:)) + "):")
approach_3_better = fit_info(1,approach_3_better)

function result = BIC(k, n, L2)
    result = log(n) * k + n * log(L2);
end

function result = AIC(k, n, L2)
    result = 2*k + n * log(L2);
end

function result = AICc(k,n,L2)
    result = AIC(k,n,L2) + ((2 * k * (k+1)) / (n-k-1));
end