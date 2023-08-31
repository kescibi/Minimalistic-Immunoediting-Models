approach = 3
prediction = 0

f = figure();
f.WindowState = 'maximized';

if prediction == 0
    load("fit_info_approach_" + string(approach) + ".mat")
else
    load("prediction_fit_info_approach_" + string(approach) + ".mat")
end

fit_info_approach_0 = load("fit_info_approach_0.mat").fit_info
fit_info_approach_1 = load("fit_info_approach_1.mat").fit_info
fit_info_approach_2 = load("fit_info_approach_2.mat").fit_info

if approach == 3
    r2 = fit_info(8,1:end);
else
    r2 = fit_info(7,1:end);
end
num_infs = sum(isinf(r2));

%num_infs = num_infs + sum(r2 < -10^3);
num_infs = num_infs + sum(r2 < -2);

r2_filtered = r2(isfinite(r2))
r2_filtered = r2_filtered(not(r2_filtered < -2))

if approach == 3
    L1 = fit_info(7,1:end);
else
    L1 = fit_info(6,1:end);
end

%%%%% Consider cases where R_2 > -0.5 (i.e. ignore outliers) %%%%%
condition = r2 > 0
r2_filtered_good = r2(condition);
L1_filtered_good = L1(condition);
assert(length(r2_filtered_good) == length(L1_filtered_good))

h = histogram(r2_filtered, 25)

ylim([0 60])

distribution_string = " - Distribution";
if prediction == 1
    distribution_string = " - Predictive Distribution"
end

title("Approach " + string(approach) + distribution_string + " of R_2 error (total: " + string(length(r2)) + "), excluded (due to R_2 < -2): " + string(num_infs) ...
   + ", mean R_2: " + string(mean(r2_filtered)) + ", mean MAE = " + string(mean(L1)) ...
   + ", median R_2: " + string(median(r2_filtered)) + ", median MAE = " + string(median(L1)) ...
   + ", R_2 > 0: " + string(sum(condition)) + ", mean R_2(R_2 > 0): " + string(mean(r2_filtered_good)) ...
   + ", mean MAE(R_2 > 0): " + string(mean(L1_filtered_good)) + ", median R_2(R_2 > 0): " + string(median(r2_filtered_good))+ ...
   ", median MAE(R_2 > 0): " + string(median(L1_filtered_good)));

if prediction == 0
    saveas(h, "figures/r_2_approach_" + string(approach) + ".png")
else
    saveas(h, "figures/prediction_r_2_approach_" + string(approach) + ".png")
end

prediction_str = ""
if prediction == 1
    prediction_str = prediction_str + "prediction ";
else
    prediction_str = prediction_str + "full fit ";
end

disp("Approach " + string(approach) + " for " + prediction_str + "achieved: "+ ...
    "median MAE = " + string(round(median(L1),3)) + ", median R^2 = " + string(round(median(r2),3)) +...
    ". Patients with R^2 > 0: " + ...
      string(sum(condition)) + ", median MAE = " + string(round(median(L1_filtered_good),3)) + ... 
    ", median R^2 = " + string(round(median(r2_filtered_good),3)))