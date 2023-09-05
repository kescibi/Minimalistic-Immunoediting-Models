load("fit_info_approach_0.mat")

r = fit_info(2, :);
y_Td = fit_info(3, :);
mu_2 = fit_info(4,:);

fprintf("Approach 0\n")
fprintf("r: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(r), median(r), std(r), min(r), max(r))
fprintf("y_Td: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(y_Td), median(y_Td), std(y_Td), min(y_Td), max(y_Td))
fprintf("mu_2: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(mu_2), median(mu_2), std(mu_2), min(mu_2), max(mu_2))

clear all;
load("fit_info_approach_1.mat")

r = fit_info(2, :);
y_Td = fit_info(3, :);
mu_2_tilde = fit_info(4,:);

fprintf("Approach 1\n")
fprintf("r: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(r), median(r), std(r), min(r), max(r))
fprintf("y_Td: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(y_Td), median(y_Td), std(y_Td), min(y_Td), max(y_Td))
fprintf("mu_2_tilde: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(mu_2_tilde), median(mu_2_tilde), std(mu_2_tilde), min(mu_2_tilde), max(mu_2_tilde))

clear all;
load("fit_info_approach_2.mat")

r = fit_info(2, :);
mu_1 = fit_info(3, :);
T_res_n_estimated = fit_info(4,:);

fprintf("Approach 2\n")
fprintf("r: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(r), median(r), std(r), min(r), max(r))
fprintf("mu_1: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(mu_1), median(mu_1), std(mu_1), min(mu_1), max(mu_1))
fprintf("T_res_n_estimated: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(T_res_n_estimated), median(T_res_n_estimated), std(T_res_n_estimated), min(T_res_n_estimated), max(T_res_n_estimated))

clear all;
load("fit_info_approach_3.mat")

r = fit_info(2, :);
T_s_n_estimated = fit_info(3, :);
E_n_estimated = fit_info(4,:);
beta_tilde = fit_info(5,:);

fprintf("Approach 3\n")
fprintf("r: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(r), median(r), std(r), min(r), max(r))
fprintf("x(T_d): mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(T_s_n_estimated), median(T_s_n_estimated), std(T_s_n_estimated), min(T_s_n_estimated), max(T_s_n_estimated))
fprintf("z(T_d): mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(E_n_estimated), median(E_n_estimated), std(E_n_estimated), min(E_n_estimated), max(E_n_estimated))
fprintf("beta_tilde: mean = %f, median = %f, std = %f, min = %f, max = %f\n", mean(beta_tilde), median(beta_tilde), std(beta_tilde), min(beta_tilde), max(beta_tilde))


