function R2_error = calculate_R2(y_true, y_fit)
    y_mean = sum(y_true) / length(y_true);
    ss_res = sum((y_true - y_fit).^2);
    ss_tot = sum((y_true - y_mean).^2);
    R2_error = 1 - (ss_res / ss_tot);
end