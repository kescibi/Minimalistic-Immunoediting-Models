function result = compute_performance_certainty_full(percentage, progression_probability_all, progression_GTs, progression_predictions_static, progression_predictions_ML)
    inv_percentage = 1 - percentage;
    result = zeros(1,11);
    progression_probability_all_binary = double(progression_probability_all > 0.5);
    condition = progression_probability_all <= inv_percentage | progression_probability_all >= (1-inv_percentage);

    metrics = compute_pred_metrics(progression_GTs(1,condition), progression_probability_all_binary(condition));

    result(1) = metrics(1);
    result(2) = metrics(2);
    result(3) = metrics(3);
    result(4) = metrics(4);
    result(5) = round((sum(condition) / length(progression_probability_all_binary))*100,1);

    metrics_static = compute_pred_metrics(progression_GTs(1,condition), progression_predictions_static(1,condition));
    result(6) = metrics_static(1);
    result(7) = metrics_static(2);
    result(8) = metrics_static(4);

    metrics_ML = compute_pred_metrics(progression_GTs(1,condition), progression_predictions_ML(1,condition));
    result(9) = metrics_ML(1);
    result(10) = metrics_ML(2);
    result(11) = metrics_ML(4);    
    %assert(length(progression_probability_all_binary) == 186);
end
