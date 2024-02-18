function result = compute_performance_certainty(percentage, progression_probability_all, progression_GTs, progression_predictions_static)
    inv_percentage = 1 - percentage;
    result = zeros(1,6);
    progression_probability_all_binary = double(progression_probability_all > 0.5);
    condition = progression_probability_all <= inv_percentage | progression_probability_all >= (1-inv_percentage);

    metrics = compute_pred_metrics(progression_GTs(1,condition), progression_probability_all_binary(condition));

    result(1) = metrics(1);
    result(2) = metrics(2);
    result(3) = metrics(3);
    result(4) = (sum(condition) / length(progression_probability_all_binary))*100;

    metrics_static = compute_pred_metrics(progression_GTs(1,condition), progression_predictions_static(1,condition));
    result(5) = metrics_static(1);
    result(6) = metrics_static(2);
    %assert(length(progression_probability_all_binary) == 186);
end