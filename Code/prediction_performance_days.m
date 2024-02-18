function [outputArg1,outputArg2] = prediction_performance_days(predicted_approaches,GT,days)
    f = figure();
    arr = 1:length(predicted_approaches);
    % 
    % chosen_colormap = colormap("hsv");  % Replace "hot" with your desired colormap
    % num_colors = length(predicted_approaches)+1;
    % indices = round(linspace(1, size(chosen_colormap, 1), num_colors));
    % cmap = chosen_colormap(indices, :);
    cmap = [
    0.00, 0.70, 0.70;  % Teal
    1.00, 0.50, 0.31;  % Coral
    0.25, 0.41, 0.88;  % Royal Blue
    0.58, 0.47, 0.71   % Olive Green
    ];

    for i = 1:length(predicted_approaches)
        predicted = predicted_approaches{i};
        sensitivity_days = [];
        specificity_days = [];
        accuracy_days = [];
        percentage_patients = [];
        days_ordered = [];
        for day = min(days) : max(days)
            indices = find(days>=day);
            if length(indices) > int64(length(GT)*0.1)
                days_ordered(end+1) = day;
                metrics = compute_pred_metrics(GT(1, indices), predicted(indices));
                sensitivity_days(end+1) = metrics(1);
                specificity_days(end+1) = metrics(2);
                accuracy_days(end+1) = metrics(4);
                percentage_patients(end+1) = (length(indices) / length(GT))*100;
            end
        end
        %f = figure();
        %plot(days_ordered, sensitivity_days);
        hold on;
        %plot(days_ordered, specificity_days);
        plot(days_ordered, accuracy_days, 'LineWidth', 2.5, 'Color', cmap(i,:));
        
        %plot(days_ordered, percentage_patients);
        xlabel("Days");
        ylabel("Performance");
        axis([50 250 30 80]);
        fontsize(12,"points")
        title("Accuracy")
    end
    legend("Model " + arr);
    colormap("cool");
end

