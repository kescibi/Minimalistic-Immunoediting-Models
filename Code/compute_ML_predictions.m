function [predictions_interpolated, models_interpolated, predictions_padded, models_padded, c] = compute_ML_predictions(t, y, Y)
    % t: time, y: tumor size, Y: label (PD or not PD)
    % Creates dataset by padding and interpolation, then uses fitcauto to find best model

    N = length(Y); % Number of patients
    max_length = 0;
    for i = 1:N
        max_length = max(max_length, length(t{i}));
        max_length = max(max_length, length(y{i}));
    end

    max_length

    X_interpolated = zeros(N, 2 * max_length);
    X_padded = zeros(N, 2 * max_length);
    
    for i = 1:N
        t_interp = interp1(1:length(t{i}), t{i}, linspace(1, length(t{i}), max_length), 'linear');
        X_interpolated(i, 1:max_length) = t_interp;
        y_interp = interp1(1:length(y{i}), y{i}, linspace(1, length(y{i}), max_length), 'nearest');
        X_interpolated(i, max_length+1:2*max_length) = y_interp;

        t_padded = [t{i}', zeros(1, max_length - length(t{i}))];
        X_padded(i, 1:max_length) = t_padded;
        y_padded = [y{i}', zeros(1, max_length - length(y{i}))];
        X_padded(i, max_length+1:2*max_length) = y_padded;
        % if i < 5
        %     f = figure();
        %     plot(t{i}, y{i}, 'o', t_interp, y_interp, 'x', t_padded, y_padded, '*');
        % end
    end

    % Normalizing the interpolated and padded data
    X_interpolated = (X_interpolated - mean(X_interpolated, 1)) ./ std(X_interpolated, 0, 1);

    X_padded = (X_padded - mean(X_padded, 1)) ./ std(X_padded, 0, 1);

    c = cvpartition(Y, 'KFold', 5);
    %c = cvpartition(Y, 'KFold', 3);
    predictions_interpolated = zeros(size(Y));
    predictions_padded = zeros(size(Y));
    %options = struct("UseParallel", true, "MaxTime", 5, "ShowPlots", false, "Verbose", 1);
    options = struct("UseParallel", true, "ShowPlots", false, "Verbose", 1, "MaxObjectiveEvaluations", 1800);
    models_interpolated = {};
    models_padded = {};

    for i = 1:c.NumTestSets
        trainIdx = training(c, i);
        testIdx = test(c, i);

        model_interpolated = fitcauto(X_interpolated(trainIdx, :), Y(trainIdx), 'HyperparameterOptimizationOptions', options);
        models_interpolated{end+1} = model_interpolated;
        predictions_interpolated(testIdx) = predict(model_interpolated, X_interpolated(testIdx, :));

        % model_padded = fitcauto(X_padded(trainIdx, :), Y(trainIdx), 'HyperparameterOptimizationOptions', options);
        % models_padded{end+1} = model_padded;
        % predictions_padded(testIdx) = predict(model_padded, X_padded(testIdx, :));
    end
end
