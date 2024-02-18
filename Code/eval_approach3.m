function L2_error = eval_approach3(days,parameters,manual_volume,loss,use_gompertz) %parameters: normalized, manual_volume: not normalized
    T_0 = 10^9;
    initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
    solpts = run_approach3(days, parameters, initial_tumor_size,use_gompertz);
    %L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume'))));
    %L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));
    %L2_error = ((1 / length(manual_volume))*sum(((log(convert2ml((solpts(1,:) + solpts(2,:))*T_0)+1) - log(manual_volume'+1))).^2));
    
    % noise_level = 0.01;
    % errors = -(noise_level / 2) + (noise_level * rand(size(manual_volume)));
    % manual_volume = manual_volume .* (1 + errors);
    % manual_volume = max(0,manual_volume); %minimum value is 0
    %if ~isreal(solpts)
    
    %r = parameters(1);
    %if any(solpts(3,:) >= (r - (log(initial_tumor_size * T_0) / (days(1) - 60))))
    if false
        L2_error = 100000;
    else
        if loss == "MAE"
            L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume'))));
        elseif loss == "MSE"
            L2_error = sqrt((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));   
            %L2_error = ((1 / length(manual_volume))*sum(((convert2ml((solpts(1,:) + solpts(2,:))*T_0) - manual_volume')).^2));   
        end
    end    
end