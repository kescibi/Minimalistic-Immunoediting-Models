function L2_error = eval_approach1(days,parameters,manual_volume,loss,use_gompertz) %parameters: normalized, manual_volume: not normalized
T_0 = 10^9;
K_0 = 10^3;
initial_tumor_size = convert2cells(manual_volume(1)) / T_0;
solpts = run_approach1(days, parameters, initial_tumor_size,use_gompertz);
%L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml(solpts(1,:)*T_0) - manual_volume'))));
%L2_error = ((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));
%L2_error = ((1 / length(manual_volume))*sum(((log(convert2ml(solpts(1,:)*T_0)+1) - log(manual_volume'+1))).^2));
if ~isreal(solpts)
    L2_error = Inf;
else
    if loss == "MAE"
        L2_error = ((1 / length(manual_volume))*sum(abs((convert2ml(solpts(1,:)*T_0) - manual_volume'))));
    elseif loss == "MSE"
        L2_error = sqrt((1 / length(manual_volume))*sum(((convert2ml(solpts(1,:)*T_0) - manual_volume')).^2));   
    end
end
end