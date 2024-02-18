function index = calcTTP_with_data(t,y,t_measured,y_measured) %returns -1 if no progression, otherwise index in y array of progression
    %%%% Input y: tumor LD measurements in mm over time
    %%%%%%%%%%%%%%%% TTP CALC (very strict, PD definition on one lesion instead of 5) %%%%%%%%%%%%%%%%
    %first we truncate simulation data
    [~, index] = min(abs(t - t_measured(end)));
    t_sim_measured = t(1:index);
    y_sim_measured = y(1:index);

    index = -1;
    nadir = min(y_measured(y_measured > 0));

    if ~any(nadir)
        nadir = 0;
    end

    nadir_timepoint = min(t_measured(y_measured == nadir));
    %disp(nadir_timepoint)
    % Assuming the timepoints in t and t_measured are comparable
    % probably code below doesn't make sense
    [~, nadir_index] = min(abs(t - nadir_timepoint));
    criterium = 1.2*nadir;
    manual_volume_after_response = y((nadir_index) : end);

    condition = (manual_volume_after_response >= criterium) & (manual_volume_after_response - nadir) >= 5;
    if any(condition)
        ttp_index = min(find(condition));
        index = ttp_index + nadir_index;
    end
end