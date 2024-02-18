function index = calcTTP(y) %returns -1 if no progression, otherwise index in y array of progression
    %%%% Input y: tumor LD measurements in mm over time
    %%%%%%%%%%%%%%%% TTP CALC (very strict, PD definition on one lesion instead of 5) %%%%%%%%%%%%%%%%
    index = -1;
    nadir = min(y);
    nadir_index = min(find(y == nadir));
    criterium = 1.2*nadir;
    manual_volume_after_response = y((nadir_index+1) : end);
    condition = (manual_volume_after_response >= criterium) & (manual_volume_after_response - y(nadir_index)) >= 5;
    if any(condition)
        ttp_index = min(find(condition));
        index = ttp_index + nadir_index;
    end
end