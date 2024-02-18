function PS = calcPS(y) %calculate Progressive Size based on observed y
    %%%% Input y: tumor LD measurements in mm over time
    %%%%%%%%%%%%%%%% TTP CALC (very strict, PD definition on one lesion instead of 5) %%%%%%%%%%%%%%%%
    nadir = min(y(y>0));
    if ~any(nadir)
        nadir=0;
    end
    PS = max(1.2 * nadir, nadir + 5);
end