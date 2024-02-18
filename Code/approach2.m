function dydt = approach2(y,t,r,mu1,use_gompertz) %using cells
    T_0 = 10^9;
    K_0 = 10^3; %10^12 / T_0 = 10^3
    if t < 0
        mu1 = 0;
    end
    dydt = zeros(2,1);
    T_p_n = y(1);
    T_res_n = y(2);
    %dydt(1) = r * T_p_n * (1 - ((T_p_n + T_res_n) / K_0)) - mu1 * T_p_n;
    %dydt(2) = r * T_res_n * (1 - ((T_p_n + T_res_n) / K_0));
    if use_gompertz
        dydt(1) = r * T_p_n * (-log(((T_p_n + T_res_n) / K_0))) - mu1 * T_p_n;
        dydt(2) = r * T_res_n * (-log(((T_p_n + T_res_n) / K_0)));
    else
        dydt(1) = r * T_p_n * (1 - ((T_p_n + T_res_n) / K_0)) - mu1 * T_p_n;
        dydt(2) = r * T_res_n * (1 - ((T_p_n + T_res_n) / K_0));
    end
end