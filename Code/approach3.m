function dydt = approach3(y,t,r,beta_tilde,use_gompertz) %using cells
    T_0 = 10^9;
    K_0 = 10^3; %10^12 / T_0 = 10^3
    dydt = zeros(3,1);
    T_p_n = y(1);

    T_res_n = y(2);
    %T_res_n = 0;
    %beta_tilde = 0;

    E_n = y(3);

    if t < 0
        E_n = 0;
        beta_tilde = 0;
    end

    if use_gompertz
        dydt(1) = r * T_p_n * (-log((T_p_n + T_res_n) / K_0)) - E_n * T_p_n;
        dydt(2) = r * T_res_n * (-log((T_p_n + T_res_n) / K_0));
    else
        dydt(1) = r * T_p_n * (1 - ((T_p_n + T_res_n) / K_0)) - E_n * T_p_n;
        dydt(2) = r * T_res_n * (1 - ((T_p_n + T_res_n) / K_0));
    end

    % if E_n >= r - (log((T_p_n + T_res_n) * T_0) / (t(1) - 60))
    %     dydt(3) = 0;
    % else
    %     dydt(3) = beta_tilde * E_n * T_p_n;
    % end
    dydt(3) = beta_tilde * E_n * T_p_n;
end