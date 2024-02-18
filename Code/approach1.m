function dydt = approach1(y,t,r,mu_2_tilde,use_gompertz) %using normalized values
    T_0 = 10^9;
    K_0 = 10^3;
    T_n = y(1);
    E_n = y(2);
    if t < 0
        E_n = 0;
        mu_2_tilde = 0;
    end
    dydt = zeros(2,1);
    if use_gompertz
        dydt(1) = r * T_n * (-log(T_n / K_0)) - T_n * E_n;
    else
        dydt(1) = r * T_n * (1 - (T_n / K_0)) - T_n * E_n;
    end
    dydt(2) = -mu_2_tilde * T_n * E_n;
end