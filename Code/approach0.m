function dydt = approach0(y,t,r,mu_star,use_gompertz) %using cells
    T_0 = 10^9;
    K_0 = 10^3; %10^12 / T_0 = 10^3
    T = y(1);
    mu = y(2);
    if t < 0
        mu = 0;
        mu_star = 0;
    end
    dydt = zeros(2,1);
    if use_gompertz
        dydt(1) = r * T * (-log(T / K_0)) - mu * T;
    else
        dydt(1) = r * T * (1 - (T / K_0)) - mu * T;
    end
    dydt(2) = -mu_star * mu;
end