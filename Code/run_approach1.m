function solpts = run_approach1(days,parameters,initial_tumor_size,use_gompertz) %accepts as parameters only normalized values
    T_0 = 10^9;
    K_0 = 10^3;
    r = parameters(1);
    y_0 = parameters(2);
    mu_2_tilde = parameters(3);
    tspan = [days(1), days(end)];
    y0 = [initial_tumor_size, y_0];
    sol = ode23s(@(t,y) approach1(y,t,r,mu_2_tilde,use_gompertz),tspan,y0);
    solpts = deval(sol, days);
end