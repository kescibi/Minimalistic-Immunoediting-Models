function solpts = run_approach2(days,parameters,initial_tumor_size,use_gompertz) %accepts as parameters only normalized values
    T_0 = 10^9;
    K_0 = 10^3;
    r = parameters(1);
    mu1 = parameters(2);
    T_res_n_estimated = parameters(3);
    tspan = [days(1), days(end)];
    y0 = [initial_tumor_size - T_res_n_estimated, T_res_n_estimated];
    sol = ode23s(@(t,y) approach2(y,t,r,mu1,use_gompertz),tspan,y0);
    solpts = deval(sol, days);
end