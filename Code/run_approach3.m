function solpts = run_approach3(days,parameters,initial_tumor_size,use_gompertz) %accepts as parameters only normalized values
    r = parameters(1);
    T_s_n_estimated = parameters(2);
    z_0 = parameters(3);
    beta_tilde = parameters(4);
    tspan = [days(1), days(end)];
    y0 = [T_s_n_estimated, initial_tumor_size - T_s_n_estimated, z_0];
    sol = ode23s(@(t,y) approach3(y,t,r,beta_tilde,use_gompertz),tspan,y0);
    solpts = deval(sol, days);
end