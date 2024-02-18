function solpts = run_approach0(days,parameters,initial_tumor_size,use_gompertz) %accepts as parameters only normalized values
    T_0 = 10^9;
    K_0 = 10^3;
    r = parameters(1);
    mu_0 = parameters(2);
    mu_star = parameters(3);
    tspan = [days(1), days(end)];
    y0 = [initial_tumor_size, mu_0];
    sol = ode23s(@(t,y) approach0(y,t,r,mu_star,use_gompertz),tspan,y0);
    solpts = deval(sol, days);
    end