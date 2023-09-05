close all;
f=figure();
r = 1;
mu_1 = 0.5;

tspan = [0 40];
y0 = [0.8, 0.00001];
%y0 = [0.00001, 0];
[t,y] = ode23s(@(t,y) calcdy(r, mu_1, t,y), tspan, y0);
subplot(3,1,1);
plot(t,y(:,1));
legend('T_s: Susceptible tumor cells', 'Location', 'best');
hold on;
subplot(3,1,2);
plot(t,y(:,2));
legend('T_{res}: Resistant tumor cells', 'Location', 'best');
hold on;
subplot(3,1,3);
plot(t,y(:,1) + y(:,2));
legend('T: Total tumor mass', 'Location', 'best');


function dydt = calcdy(r,mu_1,t,y)
    dydt = zeros(2,1);
    dydt(1) = r * y(1) * (1 - (y(1)+y(2))) - mu_1 * y(1);
    dydt(2) = r * y(2) * (1 - (y(1)+y(2)));
end