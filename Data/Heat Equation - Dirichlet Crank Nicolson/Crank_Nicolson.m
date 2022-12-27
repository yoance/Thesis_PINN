function [sol, sol_m] = Project3_Crank_Nicolson(fx, a, b, M, nu, t_end, dt)
%2.6.2. Crank-Nicolson
%sin(2*pi*x)

% M=10;
% nu=1/6;
% a = @(t) 0;
% b = @(t) 0;
% t_end=0.06;
% dt=0.02;

%Make M partitions and distance between partitions
x = linspace(0,6,M+1);
dx = x(2)-x(1);
iter = round(t_end/dt+1);

r = nu*dt/dx^2;

qa = zeros(M-2,1);
qb = zeros(M-1,1);
qc = zeros(M-2,1);

u = zeros(M+1,1);

%time 0
for k=0:M
    u(k+1)=fx(k*dx);
end
u(1) = a(0);
u(end) = b(0);

%exact solution for comparison with numeric approx
fun_pde=@(t,x) sin(2*pi*x)*exp((-2/3)*pi*pi*t);
sol_exact = fun_pde(t_end, x);
%initializing first error at first dt
err = zeros(iter,1);
err(1) = sum((u-sol_exact').^2);

qa(1:end) = -r/2;
qb(1:end) = 1+r;
qc(1:end) = -r/2;

fig=figure;
u_max = max(u);
u_min = min(u);
animate_sol(fig,x,u,u_min,u_max);

col = round(t_end/dt+1);
row = M+1;
sol = zeros(row,col); %Matrix solution
sol(:,1) = u;
sol(1,:) = a((0:dt:t_end)');
sol(end,:) = b((0:dt:t_end)');

r_bar = zeros(M-1,1);
for n=2:iter
    r_bar(1) = r/2*u(1) + (1-r)*u(2) + r/2*u(3) + r/2*a(n*dt);
    r_bar(M-1) = r/2*u(M-1) + (1-r)*u(M) + r/2*u(M+1) + r/2*b(n*dt);
    for k=2:M-2
        r_bar(k) = r/2*u(k) + (1-r)*u(k+1) + r/2*u(k+2);
    end
    
    u(1) = a((n-1)*dt);
    u(end) = b((n-1)*dt);
    u(2:M) = ThomasAlg(qa,qb,qc,r_bar);
    err(n) = sum((u-sol_exact').^2);
    animate_sol(fig,x,u,u_min,u_max);  
    
    sol(:,n) = u; % matrix sol
end

figure(1)
plot(x,u)
title('Final solution at time t\_end')
xlabel('x')
ylabel('u')

figure(2)
plot(0:dt:t_end, err)
title('Error plot for every time step')
xlabel('t')
ylabel('error')  

sol_m = sol;
sol = u;
end