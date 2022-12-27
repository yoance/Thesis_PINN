%Crank-Nicolson Scheme for 1-D Heat Equation
%
%Dependencies:
%- Crank_Nicolson.m
%- ThomasAlg.m
%- animate_sol.m

M=1000;
nu = 1.0; %1.0
dt=0.01;

a = @(t) 0;
b = @(t) 0;
L = 6;
fx = @(x) x*(L-x);

%t_end=[0.06 0.1 0.9 50];
t_end = 10;

fun_exact = @(t,x) (4*L^2)/((1)^3*pi^3)*exp(-((1)^2*pi^2*nu^2*t)/L^2)*sin(((1)*pi*x)/L)*(1-(-1)^(1))+(4*L^2)/((2)^3*pi^3)*exp(-((2)^2*pi^2*nu^2*t)/L^2)*sin(((2)*pi*x)/L)*(1-(-1)^(2))+(4*L^2)/((3)^3*pi^3)*exp(-((3)^2*pi^2*nu^2*t)/L^2)*sin(((3)*pi*x)/L)*(1-(-1)^(3))+(4*L^2)/((4)^3*pi^3)*exp(-((4)^2*pi^2*nu^2*t)/L^2)*sin(((4)*pi*x)/L)*(1-(-1)^(4)) +(4*L^2)/((5)^3*pi^3)*exp(-((5)^2*pi^2*nu^2*t)/L^2)*sin(((5)*pi*x)/L)*(1-(-1)^(5)) +(4*L^2)/((6)^3*pi^3)*exp(-((6)^2*pi^2*nu^2*t)/L^2)*sin(((6)*pi*x)/L)*(1-(-1)^(6))  +(4*L^2)/((7)^3*pi^3)*exp(-((7)^2*pi^2*nu^2*t)/L^2)*sin(((7)*pi*x)/L)*(1-(-1)^(7)) +(4*L^2)/((8)^3*pi^3)*exp(-((8)^2*pi^2*nu^2*t)/L^2)*sin(((8)*pi*x)/L)*(1-(-1)^(8)) +(4*L^2)/((9)^3*pi^3)*exp(-((9)^2*pi^2*nu^2*t)/L^2)*sin(((9)*pi*x)/L)*(1-(-1)^(9)) +(4*L^2)/((10)^3*pi^3)*exp(-((10)^2*pi^2*nu^2*t)/L^2)*sin(((10)*pi*x)/L)*(1-(-1)^(10));

for i=1:length(t_end)
    fprintf('Finding Solution at t=%s and dt=%s\n', num2str(t_end(i)), num2str(dt))
    [sol_CN, sol_CN_matrix] = Project3_Crank_Nicolson(fx, a, b, M, nu, t_end(i), dt);
    close all;

    x = linspace(0,6,M+1);
    sol_exact = fun_exact(t_end(i), x);
    plot(x,sol_CN, 'linewidth', 1.5)
    hold on    
    plot(x,sol_exact, 'linewidth', 1.5)    
    legend('Crank-Nicolson', 'Exact') %legend('BTCS', 'Crank-Nicolson', 'FTCS', 'Exact')
	
    if((i+1)<=length(t_end))
        fprintf('\nPress Enter to find solution at t=%s\n', num2str(t_end(i+1)))
    end
    input('')
end

csvwrite('u_CN.csv', sol_CN_matrix)