%% Numerical solution of 1-D viscous Burgers equation on [0, L], periodic boundary
% use Fourier collocation in modal space
% use Crank-Nicolson-Leap-Frog in time
clear all; clc; close all;
%% configurtion
nu = 0.01;
L = 2 * pi;
T = 1;
%% discretization
N = 1280;
dt = 0.001;
h = L / N;
x = (0:N - 1) * h;
%% initial value
% u_init = sin(2 * pi / L * x) + 2 * sin(10 * pi / L * x) + cos(20 * pi / L * x);
u_init = load("GP1.txt");
u_init = u_init(1:end - 1)';
%% solving
[u2, u_data, t_data] = burgers_solver(nu, L, T, dt, N, u_init);
%% ploting
figure(1)
waterfall(x, t_data, u_data), colormap(1e-6 * [1 1 1]); view(-20, 25)
xlabel x, ylabel t,
% title('solution')
axis([0 L 0 t_data(end) min(u_init) - 0.1, max(u_init) + 0.1]), grid off
set(gca, 'linewidth', 2, 'fontsize', 10, 'fontname', 'Cambria Math', 'fontweight', 'bold');

figure(2)
plot(x, u_init, 'Markersize', 5, 'Linewidth', 2)
hold on
plot(x, u2, 'Markersize', 5, 'Linewidth', 1)
xlabel('x')
ylabel('u(t=T,x)')
legend('T=0', ['T=' num2str(T)])
set(gca, 'linewidth', 2, 'fontsize', 10, 'fontname', 'Cambria Math', 'fontweight', 'bold');
