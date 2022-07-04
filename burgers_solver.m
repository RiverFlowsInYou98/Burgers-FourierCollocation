% Numerical solution of 1-D viscous Burgers equation on [0, L], periodic boundary
% use Fourier collocation in modal space
% use Crank-Nicolson-Leap-Frog in time
function [u2, u_data, t_data] = burgers_solver(nu, L, T, dt, N, u_init)
    k = [0:N / 2 - 1 0 -N / 2 + 1:-1] * 2 * pi / L;
    k1 = 1i * k;
    k2 = k1.^2;
    u0 = u_init;
    save_steps = floor(T / dt / 100);
    u_data = u_init;
    t_data = 0;
    now = 0;

    %% Use Backward Euler scheme for the first level
    ddt = min(dt / 10, N^(-2) / nu);
    while now < dt
        if now > dt - ddt
            ddt = dt - now;
        end
        u1_hat = (fft(u0) - ddt * k1 .* fft(0.5 * u0.^2)) ./ (ones(1, N) - ddt * nu * k2);
        u1 = ifft(u1_hat);
        now = now + ddt;
    end
    step = 1;
    if mod(step, save_steps) == 0
        u_data = [u_data; u1];
        t_data = [t_data; now];
    end
    step = step + 1;
    
    %% Use Crank-Nicolson-Leap-Frog
    dt1 = dt;
    dt2 = dt;
    while now < T
        if now > T - dt
            dt2 = T - now;
        end
        u2_hat = ((ones(1, N) + (dt1 + dt2) / 2 * nu * k2) .* fft(u0) - (dt1 + dt2) / 2 * k1 .* fft(0.5 * u1.^2)) ./ (ones(1, N) - (dt1 + dt2) / 2 * nu * k2);
        u2 = ifft(u2_hat);
        u0 = u1;
        u1 = u2;
        if mod(step, save_steps) == 0 && now < T - dt
            u_data = [u_data; u2];
            t_data = [t_data; now];
        end
        step = step +1;
        now = now + dt2;
    end
    u_data = [u_data; u2];
    t_data = [t_data; now];
end
