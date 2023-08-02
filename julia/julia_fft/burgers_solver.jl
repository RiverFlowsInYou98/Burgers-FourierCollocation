function burgers_solver_fft(nu, L, T, dt, N, u_init)
	"""
	Numerical solution of 1-D viscous Burgers equation on [0, L], periodic boundary
	use Fourier collocation in modal space
	use Crank-Nicolson-Leap-Frog in time
	"""
	# ordering of N wavenumbers in fft. 
	# Let fs be the sampling rate
	# In the 'fft' function, the ordering is taken to be [0 : N/2-1 -N/2 : -1]*fs for N even
	#                                                    [0 : (N-1)/2 -(N-1)/2 : -1]*fs for N odd
	# In Julia, this array is given by 'N * fftfreq(N, fs)'
	# the natural ordering usually used for plotting is [-N/2 : N/2-1]*fs, where the zero frequency is placed in the middle
	# we can recover the natural ordering by the function 'fftshift'

	k = [0:N/2-1; 0; -N/2+1:-1] * 2 * pi / L
	k1 = im * k
	k2 = k1 .^ 2
	u0 = u_init
	save_num = 100
	save_steps = floor(T / dt / save_num)
	u_data = zeros(Complex, (save_num + 1, N))
	u_data[1, :] = u_init
	t_data = zeros(Float64, save_num + 1)
	t_data[1] = 0

	global u1, u2

	now = 0
	save_idx = 1
	# # Use a backward Euler scheme for the first time step on a finer grid
	# # this is because we are dealing with oscillating initial values
	# ddt = min(dt / 10, N^(-2) / nu)
	# while now < dt
	# 	if now > dt - ddt
	# 		ddt = dt - now
	# 	end
	# 	u1_hat = (fft(u0) .- ddt * k1 .* fft(0.5 * u0 .^ 2)) ./ (ones(N) .- ddt * nu * k2)
	# 	u1 = ifft(u1_hat)
	# 	now = now + ddt
	# end
	# # otherwise just march the first time step with a single backward Euler
	u1_hat = (fft(u0) .- dt * k1 .* fft(0.5 * u0 .^ 2)) ./ (ones(N) .- dt * nu * k2)
	u1 = ifft(u1_hat)

	step = 1
	if mod(step, save_steps) == 0
		save_idx += 1
		u_data[save_idx, :] = u1
		t_data[save_idx] = now
	end

	# Use Crank-Nicolson-Leap-Frog
	dt1 = dt
	dt2 = dt
	while now < T
		if now > T - dt
			dt2 = T - now
		end
		u2_hat = ((ones(N) .+ (dt1 + dt2) / 2 * nu * k2) .* fft(u0) - (dt1 + dt2) / 2 * k1 .* fft(0.5 * u1 .^ 2)) ./ (ones(N) .- (dt1 + dt2) / 2 * nu * k2)
		u2 = ifft(u2_hat)
		u0 = u1
		u1 = u2
		if mod(step, save_steps) == 0 && now < T - dt
			save_idx += 1
			u_data[save_idx, :] = u2
			t_data[save_idx] = now
		end
		step += 1
		now += dt2
	end
	u_data[save_num+1, :] = u2
	t_data[save_num+1] = now
	u2, u_data, t_data
end

function burgers_solver_rfft(nu, L, T, dt, N, u_init)
	"""
	Numerical solution of 1-D viscous Burgers equation on [0, L], periodic boundary
	use Fourier collocation in modal space
	use Crank-Nicolson-Leap-Frog in time
	This function takes advantage of the following property of Fourier transform to gain twice the efficiency: If u is real then the û is Hermitian

	"""
	# ordering of N wavenumbers in fft. 
	# Let fs be the sampling rate
	# In the 'rfft' function, the ordering is taken to be [0 : N/2-1 ±N/2]*fs for N even
	#                                                     [0 : (N-1)/2]*fs for N odd
	# In Julia, this array is given by 'N * rfftfreq(N, fs)'

	k = (0:N/2) * 2 * pi / L
	k1 = im * k
	k2 = k1 .^ 2
	u0 = u_init
	save_num = 100
	save_steps = floor(T / dt / save_num)
	u_data = zeros(Float64, (save_num + 1, N))
	u_data[1, :] = u_init
	t_data = zeros(Float64, save_num + 1)
	t_data[1] = 0

	global u1, u2

	now = 0
	save_idx = 1
	# Use a backward Euler scheme for the first time step on a finer grid
	# this is because we are dealing with oscillating initial values
	ddt = min(dt / 10, N^(-2) / nu)
	while now < dt
		if now > dt - ddt
			ddt = dt - now
		end
		u1_hat = (rfft(u0) .- ddt * k1 .* rfft(0.5 * u0 .^ 2)) ./ (ones(N ÷ 2 + 1) .- ddt * nu * k2)
		u1 = irfft(u1_hat, N)
		now = now + ddt
	end
	# # otherwise just march the first time step with a single backward Euler
	# u1_hat = (rfft(u0) .- dt * k1 .* rfft(0.5 * u0 .^ 2)) ./ (ones(N ÷ 2 + 1) .- dt * nu * k2)
	# u1 = irfft(u1_hat, N)

	step = 1
	if mod(step, save_steps) == 0
		save_idx += 1
		u_data[save_idx, :] = u1
		t_data[save_idx] = now
	end


	# Use Crank-Nicolson-Leap-Frog
	dt1 = dt
	dt2 = dt
	while now < T
		if now > T - dt
			dt2 = T - now
		end
		u2_hat = ((ones(N ÷ 2 + 1) .+ (dt1 + dt2) / 2 * nu * k2) .* rfft(u0) - (dt1 + dt2) / 2 * k1 .* rfft(0.5 * u1 .^ 2)) ./ (ones(N ÷ 2 + 1) .- (dt1 + dt2) / 2 * nu * k2)
		u2 = irfft(u2_hat, N)
		u0 = u1
		u1 = u2
		if mod(step, save_steps) == 0 && now < T - dt
			save_idx += 1
			u_data[save_idx, :] = u2
			t_data[save_idx] = now
		end
		step += 1
		now += dt2
	end
	u_data[save_num+1, :] = u2
	t_data[save_num+1] = now
	u2, u_data, t_data
end
