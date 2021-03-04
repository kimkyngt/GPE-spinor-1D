N= 512;
% Number of Fourier modes
dt = .001;
% Time step
tfinal = 100;
% Final time
M= round(tfinal./dt); % Total number of time steps
J= 100;
% Steps between output
L= 50;
% Space period
h= L/N;
% Space step
n=( -N/2:1:N/2-1)'; % Indices
x= n*h;
% Grid points
u= exp(1i*x).*sech(x);%Intial condition
k= 2*n*pi/L;
% Wavenumbers. 
plot(abs(u).^2); hold on
tic
for m = 1:1:M % Start time loop
u = exp(dt*1i*(abs(u).*abs(u))).*u; % Solve non-linear part of NLSE 
c = fftshift(fft(u));
% Take Fourier transform
c = exp(-dt*1i*k.*k/2).*c;
% Advance in Fourier space
u = ifft(fftshift(c));
% Return to Physical Space
if mod(m, 100) == 0
    clf()
    plot(abs(u).^2); 
    pause(0.1)
end
end
toc