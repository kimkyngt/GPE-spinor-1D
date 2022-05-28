using QuantumOptics
using PyPlot

# System Parameters
m = 1.
ω = 0.5 # Strength of trapping potential

# Position Basis
xmin = -5
xmax = 5
Npoints = 1000
b_position = PositionBasis(xmin, xmax, Npoints)
b_momentum = MomentumBasis(b_position)
# Transforms a state multiplied from the right side from real space
# to momentum space.
T_px = transform(b_momentum, b_position)

T_xp = dagger(T_px)

x = position(b_position)
p = momentum(b_momentum)

H_kin = LazyProduct(T_xp, p^2/2m, T_px)
V = ω*x^2
H = LazySum(H_kin, V)

# Initial state
x0 = 1.5
p0 = 0
sigma0 = 0.6
Ψ0 = gaussianstate(b_position, x0, p0, sigma0);

# Time evolution
T = range(0, 3*pi, length = 100)
@time tout, Ψt = timeevolution.schroedinger(T, Ψ0, H);

# Plot dynamics of particle density
x_points = samplepoints(b_position)

n = abs.(Ψ0.data).^2
V = ω*x_points.^2
C = maximum(V)/maximum(n)

fig = figure(figsize=(6,6), dpi = 300)
xlabel(L"x")
ylabel(L"| \Psi(t) |^2")
plot(x_points, (V.-3)./C, "k--")

for i=1:length(T)
    Ψ = Ψt[i]
    n .= abs.(Ψ.data).^2
    plot(x_points, n.+0.005*i, "C0", alpha=0.5*(float(i)/length(T))^8+0.5)
end
display(fig)