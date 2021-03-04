#= 
Path: "Spinor1D" folder

1. Set physical parameters
2. Set simulation parameters
3. Intialize simulation
    - Build time-independenet Hamiltonian
4. Find ground state
    - Set intial condition
    - Launch imaginary time evolution
    - Visualize the result and check the validity
    - Save data
5. Compute the dynamics
    - Select the initial condition
    - Add noise if needed
    - Launch simulation
    - Visualize the results
    - Save data
=#

    ## Load packages
display("Loading pkgs...")
@time using QuantumOptics
@time using OrdinaryDiffEq, DiffEqCallbacks
@time using Plots, LaTeXStrings
@time using DrWatson
using Random
import DataFrames
@quickactivate "Spinor1D"

gr()

    ## Load physical constants

include(srcdir("Fundamental_constants_SI.jl"))

# Input parameters in SI unit
species     = "Rb87"
Natom 		= 1.1e6		    # atom number
f⊥ 			= 57.9         # [Hz]
fSI			= 0.0          # [Hz]
pSI 		= 0.0 * 1e-2*h # [J/m]
p_quenchSI 	= 0.0 * 1e-2*h  # [J/m]
qSI 		= 1e3*h        # [J]
q_quenchSI 	= 5.0*h       # [J]

    ## Set simulation parameters

fname = "test"

# Mersenne Twister seed for instability
MTseed1     = 2123123
MTseed2     = 312321

# Time domain
imTmax 		= 10 		# imaginary time max in units of [1/ω⊥]
dyTmax 		= 50
TsampleN    = 110       # number of points in time domain

# Space domain
nx  		= Int(2^9)	# number of spatial domain 
# calculate dependent constants
include(srcdir("Dependent_constants_Rb.jl"))
xmaxSI 		= π*50e-6
dxSI 		= 2*xmaxSI/nx

xmax 		= xmaxSI/a⊥
dx 			= 2*xmax/nx

# Set integrator tolerances
abstol_int  = 1e-6
reltol_int  = 1e-6
maxiters_int= Int(1e8)

display("c1n is " * string(c1_1DSI*npeakSI/h) * " Hz")

params = @dict species Natom f⊥ fSI pSI p_quenchSI qSI q_quenchSI MTseed1 MTseed2 imTmax dyTmax TsampleN nx xmaxSI


    ## Intialize 

# load functions
include(srcdir("Import_Functions.jl"))

# construct single particle Hamiltonian
include(srcdir("Construct_Hamiltonian.jl"))

# Preparing the initial state
# ψ0 = initialstate_gaussian(bx, Natom, σ = RTF/4/a⊥)
ψ0 = initialstate_uniform()
plot_wfn(xx_um, ψ0)
title!(srcdir("Initial guess for the ground state"))

    ## Find ground state

# Constructing total Hamiltonian
Hc0 = one(bx) ⊕ one(bx) ⊕ one(bx);
Hc1 = one(bx) ⊕ one(bx) ⊕ one(bx);
H_im = -1*im*LazySum(Hkin_FFT, Utrap, UZeeman, Hc0, Hc1);

# Set time domain and sample length
T = range(0., imTmax; length = 20)

# renormalization callback
norm_func(u, t, integrator) = normalize_array!(u)
ncb = FunctionCallingCallback(norm_func; func_everystep = true)

@time tout, ψt = timeevolution.schroedinger_dynamic(T, ψ0, Hgp_im,  callback=ncb, alg = DP5(), abstol=1e-4, reltol=1e-4, maxiters=maxiters_int)

# Plot the result
ψg = ψt[end]
plot_wfn(xx_um, ψg)
# nTF = max.(npeakSI*1e-6*(1 .- (xx_SI*1e-6/RTF).^2), 0*xx_SI) # Thomas fermi profile
# plot!(xx_SI, nTF, lw = 3, ls=:dot, label = "m0 Thoams-Fermi")
title!("End of the imaginary time evolution")
# plot(T, check_norm(ψt))



    ## Initialize quench dynamics

# Constructing interaction Hamiltonian
Hc0_dy = one(bx) ⊕ one(bx) ⊕ one(bx);
Hc1_dy = one(bx) ⊕ one(bx) ⊕ one(bx);
H_dy = LazySum(Hkin_FFT, Utrap, UZeeman_quench, Hc0_dy, Hc1_dy);

# intial condition
ψt_dy0 = copy(ψg)

# add noise
# ψnoise_array = sqrt(0.1)*(exp.(-(rand(3*nx).-0.5).^2) .+1*im*exp.(-(rand(3*nx).-0.5).^2))
ψp_noise = generate_vacuum_array(MTseed1)
ψm_noise = generate_vacuum_array(MTseed2)
# ψt_dy0.data[1:nx] =  ψt_dy0.data[1:nx] + ψp_noise
# ψt_dy0.data[2*nx+1:3*nx] =  ψt_dy0.data[2*nx+1:3*nx] + ψm_noise
ψt_dy0.data[1:nx] =  ψp_noise
# ψt_dy0.data[nx+1:2*nx] = 70.4 * ones(nx)
ψt_dy0.data[2*nx+1:3*nx] = ψm_noise

# check the noise
plot_wfn(xx, ψt_dy0)
plot!(xx, abs2.(ψp_noise[1:nx]))
# plot!(xx, abs2.(ψnoise_array[2*nx+1:3*nx]))

    
    ## Compute the quench dynamics

# Set time domain and sampling size
T_dy = range(0, dyTmax; length = TsampleN)

# Solve
@time tout_dy, ψt_dy = timeevolution.schroedinger_dynamic(T_dy, ψt_dy0, Hgp_dy, alg = Tsit5(), abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int)

# Check the results
plot_wfn(xx_um, ψt_dy[end])

## save data
data_dynamics = insert_ψt(params, ψt_dy)
safesave(datadir("sims","Saito",  fname*".bson") ,data_dynamics)

## check some plots
plot_pop_dynamics(T_dy/ω⊥ *1e3, ψt_dy)
ylabel!("Total atom number")
xlabel!("Time [ms]")

## test spatial distribution
plt = plot_Fplus(ψt_dy[25])
savefig(plt, plotsdir("Snapshot_"*fname*".png"))