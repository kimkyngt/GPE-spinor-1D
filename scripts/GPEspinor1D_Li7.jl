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
println("Loading pkgs...")
@time using QuantumOptics
@time using OrdinaryDiffEq, DiffEqCallbacks
@time using Plots, LaTeXStrings
@time using DrWatson
@time using JLD2
using Random
@quickactivate "Spinor1D"
gr()

## Set physical parameters
begin
    include(srcdir("Fundamental_constants_SI.jl"))

    # Input parameters in SI unit
    species     = "Li7"
    Natom 		= 1e4		    # atom number
    f⊥ 			= 1.2e3         # [Hz]
    fSI			= 10.0          # [Hz]
    pSI 		= 0.0e-3 * 1e-2*h # [J/m]
    p_quenchSI 	= 0.0 * 1e-2*h  # [J/m]
    qSI 		= 3e3*h        # [J]
    q_quenchSI 	= 0e3*h       # [J]
end

## Set simulation parameters
begin
    fname = "test"

    # Mersenne Twister seed for instability
    MTseed      = [1 2 3]
    # Time domain
    imTmax 		= 100 		# imaginary time max in units of [1/ω⊥]
    dyTmax 		= 500   
    TsampleN    = 100       # number of points in time domain

    # Space domain
    nx  		= Int(2^8)	# number of spatial domain 
    # calculate dependent constants
    include(srcdir("Dependent_constants_Li.jl"))
    xmaxSI 		= 1.5*RTF
    dxSI 		= 2*xmaxSI/nx

    xmax 		= xmaxSI/a⊥
    dx 			= 2*xmax/nx

    # Set integrator tolerances
    abstol_int  = 1e-6
    reltol_int  = 1e-6
    maxiters_int= Int(1e8)

    params = @strdict species Natom f⊥ fSI pSI p_quenchSI qSI q_quenchSI MTseed imTmax dyTmax TsampleN nx xmaxSI
    c1_1DSI*npeakSI/h
end

## Intialize simulation. generate basis and prepare initial state
begin
    # load functions
    include(srcdir("Import_Functions.jl"))

    # construct single particle Hamiltonian
    include(srcdir("Construct_Hamiltonian.jl"))

    # Preparing the initial state
    ψ0 = initialstate_gaussian(bx, Natom, σ = RTF/2/a⊥)
    # ψ0 = initialstate_uniform() 
    plot_wfn(xx_um, convert_wfn_norm2SI(ψ0, a⊥))
    xlabel!("Position [μm]")
    ylabel!("Density [1/m]")    
    title!("Initial guess for the ground state")
end

## Check potential
begin
    plot(xx_um, real([Utrap.data[ii, ii] for ii = 1:nx] ), label = "Trapping", lw = 3)
    plot!(xx_um, real([UZeeman_quench.data[ii, ii] for ii = 1:nx]) , label = "Zeeman energy" , frame = :box, legend = :top, lw = 3)
    xlabel!("Position [μm]")
    ylabel!("Energy [ħω⊥]")
end


## Find ground state
begin
    # Constructing total Hamiltonian
    Hc0 = one(bx) ⊕ one(bx) ⊕ one(bx);
    Hc1 = one(bx) ⊕ one(bx) ⊕ one(bx);
    H_im = -1*im*LazySum(Hkin_FFT, Utrap, UZeeman, Hc0, Hc1);

    # Set time domain and sample length
    T = range(0., imTmax; length = 20)

    # renormalization callback
    norm_func(u, t, integrator) = normalize_array!(u)
    ncb = FunctionCallingCallback(norm_func; func_everystep = true)

    @time tout, ψt = timeevolution.schroedinger_dynamic(T, ψ0, Hgp_im,  callback=ncb, alg = DP5(), abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int)
    
    # check the result
    plot_snapshots(xx_um, ψt, T)
end

## Check the ground state with Thomas-Fermi
begin 
    ψg = ψt[end]
    plot_wfn(xx_um, ψg)
    nTF = max.(npeakSI*1e-6*(1 .- (xx_um*1e-6/RTF).^2), 0*xx_um) # Thomas fermi profile
    plot!(xx_um, nTF, lw = 3, ls=:dot, label = "m0 Thoams-Fermi")
    title!("Found ground state")
    # plot(T, check_norm(ψt))
end

## Check the phase
begin
    plot_wfn_phase(xx_um, ψg)
end


## Initialize the quench dynamics
begin
    # intial condition
    ψt_dy0 = copy(ψg)

    # Constructing interaction Hamiltonian
    Hc0_dy = one(bx) ⊕ one(bx) ⊕ one(bx);
    Hc1_dy = one(bx) ⊕ one(bx) ⊕ one(bx);
    H_dy = LazySum(Hkin_FFT, Utrap, UZeeman_quench, Hc0_dy, Hc1_dy);

    # add noise
    noise_amplitude = 1e-3 ;#sqrt(npeakSI*1e-6)*1e-3;    

    ψt_dy0.data[1:nx] = abs.(ψt_dy0.data[1:nx]) + noise_amplitude*generate_vacuum_array(MTseed[1], uniform = 1)
    ψt_dy0.data[nx+1:2*nx] = abs.(ψt_dy0.data[nx+1:2*nx]) + noise_amplitude*generate_vacuum_array(MTseed[2], uniform = 1)
    ψt_dy0.data[2*nx+1:3*nx] = abs.(ψt_dy0.data[2*nx+1:3*nx]) + noise_amplitude*generate_vacuum_array(MTseed[3], uniform = 1)

    # check the noise
    plot_wfn(xx_um, ψt_dy0)
    # plot_wfn_phase(xx_um, ψt_dy0)
end
    

## Compute the quench dynamics
begin
    # Set time domain and sampling size
    T_dy = range(0, dyTmax; length = TsampleN)

    # Solve
    @time tout_dy, ψt_dy = timeevolution.schroedinger_dynamic(T_dy, ψt_dy0, Hgp_dy, alg = DP5(), abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int)

    # Check the results
    plot_snapshots(xx_um, ψt_dy, T_dy)
end


## save data
begin
    data_dynamics = insert_ψt(params, ψt_dy)
    safesave(datadir("sims","Li7", fname*".jld2") ,data_dynamics)
end


## check some plots
begin
    plot_pop_dynamics(T_dy/ω⊥ *1e3, ψt_dy)
    ylabel!("Total atom number")
    xlabel!("Time [ms]")
end

## Save to animation
begin
    anim = @animate for ii  = 1:length(ψt_dy)
        plot_wfn(xx_um, ψt_dy[ii]; ylims = (0, 150))
        annotate!(0, 140, string(round(T_dy[ii]/ω⊥*1e3))*" ms")
    end
    gif(anim, plotsdir("Li7", fname*".gif"), fps = 10) 
end