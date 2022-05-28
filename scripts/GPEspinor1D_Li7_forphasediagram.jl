## Load packages
println("Loading pkgs...")
@time using QuantumOptics
@time using OrdinaryDiffEq, DiffEqCallbacks
@time using Plots, LaTeXStrings
@time using DrWatson
@time using JLD2
using Random
@quickactivate "Spinor1D"
include(srcdir("default_plotting_setting.jl"))
## Set physical parameters
begin
    include(srcdir("Fundamental_constants_SI.jl"))

    # Input parameters in SI unit
    species     = "Li7"
    Natom 		= 1e4		    # atom number
    f⊥ 			= 1.2e3         # [Hz]
    fSI			= 10.0          # [Hz]
    pSI 		= 1e-3 * 1e-2*h # [J/m]
    p_quenchSI 	= 0.0 * 1e-2*h  # [J/m]
    qSI 		= -11*h        # [J]
    q_quenchSI 	= 0e3*h       # [J]
end

## Set simulation parameters
begin
    fname = "test"

    # Mersenne Twister seed for instability
    MTseed      = [1 2 3]
    # Time domain
    imTmax 		= 6000		# imaginary time max in units of [1/ω⊥]
    dyTmax 		= 1   
    TsampleN    = 10       # number of points in time domain

    # Space domain
    nx  		= Int(2^8)	# number of spatial domain 
    # calculate dependent constants
    include(srcdir("Dependent_constants_Li.jl"))
    xmaxSI 		= 1.2*RTF
    dxSI 		= 2*xmaxSI/nx

    xmax 		= xmaxSI/a⊥
    dx 			= 2*xmax/nx

    # Set integrator tolerances
    abstol_int  = 1e-6
    reltol_int  = 1e-6
    maxiters_int= Int(1e6)

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
    Hc0 = one(bx) ⊕ one(bx) ⊕ one(bx)
    Hc1 = one(bx) ⊕ one(bx) ⊕ one(bx)
    H_im = -1*im*LazySum(Hkin_FFT, Utrap, UZeeman, Hc0, Hc1)

    # Set time domain and sample length
    T = range(0., imTmax; length = 20)

    # renormalization callback
    norm_func(u, t, integrator) = normalize_array!(u)
    ncb = FunctionCallingCallback(norm_func; func_everystep = true)

    @time tout, ψt = timeevolution.schroedinger_dynamic(T, ψ0, Hgp_im,  callback=ncb, alg = DP5(), abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int)
    ψgpe = ψt[end]
end



##
function plot_spinor_density(ψ; kwargs...)
    x = range(-xmaxSI, xmaxSI, length = nx)*1e6
    n1 = abs2.(getblock(ψ,1).data) /a⊥*1e-6
    n2 = abs2.(getblock(ψ,2).data) /a⊥*1e-6
    n3 = abs2.(getblock(ψ,3).data) /a⊥*1e-6
    plt = plot(x, n1,  label = L"n_{1}", lw = 1,fillrange = 0, fillalpha = 0.35, framestyle = :box; kwargs...)
    plot!(x, n2,  label = L"n_{0}", lw = 1, fillrange = 0, fillalpha = 0.35)
    plot!(x, n3, label = L"n_{-1}", lw = 1, fillrange = 0, fillalpha = 0.35)    
    plot!(x, n1+n2+n3, label = L"n", lw = 1.5)
    # title!(string(sum(n1+n2+n3)*dxSI*1e6))
    return plt
end

plt = plot_spinor_density(ψgpe; grid = false)
xlabel!("Position [µm]")
ylabel!("Atoms/µm")
display(plt)
# savefig(plotsdir("thesis_figure", "ba_ground.svg"))