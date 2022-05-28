### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ cafebddc-b7dd-11eb-2697-f9a62e245037
using QuantumOptics, OrdinaryDiffEq, DiffEqCallbacks, 
DrWatson, JLD2, Random, PlutoUI, Statistics, Plots, Plots.PlotMeasures, 
LaTeXStrings, PhysicalConstants, PhysicalConstants.CODATA2018, Unitful

# ╔═╡ 57433548-e331-47ad-88ea-5b9a8844ac66
begin
	## Default plotting setting
	gr()
	default_font = "Arial"
	default(titlefont = (10, default_font, :black), legendfont = (8, default_font, :black), guidefont = (10, default_font, :black), tickfont = (9, default_font, :black), framestyle = :box, size = (500, 300), dpi = 200, lw = 2)
end

# ╔═╡ 29a3ff91-64fa-4a28-ad34-e5bbd8feba4e
begin
	c = ustrip(SpeedOfLightInVacuum)
	h = ustrip(PlanckConstant)
	kB = ustrip(BoltzmannConstant)
	ħ = ustrip(h/(2*π) )
	amu = ustrip(AtomicMassConstant)
	a0 = ustrip(BohrRadius)
	μB = ustrip(BohrMagneton)

	## Properties of Na23
	a0Na = 50.0*a0
	a2Na = 55.0*a0
end;

# ╔═╡ c1068a05-2b3e-4071-99de-acf4ec11f33a
md"""
### Computation control checkboxs for modification
**Suppress solver calling during the code modification** 
- Run imaginary time solver? $(@bind token_imgevol CheckBox(default=false)) 
- Run quench dynamics solver? $(@bind token_quenchdy CheckBox(default=false))
- Save data? $(@bind token_save CheckBox(default=false))
""" 

# ╔═╡ 8625a455-7c05-45b9-87e6-f25ecdbe1770
begin
	# Input parameters in SI unit
    species     = "Li7"
    Natom 		= 2e4		    # atom number
    f⊥ 			= 230        # [Hz]
    fSI			= 0.0         # [Hz]
    pSI 		= 0.1e-3 * 1e-2*h # [J/m]
    p_quenchSI 	= 0.0 * 1e-2*h  # [J/m]
    qSI 		= 3e3*h        # [J]
    q_quenchSI 	= 0*h       # [J]
	
	    # File name for saving
    fname = "Journal club"

	# Instability seed related
    MTseed      = 1234   	# Mersenne Twister seed for {+1, 0, -1}
	trWigner 	= false # select type of the noise
	Nvirtual 	= 0.1  	# number of virtual particle number feeded in noise

    # Time domain
    imTmax 		= 150 		# imaginary time max in units of [1/ω⊥]
    dyTmax 		= 300       # dynamics max time [1/ω⊥]
    TsampleN    = 100       # number of sample points for dynamics

    # Spaitial domain
    nx  		= Int(2^7)	# number of spatial domain 
	xmaxSI 		= 232e-6	# maximum single-sided domain length [m]
    
	# Set integrator tolerances
    abstol_int  = 1e-6
    reltol_int  = 1e-6
    maxiters_int= Int(1e5)
	
	# Collect parameters for saving
    params = @strdict species Natom f⊥ fSI pSI p_quenchSI qSI q_quenchSI MTseed trWigner Nvirtual imTmax dyTmax TsampleN nx xmaxSI 
end;

# ╔═╡ b03c3c39-2a7f-4a44-8518-316bb4618add
begin
	# dependent in SI
	MNa 		= 23*amu
	ω⊥ 			= 2*π*f⊥
	a⊥ 			= sqrt(ħ/MNa/ω⊥)
	ωSI  		= 2*π*fSI
	axSI 		= sqrt(ħ/MNa/ωSI)

	c0SI 		= 4*π*ħ^2/MNa * (2*a2Na + a0Na)/3
	c1SI 		= 4*π*ħ^2/MNa * (a2Na - a0Na)/3
	c0_1DSI 	= c0SI/(2*π*a⊥^2)
	c1_1DSI 	= c1SI/(2*π*a⊥^2)
	RTF 		= (3*c0_1DSI*Natom/(2*MNa*ωSI^2))^(1/3)
	if RTF == Inf
		npeakSI = Natom/(2*xmaxSI)
	else
		npeakSI = 0.5*MNa*ωSI^2*RTF^2/c0_1DSI
	end
	
	# normalized
	γ 			= ωSI/ω⊥
	p 			= pSI/(ħ*ω⊥) 
	q 			= qSI/(ħ*ω⊥)
	p_quench 	= p_quenchSI/(ħ*ω⊥) 
	q_quench 	= q_quenchSI/(ħ*ω⊥)
	c0 			= c0_1DSI/(ħ*ω⊥*a⊥)
	c1 			= c1_1DSI/(ħ*ω⊥*a⊥)
    # xmaxSI 		= xmax_factor*RTF
    dxSI 		= 2*xmaxSI/nx
    xmax 		= xmaxSI/a⊥
    dx 			= 2*xmax/nx

end;

# ╔═╡ 2cfe5770-2861-4ceb-acbf-258908f2fe7a
begin
	# basis generation
	bx = PositionBasis(-xmax, xmax, nx)
	xx = samplepoints(bx)
	xx_um = a⊥*xx *1e6
	
	bp = MomentumBasis(bx)
	pp = samplepoints(bp)
	
	# Basic operators, {+1, 0, -1} order
	x  = position(bx) ⊕ position(bx) ⊕ position(bx)
	Px = momentum(bp) ⊕ momentum(bp) ⊕ momentum(bp)
	
	# transformation operators
	Txp = transform(bx, bp) ⊕ transform(bx, bp) ⊕ transform(bx, bp)
	Tpx = transform(bp, bx) ⊕ transform(bp, bx) ⊕ transform(bp, bx)
	
	## Single particle Hamiltonian (time - independent)
	# kinetic energy opeartor
	Hkin = Px^2/2.  
	Hkin_FFT = LazyProduct(Txp, Hkin, Tpx) 
	# harmonic potential operator
	Utrap = 0.5 * (γ^2 * x^2) 
	
	# Zeeman shifts
	UZeeman = - p*( position(bx) ⊕ 0*position(bx) ⊕ -1*position(bx) ) + q*(one(bx) ⊕ (0*one(bx)) ⊕ one(bx))
	UZeeman_quench = - p_quench*( position(bx) ⊕ 0*position(bx) ⊕ -1*position(bx) ) + q_quench*(one(bx) ⊕ 0*one(bx) ⊕ one(bx))
end;

# ╔═╡ 037e195c-05e7-4cb2-88f5-857b5bedd71e
function convert_wfn_norm2SI(ψ)
	ψSI = copy(ψ)
	ψSI.data  = ψSI.data/sqrt(a⊥)
	return ψSI
end

# ╔═╡ c284e095-edc9-451d-a35e-cc107db4698c
function normalize_wfn(ψ, N)
	Δx = spacing(ψ.basis.bases[1])
	ψ = ψ/(norm(ψ) / sqrt(N/Δx))
	return ψ
end

# ╔═╡ 5394d18f-5624-4b6c-aa2b-4b9c2bf90761
function initialstate_gaussian(bx, N; σ::Float64 = 1.0, spinratio = [1,1,1])
	ϕin_p = spinratio[1]*gaussianstate(bx, 0, 0, σ)
	ϕin_0 = spinratio[2]*gaussianstate(bx, 0, 0, σ)
	ϕin_m = spinratio[3]*gaussianstate(bx, 0, 0, σ)
	ψgaussian = normalize_wfn(ϕin_p ⊕ ϕin_0 ⊕ ϕin_m, N)
	return ψgaussian
end

# ╔═╡ cace2401-882a-4b15-8db4-0c5d5075cd76
function plot_wfn(x, ψ0; kwargs...)
	ψ = convert_wfn_norm2SI(ψ0)
	n1 = abs2.(getblock(ψ,1).data)*1e-6
	n2 = abs2.(getblock(ψ,2).data)*1e-6
	n3 = abs2.(getblock(ψ,3).data)*1e-6
	plt = plot(framestyle = :box, xlabel= "Position [μm]", 
		ylabel = "Density [1/μm]" ; kwargs...)
	plot!(x, n1,  label = "+1",  ls = :solid, fillrange = 0, fillalpha = 0.35)
	plot!(x, n2,  label = "0",  ls = :dot, fillrange = 0, fillalpha = 0.35)
	plot!(x, n3, label = "-1",  ls = :dash, fillrange = 0, fillalpha = 0.35)
	plot!(x, (n1+n2+n3), label = "sum")
	return plt
end

# ╔═╡ 0dba4b2b-8b4e-40f2-bda9-c6311b609bb6
function plot_snapshots(xx, ψt, T; kwargs...)
	    function plot_one(xx, ψ; timestamp::Float64 = -1)
	        # Plot wavefunction in SI units
	        n1 = abs2.(getblock(ψ,1).data) /(a⊥/1e-6)
	        n2 = abs2.(getblock(ψ,2).data) /(a⊥/1e-6)
	        n3 = abs2.(getblock(ψ,3).data) /(a⊥/1e-6)
			plt = plot_wfn(xx,ψ; size = (300, 200))
 			annotate!(0.0, maximum(n1+n2+n3)*0.9, text( string(round(timestamp*1e3))*" ms", 12 ) )
			return plt
	    end
	
	    Npsi = length(ψt)
	
	    plt1 =  plot_one(xx, ψt[1], timestamp = T[1]/ω⊥)
		plt2 = plot_one(xx, ψt[Int(round(Npsi*2/9))], timestamp = T[Int(round(Npsi*2/9))]/ω⊥)
		plt3 = plot_one(xx, ψt[Int(round(Npsi*3/9))], timestamp = T[Int(round(Npsi*3/9))]/ω⊥)
		plt4 = plot_one(xx, ψt[Int(round(Npsi*4/9))], timestamp = T[Int(round(Npsi*4/9))]/ω⊥)
		plt5 = plot_one(xx, ψt[Int(round(Npsi*5/9))], timestamp = T[Int(round(Npsi*5/9))]/ω⊥)
		plt6 = plot_one(xx, ψt[Int(round(Npsi*6/9))], timestamp = T[Int(round(Npsi*6/9))]/ω⊥)
		plt7 = plot_one(xx, ψt[Int(round(Npsi*7/9))], timestamp = T[Int(round(Npsi*7/9))]/ω⊥)
		plt8 = plot_one(xx, ψt[Int(round(Npsi*8/9))], timestamp = T[Int(round(Npsi*8/9))]/ω⊥)
		plt9 = plot_one(xx, ψt[Int(round(Npsi*9/9))], timestamp = T[Int(round(Npsi*9/9))]/ω⊥)
	
		plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, layout = (3,3), legend = false, size = (900, 600); kwargs...)
end

# ╔═╡ Cell order:
# ╠═cafebddc-b7dd-11eb-2697-f9a62e245037
# ╠═57433548-e331-47ad-88ea-5b9a8844ac66
# ╠═29a3ff91-64fa-4a28-ad34-e5bbd8feba4e
# ╠═c1068a05-2b3e-4071-99de-acf4ec11f33a
# ╠═8625a455-7c05-45b9-87e6-f25ecdbe1770
# ╠═b03c3c39-2a7f-4a44-8518-316bb4618add
# ╠═2cfe5770-2861-4ceb-acbf-258908f2fe7a
# ╠═5394d18f-5624-4b6c-aa2b-4b9c2bf90761
# ╠═037e195c-05e7-4cb2-88f5-857b5bedd71e
# ╠═c284e095-edc9-451d-a35e-cc107db4698c
# ╠═cace2401-882a-4b15-8db4-0c5d5075cd76
# ╠═0dba4b2b-8b4e-40f2-bda9-c6311b609bb6
