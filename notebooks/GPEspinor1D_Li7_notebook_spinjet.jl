### A Pluto.jl notebook ###
# v0.14.7

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

# ╔═╡ e633a360-b643-11eb-3979-6f8b7a3551c4
using QuantumOptics, OrdinaryDiffEq, DiffEqCallbacks, 
DrWatson, JLD2, Random, PlutoUI, Statistics, Plots, Plots.PlotMeasures, 
LaTeXStrings, PhysicalConstants, PhysicalConstants.CODATA2018, Unitful

# ╔═╡ c6450a7c-ecba-4f85-98ef-3d7a007dc64f
md"""
# Gross-Piatevskii equation for 1D spinor gas of Li7 - quench dynamics
Here is GPE to solve. 
```math
\begin{equation}
    \begin{split}
        i\hbar\frac{\partial\psi_m}{\partial t} &= \frac{\delta E}{\delta \psi^*_m} \\
    & = \left[ -\frac{\hbar^2}{2M}\nabla^2 + U_{\text{trap}}(\mathbf{r}) - pm + qm^2 \right] \psi_m + c_0 n \psi_m + c_1 \sum_{m'=-1}^{1} \mathbf{F} \cdot \mathbf{f}_{mm'} \psi_{m'}
    \end{split}
\end{equation}
```
"""

# ╔═╡ bdc7e2e3-468f-4313-8154-82375ecc7c87
md"""
We normalize the equation in the following way.
```math
\begin{equation}
    \begin{split}
        i\dot{\psi_1}&=  
    \left[ 
    -\frac{1}{2}\partial^2_x + \frac{1}{2}\gamma^2 x^2 - \tilde{p} + \tilde{q} \right] \psi_1 
    + \bigg[ \tilde{c}_0 n + \tilde{c}_1(|\psi_1|^2 -|\psi_{-1}|^2 + |\psi_0|^2)\bigg]\psi_1   + \tilde{c}_1\psi^*_{-1}\psi_0^2 \\
        i\dot{\psi_0}&=  
    \left[ 
    -\frac{1}{2}\partial^2_x + \frac{1}{2}\gamma^2 x^2 \right] \psi_0 
    + \bigg[ \tilde{c}_0 n + \tilde{c}_1(|\psi_1|^2 + |\psi_{-1}|^2) \bigg] \psi_0 + 2\tilde{c}_1\psi^*_0\psi_{-1}\psi_1\\
        i\dot{\psi_{-1}} &= 
    \left[ 
    -\frac{1}{2}\partial^2_x + \frac{1}{2}\gamma^2 x^2 + \tilde{p} + \tilde{q} \right] \psi_{-1} 
    + \bigg[\tilde{c}_0 n + \tilde{c}_1 (|\psi_{-1}|^2 -|\psi_{1}|^2 + |\psi_0|^2) \bigg]\psi_{-1}   + \tilde{c}_1\psi^*_1\psi_0^2
    \end{split}
\end{equation}
```
"""

# ╔═╡ 14eaa4d0-7f90-4195-b627-6eafbd9a9e8a
md"""
- Unit conversion reminder.
```math 
\begin{gather}
x \rightarrow a_{\bot}x, \quad
t \rightarrow t/\omega_{\bot}, \quad
\psi \rightarrow \psi/\sqrt{a_\bot} \\
\gamma = \frac{\omega}{\omega_{\bot}}, \quad
\tilde{p} = \frac{p}{\hbar\omega_{\bot}}, \quad \tilde{q} = \frac{q}{\hbar\omega_{\bot}} \\
\tilde{c}_0 =\frac{c_0}{\hbar\omega_{\bot}a_{\bot}}, \quad \tilde{c}_1  =\frac{c_1}{\hbar\omega_{\bot}a_{\bot}}. \\
c_0^{(1D)} = c_0^{(3D)}/2\pi a_{\bot}^2, \quad c_1^{(1D)} = c_1^{(3D)}/2\pi a_{\bot}^2
\end{gather} 
```
"""

# ╔═╡ 5e7fce10-8003-46df-aab7-0af1046c30e0
md"""
#### Notes on computation
- Links to resources: [Intro to Julia and Pluto notebook](https://www.youtube.com/watch?v=OOjKEgbt8AI&list=PLP8iPy9hna6Q2Kr16aWPOKE0dz9OnsnIJ&index=3), [QauntumOptics.jl pakcage](https://qojulia.org/)
- The simulation time nonlinearly increase to the domain size. It is due to the stability of the solver. The time step, $\Delta t$ should be small enough to satisfy [PRE 76, 056708 (2007), PRE 93, 053309 (2016)] 

$\Delta t \leq t_{\text{stab}} \equiv \frac{\pi}{{|\frac{1}{2}k^2_{\text{max}} + q|}},$

where $k_{\text{max}} = \pi/\Delta x$ is the maximum spiatial frequency. One can expect that the computation time will increase as $N_{\text{domain}}^3$
- Sum basis order: (1's position basis) $\oplus$ (0's position basis) $\oplus$ (-1's position basis). Use `getblock()` to access each spin component. 
- Position basis has periodic boundary condition.
- See `timeevolution_base.jl` for the information about the solver. It uses DP5(ode45 in MATLAB) method in OrdinaryDiffEq.jl. We can hand over the algorithm of our choice to DifferetialEquations.jl using `alg` keyword argument.
- Relative tolerance and absolute tolerance. `reltol` impose the stopping condition like `(1-u[i]/u[i-1]) <  reltol` and `abtol` for `(u[i]-u[i-1]) <  abtol`.
- To use functionnality of `DrWatson.jl`, proejct name, `Spinor1D` should ebe activated.
- Random number for each simulation can be tracked by `MTseed` variable.
"""

# ╔═╡ 4bfb2241-5a5b-4dc8-aac1-4e8b0f53a33c
md"""
#### About initial noise for quench dynamics
We introduce a noise in order to seed the instability based on **truncated Wigner approximation** (TWA).
See the references for more details [PRA 76, 043613 (2007), PRA 94, 023608 (2016), Adv. in Phys. 47, 363 (2008)].

$\mathbf{\delta}(\mathbf{x}) = \sum_{\mathbf{k}} \begin{pmatrix}
            \alpha^{+}_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{x}} \\
            \alpha^{0}_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{x}} \\
            \alpha^{-}_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{x}}.
\end{pmatrix},\quad \langle \alpha_{m, k}^* \alpha_{m, k}\rangle = \frac{1}{2}.$

- When we consider $M$ modes, this method introduce $M/2$ virtual particle for each modes [Adv. in Phys. 47, 363 (2008)] if we choose the vacuum mode properly, i.e., orthogonal to the condensates.
- One need to introduce UV cutoff considering the energy scale of C-field region. In order to simulate jets, we need to inlcude many modes, therefore, large number of virtual particles should be added(can be thousands). Being not confident with this, we also implement uniform random noise whose input virtual particle number controls the amplitude. 
- For both method, we manually orthogonalize the vacuum mode to the condensate using Gram-Schmidt process.
"""

# ╔═╡ fd11bf5d-66c8-45e0-80f5-40386c7c9c06
md"""
## 0. Workflow
1. Define parameters such as $N_{\text{atom}}$, $f_{\bot}$, $q$, etc. 
2. Find ground state using the imaginary time evolution. 
- Prepare initial guess wavefunction.
- Evolve GPE with $-i\hat{H}$ with the number constraint and check the result.
3. Compute the quench dynamics
- Enter fluctuation seeds.
- Evolve GPE.
4. Check results, visualize, and save data. 

"""

# ╔═╡ 6d9b9a15-107e-4763-b670-264b5b019152
md"#### Initialize"

# ╔═╡ 3241b4ff-86d5-4bbb-81fa-d527398792b5
md"Load packages"

# ╔═╡ 6003638d-a1a9-4ae9-be69-232c2d1543d8
projectname()

# ╔═╡ 69cb4cbc-572d-4b57-98cd-ef297927e2b8
PlutoUI.TableOfContents()

# ╔═╡ 84baced5-832d-4855-ba5a-9380571c65fd
md"Load default plotting setting"

# ╔═╡ 8d94d626-3d97-456c-8fea-1e0feb5794d7
begin
	## Default plotting setting
	gr()
	default_font = "Arial"
	default(titlefont = (10, default_font, :black), legendfont = (8, default_font, :black), guidefont = (10, default_font, :black),guidefontcolor = :black, tickfont = (9, default_font, :black), framestyle = :box, size = (500, 300), dpi = 200, lw = 1, axiscolor = :black)
end

# ╔═╡ 709d22eb-46f7-4af9-ae5c-2c2ba49e7b06
md"""
## Computation control **checkboxs** for modification
**Suppress solver calling during the code modification** 
- Run imaginary time solver? $(@bind token_imgevol CheckBox(default=false)) 
- Run quench dynamics solver? $(@bind token_quenchdy CheckBox(default=false))
- Save data? $(@bind token_save CheckBox(default=false))
""" 

# ╔═╡ 8a9a006b-f778-4d7b-8814-4c3869d07a76
begin
	c = ustrip(SpeedOfLightInVacuum)
	h = ustrip(PlanckConstant)
	kB = ustrip(BoltzmannConstant)
	ħ = ustrip(h/(2*π) )
	amu = ustrip(AtomicMassConstant)
	a0 = ustrip(BohrRadius)
	μB = ustrip(BohrMagneton)

	## Properties of Li7
	a2Li = 6.8*a0
	a0Li = 23.9*a0
	md"#### Load physical constants"
end

# ╔═╡ d34d395d-ff80-4f72-8aac-47366ee7aa06
md"## 1. Define system parameters
All input parameters for the simulation is in the following block."

# ╔═╡ 0075ee54-d4e0-43d7-92b9-24971e92c108
md"""
### Parameter control block
"""

# ╔═╡ 55ff32a3-0e3d-4a66-8764-ce3fee1fd20c
begin
	# Input parameters in SI unit
    species     = "Li7"
    Natom 		= 1e4		    # atom number
    f⊥ 			= 1.2e3        # [Hz]
    fSI			= 10.0         # [Hz]
    pSI 		= 0.0e-3 * 0.7e6 * 1e2*h # [J/m]
    p_quenchSI 	= 0 * 1e-2*h  # [J/m]
    qSI 		= 1000*h        # [J]
    q_quenchSI 	= -1.0e3*h       # [J]
	
	    # File name for saving
    fname = "scalar_q1.0kHz_seed10"

	# Instability seed related
    MTseed      = rand(1:100000000)# Mersenne Twister seed for {+1, 0, -1}
	alpha_m 	= [1, 0, 1] 	# noise amplitude ratio of {+1, 0, -1}
	trWigner 	= false # select type of the noise
	Nvirtual 	= 10  	# number of virtual particle number feeded in noise

    # Time domain
    imTmax 		= 150 		# imaginary time max in units of [1/ω⊥]
    dyTmax 		= 400       # dynamics max time [1/ω⊥]
    TsampleN    = 100       # number of sample points for dynamics

    # Spaitial domain
    nx  		= Int(2^10)	# number of spatial domain 
	xmaxSI 		= 250e-6	# maximum single-sided domain length [m]
    
	# Set integrator tolerances
    abstol_int  = 1e-6
    reltol_int  = 1e-6
    maxiters_int= Int(1e8)
	
	# Collect parameters for saving
    params = @strdict species Natom f⊥ fSI pSI p_quenchSI qSI q_quenchSI MTseed alpha_m trWigner Nvirtual imTmax dyTmax TsampleN nx xmaxSI 
end;

# ╔═╡ 1423582b-6db1-4a00-968c-98f831058017
begin
	# dependent in SI
	MLi7 		= 7*amu
	ω⊥ 			= 2*π*f⊥
	a⊥ 			= sqrt(ħ/MLi7/ω⊥)
	ωSI  		= 2*π*fSI
	axSI 		= sqrt(ħ/MLi7/ωSI)

	c0SI 		= 4*π*ħ^2/MLi7 * (2*a2Li + a0Li)/3
	c1SI 		= 4*π*ħ^2/MLi7 * (a2Li - a0Li)/3
	c0_1DSI 	= c0SI/(2*π*a⊥^2)
	c1_1DSI 	= c1SI/(2*π*a⊥^2)
	RTF 		= (3*c0_1DSI*Natom/(2*MLi7*ωSI^2))^(1/3)
	if RTF == Inf
		npeakSI = Natom/(2*xmaxSI)
	else
		npeakSI = 0.5*MLi7*ωSI^2*RTF^2/c0_1DSI
	end
	
	# normalized
	γ 			= ωSI/ω⊥
	p 			= pSI/(ħ*ω⊥)*a⊥ 
	q 			= qSI/(ħ*ω⊥)
	p_quench 	= p_quenchSI/(ħ*ω⊥)*a⊥ 
	q_quench 	= q_quenchSI/(ħ*ω⊥)
	c0 			= c0_1DSI/(ħ*ω⊥*a⊥)
	c1 			= c1_1DSI/(ħ*ω⊥*a⊥)
    # xmaxSI 		= xmax_factor*RTF
    dxSI 		= 2*xmaxSI/nx
    xmax 		= xmaxSI/a⊥
    dx 			= 2*xmax/nx
	
	tstab = π/( 1/2*(π/dx)^2 + max(abs(q), abs(q_quench)))
	Δtmax = tstab
	md"Dependent coefficients, normalized coefficients, misc. "
end

# ╔═╡ fae7e791-7eee-4619-83cb-2ab516114f00
md"""
#### Check the scales of the resulting parameters
Phase diagram for reminder.
$(Show(MIME"image/png"(), read("phasediagram.png")))
UV cutoff energy from the choice of domain = $(round((π/dxSI*ħ)^2/2/7/amu/h 
)) Hz. 

Quadratic Zeeman energy, $$q$$: $(round(qSI/h, digits = 2)) Hz $$\rightarrow$$ $(round(q_quenchSI/h, digits = 2)) Hz.

Spin dependent interaction, $$2c_1 n_{\text{peak}}$$ is $(round(2*c1_1DSI * npeakSI/h, digits = 2)) Hz.

Spin independent interaction, $$c_0 n_{\text{peak}}$$ is $(round(c0_1DSI * npeakSI/h, digits = 2)) Hz.

The stability time $t_{\text{stab}}=$ $(round(π/( 1/2*(π/dx)^2 + max(abs(q), abs(q_quench)) ), digits = 2)) [1/$$\omega_{\bot}$$]. 

The ratio $t_{\text{max}} / t_{\text{stab}}$ = $(round(dyTmax/tstab, digits = 1)).

Healing length, $\xi= {1}/{\sqrt{8\pi n a}}$ = $(round(1/sqrt(8*π*npeakSI*(2*a2Li + a0Li)/3/2/π/a⊥^2)*1e6, digits = 2)) μm. 

Spin healing length, $\xi_{s}$ = $(round(1/sqrt(8*π*npeakSI*(a0Li - a2Li)/3/2/π/a⊥^2)*1e6, digits = 2)) μm. 

"""

# ╔═╡ 67717844-d4e7-430b-9b00-abc111ca7065
md"## 2. Find ground state
Check the trapping potentials, Zeeman energies"

# ╔═╡ 4b903823-bbe1-4245-ad8e-d0a56ccce22b
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

# ╔═╡ 82dc85f1-f3a1-4b63-9618-799affe9fd8c
begin
	plot( plot(xx_um, f⊥ * real([Utrap.data[ii, ii] for ii = 1:nx] ), label = "Trapping") ,   begin plot(xx_um, f⊥ * real([UZeeman.data[ii, ii] for ii = 1:nx]) , label = "Zeeman")
			
 	plot!(xx_um, real([UZeeman_quench.data[ii, ii] for ii = 1:nx]) , label = "Zeeman afer quench" , frame = :box, legend = :right) 
		end
		)
    xlabel!("Position [μm]")
    ylabel!("Energy [Hz]")
end


# ╔═╡ 0a43de57-7ef6-44cc-bf77-30b82f2eadce
md"### Prepare initial guess wavefunction"

# ╔═╡ 2fbfca9b-7ae8-4e88-922d-37ae839a62b9
function convert_wfn_norm2SI(ψ)
	ψSI = copy(ψ)
	ψSI.data  = ψSI.data/sqrt(a⊥)
	return ψSI
end

# ╔═╡ cdbbded1-bd40-413a-aa88-408c2ee4a364
function normalize_wfn(ψ, N)
	return ψ/norm(ψ) * sqrt(N/dx)
end

# ╔═╡ 621b5d67-a3cb-4728-8eae-743b1fb6ad27
function initialstate_gaussian(bx, N; σ::Float64 = 1.0, spinratio = [1,0,1])
	ϕin_p = spinratio[1]*gaussianstate(bx, 0, 0, σ)
	ϕin_0 = spinratio[2]*gaussianstate(bx, 0, 0, σ)
	ϕin_m = spinratio[3]*gaussianstate(bx, 0, 0, σ)
	ψgaussian = normalize_wfn(ϕin_p ⊕ ϕin_0 ⊕ ϕin_m, N)
	return ψgaussian
end

# ╔═╡ 9234bb35-0b58-4e7d-83b5-5ee66357238a
function normalize_wfn!(ψ)
	ψ[:] = ψ/norm(ψ) * sqrt(Natom/dx)
end

# ╔═╡ 840f6795-0fc1-4dcd-bd7d-aee2ee74ee0a
function plot_wfn(x, ψ0; kwargs...)
	ψ = convert_wfn_norm2SI(ψ0)
	n1 = abs2.(getblock(ψ,1).data)*1e-6
	n2 = abs2.(getblock(ψ,2).data)*1e-6
	n3 = abs2.(getblock(ψ,3).data)*1e-6
	plt = plot(framestyle = :box, xlabel= "Position [μm]", 
		ylabel = "Density [1/μm]" ; kwargs...)
	plot!(x, n1,  label = L"n_{+1}",  ls = :solid, fillrange = 0, fillalpha = 0.35; kwargs...)
	plot!(x, n2,  label = L"n_{0}",  ls = :dot, fillrange = 0, fillalpha = 0.35; kwargs...)
	plot!(x, n3, label = L"n_{-1}",  ls = :dash, fillrange = 0, fillalpha = 0.35; kwargs...)
	plot!(x, (n1+n2+n3), label = L"n"; kwargs...)
	return plt
end

# ╔═╡ 1ed291a7-d08d-4d8b-bde4-3a77451eed44
begin
	    # Preparing the initial state
    ψ0 = initialstate_gaussian(bx, Natom, σ = xmax/3, spinratio = [1, 1, 1])
    plot_wfn(xx_um, ψ0)
    title!("Initial guess for the imaginary time evolution")
end

# ╔═╡ 79694901-4257-478b-a94d-a13bf69e4463
md"### Evolve GPE with $-i\hat{H}$.
Using imaginary time evolution method, we find the ground state of the system. The noralization of the wavefunction is done by using callbacks for the solver."

# ╔═╡ 078db709-2df6-43bf-98b6-32470e74cce7
begin
    # Constructing total Hamiltonian
    Hc0 = one(bx) ⊕ one(bx) ⊕ one(bx)
    Hc1 = one(bx) ⊕ one(bx) ⊕ one(bx)
    H_im = -im*LazySum(Hkin_FFT, Utrap, UZeeman, Hc0, Hc1)
	"""
	Function to update the Hamiltonian during the imaginary time evolution.
	"""
	function Hgp_im(t, ψ) 
		ψ = normalize_wfn(ψ, Natom)
		ψ_p = ψ.data[1:nx]
		ψ_0 = ψ.data[nx+1:2*nx]
		ψ_m = ψ.data[2*nx+1:3*nx]

		np = abs2.(ψ_p)
		n0 = abs2.(ψ_0)
		nm = abs2.(ψ_m)

		c0n_dat = c0*(np+n0+nm)
		
			# c0 term
		setblock!(Hc0,diagonaloperator(bx, c0n_dat), 1, 1 )
		setblock!(Hc0,diagonaloperator(bx, c0n_dat), 2, 2 )
		setblock!(Hc0,diagonaloperator(bx, c0n_dat), 3, 3 )
			# c1 term
		setblock!(Hc1,diagonaloperator(bx, c1*(np - nm + n0) ), 1, 1 )
		setblock!(Hc1,diagonaloperator(bx, c1*(np + nm) ), 2, 2 )
		setblock!(Hc1,diagonaloperator(bx, c1*(nm - np + n0) ), 3, 3 )
		setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_m).*ψ_0) , 1,2)
		setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_0).*ψ_m) , 2,1)
		setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_0).*ψ_p) , 2,3)
		setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_p).*ψ_0) , 3,2)
		return H_im
	end

    # Set time domain and sample length
    T = range(0., imTmax; length = 30)

    # normalization callback. QOptics.jl hand the Ket's data to Diffeq.jl
	norm_func(u, t, integrator) = normalize_wfn!(u)
    ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func; func_everystep = true, func_start = false)

	# Call solver
	if token_imgevol
		tout, ψt = timeevolution.schroedinger_dynamic(T, ψ0, Hgp_im,  callback=ncb, alg = DP5(), abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int, save_everystep = false, dtmax = Δtmax)
		ψg = ψt[end]
	else
		ψg = ψ0
	end
md"Calling solver here"
end

# ╔═╡ dc86499a-70cb-4e9e-9257-2782050581ff
md"Check snapshots during the imaginary time evolution"

# ╔═╡ 33ffb2bd-9762-48aa-87bf-44f5b5044ced
function plot_snapshots(xx, ψt, T; kwargs...)
	    function plot_one(xx, ψ; timestamp::Float64 = -1)
	        # Plot wavefunction in SI units
	        n1 = abs2.(getblock(ψ,1).data) /(a⊥/1e-6)
	        n2 = abs2.(getblock(ψ,2).data) /(a⊥/1e-6)
	        n3 = abs2.(getblock(ψ,3).data) /(a⊥/1e-6)
			plt = plot_wfn(xx,ψ; size = (300, 200))
 			annotate!(-maximum(xx), maximum(n1+n2+n3)*0.9, text( string(round(timestamp*1e3, digits = 2))*" ms", 12 , :left) )
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
		plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, layout = (3,3), legend = false, size = (800, 400), xlabel = "", ylabel = ""; kwargs...)
end

# ╔═╡ 01a9053f-a9f8-4b74-ae5f-18d7ea7fb187
function plot_snapshots_3D(xx, ψarray, T; kwargs...)
	function get_density(ψ)
		# Plot wavefunction in SI units
		n1 = abs2.(getblock(ψ,1).data) /(a⊥/1e-6)
		n2 = abs2.(getblock(ψ,2).data) /(a⊥/1e-6)
		n3 = abs2.(getblock(ψ,3).data) /(a⊥/1e-6)
		return [n1, n2, n3]
	end
	plt = plot(xlabel = "Position [μm]", ylabel = "Time [ms]", cam = (70, 20),legend = false)
	for kk = 1:3
		plot!(xx, T, 
			[get_density(ψarray[ii])[kk][jj] for ii = 1:length(T), jj = 1:length(xx)]
			,  lt = :surface, c = kk, alpha = 0.3)
		for ii = 1:length(T)
			plot!(xx,fill(T[ii], length(xx)),(get_density(ψarray[ii])[kk]), c = kk, lw = 1, alpha = 0.8; kwargs...)
		end
	end
	plt
end

# ╔═╡ 8d02275a-9d96-4fa8-b14b-4c2daa59f317
# plot_snapshots_3D(xx, ψt, tout);

# ╔═╡ bfe9e487-bc60-4b6b-8bd1-5f44342b38e6
begin
	if token_imgevol
	    plt_imsnapshot = plot_snapshots(xx_um, ψt, tout)
	end
end

# ╔═╡ 461502f9-359a-42ab-a06f-36846c2be82e
md"### Check the vailidity of the found ground state
We compare the ground state with Thomas-Fermi profile."

# ╔═╡ f82490de-0b75-4151-952d-7ace5a22aa35
## Check the ground state with Thomas-Fermi
begin 
    foundgs = plot_wfn(xx_um, ψg)
	if RTF == Inf
	else
		# Thomas fermi profile
    	nTF = max.(npeakSI*(1 .- (xx_um*1e-6/RTF).^2)*1e-6, 0*xx_um) 
    	plot!(xx_um, nTF, lw = 2, ls=:solid, label ="TF profile")
	end
    # title!("Found ground state")
	foundgs
end

# ╔═╡ 67f85069-4ab1-4a9e-b204-9a81395826af
md"Optional phase profile check"

# ╔═╡ 4225d24f-0072-4950-8325-ea22cbd624a2
begin
	function plot_wfn_phase(xx, ψ)
    # Plot wavefunction in SI units
    n1 = angle.(getblock(ψ,1).data)
    n2 = angle.(getblock(ψ,2).data)
    n3 = angle.(getblock(ψ,3).data)
    plt = plot(xx, n1,  label = "+1", linewidth = 2, ls = :solid, framestyle = :box)
    plot!(xx, n2,  label = "0", linewidth = 2, ls = :dash)
    plot!(xx, n3, label = "-1", linewidth = 2, ls = :dot)
    xlabel!("Position [μm]")
    ylabel!("Phase [rad]")
    return plt
end

    # plot_wfn_phase(xx_um, ψg);
end

# ╔═╡ c915035c-c129-47d0-b1f5-7e8515bc229e
md"## 3. Compute quench dynamics
Using the ground state we found, simulate quench dynamics. 
"

# ╔═╡ 023835a4-4ff6-4b89-9b02-96282270c2ff
begin
	"""
	Generate wavefunction to seed instability.
	"""
	function get_psinoise(seed, trWigner, ψbec, alpha_m)
		
		# Generate fluctuation array using random number generator
		function generate_fluctuation(seed::Int)
			# Generate noise wavefunction for using MT seed.
			rng1r = MersenneTwister(seed)
			rng1i = MersenneTwister(seed+299792458)
			# summing two 1/4 variance Gaussian random number
			a_rand = randn(rng1r, Float64, nx)/2 .+ im*randn(rng1i, Float64, nx)/2
			ψarray = zeros(ComplexF64, nx)
			for ii = 1:nx
				for jj = 1:nx
					ψarray[ii] = ψarray[ii] + exp( im*pp[jj]*xx[ii] ) * a_rand[jj]
				end
			end 
			ψarray = ψarray/sqrt(2*xmax)
			return ψarray
		end
		
		# Make noise orthogonal to the condensate wavefunction using Gram-Schmidt
		function make_orthogonal(ψnoise, ψbec)
			ψnoise = ψnoise - ψbec * ( (dagger(ψbec)*ψnoise)  / (dagger(ψbec) * ψbec) ) 
		end	
		
		randnum = Int.(round.( rand(MersenneTwister(seed), 3) * 1000))
		
		if trWigner
			ϕin_p = Ket(bx, generate_fluctuation(randnum[1]) * alpha_m[1])
			ϕin_0 = Ket(bx, generate_fluctuation(randnum[2]) * alpha_m[2]) 
			ϕin_m = Ket(bx, generate_fluctuation(randnum[3]) * alpha_m[3])
			ψnoise = ϕin_p ⊕ ϕin_0 ⊕ ϕin_m	
			Nvacuum = sum(dagger(ψnoise)*(ψnoise))*dx
			ψnoise = make_orthogonal(ψnoise, ψbec)
			ψnoise = normalize_wfn(ψnoise, Nvacuum)
		else
			ϕin_p = Ket(bx, rand(MersenneTwister(randnum[1]), nx) * alpha_m[1])
			ϕin_0 = Ket(bx, rand(MersenneTwister(randnum[2]), nx) * alpha_m[2])
			ϕin_m = Ket(bx, rand(MersenneTwister(randnum[3]), nx) * alpha_m[3])
			ψnoise = ϕin_p ⊕ ϕin_0 ⊕ ϕin_m	
			ψnoise = make_orthogonal(ψnoise, ψbec)
			ψnoise = normalize_wfn(ψnoise, Nvirtual)
		end
	    return 	ψnoise
	end
	
	ψ_noise = get_psinoise(MTseed, trWigner, ψg, alpha_m)
		# intial condition
    ψt_dy0 = ψg + ψ_noise
	
md"### Enter fluctuation seeds
Generate noise wavefunction and check the orthogonality."
end

# ╔═╡ 2567fc9a-5a27-4c8e-89a0-0227396a818f
begin
    # check the noise
	plot_wfn(xx_um, ψ_noise)
	title!("Total number= $(round( sum(abs2.(ψ_noise.data)*dx), digits = 2))")
end

# ╔═╡ 645a8e77-e6df-487d-932a-199440e35f97
begin
	plot_wfn(xx_um, ψt_dy0)
	title!("Total number= $(round( sum(abs2.(ψt_dy0.data)*dx), digits = 2))")
end

# ╔═╡ 640be080-c000-4aeb-9e43-09f1bae085a3
function plot_psi_3D(x, ψ;kwargs...)
	ψp = getblock(ψ,1).data /sqrt(a⊥)
	ψ0 = getblock(ψ,2).data /sqrt(a⊥)
	ψm = getblock(ψ,3).data /sqrt(a⊥)
	sqrtn = sqrt.(abs2.(ψp)+ abs2.(ψ0)+ abs2.(ψm))
	plt = plot(xlabel = "position", ylabel = "Im", zlabel = "Re")
	plot!(x, imag.(ψp), real.(ψp), ls = :solid, label = "+1"; kwargs...)
	plot!(x, imag.(ψ0), real.(ψ0), ls = :dot, label = "0"; kwargs...)
	plot!(x, imag.(ψm), real.(ψm), ls = :dash, label = "-1"; kwargs...)
	plot!(ylims = (-maximum(sqrtn), maximum(sqrtn)), zlims = (-maximum(sqrtn), maximum(sqrtn)), cam = (45, 45), size = (400, 300); kwargs...)
end

# ╔═╡ 9fd8b82f-424b-42d9-8b70-df593b26d7ec
# plot_psi_3D(xx_um, ψt_dy0)

# ╔═╡ 068bd888-a0c6-4742-9e6f-0a2b21da48e4
md"### Evolve GPE"

# ╔═╡ 0485091d-d6e7-456a-b462-bd62e5166292
begin
    # Set time domain and sampling size
    T_dy = range(0, dyTmax; length = TsampleN)
	
	    # Constructing interaction Hamiltonian
    Hc0_dy = one(bx) ⊕ one(bx) ⊕ one(bx);
    Hc1_dy = one(bx) ⊕ one(bx) ⊕ one(bx);
    H_dy = LazySum(Hkin_FFT, Utrap, UZeeman_quench, Hc0_dy, Hc1_dy);

	
	function Hgp_dy(t, ψ) 
		ψ_p = ψ.data[1:nx];
		ψ_0 = ψ.data[nx+1:2*nx];
		ψ_m = ψ.data[2*nx+1:3*nx];

		np = abs2.(ψ_p)
		n0 = abs2.(ψ_0)
		nm = abs2.(ψ_m)

		c0n_dat = c0*(np+n0+nm);

			# c0 term
		setblock!(Hc0_dy,diagonaloperator(bx, c0n_dat), 1, 1 );
		setblock!(Hc0_dy,diagonaloperator(bx, c0n_dat), 2, 2 );
		setblock!(Hc0_dy,diagonaloperator(bx, c0n_dat), 3, 3 );

			# c1 term
		setblock!(Hc1_dy,diagonaloperator(bx, c1*(np - nm + n0) ), 1, 1 );
		setblock!(Hc1_dy,diagonaloperator(bx, c1*(np + nm) ), 2, 2 );
		setblock!(Hc1_dy,diagonaloperator(bx, c1*(nm - np + n0) ), 3, 3 );
		setblock!(Hc1_dy,diagonaloperator(bx, c1*conj(ψ_m).*ψ_0) , 1,2)
		setblock!(Hc1_dy,diagonaloperator(bx, c1*conj(ψ_0).*ψ_m) , 2,1)
		setblock!(Hc1_dy,diagonaloperator(bx, c1*conj(ψ_0).*ψ_p) , 2,3)
		setblock!(Hc1_dy,diagonaloperator(bx, c1*conj(ψ_p).*ψ_0) , 3,2)

		return H_dy
	end
    # Call solver
	if token_quenchdy*token_imgevol
   		tout_dy, ψt_dy = timeevolution.schroedinger_dynamic(T_dy, ψt_dy0, Hgp_dy, alg = DP5(), abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int)
	end
	md"Calling solver here"
end


# ╔═╡ 32108a28-ee94-4d30-b4c8-67a013ff1524
	# Check the results
    plot_snapshots(xx_um, ψt_dy, T_dy; size = (600, 400))

# ╔═╡ 07ea6c1a-b67d-4bdd-bbed-be34de0ae00c
md"## 4. Check results, visualize, and save data"

# ╔═╡ 1034f666-6458-49c5-bcf2-9a92b2115989
begin
	function insert_ψt(dict, ψt)
		output = copy(dict)
		output["psit"] = ψt
		return output
	end
	if token_quenchdy
		data_dynamics = insert_ψt(params, ψt_dy)
		if token_save
			safesave(datadir("sims","Li7", fname*".jld2") ,data_dynamics)
		end
	end
end

# ╔═╡ a4d11f49-950c-4b8e-b5f9-a86d123ef4d8
begin
	function plot_pop_dynamics(T_dy, ψ; kwargs...)
		n1 = zeros(length(T_dy));
		n2 = zeros(length(T_dy));
		n3 = zeros(length(T_dy));
		for ii = 1:length(T_dy)
			n1[ii] = sum(abs2.(getblock(ψ[ii],1).data))*dx
			n2[ii] = sum(abs2.(getblock(ψ[ii],2).data))*dx
			n3[ii] = sum(abs2.(getblock(ψ[ii],3).data))*dx
		end
		plt = plot(T_dy, n1,  label = L"{+1}", lw = 2, ls = :solid, fs = :box; kwargs...)
		plot!(T_dy, n2, label = L"0", lw = 2, ls = :solid; kwargs...)
		plot!(T_dy, n3, label = L"{-1}", lw = 2, ls = :dash; kwargs...)
		plot!(T_dy, n1+n2+n3, label = "total", lw = 2, ls = :dot, legend = :left; kwargs...)
		return plt, [n1, n2, n3]
	end

end

# ╔═╡ 6437597e-c5fd-4a18-b524-3ef729dfc786
md"Check with gif"

# ╔═╡ 86268d8e-25ac-4a75-9e85-1b968994066b
begin
	if token_quenchdy
		anim = @animate  for ii ∈ 1:length(ψt_dy)
			plot_wfn(xx_um, ψt_dy[ii]; ylims = (0, npeakSI*1.3*1e-6), dpi = 200)
			annotate!(-RTF*1e6, npeakSI*1e-6, string(round(T_dy[ii]/ω⊥*1e3, digits = 1))*" ms")
		end
		if token_save*token_quenchdy*token_imgevol
			gif(anim, plotsdir("Li7", fname*".gif"), fps = 10) 
		end
		gif(anim, fps = 10)
	end
end

# ╔═╡ f1905078-295b-4219-b37a-128855ec3988
function unwrap(v, inplace=false)
    # currently assuming an array
    unwrapped = inplace ? v : copy(v)
    for i in 2:length(v)
      while unwrapped[i] - unwrapped[i-1] >= pi
        unwrapped[i] -= 2pi
      end
      while unwrapped[i] - unwrapped[i-1] <= -pi
        unwrapped[i] += 2pi
      end
    end
    return unwrapped
end

# ╔═╡ 11db7f14-4484-46b7-9e52-606e56f3c554
function plot_Fplus(ψKet)
    ψp = ψKet.data[1:nx]
    ψ0 = ψKet.data[nx+1:2*nx]
    ψm = ψKet.data[2*nx+1:3*nx]
    xdomain = range(ψKet.basis.bases[1].xmin, ψKet.basis.bases[1].xmax, length = ψKet.basis.bases[1].N)*(a⊥/1e-6)
    FplusAmp = sqrt(2.0) * abs.( conj(ψp).*ψ0 + conj(ψ0).*ψm ) ./(abs2.(ψp) + abs2.(ψ0) + abs2.(ψm))
    FplusAng = angle.(conj(ψp).*ψ0 + conj(ψ0).*ψm)
    plt1 = plot(xdomain, FplusAmp, color = "red", ylabel = L"$|F^+|/\rho$",  legend = false)
    plt2 = plot(xdomain, unwrap(FplusAng)/π,  color = "blue", ylabel = "Arg [π]", legend = false,  frame = :box)
    plot(plt1, plt2, layout = (2, 1))
    xlabel!("X [μm]")

end

# ╔═╡ e093617e-9246-4e0b-99d3-212c075a900f
# plot([sum(abs2.(ψt_dy[ii].data))*dx for ii = 1:length(tout_dy)])

# ╔═╡ 4a605c05-8a85-4df2-a4d9-5699e1f0f672
# plot_Fplus(ψt_dy[1])

# ╔═╡ a03c028e-f013-4aa9-b227-08ed0d0f0f03
# plot_psi_3D(xx_um, ψt_dy[100], cam = (30, 20))

# ╔═╡ b11782bc-5bf3-432f-9a8d-138d6fb52358
function plot_snapshots_heatmap(xx, ψarray, T;spinindx::Int64 = 2 , kwargs...)
	function get_density(ψ)
		# Plot wavefunction in SI units
		n1 = abs2.(getblock(ψ,1).data) /(a⊥/1e-6)
		n2 = abs2.(getblock(ψ,2).data) /(a⊥/1e-6)
		n3 = abs2.(getblock(ψ,3).data) /(a⊥/1e-6)
		return [n1, n2, n3]
	end
	plt = plot(ylabel = "Position [μm]", xlabel = "Time [ms]", cam = (70, 20),legend = false)
		plot!(T/ω⊥*1e3, xx, 
			[get_density(ψarray[ii])[spinindx][jj] for jj = 1:length(xx),  ii = 1:length(T)]
			,  lt = :heatmap; kwargs...)
		# for ii = 1:length(T)
		# 	plot!(xx,fill(T[ii], length(xx)),(get_density(ψarray[ii])[kk]), c = kk, lw = 1; kwargs...)
		# end
	plt
end

# ╔═╡ 4af4f76e-8c5e-40bf-b0f9-75e5a6494bfe
# plot_snapshots_3D(xx_um, ψt_dy, T_dy; cam = (30, 45))

# ╔═╡ 1296de49-b0df-42eb-9942-b994960b34f5
Tindx = [35, 50, 75, 95]

# ╔═╡ 2198a036-4783-41ba-90f7-0d25b9c91190
Tslice = T_dy[Tindx]/ω⊥*1e3

# ╔═╡ 8ee6eca0-a0aa-4f94-9cc7-c99e958aecb1
	if token_quenchdy
		plt, nn = plot_pop_dynamics(T_dy/ω⊥ *1e3, ψt_dy; yscale = :log10)
		vline!(Tslice, c = :black, lw = 1, label = "")
		ylabel!("Total atom number")
		xlabel!("Time [ms]")
	end

# ╔═╡ 1185bbb0-d1e8-4653-8279-c45c3e7b4438
begin
	plot_snapshots_heatmap(xx_um, ψt_dy, T_dy;spinindx = 2, c = :Oranges_9)
	vline!(Tslice, c = :black, lw = 1)
end

# ╔═╡ a2ca69e9-50e9-4dc7-a0e6-86456c59579d
# begin
# 	plot_snapshots_heatmap(xx_um, ψt_dy, T_dy;spinindx = 1, c = :Blues_9)
# 		vline!(Tslice, c = :black, lw = 1)
	
# end

# ╔═╡ 3e2222c2-586a-4a4c-9323-7e96f9e00a2e
# begin
# 	plot_snapshots_heatmap(xx_um, ψt_dy, T_dy;spinindx = 3, c = :Greens_9)
# 		vline!(Tslice, c = :black, lw = 1)
	
# end

# ╔═╡ 3c8ecd71-f759-47a1-b411-923e4af39dc9
plot_wfn(xx_um, ψt_dy[Tindx[2]])

# ╔═╡ Cell order:
# ╠═c6450a7c-ecba-4f85-98ef-3d7a007dc64f
# ╠═bdc7e2e3-468f-4313-8154-82375ecc7c87
# ╠═14eaa4d0-7f90-4195-b627-6eafbd9a9e8a
# ╠═5e7fce10-8003-46df-aab7-0af1046c30e0
# ╠═4bfb2241-5a5b-4dc8-aac1-4e8b0f53a33c
# ╠═fd11bf5d-66c8-45e0-80f5-40386c7c9c06
# ╠═6d9b9a15-107e-4763-b670-264b5b019152
# ╠═3241b4ff-86d5-4bbb-81fa-d527398792b5
# ╠═e633a360-b643-11eb-3979-6f8b7a3551c4
# ╠═6003638d-a1a9-4ae9-be69-232c2d1543d8
# ╠═69cb4cbc-572d-4b57-98cd-ef297927e2b8
# ╠═84baced5-832d-4855-ba5a-9380571c65fd
# ╠═8d94d626-3d97-456c-8fea-1e0feb5794d7
# ╠═709d22eb-46f7-4af9-ae5c-2c2ba49e7b06
# ╠═8a9a006b-f778-4d7b-8814-4c3869d07a76
# ╠═d34d395d-ff80-4f72-8aac-47366ee7aa06
# ╠═0075ee54-d4e0-43d7-92b9-24971e92c108
# ╠═55ff32a3-0e3d-4a66-8764-ce3fee1fd20c
# ╠═1423582b-6db1-4a00-968c-98f831058017
# ╠═fae7e791-7eee-4619-83cb-2ab516114f00
# ╠═67717844-d4e7-430b-9b00-abc111ca7065
# ╠═4b903823-bbe1-4245-ad8e-d0a56ccce22b
# ╠═82dc85f1-f3a1-4b63-9618-799affe9fd8c
# ╠═0a43de57-7ef6-44cc-bf77-30b82f2eadce
# ╠═621b5d67-a3cb-4728-8eae-743b1fb6ad27
# ╠═2fbfca9b-7ae8-4e88-922d-37ae839a62b9
# ╠═cdbbded1-bd40-413a-aa88-408c2ee4a364
# ╠═9234bb35-0b58-4e7d-83b5-5ee66357238a
# ╠═840f6795-0fc1-4dcd-bd7d-aee2ee74ee0a
# ╠═1ed291a7-d08d-4d8b-bde4-3a77451eed44
# ╠═79694901-4257-478b-a94d-a13bf69e4463
# ╠═078db709-2df6-43bf-98b6-32470e74cce7
# ╠═dc86499a-70cb-4e9e-9257-2782050581ff
# ╠═33ffb2bd-9762-48aa-87bf-44f5b5044ced
# ╠═01a9053f-a9f8-4b74-ae5f-18d7ea7fb187
# ╠═8d02275a-9d96-4fa8-b14b-4c2daa59f317
# ╠═bfe9e487-bc60-4b6b-8bd1-5f44342b38e6
# ╠═461502f9-359a-42ab-a06f-36846c2be82e
# ╠═f82490de-0b75-4151-952d-7ace5a22aa35
# ╠═67f85069-4ab1-4a9e-b204-9a81395826af
# ╠═4225d24f-0072-4950-8325-ea22cbd624a2
# ╠═c915035c-c129-47d0-b1f5-7e8515bc229e
# ╠═023835a4-4ff6-4b89-9b02-96282270c2ff
# ╠═2567fc9a-5a27-4c8e-89a0-0227396a818f
# ╠═645a8e77-e6df-487d-932a-199440e35f97
# ╠═640be080-c000-4aeb-9e43-09f1bae085a3
# ╠═9fd8b82f-424b-42d9-8b70-df593b26d7ec
# ╠═068bd888-a0c6-4742-9e6f-0a2b21da48e4
# ╠═0485091d-d6e7-456a-b462-bd62e5166292
# ╠═32108a28-ee94-4d30-b4c8-67a013ff1524
# ╠═07ea6c1a-b67d-4bdd-bbed-be34de0ae00c
# ╠═1034f666-6458-49c5-bcf2-9a92b2115989
# ╠═a4d11f49-950c-4b8e-b5f9-a86d123ef4d8
# ╠═6437597e-c5fd-4a18-b524-3ef729dfc786
# ╠═86268d8e-25ac-4a75-9e85-1b968994066b
# ╠═11db7f14-4484-46b7-9e52-606e56f3c554
# ╠═f1905078-295b-4219-b37a-128855ec3988
# ╠═e093617e-9246-4e0b-99d3-212c075a900f
# ╠═4a605c05-8a85-4df2-a4d9-5699e1f0f672
# ╠═a03c028e-f013-4aa9-b227-08ed0d0f0f03
# ╠═b11782bc-5bf3-432f-9a8d-138d6fb52358
# ╠═4af4f76e-8c5e-40bf-b0f9-75e5a6494bfe
# ╠═1296de49-b0df-42eb-9942-b994960b34f5
# ╠═2198a036-4783-41ba-90f7-0d25b9c91190
# ╠═8ee6eca0-a0aa-4f94-9cc7-c99e958aecb1
# ╠═1185bbb0-d1e8-4653-8279-c45c3e7b4438
# ╠═a2ca69e9-50e9-4dc7-a0e6-86456c59579d
# ╠═3e2222c2-586a-4a4c-9323-7e96f9e00a2e
# ╠═3c8ecd71-f759-47a1-b411-923e4af39dc9
