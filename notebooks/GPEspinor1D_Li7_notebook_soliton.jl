### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e633a360-b643-11eb-3979-6f8b7a3551c4
using QuantumOptics, OrdinaryDiffEq, DiffEqCallbacks, 
DrWatson, JLD2, Random, PlutoUI, Statistics, Plots, Plots.PlotMeasures, 
LaTeXStrings, PhysicalConstants, PhysicalConstants.CODATA2018, Unitful, BenchmarkTools

# ╔═╡ c6450a7c-ecba-4f85-98ef-3d7a007dc64f
md"# Gross-Piatevskii equation for 1D spinor gas of Li7 - quench dynamics
Here is GPE to solve. 
```math
\begin{equation}
    \begin{split}
        i\hbar\frac{\partial\psi_m}{\partial t} &= \frac{\delta E}{\delta \psi^*_m} \\
    & = \left[ -\frac{\hbar^2}{2M}\nabla^2 + U_{\text{trap}}(\mathbf{r}) - pm + qm^2 \right] \psi_m + c_0 n \psi_m + c_1 \sum_{m'=-1}^{1} \mathbf{F} \cdot \mathbf{f}_{mm'} \psi_{m'}
    \end{split}
\end{equation}
```
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
- Unit conversion reminder.
```math 
\begin{gather}
x \rightarrow a_{\bot}x, \quad
t \rightarrow t/\omega_{\bot}, \quad
\psi \rightarrow \psi/\sqrt{a_\bot} \\
\gamma = \frac{\omega}{\omega_{\bot}} \\
\tilde{p} = \frac{p}{\hbar\omega_{\bot}}, \quad \tilde{q} = \frac{q}{\hbar\omega_{\bot}} \\
\tilde{c}_0 =\frac{c_0}{\hbar\omega_{\bot}a_{\bot}}, \quad \tilde{c}_1  =\frac{c_0}{\hbar\omega_{\bot}a_{\bot}}.
\end{gather} 
```

"

# ╔═╡ 6c191cd6-3a1d-4631-8737-3e647b54a1cd
md"""
We can write the equation in the form of nonlinear Schrodinger equation as follows. 
```math
 \small{ }
\begin{equation}
	\begin{split}
& i\partial_t \begin{pmatrix} \psi_{1} \\ \psi_{0} \\ \psi_{-1} \end{pmatrix}  = \hat{H}[\mathbf{\psi}] \begin{pmatrix} \psi_{1} \\ \psi_{0} \\ \psi_{-1} \end{pmatrix}\\
	&=  \begin{pmatrix} 
			-\frac{1}{2}\partial^2_x + \frac{1}{2}\gamma^2 x^2 - \tilde{p} + \tilde{q} & 0 & 0 \\
	0 & -\frac{1}{2}\partial^2_x + \frac{1}{2}\gamma^2 x^2 & 0 \\
	0 & 0 & -\frac{1}{2}\partial^2_x + \frac{1}{2}\gamma^2 x^2 + \tilde{p} + \tilde{q}
\end{pmatrix} \begin{pmatrix} \psi_{1} \\ \psi_{0} \\ \psi_{-1} \end{pmatrix}  \\
	& + 
\begin{pmatrix} \tilde{c}_0 n + \tilde{c}_1(|\psi_1|^2 -|\psi_{-1}|^2 + |\psi_0|^2) & \tilde{c}_1\psi^*_{-1}\psi_0 & 0 \\
	\tilde{c}_1\psi^*_0\psi_{-1} & \tilde{c}_0 n + \tilde{c}_1(|\psi_1|^2 + |\psi_{-1}|^2) & \tilde{c}_1\psi^*_0\psi_1 \\
	0 & \tilde{c}_1\psi^*_1\psi_0 & \tilde{c}_0 n + \tilde{c}_1 (|\psi_{-1}|^2 -|\psi_{1}|^2 + |\psi_0|^2)
\end{pmatrix}
\begin{pmatrix} \psi_{1} \\ \psi_{0} \\ \psi_{-1} \end{pmatrix} 
	\end{split}
\end{equation}

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
We introduce a noise, $\psi_{\text{n}, \pm}$, in order to seed the instability based on **truncated Wigner approximation** (TWA). We skipped the fluctuation in $m_F=0$ state since we already used the imaginary time method for searching the ground state.
See references for more details [PRA 76, 043613 (2007), PRA 94, 023608 (2016), Adv. in Phys. 47, 363 (2008)].

$\psi_{\text{n}, \pm}(x)= \sum_{k}{ \frac{1}{\sqrt{V}} e^{ikx} a_{\pm, k} },$
where $a_{\pm, k} = \alpha_{\text{rand}} + i\beta_{\text{rand}}$, and $\alpha_{\text{rand}}$ and $\beta_{\text{rand}}$ are the normal random variables of $\mathcal{N}\left(0, \frac{1}{2}^2 \right)$. This satisfies half-quantum fluctuation,

$\langle a_{\pm, k}^* a_{\pm, k}\rangle = \frac{1}{2}.$

Note that we add up all $k$ values. 
- When we consider $M$ modes, this method introduce $M/2$ virtual particle for each modes [Adv. in Phys. 47, 363 (2008)] if we choose the vacuum mode properly, i.e., orthogonal to the condensates.
- One need to introduce UV cutoff considering the energy scale of C-field region. In order to simulate jets, we need to inlcude many modes, therefore, large number of virtual particles should be added(can be thousands). Being not confident with this, we also implement uniform random noise whose input virtual particle number controls the amplitude. 
- For both method, we manually orthogonalize the vacuum mode to the condensate using Gram-Schmidt process.
"""

# ╔═╡ fd11bf5d-66c8-45e0-80f5-40386c7c9c06
md"""
## 0. Workflow
1. Define parameters such as $N_{\text{atom}}$, $f_{\bot}$, etc. 
2. Find ground state using the imaginary time evolution. 
- Prepare initial guess wavefunction.
- Evolve GPE with $i\hat{H}$ with total number constraints.
- Assess the result.
3. Compute the quench dynamics
- Enter fluctuation seeds.
- Evolve GPE properly
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
	default(titlefont = (10, default_font, :black), legendfont = (8, default_font, :black), guidefont = (10, default_font, :black), tickfont = (9, default_font, :black), framestyle = :box, size = (500, 300), dpi = 200, lw = 2)
end

# ╔═╡ d34d395d-ff80-4f72-8aac-47366ee7aa06
md"## 1. Define system parameters
All input parameters for the simulation is in the following block."

# ╔═╡ 709d22eb-46f7-4af9-ae5c-2c2ba49e7b06
md"""
### Computation control **checkboxs** for modification
**Suppress solver calling during the code modification** 
- Run imaginary time solver? $(@bind token_imgevol CheckBox(default=false)) 
- Run quench dynamics solver? $(@bind token_quenchdy CheckBox(default=false))
- Save data? $(@bind token_save CheckBox(default=false))
""" 

# ╔═╡ dc0331a0-5f9b-4dcd-b00b-5839e12130ce
md"#### Load physical constants"

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
end;

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
    qSI 		= +500*h        # [J]
    q_quenchSI 	= +500*h       # [J]
	
	    # File name for saving
    fname = "soliton"

	# Instability seed related
    MTseed      = 1234   	# Mersenne Twister seed for {+1, 0, -1}
	trWigner 	= false # select type of the noise
	Nvirtual 	= 0  	# number of virtual particle number feeded in noise

    # Time domain
    imTmax 		= 200 		# imaginary time max in units of [1/ω⊥]
    dyTmax 		= 4000       # dynamics max time [1/ω⊥]
    TsampleN    = 300       # number of sample points for dynamics

    # Spaitial domain
    nx  		= Int(2^8+1)	# number of spatial domain 
	xmaxSI 		= 100e-6	# maximum single-sided domain length [m]
    
	# Set integrator tolerances
    abstol_int  = 1e-6
    reltol_int  = 1e-6
    maxiters_int= Int(1e8)
	
	# Collect parameters for saving
    params = @strdict species Natom f⊥ fSI pSI p_quenchSI qSI q_quenchSI MTseed trWigner Nvirtual imTmax dyTmax TsampleN nx xmaxSI 
end;

# ╔═╡ c264a297-3519-4049-9a17-cac784002f08
md"Dependent coefficients, normalized coefficients, misc. "

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
end;

# ╔═╡ e97921e7-d01e-4ce4-a47c-4d50e2c60315
3000/ω⊥

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
"""

# ╔═╡ 67717844-d4e7-430b-9b00-abc111ca7065
md"## 2. Find ground state
### Construct single particle Hamiltonian
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
			
 	# plot!(xx_um, real([UZeeman_quench.data[ii, ii] for ii = 1:nx]) , label = "Zeeman afer quench" , frame = :box, legend = :best) 
		end
		)
    xlabel!("Position [μm]")
    ylabel!("Energy [Hz]")
end


# ╔═╡ 0a43de57-7ef6-44cc-bf77-30b82f2eadce
md"### Preparing intial guess for the imaginary time evolution"

# ╔═╡ ee221bb0-91b5-42d3-9dbd-dafad642a604
md"Check the shape of the potential"

# ╔═╡ 097c0ebd-aa49-4f28-8eb6-a3095d2972f0
function heaviside(x)
   0.5 * (sign(x) + 1)
end

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
function initialstate_gaussian(bx, N; σ::Float64 = 1.0, spinratio = [1,1,1])
	ϕin_p = spinratio[1]*gaussianstate(bx, 0, 0, σ)
	ϕin_0 = spinratio[2]*gaussianstate(bx, 0, 0, σ)
	ϕin_m = spinratio[3]*gaussianstate(bx, 0, 0, σ)
	ψgaussian = normalize_wfn(ϕin_p ⊕ ϕin_0 ⊕ ϕin_m, N)
return ψgaussian
end

# ╔═╡ 59d40cb1-4b4a-44e2-9491-8a9d47e4da01
function initialstate_soliton(bx, N;x0SI::Float64 = 20.0, σ::Float64 = 1.0, spinratio = [1,1,1])
	x0 = x0SI/a⊥
	ϕin_p = spinratio[1]*gaussianstate(bx, 0, 0, σ)
	ϕin_0 = spinratio[2]*gaussianstate(bx, 0, 0, σ) 
	ϕin_m = spinratio[3]*gaussianstate(bx, 0, 0, σ) 
	aa = 1
	ϕin_p.data   = ϕin_p.data* aa .* 0
	ϕin_0.data   = ϕin_0.data* aa .* (heaviside.(xx.-x0).-0.5)*2
	# ϕin_0.data   = ϕin_0.data* aa .* (heaviside.(xx.-x0).-0.5).*(heaviside.(xx.+x0).-0.5)*2
	ϕin_m.data   = ϕin_m.data* aa .* 0
	ψout = normalize_wfn(ϕin_p ⊕ ϕin_0 ⊕ ϕin_m, N)
	return ψout
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
	plot!(x, n1,  label = L"n_{+1}",  ls = :solid, fillrange = 0, fillalpha = 0.35)
	plot!(x, n2,  label = L"n_{0}",  ls = :dot, fillrange = 0, fillalpha = 0.35)
	plot!(x, n3, label = L"n_{-1}",  ls = :dash, fillrange = 0, fillalpha = 0.35)
	plot!(x, (n1+n2+n3), label = L"n")
	return plt
end

# ╔═╡ 1ed291a7-d08d-4d8b-bde4-3a77451eed44
begin
	    # Preparing the initial state
    # ψ0 = initialstate_gaussian(bx, Natom, σ = xmax/3, spinratio = [1, 1, 1])
	ψ0 = initialstate_soliton(bx, Natom, σ = xmax/2,x0SI = 20.0e-6, spinratio = [1, 1, 1])
    plot_wfn(xx_um, ψ0)
    title!("Initial guess for the ground state")
end

# ╔═╡ 79694901-4257-478b-a94d-a13bf69e4463
md"### Imaginary time evolution
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
end;

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
		plot!(xx, T/ω⊥*1e3, 
			[get_density(ψarray[ii])[kk][jj] for ii = 1:length(T), jj = 1:length(xx)]
			,  lt = :surface, c = kk, alpha = 0.5)
		for ii = 1:length(T)
			plot!(xx,fill(T[ii], length(xx)),(get_density(ψarray[ii])[kk]), c = kk, lw = 1; kwargs...)
		end
	end
	plt
end

# ╔═╡ 143eea5b-37a4-450c-9dd9-683ded1a1f93
function plot_snapshots_heatmap(xx, ψarray, T; kwargs...)
	function get_density(ψ)
		# Plot wavefunction in SI units
		n1 = abs2.(getblock(ψ,1).data) /(a⊥/1e-6)
		n2 = abs2.(getblock(ψ,2).data) /(a⊥/1e-6)
		n3 = abs2.(getblock(ψ,3).data) /(a⊥/1e-6)
		return [n1, n2, n3]
	end
	plt = plot(ylabel = "Position [μm]", xlabel = "Time [ms]", cam = (70, 20),legend = false)
	for kk = 2
		plot!(T/ω⊥*1e3, xx, 
			[get_density(ψarray[ii])[kk][jj] for jj = 1:length(xx),  ii = 1:length(T)]
			,  lt = :heatmap)
		# for ii = 1:length(T)
		# 	plot!(xx,fill(T[ii], length(xx)),(get_density(ψarray[ii])[kk]), c = kk, lw = 1; kwargs...)
		# end
	end
	plt
end

# ╔═╡ 8d02275a-9d96-4fa8-b14b-4c2daa59f317
plot_snapshots_3D(xx, ψt, tout);

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
    	# plot!(xx_um, nTF, lw = 2, ls=:solid, label ="TF profile")
	end
    # title!("Found ground state")
	foundgs
end

# ╔═╡ 67f85069-4ab1-4a9e-b204-9a81395826af
md"Optional phase profile"

# ╔═╡ 4225d24f-0072-4950-8325-ea22cbd624a2
begin
	function plot_wfn_phase(xx, ψ; kwargs...)
    # Plot wavefunction in SI units
    n1 = angle.(getblock(ψ,1).data)
    n2 = angle.(getblock(ψ,2).data)
    n3 = angle.(getblock(ψ,3).data)
    plt = plot(xx, n1,  label = "+1", linewidth = 2, ls = :solid, framestyle = :box;kwargs...)
    plot!(xx, n2,  label = "0", linewidth = 2, ls = :dash;kwargs...)
    plot!(xx, n3, label = "-1", linewidth = 2, ls = :dot;kwargs...)
    xlabel!("Position [μm]")
    ylabel!("Phase [rad]")
    return plt
end

    plot_wfn_phase(xx_um, ψg)
end

# ╔═╡ f654d1d1-e921-4e23-9474-81d460abf753
    plot_wfn_phase(xx_um, ψ0; lt = :line)

# ╔═╡ c915035c-c129-47d0-b1f5-7e8515bc229e
md"## 3. Compute quench dynamics
Using the ground state we found, simulate quench dynamics. 
"

# ╔═╡ e685a6f6-0e27-4395-9f49-8e156cce4534
md"### Inject seed (vacuum) fluctuation 
Generate noise wavefunction and check the orthogonality."

# ╔═╡ 01cd7465-0cad-468d-bc5a-dd9f8b5ed34a
begin
	"""
	Generate wavefunction to seed instability.
	"""
	function get_psinoise(seed, trWigner, ψbec; spin0::Bool = false)
		
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
			ϕin_p = Ket(bx, generate_fluctuation(randnum[1]))
			if spin0
				ϕin_0 = Ket(bx, generate_fluctuation(randnum[2])) 
			else
				ϕin_0 = Ket(bx, generate_fluctuation(randnum[2])) 
			end
			ϕin_m = Ket(bx, generate_fluctuation(randnum[3]))
			ψnoise = ϕin_p ⊕ ϕin_0 ⊕ ϕin_m	
			Nvacuum = sum(dagger(ψnoise)*(ψnoise))*dx
			ψnoise = make_orthogonal(ψnoise, ψbec)
			ψnoise = normalize_wfn(ψnoise, Nvacuum)
		else
			ϕin_p = Ket(bx, rand(MersenneTwister(randnum[1]), nx))
			if spin0
				ϕin_0 = Ket(bx, rand(MersenneTwister(randnum[2]), nx))
			else
				ϕin_0 = Ket(bx, rand(MersenneTwister(randnum[2]), nx))
			end
			ϕin_m = Ket(bx, rand(MersenneTwister(randnum[3]), nx))
			ψnoise = ϕin_p ⊕ ϕin_0 ⊕ ϕin_m	
			ψnoise = make_orthogonal(ψnoise, ψbec)
			ψnoise = normalize_wfn(ψnoise, Nvirtual)
		end
	    return 	ψnoise
	end
	
	ψ_noise = get_psinoise(MTseed, trWigner, ψg, spin0 = false)
	
    # check the noise
	plot_wfn(xx_um, ψ_noise)
	title!("Total number= $(round( sum(abs2.(ψ_noise.data)*dx), digits = 4))")
end
    

# ╔═╡ 645a8e77-e6df-487d-932a-199440e35f97
begin
	# intial condition
    ψt_dy0 = ψg + ψ_noise
	plot_wfn(xx_um, ψt_dy0)
	title!("Total number= $(round( sum(abs2.(ψt_dy0.data)*dx), digits = 7))")
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
plot_psi_3D(xx_um, ψt_dy0)

# ╔═╡ 068bd888-a0c6-4742-9e6f-0a2b21da48e4
md"### Compute and check quench dynamics"

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
	if token_quenchdy
   		tout_dy, ψt_dy = timeevolution.schroedinger_dynamic(T_dy, ψt_dy0, Hgp_dy, alg = DP5(), abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int)
	# Check the results
    plot_snapshots(xx_um, ψt_dy, T_dy; size = (600, 600))
	end
end

# ╔═╡ 07ea6c1a-b67d-4bdd-bbed-be34de0ae00c
md"## 4. Save data and calculate some observables"

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
	function plot_pop_dynamics(T_dy, ψ)
		n1 = zeros(length(T_dy));
		n2 = zeros(length(T_dy));
		n3 = zeros(length(T_dy));
		for ii = 1:length(T_dy)
			n1[ii] = sum(abs2.(getblock(ψ[ii],1).data))*dx
			n2[ii] = sum(abs2.(getblock(ψ[ii],2).data))*dx
			n3[ii] = sum(abs2.(getblock(ψ[ii],3).data))*dx
		end
		plt = plot(T_dy, n1,  label = "+1", lw = 2, ls = :solid, fs = :box)
		plot!(T_dy, n2, label = "0", lw = 2, ls = :solid)
		plot!(T_dy, n3, label = "-1", lw = 2, ls = :dash)
		plot!(T_dy, n1+n2+n3, label = "total", lw = 2, ls = :dot, legend = :left)
		return plt
	end
	if token_quenchdy
		plot_pop_dynamics(T_dy/ω⊥ *1e3, ψt_dy)
		ylabel!("Total atom number")
		xlabel!("Time [ms]")
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
		if token_save
			gif(anim, plotsdir("Li7", fname*".gif"), fps = 10) 
		end
		gif(anim, fps = 10)
	end
end

# ╔═╡ 1dd33a10-6e92-4343-9cbf-44e1a83e8bff
ξ = √(1/(8*π*(2*a2Li + a0Li)/3/2/π/a⊥^2*npeakSI))

# ╔═╡ fd925395-38b2-4ee5-820a-bcb56a638323
npeakSI

# ╔═╡ 8031a1fa-ac73-495f-8cea-e8365e587d40
begin
	plot_snapshots_heatmap(xx_um, ψt_dy, tout_dy)
	# plot!(tout_dy/ω⊥*1e3, 15*cos.(2*π*fSI*tout_dy/ω⊥*0.8) , c = :white, ls = :dash)
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
plot([sum(abs2.(ψt_dy[ii].data))*dx for ii = 1:length(tout_dy)])

# ╔═╡ 4a605c05-8a85-4df2-a4d9-5699e1f0f672
plot_Fplus(ψt_dy[1])

# ╔═╡ a03c028e-f013-4aa9-b227-08ed0d0f0f03
plot_psi_3D(xx_um, ψt_dy[100], cam = (45, 20))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DiffEqCallbacks = "459566f4-90b8-5000-8ac3-15dfb0a30def"
DrWatson = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
PhysicalConstants = "5ad8b20f-a522-5ce9-bfc9-ddf1d5bda6ab"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuantumOptics = "6e0679c1-51ea-5a7c-ac74-d61b76210b0c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[compat]
BenchmarkTools = "~1.2.2"
DiffEqCallbacks = "~2.18.0"
DrWatson = "~2.7.5"
JLD2 = "~0.4.15"
LaTeXStrings = "~1.3.0"
OrdinaryDiffEq = "~5.70.0"
PhysicalConstants = "~0.2.1"
Plots = "~1.25.2"
PlutoUI = "~0.7.22"
QuantumOptics = "~1.0.1"
Unitful = "~1.9.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "265b06e2b1f6a216e0e8f183d28e4d354eab3220"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "940001114a0147b6e4d10624276d56d531dd9b49"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.2"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "bc1317f71de8dce26ea67fcdf7eccc0d0693b75b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CPUSummary]]
deps = ["Hwloc", "IfElse", "Static"]
git-tree-sha1 = "87b0c9c6ee0124d6c1f4ce8cb035dcaf9f90b803"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.6"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4c26b4e9e91ca528ea212927326ece5918a04b47"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.2"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "7b8f09d58294dc8aa13d91a8544b37c8a1dcbc06"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DEDataArrays]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "RecursiveArrayTools", "SciMLBase", "StaticArrays"]
git-tree-sha1 = "31186e61936fbbccb41d809ad4338c9f7addf7ae"
uuid = "754358af-613d-5f8d-9788-280bf1605d4c"
version = "0.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "ChainRulesCore", "DEDataArrays", "DataStructures", "Distributions", "DocStringExtensions", "FastBroadcast", "ForwardDiff", "FunctionWrappers", "IterativeSolvers", "LabelledArrays", "LinearAlgebra", "Logging", "MuladdMacro", "NonlinearSolve", "Parameters", "PreallocationTools", "Printf", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "Requires", "SciMLBase", "Setfield", "SparseArrays", "StaticArrays", "Statistics", "SuiteSparse", "ZygoteRules"]
git-tree-sha1 = "9a309839a580a40111ff7ac1d3c19792a91985ba"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.77.0"

[[deps.DiffEqCallbacks]]
deps = ["DataStructures", "DiffEqBase", "ForwardDiff", "LinearAlgebra", "NLsolve", "OrdinaryDiffEq", "Parameters", "RecipesBase", "RecursiveArrayTools", "SciMLBase", "StaticArrays"]
git-tree-sha1 = "a615f494f0c10d0a21f895aa0b65b2056c37d17b"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "2.18.0"

[[deps.DiffEqJump]]
deps = ["ArrayInterface", "Compat", "DataStructures", "DiffEqBase", "FunctionWrappers", "Graphs", "LinearAlgebra", "PoissonRandom", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "StaticArrays", "TreeViews", "UnPack"]
git-tree-sha1 = "0aa2d003ec9efe2a93f93ae722de05a870ffc0b2"
uuid = "c894b116-72e5-5b58-be3c-e6d8d4ac2b12"
version = "8.0.0"

[[deps.DiffEqNoiseProcess]]
deps = ["DiffEqBase", "Distributions", "LinearAlgebra", "Optim", "PoissonRandom", "QuadGK", "Random", "Random123", "RandomNumbers", "RecipesBase", "RecursiveArrayTools", "Requires", "ResettableStacks", "SciMLBase", "StaticArrays", "Statistics"]
git-tree-sha1 = "d6839a44a268c69ef0ed927b22a6f43c8a4c2e73"
uuid = "77a26b50-5914-5dd7-bc55-306e6241c503"
version = "5.9.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "41660d34a73b9983b9a98bafafcee44539c72bb2"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.6.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "d6cc7abd52ebae5815fd75f6004a44abcf7a6b00"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.35"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DrWatson]]
deps = ["Dates", "FileIO", "JLD2", "LibGit2", "MacroTools", "Pkg", "Random", "Requires", "Scratch", "UnPack"]
git-tree-sha1 = "dfc6c06fa560e6a7658245bacd90fc6a3a6c1cce"
uuid = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
version = "2.7.5"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[deps.ExponentialUtilities]]
deps = ["ArrayInterface", "LinearAlgebra", "Printf", "Requires", "SparseArrays"]
git-tree-sha1 = "1b873816d2cfc8c0fcb1edcb08e67fdf630a70b7"
uuid = "d4d017d3-3776-5f7e-afef-a10c40355c18"
version = "1.10.2"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FastBroadcast]]
deps = ["LinearAlgebra", "Polyester", "Static"]
git-tree-sha1 = "e32a81c505ab234c992ca978f31ed8b0dabbc327"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.1.11"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2db648b6712831ecb333eae76dbfd1c156ca13bb"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.2"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "6406b5112809c08b1baa5703ad274e1dded0652f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.23"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "92243c07e786ea3458532e199eb3feee0e7e08eb"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.4.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HalfIntegers]]
git-tree-sha1 = "dc0ce9efc3d88c6cefc4e1f9c29b397be8734cfc"
uuid = "f0d1745a-41c9-11e9-1dd9-e5d34d218721"
version = "1.4.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8f0dc80088981ab55702b04bba38097a44a1a3a9"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.5"

[[deps.Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3395d4d4aeb3c9d31f5929d32760d8baeee88aaf"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.5.0+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["DataStructures", "FileIO", "MacroTools", "Mmap", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "46b7834ec8165c541b0b5d1c8ba63ec940723ffb"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.15"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LRUCache]]
git-tree-sha1 = "d64a0aff6691612ab9fb0117b0995270871c5dfc"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.3.0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "3609bbf5feba7b22fb35fe7cb207c8c8d2e2fc5b"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.6.7"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "83b56449c39342a47f3fcdb3bc782bd6d66e1d97"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.4"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "dbb14c604fc47aa4f2e19d0ebb7b6416f3cfa5f5"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.5.1"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "Requires", "SIMDDualNumbers", "SLEEFPirates", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "9e10579c154f785b911d9ceb96c33fcc1a661171"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.99"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.ManualMemory]]
git-tree-sha1 = "9cb207b18148b2199db259adfa923b45593fe08e"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.6"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf", "RecipesBase", "Requires"]
git-tree-sha1 = "31c8c0569b914111c94dd31149265ed47c238c5b"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.6.0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MuladdMacro]]
git-tree-sha1 = "c6190f9a7fc5d9d5915ab29f2134421b12d24a68"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.2"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[deps.NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "8dc3be3e9edf976a3e79363b3bd2ad776a627c31"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.12"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "35d435b512fbab1d1a29138b5229279925eba369"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.5.0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.OrdinaryDiffEq]]
deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "ExponentialUtilities", "FastClosures", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "Logging", "LoopVectorization", "MacroTools", "MuladdMacro", "NLsolve", "Polyester", "PreallocationTools", "RecursiveArrayTools", "Reexport", "SparseArrays", "SparseDiffTools", "StaticArrays", "UnPack"]
git-tree-sha1 = "c84fb58ba1308db7a95511c7f0eba1a0bf2bc829"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "5.70.0"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[deps.PhysicalConstants]]
deps = ["Measurements", "Roots", "Unitful"]
git-tree-sha1 = "2bc26b693b5cbc823c54b33ea88a9209d27e2db7"
uuid = "5ad8b20f-a522-5ce9-bfc9-ddf1d5bda6ab"
version = "0.2.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "65ebc27d8c00c84276f14aaf4ff63cbe12016c70"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "565564f615ba8c4e4f40f5d29784aa50a8f7bbaf"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.22"

[[deps.PoissonRandom]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "44d018211a56626288b5d3f8c6497d28c26dc850"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.0"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "892b8d9dd3c7987a4d0fd320f0a421dd90b5d09d"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.5.4"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "a3ff99bf561183ee20386aec98ab8f4a12dc724a"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "LabelledArrays"]
git-tree-sha1 = "435379f01c1e6f7ca65cf46fdd403226f1d36e37"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[deps.Primes]]
git-tree-sha1 = "984a3ee07d47d401e0b823b7d30546792439070a"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.QuantumOptics]]
deps = ["Arpack", "DiffEqCallbacks", "FFTW", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "OrdinaryDiffEq", "QuantumOpticsBase", "Random", "RecursiveArrayTools", "Reexport", "SparseArrays", "StochasticDiffEq", "WignerSymbols"]
git-tree-sha1 = "eea423799266abf5c22a41313f26c11b9bb54488"
uuid = "6e0679c1-51ea-5a7c-ac74-d61b76210b0c"
version = "1.0.1"

[[deps.QuantumOpticsBase]]
deps = ["Adapt", "FFTW", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "5506df67d351551b521c3099e19e06b6ea64e038"
uuid = "4f57444f-1401-5e15-980d-4471b28d5678"
version = "0.3.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Libdl", "Random", "RandomNumbers"]
git-tree-sha1 = "0e8b146557ad1c6deb1367655e052276690e71a3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.4.2"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RationalRoots]]
git-tree-sha1 = "52315cf3098691c1416a356925027af5ab5bf548"
uuid = "308eb6b3-cc68-5ff3-9e97-c3c4da4fa681"
version = "0.2.0"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[deps.RecursiveArrayTools]]
deps = ["ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "c944fa4adbb47be43376359811c0a14757bdc8a8"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.20.0"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "b7edd69c796b30985ea6dfeda8504cdb7cf77e9f"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.5"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[deps.ResettableStacks]]
deps = ["StaticArrays"]
git-tree-sha1 = "256eeeec186fa7f26f2801732774ccf277f05db9"
uuid = "ae5879a3-cd67-5da8-be7f-38c6eb64a37b"
version = "1.1.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "51ee572776905ee34c0568f5efe035d44bf59f74"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "1.3.11"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SIMDDualNumbers]]
deps = ["ForwardDiff", "IfElse", "SLEEFPirates", "VectorizationBase"]
git-tree-sha1 = "62c2da6eb66de8bb88081d20528647140d4daa0e"
uuid = "3cdde19b-5bb0-4aaf-8931-af3e248e098b"
version = "0.1.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "1410aad1c6b35862573c01b96cd1f6dbe3979994"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.28"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "5fc45c4a693d96698961e4a48c5a5fcfaf8ff876"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.22.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "0afd9e6c623e379f593da01f20590bacc26d1d14"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SparseDiffTools]]
deps = ["Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays", "VertexSafeGraphs"]
git-tree-sha1 = "5e86e10d8a833e792d27c5db9a172d002cb4c4e2"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "1.18.3"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "e7bc80dc93f50857a5d1e3c8121495852f407e6a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[deps.StochasticDiffEq]]
deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DiffEqJump", "DiffEqNoiseProcess", "DocStringExtensions", "FillArrays", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "Logging", "MuladdMacro", "NLsolve", "OrdinaryDiffEq", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "SparseArrays", "SparseDiffTools", "StaticArrays", "UnPack"]
git-tree-sha1 = "d6756d0c66aecd5d57ad9d305d7c2526fb5922d9"
uuid = "789caeaf-c7a9-5a7d-9973-96adeb23e2a0"
version = "6.41.0"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "Requires", "SIMDTypes", "Static", "ThreadingUtilities"]
git-tree-sha1 = "12cf3253ebd8e2a3214ae171fbfe51e7e8d8ad28"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.2.9"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "03013c6ae7f1824131b2ae2fc1d49793b51e8394"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.4.6"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "ec9a310324dd2c546c07f33a599ded9c1d00a420"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "0992ed0c3ef66b0390e5752fe60054e5ff93b908"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.9.2"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "Hwloc", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "17e5847bb36730d90801170ecd0ce4041a3dde86"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.22"

[[deps.VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[deps.WignerSymbols]]
deps = ["HalfIntegers", "LRUCache", "Primes", "RationalRoots"]
git-tree-sha1 = "960e5f708871c1d9a28a7f1dbcaf4e0ee34ee960"
uuid = "9f57e263-0b3d-5e2e-b1be-24f2bb48858b"
version = "2.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─c6450a7c-ecba-4f85-98ef-3d7a007dc64f
# ╠═6c191cd6-3a1d-4631-8737-3e647b54a1cd
# ╟─5e7fce10-8003-46df-aab7-0af1046c30e0
# ╟─4bfb2241-5a5b-4dc8-aac1-4e8b0f53a33c
# ╟─fd11bf5d-66c8-45e0-80f5-40386c7c9c06
# ╟─6d9b9a15-107e-4763-b670-264b5b019152
# ╟─3241b4ff-86d5-4bbb-81fa-d527398792b5
# ╠═e633a360-b643-11eb-3979-6f8b7a3551c4
# ╠═6003638d-a1a9-4ae9-be69-232c2d1543d8
# ╠═69cb4cbc-572d-4b57-98cd-ef297927e2b8
# ╟─84baced5-832d-4855-ba5a-9380571c65fd
# ╠═8d94d626-3d97-456c-8fea-1e0feb5794d7
# ╠═d34d395d-ff80-4f72-8aac-47366ee7aa06
# ╟─709d22eb-46f7-4af9-ae5c-2c2ba49e7b06
# ╟─dc0331a0-5f9b-4dcd-b00b-5839e12130ce
# ╠═8a9a006b-f778-4d7b-8814-4c3869d07a76
# ╠═0075ee54-d4e0-43d7-92b9-24971e92c108
# ╠═55ff32a3-0e3d-4a66-8764-ce3fee1fd20c
# ╠═e97921e7-d01e-4ce4-a47c-4d50e2c60315
# ╠═c264a297-3519-4049-9a17-cac784002f08
# ╠═1423582b-6db1-4a00-968c-98f831058017
# ╠═fae7e791-7eee-4619-83cb-2ab516114f00
# ╟─67717844-d4e7-430b-9b00-abc111ca7065
# ╠═4b903823-bbe1-4245-ad8e-d0a56ccce22b
# ╠═82dc85f1-f3a1-4b63-9618-799affe9fd8c
# ╠═0a43de57-7ef6-44cc-bf77-30b82f2eadce
# ╟─ee221bb0-91b5-42d3-9dbd-dafad642a604
# ╠═621b5d67-a3cb-4728-8eae-743b1fb6ad27
# ╠═097c0ebd-aa49-4f28-8eb6-a3095d2972f0
# ╠═59d40cb1-4b4a-44e2-9491-8a9d47e4da01
# ╠═2fbfca9b-7ae8-4e88-922d-37ae839a62b9
# ╠═cdbbded1-bd40-413a-aa88-408c2ee4a364
# ╠═9234bb35-0b58-4e7d-83b5-5ee66357238a
# ╠═840f6795-0fc1-4dcd-bd7d-aee2ee74ee0a
# ╠═1ed291a7-d08d-4d8b-bde4-3a77451eed44
# ╠═f654d1d1-e921-4e23-9474-81d460abf753
# ╟─79694901-4257-478b-a94d-a13bf69e4463
# ╠═078db709-2df6-43bf-98b6-32470e74cce7
# ╟─dc86499a-70cb-4e9e-9257-2782050581ff
# ╠═33ffb2bd-9762-48aa-87bf-44f5b5044ced
# ╠═01a9053f-a9f8-4b74-ae5f-18d7ea7fb187
# ╠═143eea5b-37a4-450c-9dd9-683ded1a1f93
# ╠═8d02275a-9d96-4fa8-b14b-4c2daa59f317
# ╠═bfe9e487-bc60-4b6b-8bd1-5f44342b38e6
# ╟─461502f9-359a-42ab-a06f-36846c2be82e
# ╠═f82490de-0b75-4151-952d-7ace5a22aa35
# ╟─67f85069-4ab1-4a9e-b204-9a81395826af
# ╠═4225d24f-0072-4950-8325-ea22cbd624a2
# ╟─c915035c-c129-47d0-b1f5-7e8515bc229e
# ╟─e685a6f6-0e27-4395-9f49-8e156cce4534
# ╟─01cd7465-0cad-468d-bc5a-dd9f8b5ed34a
# ╠═645a8e77-e6df-487d-932a-199440e35f97
# ╠═640be080-c000-4aeb-9e43-09f1bae085a3
# ╠═9fd8b82f-424b-42d9-8b70-df593b26d7ec
# ╟─068bd888-a0c6-4742-9e6f-0a2b21da48e4
# ╠═0485091d-d6e7-456a-b462-bd62e5166292
# ╟─07ea6c1a-b67d-4bdd-bbed-be34de0ae00c
# ╠═1034f666-6458-49c5-bcf2-9a92b2115989
# ╟─a4d11f49-950c-4b8e-b5f9-a86d123ef4d8
# ╟─6437597e-c5fd-4a18-b524-3ef729dfc786
# ╠═86268d8e-25ac-4a75-9e85-1b968994066b
# ╠═1dd33a10-6e92-4343-9cbf-44e1a83e8bff
# ╠═fd925395-38b2-4ee5-820a-bcb56a638323
# ╠═8031a1fa-ac73-495f-8cea-e8365e587d40
# ╟─11db7f14-4484-46b7-9e52-606e56f3c554
# ╟─f1905078-295b-4219-b37a-128855ec3988
# ╠═e093617e-9246-4e0b-99d3-212c075a900f
# ╠═4a605c05-8a85-4df2-a4d9-5699e1f0f672
# ╠═a03c028e-f013-4aa9-b227-08ed0d0f0f03
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
