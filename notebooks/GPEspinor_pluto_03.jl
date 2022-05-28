### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 09a5aae0-70d1-11eb-2e28-31e5d0d7daed
begin
	@time using QuantumOptics
	@time using OrdinaryDiffEq, DiffEqCallbacks
	@time using Plots
	plotly()
end

# ╔═╡ 87bdeb30-637b-11eb-2b40-4b885e156281
md"# Spinor BEC in 1D
by kT"

# ╔═╡ 6f01e4a0-646d-11eb-372a-81a875a291ed
md"[Reference for Julia implementation](https://docs.qojulia.org/examples/spin-orbit-coupled-BEC1D/#Including-two-body-interactions-(stripe-phase))"

# ╔═╡ 38f91830-6777-11eb-0ed1-2fb2ca451162
md"
## Spin-1 BEC GPE
ref: Kawaguchi, Ueda (2012)

```math 
E[\psi] = \int d\mathbf{r}\left\{ \sum_{m=-1}^{1} \psi^*_m \left[ -\frac{\hbar^2 \nabla^2}{2M} + U_{\mathrm{trap}}(\mathbf{r}) - p(x)m + qm^2 \right] \psi_m + \frac{c_0^{(3D)}}{2} n^2 + \frac{c_1^{(3D)}}{2} |\mathbf{F}|^2 \right\}
```
"

# ╔═╡ cd6af090-677e-11eb-023c-8d0d2ba5df32
md"
For 1D,

```math 
E[\psi] = \int d\mathbf{r}\left\{ \sum_{m=-1}^{1} \psi^*_m(x) \left[ -\frac{\hbar^2}{2M}\frac{d^2}{dx^2} + U_{\text{trap}}(x) - p(x)m + qm^2 \right] \psi_m(x) + \frac{c_0}{2} n^2 + \frac{c_1}{2} |\mathbf{F}|^2 \right\}.
```
Here, we redefine the interaction coefficients. 
$c_0 = 2\hbar\omega_{\bot}c_0^{(3D)}$, $c_1 = 2\hbar\omega_{\bot}c_1^{(3D)}$. 
$\psi$ becomes 1-dimensional probability amplitude.

"

# ╔═╡ fe7a6a80-636e-11eb-0e80-8137669fdd2c
md"
The time evolution is 
``` math
\begin{align}
i\hbar\frac{\partial\psi_m(x)}{\partial t} &= \frac{\delta E}{\delta \psi^*_m(x)} \\
&= \left[ -\frac{\hbar^2}{2M}\frac{d^2}{dx^2} + U_{\text{trap}}(x) - p(x)m + qm^2 \right] \psi_m + c_0 n \psi_m + c_1 \sum_{m'=-1}^{1} \mathbf{F} \cdot \mathbf{f}_{mm'} \psi_{m'}
\end{align}
```

We explicitly write the spin density vector, $\mathbf{F}$
``` math
\begin{align}
F_x &= \frac{1}{\sqrt{2}} \left[ \psi^*_1 \psi_0 + \psi^*_0 (\psi_1 + \psi_{-1}) + \psi^*_{-1} \psi_0 \right] \\
F_y &= \frac{i}{\sqrt{2}} \left[ -\psi^*_1 \psi_0 + \psi^*_0 (\psi_1 - \psi_{-1}) + \psi^*_{-1} \psi_0 \right] \\
F_z &= |\psi_1|^2 - |\psi_{-1}|^2 \\
F_+ &= \sqrt{2}\left[ \psi^*_1 \psi_0 + \psi^*_0 \psi_{-1} \right] \\
F_- &= \sqrt{2}\left[  \psi^*_0\psi_1 + \psi^*_{-1} \psi_0  \right] 
\end{align}
```
and the spin matrices

``` math
\begin{equation*}
    \mathsf{f}_{x} = \frac{1}{\sqrt{2}} 
    \begin{pmatrix}
        0 & 1 & 0 \\
        1 & 0 & 1 \\
        0 & 1 & 0 \\
    \end{pmatrix}, \quad
    \mathsf{f}_{y} = \frac{i}{\sqrt{2}} 
    \begin{pmatrix}
            0 & -1  & 0  \\
            1 &  0  & -1 \\
            0 &  1  & 0  \\
    \end{pmatrix}, \quad
    \mathsf{f}_{z} = 
    \begin{pmatrix}
            1 &  0  & 0  \\
            0 &  0  & 0  \\
            0 &  0  & -1 \\
    \end{pmatrix}.
\end{equation*}
```

"

# ╔═╡ 1e07c6c0-6466-11eb-0b4f-4f433f18e09e
md"
We write down the equation we have to solve explicitly.
```math
\begin{align}
	i\hbar\partial_t\psi_1    &=  
\left[ 
-\frac{\hbar^2}{2M}\partial^2_x + U_{\text{trap}}(x) - p(x) + q \right] \psi_1 
+ c_0 n \psi_1 + c_1 \left[F_z\psi_1   + \frac{F_-}{\sqrt{2}} \psi_0
\right]\\

	i\hbar\partial_t\psi_0    &=  
\left[ 
-\frac{\hbar^2}{2M}\partial^2_x + U_{\text{trap}}(x)\right] \psi_0 
+ c_0 n \psi_0 + c_1 \left[\frac{F_+}{\sqrt{2}} \psi_1  + \frac{F_-}{\sqrt{2}} \psi_{-1}
\right]\\

	i\hbar\partial_t\psi_{-1} &= 
\left[ 
-\frac{\hbar^2}{2M}\partial^2_x + U_{\text{trap}}(x) + p(x) + q \right] \psi_{-1} 
+ c_0 n \psi_{-1} + c_1 \left[-F_z\psi_{-1}   + \frac{F_+}{\sqrt{2}} \psi_0
\right] \\

\end{align}
```

n is the total density
"

# ╔═╡ ec64e12e-6469-11eb-0880-33ea64337e40
md"
Spin-dependent interaction terms are
```math
\begin{gather}
%c_1\bigg[(|\psi_1|^2 -|\psi_{-1}|^2)\psi_1   + (\psi^*_0\psi_1 + \psi^*_{-1}\psi_0) \psi_0 \bigg] = 
c_1\bigg[(|\psi_1|^2 -|\psi_{-1}|^2 + |\psi_0|^2)\psi_1   + \psi^*_{-1}\psi_0^2
\bigg]\\

% c_1 \bigg[(\psi^*_1\psi_0 + \psi^*_0\psi_{-1}) \psi_1  + (\psi^*_0\psi_1 + \psi^*_{-1}\psi_0) \psi_{-1}\bigg] = 
c_1 \bigg[(|\psi_1|^2 + |\psi_{-1}|^2)\psi_0 + 2\psi^*_0\psi_{-1}\psi_1 \bigg]\\

% c_1 \bigg[-(|\psi_1|^2 -|\psi_{-1}|^2)\psi_{-1}   + (\psi^*_1\psi_0 + \psi^*_0\psi_{-1}) \psi_0 \bigg] = 
 c_1 \bigg[(-|\psi_1|^2 +|\psi_{-1}|^2 + |\psi_0|^2)\psi_{-1}   + \psi^*_1\psi_0^2
\bigg] \\

\end{gather}
```
Offdiagonal terms can be expressed in a hermitian matrix form.

```math
\begin{pmatrix}
	0 & \psi^*_{-1}\psi_0 & 0 \\
	\psi^*_{0}\psi_{-1} & 0 & \psi^*_{0}\psi_{1} \\
	0 & \psi^*_{1}\psi_{0} & 0 \\
\end{pmatrix}
\begin{pmatrix}
	\psi_1\\ \psi_0 \\ \psi_{-1}
\end{pmatrix}
```
"

# ╔═╡ 41ff0070-6a79-11eb-1166-bf38b9b0dbe5
md"## The dimensionless GPE
Small numbers such as $\hbar$ can cause numerical error and we prevent it by properly normalize the equation. It also provides understanding on the relavant physical scales of the equation.
Now define the dimensionless GPE. Use change of variable as follows.
```math 
\begin{gather}
x \rightarrow a_{\bot}x \\
t \rightarrow t/\omega_{\bot} \\
\psi \rightarrow \psi/\sqrt{a_\bot}
\end{gather} 
```
Here, $a_{\bot} = \sqrt{\frac{\hbar}{M \omega_{\bot}}}$, $\omega_{\bot}$ is the radial frequency of quasi-1D trap (in 1D mean-field regime). GPE becomes

```math
\begin{align}
	i\partial_t\psi_1    &=  
\left[ 
-\frac{1}{2}\partial^2_x + \frac{1}{2}\gamma^2 x^2 - \tilde{p} + \tilde{q} \right] \psi_1 
+ \bigg[ \tilde{c}_0 n + \tilde{c}_1(|\psi_1|^2 -|\psi_{-1}|^2 + |\psi_0|^2)\bigg]\psi_1   + \tilde{c}_1\psi^*_{-1}\psi_0^2 \\

	i\partial_t\psi_0    &=  
\left[ 
-\frac{1}{2}\partial^2_x + \frac{1}{2}\gamma^2 x^2 \right] \psi_0 
+ \bigg[ \tilde{c}_0 n + \tilde{c}_1[(|\psi_1|^2 + |\psi_{-1}|^2) \bigg] \psi_0 + 2\tilde{c}_1\psi^*_0\psi_{-1}\psi_1\\

	i\partial_t\psi_{-1} &= 
\left[ 
-\frac{1}{2}\partial^2_x + \frac{1}{2}\gamma^2 x^2 + \tilde{p} + \tilde{q} \right] \psi_{-1} 
+ \bigg[\tilde{c}_0 n + \tilde{c}_1 (|\psi_{-1}|^2 -|\psi_{1}|^2 + |\psi_0|^2) \bigg]\psi_{-1}   + \tilde{c}_1\psi^*_1\psi_0^2 \\

\end{align}
```

"

# ╔═╡ 3d1db920-701e-11eb-3a4d-7da891d70116
md"
Here, $\gamma = \frac{\omega}{\omega_{\bot}}$, $\tilde{p} = \frac{p}{\hbar\omega_{\bot}}$, $\tilde{q} = \frac{q}{\hbar\omega_{\bot}}$, $\tilde{c}_0 =\frac{c_0}{\hbar\omega_{\bot}a_{\bot}}$ and $\tilde{c}_1  =\frac{c_0}{\hbar\omega_{\bot}a_{\bot}}$.
Subscript $\bot$ denotes the radial trap frequency that freeze the motion in the directions.

"

# ╔═╡ 3a9c8750-7030-11eb-34ac-5f1a11849622
md"### Some analytic results of 1D single component
$R_{TF} = \left[ \frac{3c_0 N}{2M\omega^2} \right]^{1/3}$
$n_{\text{peak}} = \frac{1}{2}M\omega^2 R_{TF}^2 / c_0$
$n(x) = n_{\text{peak}} \bigg[1-\frac{x^2}{R_{TF}^2}\bigg]$
$\mu = \frac{M\omega}{2} \bigg[\frac{3c_0 N}{2M\omega^2}\bigg]^{2/3}$
To be valid with 1D mean field regime,

$\frac{Naa_{\bot}}{a_z^2} \ll 1$

To enter mean-field Lieb-Liniger,

$|a_{1D}| n_1 \gg 1$

The opposite limit is for Tonks gas.
Ref: Pitaevskii, Stringari (2016)
"

# ╔═╡ ecdfa9d0-6395-11eb-159e-9168e235b775
md"
### Thechnical notes
- Solve GPE in position basis.
- Spin order of the basis: (1's position) $\oplus$ (0's position) $\oplus$ (-1's position)
- Position basis has periodic boundary condition.
- See 'timeevolution_base.jl' for the information about the solver. It uses DP5(ode45 in MATLAB) method in OrdinaryDiffEq.jl. We can hand over the algorithm of our choice to DifferetialEquations.jl using `alg` kwarg. DP5() works well though.
- Relative tolerance and absolute tolerance. `reltol` impose the stopping condition like `(1-u[i]/u[i-1]) <  reltol` and `abtol` for `(u[i]-u[i-1]) <  abtol`. I may need to check the exact criteria.
"

# ╔═╡ 4d5420ee-6398-11eb-036e-6953db9fb18a
md"## Computation
### Pacakages"

# ╔═╡ 0dca7560-70d1-11eb-2db2-cd726c7b4379
md"### Unit conversion"

# ╔═╡ 00e0f082-7023-11eb-3990-3fc3b0282bca
begin
	# Constants
	c 			= 299792458
	ħ 			= 1.0545718e-34
	h 			= ħ*2*π
	kB 			= 1.38064852e-23
	amu 		= 1.660538921e-27
	a0 			= 5.29e-11
	μB 			= 9.274009994e-24
	Hartree 	= 4.35974e-18
	
	a2Li = 6.8*a0
	a0Li = 23.9*a0
	
	# INPUT in SI unit
	f⊥ 			= 1.2e3		
	fSI			= 10
	pSI 		= 0.00 * 1e-2*h # [T/m]
	p_quenchSI 	= 0.0 * 1e-2*h 
	qSI 		= 1e3*h
	q_quenchSI 	= -1.0e3*h
	
	c0SI 		= 4*π*ħ^2/7/amu * (2*a2Li + a0Li)/3
	c1SI 		= 4*π*ħ^2/7/amu * (a2Li - a0Li)/3
	Natom 		= 2e4 		# atom number
	
	nx  		= Int(2e3)	# number of spatial domain 
	imTmax 		= 100. 		# imaginary time max in units of [1/ω⊥]
	dyTmax 		= 400.
	
	# dependent in SI
	ω⊥ 			= 2*π*f⊥
	a⊥ 			= sqrt(ħ/7/amu/ω⊥)
	ωSI  		= 2*π*fSI
	ax 			= sqrt(ħ/7/amu/ωSI)
	c0_1D_SI 	= c0SI/(2*π*a⊥^2)
	c1_1D_SI 	= c1SI/(2*π*a⊥^2)
	RTF 		= (3*c0_1D_SI*Natom/(2*7*amu*ωSI^2))^(1/3)
	npeakSI 	= 0.5*7*amu*ωSI^2*RTF^2/c0_1D_SI
	xmaxSI 		= 4*RTF
	dxSI 		= 2*xmaxSI/nx
	
	# normalized
	γ 			= ωSI/ω⊥
	xmax 		= xmaxSI/a⊥
	dx 			= 2*xmax/nx
	p 			= pSI/(ħ*ω⊥) 
	q 			= qSI/(ħ*ω⊥)
	p_quench 	= p_quenchSI/(ħ*ω⊥) 
	q_quench 	= q_quenchSI/(ħ*ω⊥)
	c0 			= c0_1D_SI/(ħ*ω⊥*a⊥)
	c1 			= c1_1D_SI/(ħ*ω⊥*a⊥)

	# integrator tolerances
	abstol_int = 1e-6
	reltol_int = 1e-6
	maxiters_int = Int(1e8)
	
end;

# ╔═╡ a817d7f0-7033-11eb-1f2a-ddb317fd14ae
md"### Check some numbers
The ratio $\lambda_q / Δx$ is"

# ╔═╡ 9cf22510-7033-11eb-2cb0-036b14e102e6
begin
	λq = sqrt(2*π^2*ħ^2/7/amu/abs(q_quenchSI)) 
	λq/dxSI
end

# ╔═╡ fe93c5c0-70de-11eb-03f7-632727c5ae36
md"Dimension parameter, $Naa_{\bot}/a_z^2$"

# ╔═╡ 11db1340-70df-11eb-26fe-7f7a79370a67
Natom*a0Li*a⊥/ax^2

# ╔═╡ 5117dbd0-70e2-11eb-2f61-45c584857e7a
md"LL parameter, $a_{1D}n_1$"

# ╔═╡ 5f1ce0e0-70e2-11eb-188e-c34aa6b7f0ad
begin
	a1D = -a⊥^2/a0Li
	a1D * npeakSI
end

# ╔═╡ b00d0b60-7033-11eb-3beb-6b2666557f93
md"### Launch simulations"

# ╔═╡ efe3fae0-636e-11eb-2f69-6d9cca6ceecf
begin
	# basis generation
	bx = PositionBasis(-xmax, xmax, nx); xx = samplepoints(bx);
	bp = MomentumBasis(bx);				 pp = samplepoints(bp);
	bs = SpinBasis(1);
	
	# Basic operators
	x  = position(bx) ⊕ position(bx) ⊕ position(bx);	
	Px = momentum(bp) ⊕ momentum(bp) ⊕ momentum(bp);	

	# transformation operators
	Txp = transform(bx, bp) ⊕ transform(bx, bp) ⊕ transform(bx, bp);
	Tpx = transform(bp, bx) ⊕ transform(bp, bx) ⊕ transform(bp, bx);
	
	# Single particle Hamiltonian (time - independent)
		# kinetic energy opeartor
	Hkin = Px^2/2.  ;
	Hkin_FFT = LazyProduct(Txp, Hkin, Tpx) ;
		# harmonic potential operator
	Utrap = 0.5 * (γ^2 * x^2) ;
		# zeeman shifts
	UZeeman = - p*( position(bx) ⊕ 0*position(bx) ⊕ -1*position(bx) ) + q*(one(bx) ⊕ 0*one(bx) ⊕ one(bx)); 
	
	UZeeman_quench = - p_quench*( position(bx) ⊕ 0*position(bx) ⊕ -1*position(bx) ) + q_quench*(one(bx) ⊕ 0*one(bx) ⊕ one(bx)); # check: heatmap(Array(real(UZeeman.data)), box = "off")
	"Operators generated"
end

# ╔═╡ 3c25cb70-6524-11eb-21ac-2d2b83252360
md"
Check the shape of the potential.
"

# ╔═╡ bf366a80-6522-11eb-3aca-57bbe8979881
begin
	pl1 = plot(xx, real([Utrap.data[ii, ii] for ii = 1:nx] ), label = "trapping", color = "red")
	pl2 = plot(xx, real([UZeeman.data[ii, ii] for ii = 1:nx] ) .- q, label= "gradient", color = "blue")
	plot(pl1, pl2, layout = (1, 2))
end

# ╔═╡ a61eb7be-6395-11eb-0ee2-759b617deae2
md"## Initial state preparation
Check the shape of the potential."

# ╔═╡ b5054330-6395-11eb-0858-6b8cd3099f4a
begin
	p0 = 0; 
	σ0 = RTF/a⊥/2;
	spinratio = [1, 1, 1];
	ϕin_p = spinratio[1]*gaussianstate(bx, 0, p0, σ0);
	ϕin_0 = spinratio[2]*gaussianstate(bx, 0, p0, σ0);
	ϕin_m = spinratio[3]*gaussianstate(bx, 0, p0, σ0);
	# ϕin_p = 1.0*Ket(bx, ones(nx).+0.01im);
	# ϕin_0 = 1.0*Ket(bx, ones(nx).+0.01im);
	# ϕin_m = 1.0*Ket(bx, ones(nx).+0.01im);
	
	function normalize_wfn(ψ)
		return ψ/(norm(ψ) / sqrt(Natom/dx))
	end
	
	function normalize_array!(ψ)
		ψ[:] = ψ/(norm(ψ) / sqrt(Natom/dx))
		return ψ
	end
	
	function plotWfn(xx, ψ)
		n1 = abs2.(getblock(ψ,1).data) /(a⊥/1e-6)
		n2 = abs2.(getblock(ψ,2).data) /(a⊥/1e-6)
		n3 = abs2.(getblock(ψ,3).data) /(a⊥/1e-6)
		plt = plot(xx, n1,  label = "+1", linewidth = 2, linestyle = :solid, framestyle = :box)
		plot!(xx, n2,  label = "0", linewidth = 2, linestyle = :dash)
		plot!(xx, n3, label = "-1", linewidth = 2, linestyle = :solid)
		plot!(xx, (n1+n2+n3), label = "sum")
		# xlabel!("x")
		ylims!((0,Inf))
		return plt
	end
	
	# initial state
	ψ0 = normalize_wfn(ϕin_p ⊕ ϕin_0 ⊕ ϕin_m)
	plotWfn(xx, ψ0)
end

# ╔═╡ 83040b10-769f-11eb-344a-01d1e7203e7f


# ╔═╡ 079bbba0-6450-11eb-1aa7-3f1fbd08ae6f
md"## Find ground state"

# ╔═╡ 95f9b180-6398-11eb-0b53-59102afb3ece
# begin
# 	# Constructing interaction Hamiltonian
# 	Hc0 = one(bx) ⊕ one(bx) ⊕ one(bx);
# 	Hc1 = one(bx) ⊕ one(bx) ⊕ one(bx);
# 	H_im = -1*im*LazySum(Hkin_FFT, Utrap, UZeeman, Hc0, Hc1);
	
# 	function Hgp_im(t, ψ) 
# 		ψ = normalize_wfn(ψ)
# 		ψ_p = ψ.data[1:nx];
# 		ψ_0 = ψ.data[nx+1:2*nx];
# 		ψ_m = ψ.data[2*nx+1:3*nx];
		
# 		np = abs2.(ψ_p)
# 		n0 = abs2.(ψ_0)
# 		nm = abs2.(ψ_m)
		
# 		c0n_dat = c0*(np+n0+nm);

# 			# c0 term
# 		setblock!(Hc0,diagonaloperator(bx, c0n_dat), 1, 1 );
# 		setblock!(Hc0,diagonaloperator(bx, c0n_dat), 2, 2 );
# 		setblock!(Hc0,diagonaloperator(bx, c0n_dat), 3, 3 );

# 			# c1 term
# 		setblock!(Hc1,diagonaloperator(bx, c1*(np - nm + n0) ), 1, 1 );
# 		setblock!(Hc1,diagonaloperator(bx, c1*(np + nm) ), 2, 2 );
# 		setblock!(Hc1,diagonaloperator(bx, c1*(nm - np + n0) ), 3, 3 );
# 		setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_m).*ψ_0) , 1,2)
# 		setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_0).*ψ_m) , 2,1)
# 		setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_0).*ψ_p) , 2,3)
# 		setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_p).*ψ_0) , 3,2)

# 		return H_im
# 	end
	
# 	# renormalization callback
# 	norm_func(u, t, integrator) = normalize_array!(u)
# 	ncb = FunctionCallingCallback(norm_func; func_everystep = true)
	
# 	# Solve
# 	T = range(0., imTmax; length = 9)
# 	@time tout, ψt = timeevolution.schroedinger_dynamic(T, ψ0, Hgp_im,  callback=ncb, alg = DP5(), abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int,)
# 	# Check the result	
# 	function check_norm(ψt)
# 		zz = zeros(length(ψt));
# 		for ii = 1:length(ψt)
# 			zz[ii] = sum(abs2.(getblock(ψt[ii], 1).data)) + sum(abs2.(getblock(ψt[ii], 2).data)) + sum(abs2.(getblock(ψt[ii], 3).data));

# 		end
# 		return zz
# 	end
# 	"Ground state calculated"
# end

# ╔═╡ deb42060-70cf-11eb-3a4b-b5c17fe3e479
md"#### Snapshots of the imaginary time evolution"

# ╔═╡ d46b1f80-63b6-11eb-0837-9d7f5f8a8127
# begin
# 	xx_real = a⊥*xx *1e6
# 	plt1 = plotWfn(xx_real, ψt[1]); 
# 	plt2 = plotWfn(xx_real, ψt[2]); 
# 	plt3 = plotWfn(xx_real, ψt[3]);
# 	plt4 = plotWfn(xx_real, ψt[4]); 
# 	plt5 = plotWfn(xx_real, ψt[5]);
# 	plt6 = plotWfn(xx_real, ψt[6]);
# 	plt7 = plotWfn(xx_real, ψt[7]); 
# 	plt8 = plotWfn(xx_real, ψt[8]); 
# 	plt9 = plotWfn(xx_real, ψt[9]); 
# 	plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, layout = (3,3), legend = false)
# end

# ╔═╡ 4b853c00-6456-11eb-3e7b-dfa24174ec08
# begin
# 	ψg = ψt[end]
# 	plotWfn(xx_real, ψg)
# 	nTF = max.(npeakSI*(1 .- (xx_real*1e-6/RTF).^2), 0)
# 	plot!(xx_real, nTF*1e-6, lw = 3, ls=:dot, label = "m0 Thoams-Fermi")
# 	title!("End of the imaginary time evolution")
# end

# ╔═╡ 870e29f0-7273-11eb-2256-fba26a2b6340
# sum(abs2.(ψg.data))*dx

# ╔═╡ 8e9691d0-7273-11eb-1ba8-7d1a8c31790c
# sum(nTF)*dxSI

# ╔═╡ 4555d330-6500-11eb-3fc9-3f06ec96b349
md"
## Dynamics with the ground state"

# ╔═╡ d4306980-6500-11eb-2043-413c6847db82
# begin
# 	# Constructing interaction Hamiltonian
# 	Hc0_dy = one(bx) ⊕ one(bx) ⊕ one(bx);
# 	Hc1_dy = one(bx) ⊕ one(bx) ⊕ one(bx);
# 	H_dy = LazySum(Hkin_FFT, Utrap, UZeeman_quench, Hc0_dy, Hc1_dy);
	
# 	function Hgp_dy(t, ψ) 
# 		# ψ = normalize_wfn(ψ) --> no more normalization needed.
# 		ψ_p = getblock(ψ,1);
# 		ψ_0 = getblock(ψ,2);
# 		ψ_m = getblock(ψ,3);
# 		n_dat = abs2.(ψ_p.data) + abs2.(ψ_0.data) + abs2.(ψ_m.data);
# 			# c0 term
# 		setblock!(Hc0_dy,diagonaloperator(bx, c0*n_dat), 1, 1 );
# 		setblock!(Hc0_dy,diagonaloperator(bx, c0*n_dat), 2, 2 );
# 		setblock!(Hc0_dy,diagonaloperator(bx, c0*n_dat), 3, 3 );
# 			# c1 term 
# 		setblock!(Hc1_dy,diagonaloperator(bx, c1*(abs2.(ψ_p.data) - abs2.(ψ_m.data) + abs2.(ψ_0.data)) ), 1, 1 );
# 		setblock!(Hc1_dy,diagonaloperator(bx, c1*(abs2.(ψ_p.data) + abs2.(ψ_m.data)) ), 2, 2 );
# 		setblock!(Hc1_dy,diagonaloperator(bx, c1*(-abs2.(ψ_p.data)+abs2.(ψ_m.data) + abs2.(ψ_0.data)) ), 3, 3 );	
# 		setblock!(Hc1_dy,diagonaloperator(bx, c1*conj(ψ_m.data).*ψ_0.data) , 1,2)
# 		setblock!(Hc1_dy,diagonaloperator(bx, c1*conj(ψ_0.data).*ψ_m.data) , 2,1)
# 		setblock!(Hc1_dy,diagonaloperator(bx, c1*conj(ψ_0.data).*ψ_p.data) , 2,3)
# 		setblock!(Hc1_dy,diagonaloperator(bx, c1*conj(ψ_p.data).*ψ_0.data) , 3,2)

# 		return H_dy
# 	end

# 	# intial condition
# 	ψt_dy0 = copy(ψt[end]);
# 	# ψt_dy0 = gaussianstate(bx, 0, 3, 2) ⊕ 0*gaussianstate(bx, 0, 1, σ0) ⊕ 0*gaussianstate(bx, 0, 1, σ0) 
# 	ψt_dy0.data =  ψt_dy0.data + sqrt(Natom)*1e-4*(exp.(-(rand(3*nx)).^2) .+1*im*exp.(-(rand(3*nx)).^2))
	
	
# 	# Solve
# 	T_dy = range(0, dyTmax; length = 100)
# 	tout_dy, ψt_dy = timeevolution.schroedinger_dynamic(T_dy, ψt_dy0, Hgp_dy, abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int);
# 	"Dynamics calculated"
# end

# ╔═╡ d9bd26e0-70d1-11eb-1af8-851f530b4354
md"Snapshots."

# ╔═╡ aff2b170-6502-11eb-39bb-030e58766b55
# begin
# 	plt1_dy = plotWfn(xx_real, ψt_dy[1]); 
# 	plt2_dy = plotWfn(xx_real, ψt_dy[Int(round(length(T_dy)*2/9))]); 
# 	plt3_dy = plotWfn(xx_real, ψt_dy[Int(round(length(T_dy)*3/9))]); 
# 	plt4_dy = plotWfn(xx_real, ψt_dy[Int(round(length(T_dy)*4/9))]);  
# 	plt5_dy = plotWfn(xx_real, ψt_dy[Int(round(length(T_dy)*5/9))]); 
# 	plt6_dy = plotWfn(xx_real, ψt_dy[Int(round(length(T_dy)*6/9))]); 
# 	plt7_dy = plotWfn(xx_real, ψt_dy[Int(round(length(T_dy)*7/9))]);  
# 	plt8_dy = plotWfn(xx_real, ψt_dy[Int(round(length(T_dy)*8/9))]);  
# 	plt9_dy = plotWfn(xx_real, ψt_dy[Int(round(length(T_dy)*9/9))]);  
# 	plot(plt1_dy, plt2_dy, plt3_dy, plt4_dy, plt5_dy, plt6_dy, plt7_dy, plt8_dy, plt9_dy, layout = (3,3), legend = false)
# end

# ╔═╡ 88ec99a2-660c-11eb-2a61-0929a6150d8f
# plotWfn(xx, ψt_dy[65])

# ╔═╡ 4848b660-70d2-11eb-383f-d9638a02cbf9
md"Check the total atom number evolution.
"

# ╔═╡ 012d3910-651d-11eb-20e0-692c67af0d47
# begin
# 	function plot_pop_dynamics(T_dy, ψ)
# 		n1 = zeros(length(T_dy));
# 		n2 = zeros(length(T_dy));
# 		n3 = zeros(length(T_dy));
# 		for ii = 1:length(T_dy)
# 			n1[ii] = sum(abs2.(getblock(ψ[ii],1).data))*dx
# 			n2[ii] = sum(abs2.(getblock(ψ[ii],2).data))*dx
# 			n3[ii] = sum(abs2.(getblock(ψ[ii],3).data))*dx
# 		end
		

# 		plt = plot(T_dy, n1,  label = "+1", linewidth = 2, linestyle = :solid, framestyle = :box)
# 		plot!(T_dy, n2, label = "0", linewidth = 2, linestyle = :solid)
# 		plot!(T_dy, n3, label = "-1", linewidth = 2, linestyle = :dash)
# 		plot!(T_dy, n1+n2+n3, label = "total", linewidth = 2, linestyle = :dot)

# 		return plt
# 	end
# 	plot_pop_dynamics(T_dy/ω⊥ *1e3, ψt_dy)
# 	ylabel!("Total atom number")
# 	xlabel!("Time [ms]")

# end

# ╔═╡ Cell order:
# ╠═87bdeb30-637b-11eb-2b40-4b885e156281
# ╠═6f01e4a0-646d-11eb-372a-81a875a291ed
# ╠═38f91830-6777-11eb-0ed1-2fb2ca451162
# ╠═cd6af090-677e-11eb-023c-8d0d2ba5df32
# ╠═fe7a6a80-636e-11eb-0e80-8137669fdd2c
# ╠═1e07c6c0-6466-11eb-0b4f-4f433f18e09e
# ╠═ec64e12e-6469-11eb-0880-33ea64337e40
# ╠═41ff0070-6a79-11eb-1166-bf38b9b0dbe5
# ╠═3d1db920-701e-11eb-3a4d-7da891d70116
# ╠═3a9c8750-7030-11eb-34ac-5f1a11849622
# ╠═ecdfa9d0-6395-11eb-159e-9168e235b775
# ╠═4d5420ee-6398-11eb-036e-6953db9fb18a
# ╠═09a5aae0-70d1-11eb-2e28-31e5d0d7daed
# ╠═0dca7560-70d1-11eb-2db2-cd726c7b4379
# ╠═00e0f082-7023-11eb-3990-3fc3b0282bca
# ╠═a817d7f0-7033-11eb-1f2a-ddb317fd14ae
# ╠═9cf22510-7033-11eb-2cb0-036b14e102e6
# ╠═fe93c5c0-70de-11eb-03f7-632727c5ae36
# ╠═11db1340-70df-11eb-26fe-7f7a79370a67
# ╠═5117dbd0-70e2-11eb-2f61-45c584857e7a
# ╠═5f1ce0e0-70e2-11eb-188e-c34aa6b7f0ad
# ╠═b00d0b60-7033-11eb-3beb-6b2666557f93
# ╠═efe3fae0-636e-11eb-2f69-6d9cca6ceecf
# ╠═3c25cb70-6524-11eb-21ac-2d2b83252360
# ╠═bf366a80-6522-11eb-3aca-57bbe8979881
# ╠═a61eb7be-6395-11eb-0ee2-759b617deae2
# ╠═b5054330-6395-11eb-0858-6b8cd3099f4a
# ╠═83040b10-769f-11eb-344a-01d1e7203e7f
# ╠═079bbba0-6450-11eb-1aa7-3f1fbd08ae6f
# ╠═95f9b180-6398-11eb-0b53-59102afb3ece
# ╠═deb42060-70cf-11eb-3a4b-b5c17fe3e479
# ╠═d46b1f80-63b6-11eb-0837-9d7f5f8a8127
# ╠═4b853c00-6456-11eb-3e7b-dfa24174ec08
# ╠═870e29f0-7273-11eb-2256-fba26a2b6340
# ╠═8e9691d0-7273-11eb-1ba8-7d1a8c31790c
# ╠═4555d330-6500-11eb-3fc9-3f06ec96b349
# ╠═d4306980-6500-11eb-2043-413c6847db82
# ╠═d9bd26e0-70d1-11eb-1af8-851f530b4354
# ╠═aff2b170-6502-11eb-39bb-030e58766b55
# ╠═88ec99a2-660c-11eb-2a61-0929a6150d8f
# ╠═4848b660-70d2-11eb-383f-d9638a02cbf9
# ╠═012d3910-651d-11eb-20e0-692c67af0d47
