### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ dc0bdf10-636e-11eb-0a3b-e1996ef3583a
begin
	using QuantumOptics
	using OrdinaryDiffEq, DiffEqCallbacks
	using Plots
	plotly()
end

# ╔═╡ 87bdeb30-637b-11eb-2b40-4b885e156281
md"# Spinor BEC in 1D"

# ╔═╡ 6f01e4a0-646d-11eb-372a-81a875a291ed
md"[Reference for Julia implementation](https://docs.qojulia.org/examples/spin-orbit-coupled-BEC1D/#Including-two-body-interactions-(stripe-phase))"

# ╔═╡ fe7a6a80-636e-11eb-0e80-8137669fdd2c
md"
## Spin-1 BEC GPE
ref: Kawaguchi, Ueda (2012)

```math 
E[\psi] = \langle \hat{H} \rangle_0 = \int d\mathbf{r}\left\{ \sum_{m=-1}^{1} \psi^*_m \left[ -\frac{\hbar^2 \nabla^2}{2M} + U_{\mathrm{trap}}(\mathbf{r}) - p(x)m + qm^2 \right] \psi_m + \frac{c_0}{2} n^2 + \frac{c_1}{2} |\mathbf{F}|^2 \right\}
```

For 1D,

```math 
E[\psi] = \int d\mathbf{r}\left\{ \sum_{m=-1}^{1} \psi^*_m(x) \left[ -\frac{\hbar^2}{2M}\frac{d^2}{dx^2} + U_{\text{trap}}(x) - p(x)m + qm^2 \right] \psi_m(x) + \frac{c_0}{2} n^2 + \frac{c_1}{2} |\mathbf{F}|^2 \right\}
```

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

# ╔═╡ ecdfa9d0-6395-11eb-159e-9168e235b775
md"
Notes
- Order of the basis: (1's position) $\oplus$ (0's position) $\oplus$ (-1's position)
- Position basis has periodic boundary condition.
- See 'timeevolution_base.jl' for the information about the solver. It uses DP5(ode45 in MATLAB) method in OrdinaryDiffEq.jl
"

# ╔═╡ 4d5420ee-6398-11eb-036e-6953db9fb18a
md"## Basic construction"

# ╔═╡ efe3fae0-636e-11eb-2f69-6d9cca6ceecf
begin
	# Parameters
	ω  = 1.;
	m  = 1.;
	p  = 1e-3;
	p_quench  = 0;
	q  = 50;
	q_quench = -20;
	c0 = 50;
	c1 = -c0*0.4;
	
	# Domain
	xmax = 13. ;
	nx = 2^8; 
	
	# integrator tolerances
	abstol_int = 1e-5
	reltol_int = 1e-5
	maxiters_int = 1e8
	
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
	Hkin = Px^2/2m  ;
	Hkin_FFT = LazyProduct(Txp, Hkin, Tpx) ;
		# harmonic potential operator
	Utrap = 0.5 * (ω^2 * x^2) ;
	Ubarrier = zeros(nx)
	Ubarrier[findall(x->abs.(x) > 0.95*xmax, xx)] .= 0;
	for ii = 1:nx 
		Utrap.data[ii, ii]  = Utrap.data[ii, ii] + Ubarrier[ii]
		Utrap.data[nx + ii,nx + ii]  = Utrap.data[nx + ii,nx + ii] + Ubarrier[ii]
		Utrap.data[2*nx+ii, 2*nx+ii]  = Utrap.data[2*nx+ii, 2*nx+ii] + Ubarrier[ii]
	end
				
		# zeeman shifts
	UZeeman = - p*( position(bx) ⊕ 0*position(bx) ⊕ -1*position(bx) ) + q*(one(bx) ⊕ 0*one(bx) ⊕ one(bx)); 
	
	UZeeman_quench = - p_quench*( position(bx) ⊕ 0*position(bx) ⊕ -1*position(bx) ) + q_quench*(one(bx) ⊕ 0*one(bx) ⊕ one(bx)); # check: heatmap(Array(real(UZeeman.data)), box = "off")
	
end;


# ╔═╡ 3c25cb70-6524-11eb-21ac-2d2b83252360
md"
Check the shape of the potential.
"

# ╔═╡ bf366a80-6522-11eb-3aca-57bbe8979881
begin
	aa = real([Utrap.data[ii, ii] for ii = 1:nx] )
	plot(xx, aa)
end

# ╔═╡ a61eb7be-6395-11eb-0ee2-759b617deae2
md"## Initial state preparation"

# ╔═╡ b5054330-6395-11eb-0858-6b8cd3099f4a
begin
	p0 = 0; 
	σ0 = 2;
	spinratio = [1, 1, 1];
	ϕin_p = spinratio[1]*gaussianstate(bx, 0, p0, σ0);
	ϕin_0 = spinratio[2]*gaussianstate(bx, 0, p0, σ0);
	ϕin_m = spinratio[3]*gaussianstate(bx, 0, p0, σ0);
# 	ϕin_p = 1.0*Ket(bx, ones(nx).+0.01im);
# 	ϕin_0 = 1.0*Ket(bx, ones(nx).+0.01im);
# 	ϕin_m = 1.0*Ket(bx, ones(nx).+0.01im);
	
	function normalize_wfn(ψ)
		ψn = ψ/sqrt(sum(abs2.(ψ.data)))
		return ψn
	end
	
	function normalize_array!(ψ)
		ψ[:] = ψ/norm(ψ)
		return ψ
	end
	
	function plotWfn(xx, ψ)
		n1 = abs2.(getblock(ψ,1).data)
		n2 = abs2.(getblock(ψ,2).data)
		n3 = abs2.(getblock(ψ,3).data)
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

# ╔═╡ 079bbba0-6450-11eb-1aa7-3f1fbd08ae6f
md"## Find ground state"

# ╔═╡ 95f9b180-6398-11eb-0b53-59102afb3ece
begin
	# Constructing interaction Hamiltonian
	dx = 2*xmax/nx;
	H0 = diagonaloperator(bx, Ket(bx).data);
	Hc0 = H0 ⊕ H0 ⊕ H0;
	Hc1 = H0 ⊕ H0 ⊕ H0;
	H_im = -1*im*LazySum(Hkin_FFT, Utrap, UZeeman, Hc0, Hc1);
	
	function Hgp_im(t, ψ) 
		ψ = normalize_wfn(ψ)
		ψ_p = getblock(ψ,1);
		ψ_0 = getblock(ψ,2);
		ψ_m = getblock(ψ,3);
		n_dat = abs2.(ψ_p.data) + abs2.(ψ_0.data) + abs2.(ψ_m.data);

			# c0 term
		setblock!(Hc0,diagonaloperator(bx, c0/dx*n_dat), 1, 1 );
		setblock!(Hc0,diagonaloperator(bx, c0/dx*n_dat), 2, 2 );
		setblock!(Hc0,diagonaloperator(bx, c0/dx*n_dat), 3, 3 );

			# c1 term
		setblock!(Hc1,diagonaloperator(bx, c1/dx*(abs2.(ψ_p.data) - abs2.(ψ_m.data) + abs2.(ψ_0.data)) ), 1, 1 );
		setblock!(Hc1,diagonaloperator(bx, c1/dx*(abs2.(ψ_p.data) + abs2.(ψ_m.data)) ), 2, 2 );
		setblock!(Hc1,diagonaloperator(bx, c1/dx*(-abs2.(ψ_p.data)+abs2.(ψ_m.data) + abs2.(ψ_0.data)) ), 3, 3 );
		setblock!(Hc1,diagonaloperator(bx, c1/dx.*conj(ψ_m.data).*ψ_0.data) , 1,2)
		setblock!(Hc1,diagonaloperator(bx, c1/dx.*conj(ψ_0.data).*ψ_m.data) , 2,1)
		setblock!(Hc1,diagonaloperator(bx, c1/dx.*conj(ψ_0.data).*ψ_p.data) , 2,3)
		setblock!(Hc1,diagonaloperator(bx, c1/dx.*conj(ψ_p.data).*ψ_0.data) , 3,2)

		return H_im
	end
	
	# renormalization callback
	norm_func(u, t, integrator) = normalize_array!(u)
	ncb = FunctionCallingCallback(norm_func; func_everystep = true)
	

	# Solve
	T = range(0., 1.; length = 9)
	tout, ψt = timeevolution.schroedinger_dynamic(T, ψ0, Hgp_im, callback=ncb, abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int);
	
	# Check the result	
	function check_norm(ψt)
		zz = zeros(length(ψt));
		for ii = 1:length(ψt)
			zz[ii] = sum(abs2.(getblock(ψt[ii], 1).data)) + sum(abs2.(getblock(ψt[ii], 2).data)) + sum(abs2.(getblock(ψt[ii], 3).data));

		end
		return zz
	end
	
	plot(T, check_norm(ψt))
	xlabel!("T")
	ylabel!("Norm")
	title!("Normalization check")
end ;


# ╔═╡ 39d24650-64a7-11eb-2744-db38bcfec8db
begin
	# heatmap(real(Array(H_im.operators[5].data)), aspect_ratio = 1)
end

# ╔═╡ d46b1f80-63b6-11eb-0837-9d7f5f8a8127
begin
	plt1 = plotWfn(xx, ψt[1]); 
	plt2 = plotWfn(xx, ψt[2]); 
	plt3 = plotWfn(xx, ψt[3]);
	plt4 = plotWfn(xx, ψt[4]); 
	plt5 = plotWfn(xx, ψt[5]);
	plt6 = plotWfn(xx, ψt[6]);
	plt7 = plotWfn(xx, ψt[7]); 
	plt8 = plotWfn(xx, ψt[8]); 
	plt9 = plotWfn(xx, ψt[9]); 
	plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, layout = (3,3), legend = false)
end

# ╔═╡ 4b853c00-6456-11eb-3e7b-dfa24174ec08
plotWfn(xx, ψt[end]); title!("End of the evolution")

# ╔═╡ 4555d330-6500-11eb-3fc9-3f06ec96b349
md"
## Dynamics with the ground state"

# ╔═╡ d4306980-6500-11eb-2043-413c6847db82
begin
	# Constructing interaction Hamiltonian
	Hc0_dy = H0 ⊕ H0 ⊕ H0;
	Hc1_dy = H0 ⊕ H0 ⊕ H0;
	H_dy = LazySum(Hkin_FFT, Utrap, UZeeman_quench, Hc0_dy, Hc1_dy);
	
	function Hgp_dy(t, ψ) 
		# ψ = normalize_wfn(ψ) --> no more normalization needed.
		ψ_p = getblock(ψ,1);
		ψ_0 = getblock(ψ,2);
		ψ_m = getblock(ψ,3);
		n_dat = abs2.(ψ_p.data) + abs2.(ψ_0.data) + abs2.(ψ_m.data);
			# c0 term
		setblock!(Hc0_dy,diagonaloperator(bx, c0/dx*n_dat), 1, 1 );
		setblock!(Hc0_dy,diagonaloperator(bx, c0/dx*n_dat), 2, 2 );
		setblock!(Hc0_dy,diagonaloperator(bx, c0/dx*n_dat), 3, 3 );
			# c1 term 
		setblock!(Hc1_dy,diagonaloperator(bx, c1/dx*(abs2.(ψ_p.data) - abs2.(ψ_m.data) + abs2.(ψ_0.data)) ), 1, 1 );
		setblock!(Hc1_dy,diagonaloperator(bx, c1/dx*(abs2.(ψ_p.data) + abs2.(ψ_m.data)) ), 2, 2 );
		setblock!(Hc1_dy,diagonaloperator(bx, c1/dx*(-abs2.(ψ_p.data)+abs2.(ψ_m.data) + abs2.(ψ_0.data)) ), 3, 3 );	
		setblock!(Hc1_dy,diagonaloperator(bx, c1/dx.*conj(ψ_m.data).*ψ_0.data) , 1,2)
		setblock!(Hc1_dy,diagonaloperator(bx, c1/dx.*conj(ψ_0.data).*ψ_m.data) , 2,1)
		setblock!(Hc1_dy,diagonaloperator(bx, c1/dx.*conj(ψ_0.data).*ψ_p.data) , 2,3)
		setblock!(Hc1_dy,diagonaloperator(bx, c1/dx.*conj(ψ_p.data).*ψ_0.data) , 3,2)

		return H_dy
	end

	
	
	# intial condition
	ψt_dy0 = copy(ψt[end]);
	# ψt_dy0 = gaussianstate(bx, 0, 3, 2) ⊕ 0*gaussianstate(bx, 0, 1, σ0) ⊕ 0*gaussianstate(bx, 0, 1, σ0) 
	ψt_dy0.data =  ψt_dy0.data + 1e-4*(exp.(-(rand(3*nx)).^2) .+1*im*exp.(-(rand(3*nx)).^2))
	
	
	# Solve
	T_dy = range(0, 3*pi; length = 9)
	tout_dy, ψt_dy = timeevolution.schroedinger_dynamic(T_dy, ψt_dy0, Hgp_dy, abstol=abstol_int, reltol=reltol_int, maxiters=maxiters_int);
end ;

# ╔═╡ aff2b170-6502-11eb-39bb-030e58766b55
begin
	plt1_dy = plotWfn(xx, ψt_dy[1]); 
	plt2_dy = plotWfn(xx, ψt_dy[2]); 
	plt3_dy = plotWfn(xx, ψt_dy[3]);
	plt4_dy = plotWfn(xx, ψt_dy[4]); 
	plt5_dy = plotWfn(xx, ψt_dy[5]);
	plt6_dy = plotWfn(xx, ψt_dy[6]);
	plt7_dy = plotWfn(xx, ψt_dy[7]); 
	plt8_dy = plotWfn(xx, ψt_dy[8]); 
	plt9_dy = plotWfn(xx, ψt_dy[9]); 
	plot(plt1_dy, plt2_dy, plt3_dy, plt4_dy, plt5_dy, plt6_dy, plt7_dy, plt8_dy, plt9_dy, layout = (3,3), legend = false)
end

# ╔═╡ 88ec99a2-660c-11eb-2a61-0929a6150d8f
plotWfn(xx, ψt_dy[6])

# ╔═╡ fed42fe0-6506-11eb-38c1-eb3034b73047
# begin	
		
# 	function plotWfnpm(xx, ψ)
# 		n1 = abs2.(getblock(ψ,1).data)
# 		n2 = abs2.(getblock(ψ,2).data)
# 		n3 = abs2.(getblock(ψ,3).data)
# 		plt = plot(xx, n1,  label = "+1", linewidth = 2, linestyle = :solid, framestyle = :box)
# 		plot!(xx, n3, label = "-1", linewidth = 2, linestyle = :dash)
# 		# xlabel!("x")
# 		ylims!((0,Inf))
# 		return plt
# 	end
# 	plt1_dypm = plotWfnpm(xx, ψt_dy[1]); 
# 	plt2_dypm = plotWfnpm(xx, ψt_dy[2]); 
# 	plt3_dypm = plotWfnpm(xx, ψt_dy[3]);
# 	plt4_dypm = plotWfnpm(xx, ψt_dy[4]); 
# 	plt5_dypm = plotWfnpm(xx, ψt_dy[5]);
# 	plt6_dypm = plotWfnpm(xx, ψt_dy[6]);
# 	plt7_dypm = plotWfnpm(xx, ψt_dy[7]); 
# 	plt8_dypm = plotWfnpm(xx, ψt_dy[8]); 
# 	plt9_dypm = plotWfnpm(xx, ψt_dy[9]); 
# 	plot(plt1_dypm, plt2_dypm, plt3_dypm, plt4_dypm, plt5_dypm, plt6_dypm, plt7_dypm, plt8_dypm, plt9_dypm, layout = (3,3), legend = false, ylims = (0, Inf))
	
# end

# ╔═╡ 012d3910-651d-11eb-20e0-692c67af0d47
begin
	function plot_pop_dynamics(T_dy, ψ)
		n1 = zeros(length(T_dy));
		n2 = zeros(length(T_dy));
		n3 = zeros(length(T_dy));
		for ii = 1:length(T_dy)
			n1[ii] = sum(abs2.(getblock(ψ[ii],1).data))
			n2[ii] = sum(abs2.(getblock(ψ[ii],2).data))
			n3[ii] = sum(abs2.(getblock(ψ[ii],3).data))
		end
		

		plt = plot(T_dy, n1,  label = "+1", linewidth = 2, linestyle = :solid, framestyle = :box)
		plot!(T_dy, n2, label = "0", linewidth = 2, linestyle = :dash)
		plot!(T_dy, n3, label = "-1", linewidth = 2, linestyle = :dash)
		return plt
	end
	plot_pop_dynamics(T_dy, ψt_dy)

end

# ╔═╡ Cell order:
# ╟─87bdeb30-637b-11eb-2b40-4b885e156281
# ╟─6f01e4a0-646d-11eb-372a-81a875a291ed
# ╠═dc0bdf10-636e-11eb-0a3b-e1996ef3583a
# ╟─fe7a6a80-636e-11eb-0e80-8137669fdd2c
# ╟─1e07c6c0-6466-11eb-0b4f-4f433f18e09e
# ╟─ec64e12e-6469-11eb-0880-33ea64337e40
# ╟─ecdfa9d0-6395-11eb-159e-9168e235b775
# ╠═4d5420ee-6398-11eb-036e-6953db9fb18a
# ╠═efe3fae0-636e-11eb-2f69-6d9cca6ceecf
# ╟─3c25cb70-6524-11eb-21ac-2d2b83252360
# ╠═bf366a80-6522-11eb-3aca-57bbe8979881
# ╠═a61eb7be-6395-11eb-0ee2-759b617deae2
# ╠═b5054330-6395-11eb-0858-6b8cd3099f4a
# ╟─079bbba0-6450-11eb-1aa7-3f1fbd08ae6f
# ╠═95f9b180-6398-11eb-0b53-59102afb3ece
# ╠═39d24650-64a7-11eb-2744-db38bcfec8db
# ╠═d46b1f80-63b6-11eb-0837-9d7f5f8a8127
# ╠═4b853c00-6456-11eb-3e7b-dfa24174ec08
# ╟─4555d330-6500-11eb-3fc9-3f06ec96b349
# ╠═d4306980-6500-11eb-2043-413c6847db82
# ╠═aff2b170-6502-11eb-39bb-030e58766b55
# ╠═88ec99a2-660c-11eb-2a61-0929a6150d8f
# ╟─fed42fe0-6506-11eb-38c1-eb3034b73047
# ╠═012d3910-651d-11eb-20e0-692c67af0d47
