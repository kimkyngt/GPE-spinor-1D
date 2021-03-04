function normalize_wfn(ψ, N)
    Δx = spacing(ψ.basis.bases[1])
    ψ = ψ/(norm(ψ) / sqrt(N/Δx))
    return ψ
end

function normalize_array!(ψ)
    ψ[:] = ψ/(norm(ψ) / sqrt(Natom/dx))
    return ψ
end

function convert_wfn_norm2SI(ψ, a⊥)
    ψSI = copy(ψ)
    ψSI.data  = ψSI.data/sqrt(a⊥)
    return ψSI
end

function plot_wfn(x, ψ; kwargs...)
    # Plot ψ (x)
    n1 = abs2.(getblock(ψ,1).data) 
    n2 = abs2.(getblock(ψ,2).data) 
    n3 = abs2.(getblock(ψ,3).data) 
    plt = plot(x, n1,  label = "+1", lw = 2, ls = :solid, framestyle = :box)
    plot!(x, n2,  label = "0", lw = 2, ls = :dash; kwargs...)
    plot!(x, n3, label = "-1", lw = 2, ls = :dot; kwargs...)
    plot!(x, (n1+n2+n3), label = "sum"; kwargs...)
    return plt
end

function plot_spinor_n(xx, ψ; kwargs...)
    # Plot wavefunction in SI units
    n1 = abs2.(getblock(ψ,1).data) /(a⊥/1e-6)
    n2 = abs2.(getblock(ψ,2).data) /(a⊥/1e-6)
    n3 = abs2.(getblock(ψ,3).data) /(a⊥/1e-6)
    plt = plot(xx, n1,  label = "+1", linewidth = 2, ls = :solid, framestyle = :box)
    plot!(xx, n2,  label = "0", linewidth = 2, ls = :dash; kwargs...)
    plot!(xx, n3, label = "-1", linewidth = 2, ls = :dot; kwargs...)
    plot!(xx, (n1+n2+n3), label = "sum"; kwargs...)
    xlabel!("Position [μm]")
    ylabel!("Density [1/μm]")
    return plt
end



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


function initialstate_gaussian(bx, N; σ::Float64 = 1.0, spinratio = [1, 1, 1])
    ϕin_p = spinratio[1]*gaussianstate(bx, 0, 0, σ);
    ϕin_0 = spinratio[2]*gaussianstate(bx, 0, 0, σ);
    ϕin_m = spinratio[3]*gaussianstate(bx, 0, 0, σ);
    return normalize_wfn(ϕin_p ⊕ ϕin_0 ⊕ ϕin_m, N)
end


function initialstate_uniform(N; spinratio = [1.0, 1.0, 1.0])
	ϕin_p = spinratio[1]*Ket(bx, ones(nx).+0.01im);
	ϕin_0 = spinratio[2]*Ket(bx, ones(nx).+0.01im);
	ϕin_m = spinratio[3]*Ket(bx, ones(nx).+0.01im);
   
    # initial state
    ψ0 = normalize_wfn(ϕin_p ⊕ ϕin_0 ⊕ ϕin_m, N)
    return ψ0
end

function Hgp_im(t, ψ) 
    # Functions to update the Hamiltonian during the imaginary time evolution.
    # display(sum(abs2.(ψ.data))*dx)
    ψ = normalize_wfn(ψ, Natom)
    ψ_p = ψ.data[1:nx];
    ψ_0 = ψ.data[nx+1:2*nx];
    ψ_m = ψ.data[2*nx+1:3*nx];
    
    np = abs2.(ψ_p)
    n0 = abs2.(ψ_0)
    nm = abs2.(ψ_m)
    
    c0n_dat = c0*(np+n0+nm);

        # c0 term
    setblock!(Hc0,diagonaloperator(bx, c0n_dat), 1, 1 );
    setblock!(Hc0,diagonaloperator(bx, c0n_dat), 2, 2 );
    setblock!(Hc0,diagonaloperator(bx, c0n_dat), 3, 3 );

        # c1 term
    setblock!(Hc1,diagonaloperator(bx, c1*(np - nm + n0) ), 1, 1 );
    setblock!(Hc1,diagonaloperator(bx, c1*(np + nm) ), 2, 2 );
    setblock!(Hc1,diagonaloperator(bx, c1*(nm - np + n0) ), 3, 3 );
    setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_m).*ψ_0) , 1,2)
    setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_0).*ψ_m) , 2,1)
    setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_0).*ψ_p) , 2,3)
    setblock!(Hc1,diagonaloperator(bx, c1*conj(ψ_p).*ψ_0) , 3,2)

    return H_im
end


function check_norm(ψ)
    zz = zeros(length(ψ));
    for ii = 1:length(ψ)
        zz[ii] = sum(abs2.(ψ[ii].data))*dx
    end
    return zz
end


function Hgp_dy(t, ψ) 
    # ψ = normalize_wfn(ψ) --> no more normalization needed.

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

function plot_pop_dynamics(T_dy, ψ)
    n1 = zeros(length(T_dy));
    n2 = zeros(length(T_dy));
    n3 = zeros(length(T_dy));
    for ii = 1:length(T_dy)
        n1[ii] = sum(abs2.(getblock(ψ[ii],1).data))*dx
        n2[ii] = sum(abs2.(getblock(ψ[ii],2).data))*dx
        n3[ii] = sum(abs2.(getblock(ψ[ii],3).data))*dx
    end
    

    plt = plot(T_dy, n1,  label = "+1", linewidth = 2, ls = :solid, framestyle = :box)
    plot!(T_dy, n2, label = "0", linewidth = 2, ls = :solid)
    plot!(T_dy, n3, label = "-1", linewidth = 2, ls = :dash)
    plot!(T_dy, n1+n2+n3, label = "total", linewidth = 2, ls = :dot)

    return plt
end

function generate_vacuum_array(sd::Int ;uniform::Int = 0)
    rng1r = MersenneTwister(sd)
    rng1i = MersenneTwister(sd+1)
    a_rand = randn(rng1r, Float64, nx)/2 .+ im*randn(rng1i, Float64, nx)/2

    if uniform == 0
        ψ = zeros(ComplexF64, nx)
        for ii = 1:nx
            for jj = 1:nx
                ψ[ii] = ψ[ii] + exp( im*pp[jj]*xx[ii] ) * a_rand[jj]
            end
        end 
        ψ = 1/sqrt(2*xmax) * ψ
    elseif uniform == 1
        ψ = a_rand
    end

    return ψ
end



function plot_Fplus(ψKet)
    ψp = ψKet.data[1:nx]
    ψ0 = ψKet.data[nx+1:2*nx]
    ψm = ψKet.data[2*nx+1:3*nx]

    xdomain = range(ψKet.basis.bases[1].xmin, ψKet.basis.bases[1].xmax, length = ψKet.basis.bases[1].N)
    FplusAmp = sqrt(2.0) * abs.( conj(ψp).*ψ0 + conj(ψ0).*ψm ) ./(abs2.(ψp) + abs2.(ψ0) + abs2.(ψm))
    FplusAng = angle.(conj(ψp).*ψ0 + conj(ψ0).*ψm)
    plt1 = plot(xdomain, FplusAmp, frame = :box, color = "red", ylabel = L"$|F^+|/\rho $",  legend = false, dpi = 300)
    plt2 = plot(xdomain, unwrap(FplusAng)/π, ymirror = :true, color = "blue", ylabel = "Arg [π]", legend = false, dpi = 300, frame = :box)
    plot(plt1, plt2)
    xlabel!("X [μm]")


end

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

function insert_ψt(dict, ψt)
    output = copy(dict)
    output[:psit] = ψt
    return output
end


function plot_snapshots(xx, ψt, T)
    function plot_one(xx, ψ; timestamp::Float64 = -1.)
        # Plot wavefunction in SI units
        n1 = abs2.(getblock(ψ,1).data) /(a⊥/1e-6)
        n2 = abs2.(getblock(ψ,2).data) /(a⊥/1e-6)
        n3 = abs2.(getblock(ψ,3).data) /(a⊥/1e-6)
        plt = plot(xx, n1,  label = "+1", linewidth = 2, ls = :solid, framestyle = :box)
        plot!(xx, n2,  label = "0", linewidth = 2, ls = :dash)
        plot!(xx, n3, label = "-1", linewidth = 2, ls = :dot)
        plot!(xx, (n1+n2+n3), label = "sum")
        if timestamp >= 0
            annotate!(100, maximum(n1+n2+n3), string(round(timestamp*1e3))*" ms") 
        end
        return plt
    end
    Npsi = length(ψt)
    plt1 = plot_one(xx, ψt[1], timestamp = T[1]/ω⊥);
	plt2 = plot_one(xx, ψt[Int(round(Npsi*2/9))], timestamp = T[Int(round(Npsi*2/9))]/ω⊥); 
	plt3 = plot_one(xx, ψt[Int(round(Npsi*3/9))], timestamp = T[Int(round(Npsi*3/9))]/ω⊥); 
	plt4 = plot_one(xx, ψt[Int(round(Npsi*4/9))], timestamp = T[Int(round(Npsi*4/9))]/ω⊥);  
	plt5 = plot_one(xx, ψt[Int(round(Npsi*5/9))], timestamp = T[Int(round(Npsi*5/9))]/ω⊥); 
	plt6 = plot_one(xx, ψt[Int(round(Npsi*6/9))], timestamp = T[Int(round(Npsi*6/9))]/ω⊥); 
	plt7 = plot_one(xx, ψt[Int(round(Npsi*7/9))], timestamp = T[Int(round(Npsi*7/9))]/ω⊥);  
	plt8 = plot_one(xx, ψt[Int(round(Npsi*8/9))], timestamp = T[Int(round(Npsi*8/9))]/ω⊥);  
	plt9 = plot_one(xx, ψt[Int(round(Npsi*9/9))], timestamp = T[Int(round(Npsi*9/9))]/ω⊥);  
	plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, layout = (3,3), legend = false)


end
