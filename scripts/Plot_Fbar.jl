using Plots
using QuantumOptics, LaTeXStrings
using DrWatson
@quickactivate "Spinor1D"
gr()

function calc_Fbar(ψKet)
    nx = ψKet.basis.bases[1].N
    dx = (ψKet.basis.bases[1].xmax - ψKet.basis.bases[1].xmin)/nx
    xmax = ψKet.basis.bases[1].xmax

    ψp = ψKet.data[1:nx]
    ψ0 = ψKet.data[nx+1:2*nx]
    ψm = ψKet.data[2*nx+1:3*nx]
    return sum(2*abs2.( conj(ψp).*ψ0 + conj(ψ0).*ψm ) ./(abs2.(ψp) + abs2.(ψ0) + abs2.(ψm)).^2  ) *dx/(2*xmax)
end

# load data
fname = readdir(datadir("sims", "Saito"))
function saveFbar(fname)
    for ii = 1:length(fname)
        datadict = wload(datadir("sims", "Saito", fname[ii]))

        psit      = datadict[:psit]
        Tdomain    = range(0, datadict[:dyTmax], length = datadict[:TsampleN]) /(2*π*datadict[:f⊥])

        # plot
        Fbar_t = [calc_Fbar(psit[ii]) for ii = 1:length(Tdomain) ]
        fig = plot(Tdomain*1e3, Fbar_t, frame= :box, legend = false, lw = 3, dpi = 300, ylims = (0, 1))
        xlabel!("Time [ms]")
        ylabel!(L"$\bar{F}$")

        # save 
        savefig(fig,plotsdir("Fbar_"*fname[ii]*".png"))
    end
    return
end

saveFbar(fname)