using Plots
using QuantumOptics, LaTeXStrings
using DrWatson
using JLD2
@quickactivate "Spinor1D"
gr()

function calc_density(ψKet)
    nx = ψKet[1].basis.bases[1].N
    dx = (ψKet[1].basis.bases[1].xmax - ψKet[1].basis.bases[1].xmin)/nx

    np = [sum(abs2.(ψKet[ii].data[1:nx]))*dx for ii = 1:length(ψKet)]
    n0 = [sum(abs2.(ψKet[ii].data[nx+1:2*nx]))*dx for ii = 1:length(ψKet)]
    nm = [sum(abs2.(ψKet[ii].data[2*nx+1:3*nx]))*dx for ii = 1:length(ψKet)]

    return np, n0, nm
end

# load data

function saveTotPop(file_name)
    for ii = 1:length(file_name)
        datadict = wload(datadir("sims", "Li7", file_name[ii]))

        psit      = datadict[:psit]
        Tdomain   = range(0, datadict[:dyTmax], length = datadict[:TsampleN])/(2*π*datadict[:f⊥])

        np, n0, nm = calc_density(psit)
        # plot
        plt = plot(Tdomain *1e3, np, frame= :box, legend = :true, label = "plus", lw = 3, dpi = 300)
        plot!(Tdomain *1e3, n0, frame= :box, label = "zero", legend = false, lw = 3, dpi = 300)
        plot!(Tdomain*1e3, nm, frame= :box,ls = :dash, label = "minus", legend = false, lw = 3, dpi = 300)
        plot!(Tdomain*1e3, np+n0+nm, ls = :dash, label = "total", frame= :box, legend = false, lw = 3, dpi = 300)


        xlabel!("Time [ms]")
        ylabel!(L"$\bar{\rho}_{+1} + \bar{\rho}_{-1}$")

        # save 
        savefig(plt,plotsdir(folder_name, "PopDynamics_"*file_name[ii]*".png"))
    end
    return
end

folder_name = "Li7"
file_name = readdir(datadir("sims", folder_name))
saveTotPop(file_name)