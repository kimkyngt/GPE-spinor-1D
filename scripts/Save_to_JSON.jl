using QuantumOptics, LaTeXStrings
import JSON
using DrWatson
@quickactivate "Spinor1D"


# load data
fname = readdir(datadir("sims", "Saito"))
datadict = wload(datadir("sims", "Saito", fname[2]))

a = JSON.json(datadict);

length(a)
##
open("myfile.txt", "w") do io
    write(io, a)
end;