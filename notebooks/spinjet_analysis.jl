### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ e7ca4ba6-ba04-11eb-18eb-a300177dd1d9
using DrWatson, Plots, LaTeXStrings, JLD2, DataFrames, QuantumOptics, Statistics, PhysicalConstants, PhysicalConstants.CODATA2018, Unitful

# ╔═╡ f1b977f8-8fa4-4093-a50e-7d955464e14e
projectname()

# ╔═╡ 25c2fdff-d9d3-4e12-8681-7381604dc942
begin
	## Default plotting setting
	gr()
	default_font = "Sans"
	default(titlefont = (10, default_font, :black), legendfont = (8, default_font, :black), guidefont = (10, default_font, :black), tickfont = (9, default_font, :black), framestyle = :box, size = (500, 300), dpi = 300, lw = 1, axiscolor = :black)
end

# ╔═╡ cb830da3-0627-4090-b3ca-52bf573d423e
begin
	c = (SpeedOfLightInVacuum)
	h = (PlanckConstant)
	kB = (BoltzmannConstant)
	ħ = (h/(2*π) )
	amu = (AtomicMassConstant)
	a0 = (BohrRadius)
	μB = (BohrMagneton)

	## Properties of Li7
	a2Li = 6.8*a0
	a0Li = 23.9*a0
	
		# dependent in SI
	MLi7 		= 7*amu

end;

# ╔═╡ 99bff367-5634-449f-92aa-57231011055e
function plot_pop_dynamics_DataFrames(df;alp::Float64 = 0.1,  kwargs...)
	ψ  = df[1, :psit][1]
	bx = ψ.basis.bases[1]
	ω⊥ = 2π*df[1, :f⊥]
	dx = (bx.xmax-bx.xmin)/bx.N
	T_dy = range(0, df[1, :dyTmax], length = df[1, :TsampleN])/ω⊥*1e3

	plt = plot()
	navgp = zeros(length(T_dy))
	navg0 = zeros(length(T_dy))
	for kk = 1:size(df)[1]
		n1 = zeros(length(T_dy));
		n2 = zeros(length(T_dy));
		n3 = zeros(length(T_dy));
		for ii = 1:length(T_dy)
			n1[ii] = sum(abs2.(getblock(df[kk, :psit][ii],1).data))*dx
			n2[ii] = sum(abs2.(getblock(df[kk, :psit][ii],2).data))*dx
			n3[ii] = sum(abs2.(getblock(df[kk, :psit][ii],3).data))*dx
		end
		
# 		n1  = n1.- n1[1]
# 		n2  = n2.- n2[1]
# 		n3  = n3.- n3[1]
		
		plot!(T_dy, n1,  label = L"{+1}", c= 1, alpha = alp ,lw = 2, ls = :solid, fs = :box; kwargs...)
		plot!(T_dy, n2, label = L"0", c= 2, alpha = alp , lw = 2, ls = :solid; kwargs...)
		plot!(T_dy, n3, label = L"{-1}", c= 3, alpha =alp ,lw = 2, ls = :dash; kwargs...)
		plot!(T_dy, n1+n2+n3, label = "total", c= 4, alpha = alp ,lw = 2, ls = :dot, legend = false; kwargs...)
		navgp += n1
		navg0 += n2
	end
	navgp = navgp/size(df)[1]
	navg0 = navg0/size(df)[1]

	plot!(T_dy, navgp,  c= :black, lw = 1, legend = false; kwargs...)
	plot!(T_dy, navg0,  c= :black, lw = 1, legend = false; kwargs...)
	xlabel!("Time [ms]")
	ylabel!("Atom number")
	plt
end

# ╔═╡ 3ee96a64-bdcc-4334-aef0-1379b367a133
df = collect_results(datadir("sims", "Li7", "spinjet"));

# ╔═╡ 461fc005-9dd1-44b1-b335-152dfa5ea960
size(df)

# ╔═╡ 113d346e-2ac8-4a81-a2e0-4c54d973b18e
names(df)

# ╔═╡ b9213e12-254c-47ad-a211-3ccf8ee66e8c
plot_pop_dynamics_DataFrames(df, yscale = :log10)

# ╔═╡ 24b6fc3e-8104-4aa6-b736-2579e2eadfa7
function get_halfpolarization_DataFrames(df)
	ψ  = df[1, :psit][1]
	bx = ψ.basis.bases[1]
	ω⊥ = 2π*df[1, :f⊥]
	dx = (bx.xmax-bx.xmin)/bx.N
	T_dy = range(0, df[1, :dyTmax], length = df[1, :TsampleN])/ω⊥*1e3

	plt = plot()
	alp = 0.1
	
	jhalf = zeros(size(df)[1])
	tindx = 60
	for kk = 1:size(df)[1]
		ψp = getblock(df[kk, :psit][tindx],1).data
		ψm = getblock(df[kk, :psit][tindx],3).data
		
		jp = real.(1/2*(conj.(ψp).*ψm + conj.(ψm).*ψp))
		jz = abs2.(ψp) - abs2.(ψm)
		n = abs2.(ψp) + abs2.(ψm)
		jhalf[kk] = sum(jp[1:Int(bx.N/2)])/sum(n[1:Int(bx.N/2)])

	end
	jhalf
end

# ╔═╡ bc743db6-3b45-4cea-9e6b-3938e85cab1d
histogram(get_halfpolarization_DataFrames(df), nbins=20)

# ╔═╡ bac6867c-e9d4-4216-837c-a1cfffb79a2e
df2 = collect_results(datadir("sims", "Li7", "scalarjet"));

# ╔═╡ ed4bf4c0-8e5b-43e8-b9c8-b2fa28933b9f
size(df2)

# ╔═╡ 7606ba9e-59c8-41f4-97bc-1dee4a83ddaa
plot_pop_dynamics_DataFrames(df2,alp = 0.4, yscale = :log10)

# ╔═╡ 1a77fbe2-b1d6-4d5c-bb9c-aa47290bc86d
df3 = collect_results(datadir("sims", "Li7", "spinjetTWA"));

# ╔═╡ 96729825-3ef9-4f1f-a628-6d6365ae2529
size(df3)

# ╔═╡ 4576dee4-f748-48b2-9f38-fbd515060a14
plot_pop_dynamics_DataFrames(df3,alp = 0.3, yscale = :log10)

# ╔═╡ b9b8eb50-eab8-480b-aa68-6fc9749e0281
histogram(get_halfpolarization_DataFrames(df3), nbins=10)

# ╔═╡ 41dabceb-e5eb-4f53-95c8-7bf183ef2480
df4 = collect_results(datadir("sims", "Li7", "scalarjetTWA"));

# ╔═╡ 8a597f76-a522-4dbf-8540-8614c2cbcfc0
size(df4) 

# ╔═╡ f0e90e18-88da-47d6-b110-9bc94b14c026
plot_pop_dynamics_DataFrames(df4,alp = 0.3, yscale = :log10)

# ╔═╡ 2c6aefec-b008-40b5-b12a-31137cdb6e17
df5 = collect_results(datadir("sims", "Li7", "scalarjetTWA1.5e4"));

# ╔═╡ 145bd881-208f-4728-8745-a195fce20646
size(df5)

# ╔═╡ c3b0e0d8-90c3-4588-89e9-54cb7aafa123
plot_pop_dynamics_DataFrames(df5,alp = 0.3, yscale = :log10)

# ╔═╡ 09d14032-828e-4266-bb8e-12f16bbada23
begin
	ψ  = df[1, :psit][1]
	bx = ψ.basis.bases[1]
	ω⊥ = 2π*df[1, :f⊥]
	a⊥ = ustrip(sqrt(ħ/MLi7/ω⊥))
	xum = range(bx.xmin, bx.xmax,length =  bx.N)*a⊥*1e6
	
	t  = range(0, df[1, :dyTmax], length = df[1, :TsampleN])
	
end

# ╔═╡ a6c6297c-bbd8-4539-ba68-3a6e7cf74552
a⊥

# ╔═╡ c56b485b-c69e-4f69-b33e-5d4ab678f25e
function convert_wfn_norm2SI(ψ)
	ψSI = copy(ψ)
	ψSI.data  = ψSI.data/sqrt(a⊥)
	return ψSI
end

# ╔═╡ c3ebba53-f164-44a4-8ae6-fd51bdfcc42a
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
	# plot!(x, (n1+n2+n3), label = L"n"; kwargs...)
	return plt
end

# ╔═╡ dceef314-de55-4ee8-88f2-37f4b1d0f5bb
indx = 55

# ╔═╡ 34c3ca35-2a6f-430c-80c3-585427fe1504
ψtest = df3[1, :psit][indx];

# ╔═╡ 2c513761-9d84-4247-a3d9-433e4347db4c
t[indx]/ω⊥*1e3

# ╔═╡ f9bf1e47-3d75-4c30-a634-aa0690743c14
plot_wfn(xum, ψtest)

# ╔═╡ Cell order:
# ╠═e7ca4ba6-ba04-11eb-18eb-a300177dd1d9
# ╠═f1b977f8-8fa4-4093-a50e-7d955464e14e
# ╠═25c2fdff-d9d3-4e12-8681-7381604dc942
# ╠═cb830da3-0627-4090-b3ca-52bf573d423e
# ╠═99bff367-5634-449f-92aa-57231011055e
# ╠═3ee96a64-bdcc-4334-aef0-1379b367a133
# ╠═461fc005-9dd1-44b1-b335-152dfa5ea960
# ╠═113d346e-2ac8-4a81-a2e0-4c54d973b18e
# ╠═b9213e12-254c-47ad-a211-3ccf8ee66e8c
# ╠═24b6fc3e-8104-4aa6-b736-2579e2eadfa7
# ╠═bc743db6-3b45-4cea-9e6b-3938e85cab1d
# ╠═bac6867c-e9d4-4216-837c-a1cfffb79a2e
# ╠═ed4bf4c0-8e5b-43e8-b9c8-b2fa28933b9f
# ╠═7606ba9e-59c8-41f4-97bc-1dee4a83ddaa
# ╠═1a77fbe2-b1d6-4d5c-bb9c-aa47290bc86d
# ╠═96729825-3ef9-4f1f-a628-6d6365ae2529
# ╠═4576dee4-f748-48b2-9f38-fbd515060a14
# ╠═b9b8eb50-eab8-480b-aa68-6fc9749e0281
# ╠═41dabceb-e5eb-4f53-95c8-7bf183ef2480
# ╠═8a597f76-a522-4dbf-8540-8614c2cbcfc0
# ╠═f0e90e18-88da-47d6-b110-9bc94b14c026
# ╠═2c6aefec-b008-40b5-b12a-31137cdb6e17
# ╠═145bd881-208f-4728-8745-a195fce20646
# ╠═c3b0e0d8-90c3-4588-89e9-54cb7aafa123
# ╠═09d14032-828e-4266-bb8e-12f16bbada23
# ╠═a6c6297c-bbd8-4539-ba68-3a6e7cf74552
# ╠═c56b485b-c69e-4f69-b33e-5d4ab678f25e
# ╠═c3ebba53-f164-44a4-8ae6-fd51bdfcc42a
# ╠═dceef314-de55-4ee8-88f2-37f4b1d0f5bb
# ╠═34c3ca35-2a6f-430c-80c3-585427fe1504
# ╠═2c513761-9d84-4247-a3d9-433e4347db4c
# ╠═f9bf1e47-3d75-4c30-a634-aa0690743c14
