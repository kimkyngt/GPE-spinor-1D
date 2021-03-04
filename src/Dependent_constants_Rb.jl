# dependent in SI
ω⊥ 			= 2*π*f⊥
a⊥ 			= sqrt(ħ/87/amu/ω⊥)
ωSI  		= 2*π*fSI
axSI 		= sqrt(ħ/87/amu/ωSI)

c0SI 		= 4*π*ħ^2/87/amu * (2*a2Rb + a0Rb)/3
c1SI 		= 4*π*ħ^2/87/amu * (a2Rb - a0Rb)/3
c0_1DSI 	= c0SI/(2*π*a⊥^2)
c1_1DSI 	= c1SI/(2*π*a⊥^2)
# RTF 		= (3*c0_1DSI*Natom/(2*87*amu*ωSI^2))^(1/3)
npeakSI 	= 3500/1e-6


# normalized
γ 			= ωSI/ω⊥
p 			= pSI/(ħ*ω⊥) 
q 			= qSI/(ħ*ω⊥)
p_quench 	= p_quenchSI/(ħ*ω⊥) 
q_quench 	= q_quenchSI/(ħ*ω⊥)
c0 			= c0_1DSI/(ħ*ω⊥*a⊥)
c1 			= c1_1DSI/(ħ*ω⊥*a⊥)