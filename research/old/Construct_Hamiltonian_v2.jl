# basis generation
bx = PositionBasis(-xmax, xmax, nx)
xx = samplepoints(bx)
xx_um = a⊥*xx *1e6

bp = MomentumBasis(bx)
pp = samplepoints(bp)

bs = SpinBasis(1)
σx, σy, σz, σp, σm = sigmax(bs), sigmay(bs), sigmaz(bs), sigmap(bs), sigmam(bs) ;

# transformation operators
Txp = transform(bx ⊗ bs, bp ⊗ bs)
Tpx = dagger(Txp)

# Basic operators
x  = position(bx) ⊗ one(bs)
Px = momentum(bp) ⊗ one(bs)

# Single particle Hamiltonian (time - independent)
# kinetic energy opeartor
Hkin = Px^2/2.0  
Hkin_FFT = LazyProduct(Txp, Hkin, Tpx) 
# harmonic potential operator
Utrap = 0.5 * (γ^2 * x^2) 
# zeeman shifts
UZeeman = - p*(one(bx) ⊗ σz) + q*( one(bx) ⊗ σz)^2
UZeeman_quench = - p_quench*(one(bx) ⊗ σz ) + q_quench*(one(bx) ⊗ σz)^2