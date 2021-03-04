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
# zeeman shifts
UZeeman = - p*( position(bx) ⊕ 0*position(bx) ⊕ -1*position(bx) ) + q*(one(bx) ⊕ 0*one(bx) ⊕ one(bx)); 
UZeeman_quench = - p_quench*( position(bx) ⊕ 0*position(bx) ⊕ -1*position(bx) ) + q_quench*(one(bx) ⊕ 0*one(bx) ⊕ one(bx)); 