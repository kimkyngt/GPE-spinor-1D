# Angular momentum algebra of the collective spins

Table 11.1 of Agarwal

\\[
[J_{\alpha}, J_{\beta}] = i\epsilon_{\alpha\beta\gamma}J_{\gamma} \\
 D(\xi) = \exp(\xi J_+ - \xi^*J_-) \\
 = \exp(\zeta J_+) \exp \{ J_z \ln(1+|\zeta|^2) \} \exp(-\zeta^*J_-), \quad \xi =\frac{\theta}{2}e^{-i \phi }
 \\]

\\[
e^{i\theta \mathbf{n}\cdot \mathbf{J}} \mathbf{A} e^{-i\theta \mathbf{n}\cdot \mathbf{J}} = \mathbf{n}(\mathbf{n} \cdot \mathbf{A}) - \mathbf{n} \times  (\mathbf{n} \times \mathbf{A})\cos{\theta} + (\mathbf{n}\times\mathbf{A})\sin{\theta} 
\\]

For $\mathbf{n} = \hat{z}$ and $\mathbf{A} = \hat{z}J_z$,

\\[
e^{i\theta J_z} \hat{z}J_z e^{-i\theta J_z} = \hat{z}J_z  
\\]

For $\mathbf{n} = \hat{z}$ and $\mathbf{A} = \hat{x}J_x$,

\\[
e^{i\theta J_z} \hat{x}J_x e^{-i\theta J_z} = 0 - (-\hat{x}J_x)\cos{\theta} + (\hat{y}J_x)sin{\theta} = \hat{x}J_x \cos{\theta} + \hat{y}J_x \sin{\theta}
\\]

The coherent spin state, $|\theta, \phi\rangle$ is
\\[
|\theta, \phi\rangle = D(\xi)|j, -j\rangle \\
 = \sum^{+j}_{m = -j} \begin{pmatrix}2j \\ j+m\end{pmatrix}^{\frac{1}{2}} \sin^{j+m}{\frac{\theta}{2}} \cos^{j-m}{\frac{\theta}{2}} e^{-(j+m)\phi} |j, m\rangle
\\]

The phase space funciton is 
\\[
\Phi(\theta, \phi) = \sum_{K, Q}\rho_{KQ}Y_{KQ}f_{KQ}
\\]
$f_{KQ}$ depends on the representation methods. For Wigner function, $f_{KQ} = \sqrt{ \frac{2j+1}{4\pi} }$