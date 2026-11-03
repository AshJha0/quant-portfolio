# Linear algebra and numerical methods: knowledge check (questions only)

Twenty-five problems, written fresh for this repository, on the numerics
behind pricing and risk code: eigenvalues and PCA, positive semidefinite
(PSD) repair of correlation matrices, LU, QR and Cholesky, conditioning,
root finding (Newton, bisection, secant), quadrature, finite differences and
the stability of heat-equation schemes such as Crank-Nicolson. The answers
are in [`keys/answers.md`](keys/answers.md).

Each answer in the key is recomputed by [`verify.py`](verify.py), which runs
the method itself: a real Newton loop, a real bisection, a real trapezoid
rule, the full finite-difference step matrix and its spectrum. The CI test
`career-prep/tests/test_numerics_pack.py` fails if a key and its check
disagree.

Tags: `[eig]` eigenvalues and PCA, `[psd]` correlation repair, `[fact]`
factorisations, `[cond]` conditioning and floating point, `[root]` root
finding, `[quad]` quadrature, `[fd]` finite differences, `[pde]` PDE
stability.

## Eigenvalues, PCA and correlation repair

- **N01** `[eig]` Largest eigenvalue of [[2, 1], [1, 2]]?
- **N02** `[eig]` Three assets, every pairwise correlation 0.5. Share of total variance explained by the first principal component of the correlation matrix?
- **N03** `[psd]` What is the smallest rho for which the 3 x 3 matrix with ones on the diagonal and rho everywhere else is still positive semidefinite?
- **N04** `[psd]` Determinant of the "correlation matrix" [[1, 0.9, 0.7], [0.9, 1, 0.3], [0.7, 0.3, 1]]? What does its sign tell you?
- **N06** `[psd]` Repair [[1, 2], [2, 1]] by setting its negative eigenvalues to zero and rebuilding V diag(lambda) V^T. Diagonal entry of the result, before any rescaling?
- **N25** `[eig]` Power iteration on a symmetric matrix with eigenvalues 3 and 1. By what factor does the error shrink each step?

## Factorisations and conditioning

- **N05** `[fact]` Cholesky factor L of [[4, 2], [2, 5]] (A = L L^T, L lower triangular with positive diagonal). Entry L22?
- **N07** `[cond]` 2-norm condition number of [[1, 1], [1, 1.0001]]?
- **N08** `[fact]` Use Gaussian elimination (LU) to find the determinant of [[2, 1, 1], [4, 3, 3], [8, 7, 9]].
- **N09** `[fact]` QR factorisation of [[3, 1], [4, 2]] by Gram-Schmidt. Value of |R22|?
- **N24** `[cond]` Machine epsilon of IEEE double precision (the gap between 1 and the next larger double)?

## Root finding

- **N10** `[root]` Newton's method for x^2 - 2 = 0 from x0 = 1. Value of x2?
- **N11** `[root]` Newton's method for x^2 = 0 (a double root) from x0 = 1. Value of x10? What rate of convergence is that?
- **N12** `[root]` Bisection for sqrt(2) on [0, 2]. How many halvings until the bracket is narrower than 1e-6?
- **N13** `[root]` One secant step for x^2 - 2 = 0 from x0 = 1, x1 = 2. Value of x2?

## Quadrature

- **N14** `[quad]` Trapezoid rule with two equal intervals for the integral of x^2 on [0, 1]?
- **N15** `[quad]` Simpson's rule on the same grid?
- **N16** `[quad]` Two-point Gauss-Legendre rule for the integral of x^4 on [-1, 1]? (The exact value is 2/5.)
- **N17** `[quad]` Composite trapezoid rule on a smooth integrand: by what factor does the error fall when the step is halved?

## Finite differences

- **N18** `[fd]` Forward difference (f(h) - f(0)) / h for f = exp at h = 0.1?
- **N19** `[fd]` Central difference (f(h) - f(-h)) / (2h) for f = exp at h = 0.1?
- **N20** `[fd]` Balancing truncation error (order h) against rounding error (order eps / h) in a forward difference gives the usual step h = sqrt(eps) for double precision near x = 1. What is it, as a power of 2?

## PDE stability (heat equation u_t = D u_xx)

- **N21** `[pde]` Explicit (forward Euler) scheme with dx = 0.01 and D = 1. Largest stable time step?
- **N22** `[pde]` Crank-Nicolson with r = D dt / dx^2 = 10. Amplification factor of the highest-frequency (sawtooth) mode? What does its sign mean for a payoff with a kink?
- **N23** `[pde]` Fully implicit scheme, same r = 10. Amplification factor of the sawtooth mode?
