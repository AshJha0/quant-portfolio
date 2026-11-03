# Linear algebra and numerical methods: answer key

Questions are in [`../questions.md`](../questions.md). Each line gives the
answer in backticks, then the shortest argument we know. Each value is
recomputed by [`../verify.py`](../verify.py) and compared with this file by
`career-prep/tests/test_numerics_pack.py`. Where the check is a limit (a
finite grid tending to the answer) the test allows the relative tolerance
listed in `REL_TOL` in that file.

- **N01** `3`. Eigenvectors (1, 1) and (1, -1) give 2 + 1 and 2 - 1.
- **N02** `2/3`. Equicorrelation has eigenvalues 1 + (n - 1) rho = 2 (the "market" vector) and 1 - rho = 0.5 twice; 2 / 3.
- **N03** `-1/2`. The market eigenvalue 1 + (n - 1) rho must be >= 0, so rho >= -1/(n - 1). Three assets cannot all be strongly negatively correlated with each other.
- **N04** `-3/250`. Cofactor expansion: 0.91 - 0.9 x 0.69 + 0.7 x (-0.43) = -0.012. A negative determinant of a 3 x 3 symmetric matrix means an odd number of negative eigenvalues, so this is not a valid correlation matrix: pairwise estimates or hand edits produced an impossible set. Repair it (N06, or the nearest-correlation method of Higham 2002, *IMA J. Numer. Anal.* 22) before simulating from it.
- **N05** `2`. L11 = sqrt(4) = 2, L21 = 2 / 2 = 1, L22 = sqrt(5 - 1) = 2.
- **N06** `3/2`. Eigenvalues 3 (vector (1, 1)/sqrt 2) and -1. Keeping only 3 gives 3/2 [[1, 1], [1, 1]]. The diagonal is no longer 1, so clipping must be followed by rescaling to unit diagonal, which here gives perfect correlation.
- **N07** `40002.0000749`. Symmetric, so the condition number is lambda_max / lambda_min; the eigenvalues are 1.00005 +- sqrt(1 + 0.00005^2), about 2.00005 and 5e-5. Expect to lose about 4.6 decimal digits solving with it.
- **N08** `4`. Subtract 2 x row 1 and 4 x row 1: rows (0, 1, 1) and (0, 3, 5); subtract 3 x row 2: (0, 0, 2). Product of pivots 2 x 1 x 2.
- **N09** `2/5`. R11 = |(3, 4)| = 5 and |det A| = |R11 R22| = |3 x 2 - 4 x 1| = 2.
- **N10** `17/12`. x1 = 1 - (-1)/2 = 3/2; x2 = 3/2 - (1/4)/3 = 17/12, already within 2.5e-3 of sqrt 2.
- **N11** `1/1024`. x_{n+1} = x_n - x_n^2 / (2 x_n) = x_n / 2. At a multiple root Newton is only linear, with rate 1/2.
- **N12** `21`. The width after n halvings is 2 / 2^n; need 2^n > 2 x 10^6, and 2^21 = 2,097,152.
- **N13** `4/3`. x2 = x1 - f(x1)(x1 - x0) / (f(x1) - f(x0)) = 2 - 2 x 1 / 3.
- **N14** `3/8`. h/2 (f0 + 2 f1 + f2) = 0.25 (0 + 0.5 + 1); the exact value is 1/3.
- **N15** `1/3`. h/3 (f0 + 4 f1 + f2) = (1/6)(0 + 1 + 1). Simpson is exact for cubics.
- **N16** `2/9`. Nodes +-1/sqrt 3, weights 1: 2 x (1/9). Two-point Gauss is exact to degree 3, so degree 4 is where it first errs.
- **N17** `4`. The composite trapezoid error is O(h^2) for smooth integrands (Euler-Maclaurin formula); halving h divides it by 4.
- **N18** `1.05170918076`. (e^0.1 - 1) / 0.1. The error, about 0.05, is about (h/2) f'': first order.
- **N19** `1.0016675002`. sinh(0.1) / 0.1; the error is about h^2/6, thirty times smaller than in N18 at the same h.
- **N20** `1/67108864`. eps = 2^-52, so sqrt(eps) = 2^-26, about 1.5e-8 (scale it by |x| away from x = 1).
- **N21** `1/20000`. Von Neumann analysis: the sawtooth mode is multiplied by 1 - 4r each step, which needs r <= 1/2, so dt <= dx^2 / (2D) = 5e-5.
- **N22** `-19/21`. For the sawtooth g = (1 - 2r)/(1 + 2r). Since |g| < 1 the scheme is stable, but g is close to -1, so that mode flips sign each step and barely decays. A kinked payoff excites it, which is why Crank-Nicolson Greeks oscillate near the strike; the usual remedy is a few fully implicit steps first (Rannacher 1984, *Numer. Math.* 43).
- **N23** `1/41`. g = 1 / (1 + 4r): strongly damped and never oscillating, but only first order in time.
- **N24** `1/4503599627370496`. 2^-52, about 2.22e-16: a double stores 52 fraction bits (IEEE 754-2008).
- **N25** `1/3`. The component along the second eigenvector falls by lambda_2 / lambda_1 each step relative to the first.
