# Stochastic calculus: answer key

Questions are in [`../questions.md`](../questions.md). Each line gives the
answer in backticks, then the shortest argument we know. Each value is
recomputed by a different route in [`../verify.py`](../verify.py) and
compared with this file by `career-prep/tests/test_quant_packs.py`. If you
find an error, open an issue: the test is the arbiter, not this prose.

- **S01** `3`. W_t ~ N(0, t), so E[W_t^2] = t.
- **S02** `2`. Write W_5 = W_2 + (W_5 - W_2); the increment is independent of W_2 with mean 0, so E[W_2 W_5] = E[W_2^2] = min(2, 5).
- **S03** `1/3`. Var = double integral of Cov(W_s, W_u) = min(s, u) over the unit square = 1/3.
- **S04** `3`. The fourth moment of N(0, 1) is 3 (the kurtosis of a normal).
- **S18** `4`. With W_2 = W_1 + Z: E[W_1^2 (W_1 + Z)^2] = E[W_1^4] + E[W_1^2] E[Z^2] = 3 + 1.
- **S22** `4.67077427047`. E[e^{2W}] - E[e^W]^2 = e^2 - e, from E[e^{aW_1}] = e^{a^2/2}.
- **S05** `2`. Ito isometry: E[(int W dW)^2] = int_0^2 E[W_s^2] ds = int_0^2 s ds = 2.
- **S14** `1/2`. Fubini: int_0^1 E[W_s^2] ds = int_0^1 s ds.
- **S19** `1/2`. The Stratonovich integral obeys ordinary calculus: int W o dW = W_1^2 / 2, mean 1/2. The Ito integral is (W_1^2 - 1)/2 with mean 0; the gap is half the quadratic variation.
- **S06** `2.71828182846`. Normal moment generating function: E[e^{W_2}] = e^{2/2} = e.
- **S07** `0.317310507863`. Reflection principle: P(max >= a) = 2 P(W_1 >= a) = 2(1 - Phi(1)).
- **S15** `4`. X_t - 0.5 t is a martingale; optional stopping at the hitting time gives 2 = 0.5 E[tau]. The hitting time is inverse Gaussian with mean a/mu.
- **S16** `0.367879441171`. With drift mu, e^{-2 mu X_t} is a martingale; here that is e^{X_t}. Paths that never hit 1 drift to minus infinity where e^X is 0, so optional stopping gives 1 = P(hit) e^1, and P = e^{-1}.
- **S08** `105.127109638`. E[S_t] = S_0 e^{mu t}.
- **S09** `103.045453395`. log S_1 is normal with mean log 100 + mu - sigma^2/2 = log 100 + 0.03; the median is e^{that}. The mean is higher because the lognormal is right-skewed (the median lacks the e^{sigma^2/2} factor).
- **S10** `451.028807816`. S_0^2 e^{2 mu t}(e^{sigma^2 t} - 1) = 10^4 e^{0.1}(e^{0.04} - 1).
- **S11** `0.864664716763`. E[X_t] = theta + (X_0 - theta) e^{-kappa t} = 1 - e^{-2}.
- **S12** `0.0613552725695`. Var(X_t) = sigma^2 / (2 kappa) (1 - e^{-2 kappa t}) = 0.0625 (1 - e^{-4}).
- **S13** `1/16`. Let t go to infinity in S12: sigma^2 / (2 kappa) = 0.25 / 4.
- **S17** `1`. e^{W_1 - 1/2} is the density of a measure under which W_1 ~ N(1, 1) (Girsanov), so the expectation is that mean.
- **S20** `10941.7428371`. e^{-r} E[S_1^2] = e^{-r} S_0^2 e^{2r + sigma^2} = 10^4 e^{0.09}.
- **S21** `3`. Feynman-Kac: u(x, t) = E[(x + W_{T-t})^2] = x^2 + (T - t) = 1 + 2.
