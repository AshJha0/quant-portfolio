# Statistics and econometrics: answer key

Questions are in [`../questions.md`](../questions.md). Each line gives the
answer in backticks, then the shortest argument we know. Each value is
recomputed from the question's data by [`../verify.py`](../verify.py) and
compared with this file by `career-prep/tests/test_stats_pack.py`; if prose
and test disagree, the test is the arbiter.

- **S01** `8/3`. Mean 4; squared deviations 4, 0, 0, 4 sum to 8; divide by n - 1 = 3.
- **S02** `2`. s / sqrt(n) = 20 / 10.
- **S03** `2`. Under H0 the count has mean 50 and standard deviation sqrt(100 x 1/4) = 5; (60 - 50) / 5.
- **S04** `0.0455002638964`. 2(1 - Phi(2)). The familiar "about 5%" is for 1.96, not 2.
- **S05** `9604`. n >= z^2 p(1 - p) / h^2 = 1.959964^2 x 0.25 / 0.0001 = 9603.6, so 9604.
- **S06** `7/5`. Sxy = 7 and Sxx = 5 about the means (2.5, 4); slope = Sxy / Sxx.
- **S07** `49/50`. Explained sum of squares = Sxy^2 / Sxx = 9.8; total = 10.
- **S08** `0.158745078664`. Variances add over independent days: 0.01 sqrt(252), about 15.9%.
- **S09** `0.612372435696`. sqrt(1.5 / 4). A Sharpe of 1 over four years is under two standard errors from zero (Lo 2002, *Financial Analysts Journal* 58(4)).
- **S10** `4/3`. Var = sigma^2 / (1 - phi^2) = 1 / 0.75.
- **S11** `3.10628371951`. Solve 0.8^h = 1/2: h = ln 0.5 / ln 0.8.
- **S12** `1/3`. Log-likelihood n ln(lambda) - lambda sum(x); setting the derivative to zero gives n / sum(x) = 4 / 12.
- **S13** `1/400`. 0.05 / 20.
- **S14** `0.641514077591`. 1 - 0.95^20, about 64%: test enough ideas and something "works".
- **S15** `0.803764940015`. Reject when Z + 2.5 > 1.644854; power = Phi(2.5 - 1.644854) = Phi(0.855).
- **S16** `19`. 4 + 9 + 2 x 0.5 x 2 x 3.
- **S17** `7/25`. (1 + (n - 1) rho) / n = (1 + 1.8) / 10. As n grows it tends to rho, not zero: correlation is the floor on diversification.
- **S18** `100/3`. The long-run variance factor is (1 + phi) / (1 - phi) = 3, so n_eff = 100 / 3.
- **S19** `67/5`. Expected 10 per face; squared deviations 25, 4, 1, 4, 0, 100 sum to 134; divide by 10. Well above the 5-degree-of-freedom 5% critical value of 11.07.
- **S20** `25/3`. JB = n/6 (S^2 + K^2/4) = 100/6 x (0.25 + 0.25), above the 2-degree-of-freedom 5% critical value of 5.99.
- **S21** `2/3`. Beta(1, 1) prior updates to Beta(8, 4), mean 8 / 12.
- **S22** `1`. Precision-weighted: (0 x 1 + 2 x 1) / (1 + 1). Equal precisions shrink the observation half way to the prior.
- **S23** `0.3486784401`. (9/10)^10; for large n it tends to 1/e, about 0.368.
- **S24** `0.339307522483`. atanh(0.5) = 0.5493; subtract 1.959964 / sqrt(100) and take tanh.
- **S25** `-1/100`. 1.1 x 0.9 = 0.99. Log returns would add to ln 0.99 exactly; simple returns do not add.
