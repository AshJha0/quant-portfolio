# Machine learning for finance: answer key

Questions are in [`../questions.md`](../questions.md). Each line gives the
answer in backticks, then the shortest argument we know. Each value is
recomputed by [`../verify.py`](../verify.py) and compared with this file by
`career-prep/tests/test_ml_pack.py`; if prose and test disagree, the test is
the arbiter.

- **M01** `1`. beta = sum(xy) / (sum(x^2) + lambda) = 28 / (14 + 14). OLS would give 2; the penalty halves it.
- **M02** `1/2`. With an orthonormal design the lasso is soft thresholding: sign(b) max(|b| - lambda, 0) = 0.8 - 0.3.
- **M03** `0`. |0.2| < 0.3, so the coefficient is set exactly to zero. This is why the lasso selects features and ridge does not.
- **M04** `780`. Remove the 200 test points, the 10 points just before the test fold whose labels reach into it (390-399), and the 10 embargoed points after it (600-609): 1000 - 220. Without purging, the model is trained on labels built from test-period prices (Lopez de Prado 2018, ch. 7).
- **M05** `300`. 60 settings x 5 folds; plus one final refit if you count it. Every fit is another look at the data.
- **M06** `1/5`. The average of 5 independent unit-variance noises has variance 1/5: larger k lowers variance and raises bias.
- **M07** `21/50`. 1 - 0.3^2 - 0.7^2 = 0.42.
- **M08** `0.811278124459`. -(1/4) log2(1/4) - (3/4) log2(3/4) = 0.5 + 0.311.
- **M09** `0.278071905113`. Parent entropy 1; each child has entropy H(0.8) = 0.7219; gain 1 - 0.7219.
- **M10** `37/100`. rho + (1 - rho)/B = 0.3 + 0.07. More trees cannot push it below 0.3; only decorrelating the trees can (Breiman 2001, *Machine Learning* 45).
- **M11** `0.367879441171`. (1 - 1/n)^n tends to 1/e. In finance the out-of-bag score is optimistic when labels overlap, because the "unseen" points share information with in-bag neighbours.
- **M12** `33/10`. The best constant is the mean, 3; the residual at y = 6 is 3; 3 + 0.1 x 3.
- **M13** `0.223143551314`. -ln(0.8).
- **M14** `2.01375270747`. The odds are exp(linear score), so the ratio is e^0.7.
- **M15** `2/3`. Precision 0.75, recall 0.6; F1 = 2TP / (2TP + FP + FN) = 60 / 90.
- **M16** `5/6`. AUC is the fraction of positive-negative pairs ranked correctly: 0.9 beats all 3 negatives, 0.6 beats 2, so 5 of 6.
- **M17** `8`. Before shuffling the error is x2, with mean square 1. After, it is 2 x1 + x2 - 2 x1' with x1' independent, mean square 4 + 1 + 4 = 9. The rise is 8 = 2 x 2^2: permutation importance double-counts because the shuffled copy adds its own variance.
- **M18** `3.29052673149`. Phi^{-1}(1 - 0.05/100). A Sharpe "t-stat" of 2 is not enough once you have tried 100 variants (Harvey, Liu and Zhu 2016, *Review of Financial Studies* 29).
- **M19** `50`. Away from the ends each day is covered by 5 labels, so each label has uniqueness 1/5; 250 x 1/5 (Lopez de Prado 2018, ch. 4).
- **M20** `13/10`. df = sum d_j^2 / (d_j^2 + lambda) = 4/5 + 1/2 (Hastie, Tibshirani and Friedman 2009, eq. 3.50).
- **M21** `10/3`. Point 0 goes to centre 0; points 1, 4 and 5 go to centre 1, whose new position is their mean.
- **M22** `1/2`. The update is w <- (1 - 4 eta) w, which needs |1 - 4 eta| <= 1, so eta <= 2/L with L = 4.
- **M23** `49/50`. 98% accuracy with no skill at all; report precision and recall on the rare class instead.
- **M24** `7`. Test years 4 to 10.
- **M25** `1/10`. Under the null R^2 ~ Beta(p/2, (n - p - 1)/2), with mean p / (n - 1) = 10 / 100. Ten noise features "explain" 10% in sample.
