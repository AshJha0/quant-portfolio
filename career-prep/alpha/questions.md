# Alpha research: knowledge check (questions only)

Fourteen graded problems and five discussion prompts on signal evaluation,
statistical significance and overfitting. Answers are in
[`keys/answers.md`](keys/answers.md); [`verify.py`](verify.py) recomputes each
graded answer by simulation, numerical integration, enumeration or a
root-find, and `career-prep/tests/test_numeric_packs.py` compares them.

Code in this repository that does these things for real: the prereg and
trials ledger (`research/`), and the purged cross-validation in
`python/equity/05-pairs-trading/`, `python/equity/10-regime-switching/`, `research/`. References: Grinold and Kahn (2000); Lo (2002),
"The statistics of Sharpe ratios"; Bailey and Lopez de Prado (2012, 2014) on
the probabilistic and deflated Sharpe ratio; Lopez de Prado (2018), *Advances
in Financial Machine Learning*, ch. 7; Harvey, Liu and Zhu (2016).

## Breadth and skill

- **A01** Fundamental law: IC 0.05 with 400 independent bets a year. Information ratio?
- **A02** What IC do you need for IR 1.5 with 100 independent bets a year?
- **A09** Monthly cross-sectional IC averages 0.03 with sd 0.06 over 60 months. t-statistic of the mean IC?
- **A10** Signal values 1.2, -0.4, 0.3, 2.0, -1.1, 0.8 and next-period returns 0.5, 0.1, -0.2, 0.9, -0.3, 0.2 for six stocks. Rank IC (Spearman)?
- **A07** A signal follows an AR(1) with daily coefficient 0.9. Half-life in days?
- **A08** Gross alpha 8% a year with one-way turnover of 20 times a year. Cost per unit traded (as a fraction) at which net alpha is zero?

## Is it real?

- **A03** Annualised Sharpe 1.0 measured over 4 years of iid returns. t-statistic for SR > 0?
- **A04** Normal iid monthly returns, true monthly Sharpe 0.3, 60 months. Standard error of the estimated monthly Sharpe (Lo 2002)?
- **A11** A zero-skill strategy with known vol is run for 3 years. Probability its annualised Sharpe exceeds 1?
- **A12** Normal returns, monthly Sharpe 0.1. Minimum track record length (months) for 95% confidence that the true Sharpe exceeds 0?
- **A05** You test 20 strategies and want a 5% family-wise error rate with Bonferroni. One-sided z threshold per test?
- **A13** Twenty independent null strategies, each tested at 5%. Probability at least one looks significant?
- **A06** Expected maximum of 100 independent standard normal t-statistics (all nulls)?

## Validation

- **A14** 1000 daily observations, 5 contiguous folds; the test fold is days 400-599. Each label uses returns over the next 10 days, and you embargo 1% of the sample after the test fold. How many training observations remain after purging and embargo?

## Discussion (not graded)

- **AD1** List three ways future information leaks into a backtest.
- **AD2** Why should you count every variant you tried, not only the ones you kept?
- **AD3** A signal has a high IC but loses money after costs. What do you check?
- **AD4** How would you tell decay of an edge from bad luck?
- **AD5** When is a rank IC better than a Pearson IC?
