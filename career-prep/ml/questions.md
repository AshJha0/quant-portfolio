# Machine learning for finance: knowledge check (questions only)

Twenty-five problems, written fresh for this repository, on the parts of
machine learning that decide whether a finance model is real or an
artefact: purged and embargoed cross-validation, overlapping labels,
leakage, regularisation, tree ensembles, feature importance, evaluation
metrics and multiple testing. The answers are in
[`keys/answers.md`](keys/answers.md).

Each answer is recomputed by [`verify.py`](verify.py), which runs the
procedure on the question's numbers using only `numpy` and `scipy` (no ML
library defaults), and the population answers are also simulated with a
fixed seed. The CI test `career-prep/tests/test_ml_pack.py` fails if a key
and its check disagree. The cross-validation questions follow Lopez de
Prado, *Advances in Financial Machine Learning* (2018), chapters 4 and 7.

Tags: `[cv]` validation design, `[reg]` regularisation, `[tree]` trees and
ensembles, `[metric]` evaluation, `[imp]` feature importance, `[opt]`
optimisation, `[leak]` leakage and overfitting.

## Validation design and leakage

- **M04** `[cv]` 1,000 daily observations, 5 contiguous folds. Each label uses the next 10 days of prices. The test fold is the third (observations 400-599). Purge training points whose label window reaches into the test fold, and embargo the 10 observations after it. How many training observations remain?
- **M05** `[cv]` A grid search tries 3 x 4 x 5 hyperparameter settings with 5-fold CV. How many model fits?
- **M19** `[cv]` 250 daily labels, each spanning 5 consecutive days, one label starting each day. Ignoring the ends of the sample, what is the effective number of non-overlapping labels (250 times the average uniqueness)?
- **M24** `[cv]` Ten years of data. Walk-forward: train on 3 years, test on the next 1, roll forward 1 year. How many test years?
- **M25** `[leak]` Regress 101 observations of pure noise on 10 pure-noise features plus an intercept. Expected in-sample R-squared?
- **M18** `[leak]` 100 independent strategy variants are backtested. To keep the chance of any false discovery at 5%, what one-sided z threshold (Bonferroni) must the best one clear?
- **M23** `[metric]` 2% of trades are labelled "bad". Accuracy of a model that always predicts "good"?

## Regularisation

- **M01** `[reg]` Ridge with no intercept on x = (1, 2, 3), y = (2, 4, 6), penalty lambda = 14 on beta^2 (objective sum of squared errors + lambda beta^2). Fitted beta?
- **M02** `[reg]` Minimise (1/2)(b - 0.8)^2 + 0.3 |b|. Minimiser?
- **M03** `[reg]` Same with 0.2 in place of 0.8?
- **M20** `[reg]` A design matrix has singular values 2 and 1. Effective degrees of freedom of ridge with lambda = 1?

## Trees and ensembles

- **M07** `[tree]` A node holds 30 positives and 70 negatives. Gini impurity?
- **M08** `[tree]` Entropy in bits of a node with class proportions 1/4 and 3/4?
- **M09** `[tree]` A parent of 50 positives and 50 negatives splits into (40 positive, 10 negative) and (10 positive, 40 negative). Information gain in bits?
- **M10** `[tree]` A forest averages 10 trees; each prediction has variance 1 and every pair of trees has correlation 0.3. Variance of the average?
- **M11** `[tree]` In a large bootstrap sample (n draws with replacement from n), what fraction of the original observations is out of bag?
- **M12** `[tree]` Gradient boosting with squared loss on y = (1, 2, 6). Start from the best constant, fit one tree that matches every residual exactly, learning rate 0.1. New prediction for the point with y = 6?
- **M06** `[tree]` k-nearest-neighbour regression with k = 5, labels with independent noise of variance 1. Variance of the noise part of a prediction?

## Evaluation

- **M13** `[metric]` Log-loss (natural log) of predicting 0.8 for an example whose label is 1?
- **M14** `[metric]` A logistic regression coefficient on a feature is 0.7. By what factor do the odds change when the feature rises by 1?
- **M15** `[metric]` 30 true positives, 10 false positives, 20 false negatives. F1 score?
- **M16** `[metric]` Scores for positives 0.9 and 0.6; for negatives 0.7, 0.3 and 0.2. ROC AUC?

## Feature importance and optimisation

- **M17** `[imp]` y = 2 x1 + x2 with x1, x2 independent standard normals; the model is y_hat = 2 x1. Permutation importance of x1 measured as the rise in mean squared error when x1 is shuffled (large sample)?
- **M21** `[opt]` k-means, one Lloyd step: points 0, 1, 4, 5 and starting centres 0 and 1. New position of the second centre?
- **M22** `[opt]` Gradient descent on f(w) = 2 w^2. Largest step size for which the iterates do not diverge?
