// Matrix, Cholesky (clean + jitter fallback), covariance estimators.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "eqvar/matrix.hpp"

using namespace eqvar;

namespace {

Matrix spd3() {
    // SPD matrix built as A = B B^T + I from a fixed B.
    return Matrix(3, 3,
                  {4.0, 2.0, 0.6,
                   2.0, 5.0, 1.5,
                   0.6, 1.5, 3.0});
}

Matrix reconstruct(const Matrix& l) {
    const std::size_t n = l.rows();
    Matrix a(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += l(i, k) * l(j, k);
            a(i, j) = s;
        }
    return a;
}

}  // namespace

TEST(Cholesky, ReconstructsSpdTo1e12) {
    const Matrix a = spd3();
    const CholeskyResult res = cholesky(a);
    EXPECT_DOUBLE_EQ(res.jitter_added, 0.0);
    EXPECT_LT(Matrix::max_abs_diff(reconstruct(res.lower), a), 1e-12);
    // Lower-triangular: strict upper part is zero.
    EXPECT_DOUBLE_EQ(res.lower(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(res.lower(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(res.lower(1, 2), 0.0);
}

TEST(Cholesky, JitterPathOnSingularMatrix) {
    // Perfectly correlated 2x2 — exactly singular, plain Cholesky must fail
    // and the diagonal-jitter fallback must engage.
    const Matrix a(2, 2, {1.0, 1.0, 1.0, 1.0});
    const CholeskyResult res = cholesky(a);
    EXPECT_GT(res.jitter_added, 0.0);
    EXPECT_LE(res.jitter_added, 1e-6);  // tiny relative to unit variances
    EXPECT_LT(Matrix::max_abs_diff(reconstruct(res.lower), a), 1e-6);
}

TEST(Cholesky, JitterPathOnZeroVarianceAsset) {
    const Matrix a(2, 2, {0.01, 0.0, 0.0, 0.0});  // second asset has zero variance
    const CholeskyResult res = cholesky(a);
    EXPECT_GT(res.jitter_added, 0.0);
    EXPECT_LT(Matrix::max_abs_diff(reconstruct(res.lower), a), 1e-8);
}

TEST(Cholesky, RejectsBadInput) {
    EXPECT_THROW(cholesky(Matrix(2, 3, 1.0)), std::invalid_argument);           // non-square
    EXPECT_THROW(cholesky(Matrix(2, 2, {1.0, 0.5, -0.5, 1.0})), std::invalid_argument);  // asymmetric
    // Badly indefinite: jitter cannot rescue a huge negative eigenvalue.
    EXPECT_THROW(cholesky(Matrix(2, 2, {1.0, 1e8, 1e8, 1.0})), std::runtime_error);
}

TEST(MatVec, KnownProductAndShapeCheck) {
    const Matrix a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const std::vector<double> x = {1.0, 0.5, -1.0};
    const std::vector<double> y = matvec(a, x);
    ASSERT_EQ(y.size(), 2u);
    EXPECT_DOUBLE_EQ(y[0], -1.0);
    EXPECT_DOUBLE_EQ(y[1], 0.5);
    EXPECT_THROW(matvec(a, std::vector<double>{1.0, 2.0}), std::invalid_argument);
}

TEST(QuadForm, MatchesHandComputation) {
    const Matrix s(2, 2, {2.0, 0.5, 0.5, 1.0});
    const std::vector<double> w = {1.0, -2.0};
    // w'Sw = 2 - 2*0.5*2 + 4 = 4
    EXPECT_DOUBLE_EQ(quad_form(w, s), 4.0);
}

TEST(SampleCovariance, HandComputed2x2) {
    // Two assets, three days: covariance ddof=1 by hand.
    const Matrix r(3, 2, {0.01, 0.02, -0.01, 0.00, 0.03, 0.01});
    const Matrix s = sample_covariance(r);
    EXPECT_NEAR(s(0, 0), 4.0e-4, 1e-18);
    EXPECT_NEAR(s(1, 1), 1.0e-4, 1e-18);
    EXPECT_NEAR(s(0, 1), 1.0e-4, 1e-18);
    EXPECT_DOUBLE_EQ(s(0, 1), s(1, 0));
    EXPECT_THROW(sample_covariance(Matrix(1, 2, 0.0)), std::invalid_argument);
}

TEST(EwmaCovariance, SinglePointLimitAndValidation) {
    // With lam -> 1 the EWMA forecast stays near the sample seed.
    const Matrix r(3, 2, {0.01, 0.02, -0.01, 0.00, 0.03, 0.01});
    const Matrix s = sample_covariance(r);
    const Matrix e = ewma_covariance(r, 0.999999);
    EXPECT_LT(Matrix::max_abs_diff(e, s), 1e-8);
    EXPECT_THROW(ewma_covariance(r, 1.0), std::invalid_argument);
    EXPECT_THROW(ewma_covariance(r, 0.0), std::invalid_argument);
}

TEST(CovarianceFromVols, Known) {
    const std::vector<double> vols = {0.01, 0.02};
    const Matrix corr(2, 2, {1.0, 0.5, 0.5, 1.0});
    const Matrix cov = covariance_from_vols(vols, corr);
    EXPECT_DOUBLE_EQ(cov(0, 0), 1e-4);
    EXPECT_DOUBLE_EQ(cov(0, 1), 1e-4);
    EXPECT_DOUBLE_EQ(cov(1, 1), 4e-4);
}
