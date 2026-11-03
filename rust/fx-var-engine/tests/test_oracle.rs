//! W5: the Rust FX engine against the independent oracle.
//!
//! Expected values come from `tests/oracle/mod.rs`, generated from
//! `python/equity/03-var-es-engine/tests/golden/var_oracle.py` (mpmath at
//! 50 digits and exact rationals; it imports nothing from `eq_var` or
//! `fx_var`) by `emit_oracle_consts.py` in the same directory. Closed forms
//! must agree to 1e-10 relative; the historical VaR order statistic
//! exactly; Monte Carlo VaR of a Gaussian book within 3 standard errors.

mod oracle;

use fx_var_engine::monte_carlo::simulate_factor_returns;
use fx_var_engine::prelude::*;
use fx_var_engine::returns::FactorCov;
use oracle::*;

const REL: f64 = 1e-10;
const SIG: f64 = 12345.678;
const MEAN: f64 = -37.5;

fn close(actual: f64, expected: f64, what: &str) {
    close_rel(actual, expected, what, REL);
}

fn close_rel(actual: f64, expected: f64, what: &str, rel: f64) {
    assert!(actual.is_finite(), "{what}: not finite");
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{what}: engine {actual:e} oracle {expected:e} rel {:e}",
        (actual / expected - 1.0).abs()
    );
}

fn conv_sample() -> Vec<f64> {
    (0..250)
        .map(|i| {
            let t = f64::from(i);
            let x = 1000.0 * (0.37 * t + 0.1).sin() * (1.0 + 0.5 * (0.11 * t).cos());
            (x * 100.0 + 0.5).floor() / 100.0
        })
        .collect()
}

fn case_b() -> (Vec<f64>, Matrix) {
    let cov = Matrix::from_rows(&[
        vec![3.6e-5, 1.1e-5, -2e-6],
        vec![1.1e-5, 4.9e-5, -1e-6],
        vec![-2e-6, -1e-6, 2.5e-7],
    ])
    .unwrap();
    (vec![11.0e6, -4.5e6, -2.4e6], cov)
}

macro_rules! t_rows {
    ($($a:expr, $df:expr, $vu:ident, $eu:ident, $vo:ident, $eo:ident;)*) => {
        [$(($a, $df, $vu, $eu, $vo, $eo)),*]
    };
}

#[test]
fn normal_and_student_t_var_es() {
    let rows = [
        (
            0.001,
            NORMAL_VAR_UNIT_A0001,
            NORMAL_ES_UNIT_A0001,
            NORMAL_VAR_ODD_A0001,
            NORMAL_ES_ODD_A0001,
        ),
        (
            0.01,
            NORMAL_VAR_UNIT_A001,
            NORMAL_ES_UNIT_A001,
            NORMAL_VAR_ODD_A001,
            NORMAL_ES_ODD_A001,
        ),
        (
            0.025,
            NORMAL_VAR_UNIT_A0025,
            NORMAL_ES_UNIT_A0025,
            NORMAL_VAR_ODD_A0025,
            NORMAL_ES_ODD_A0025,
        ),
        (
            0.05,
            NORMAL_VAR_UNIT_A005,
            NORMAL_ES_UNIT_A005,
            NORMAL_VAR_ODD_A005,
            NORMAL_ES_ODD_A005,
        ),
    ];
    for (a, vu, eu, vo, eo) in rows {
        close(normal_var(1.0, a, 0.0).unwrap(), vu, "N VaR unit");
        close(normal_var(SIG, a, MEAN).unwrap(), vo, "N VaR odd");
        close(normal_es(1.0, a, 0.0).unwrap(), eu, "N ES unit");
        close(normal_es(SIG, a, MEAN).unwrap(), eo, "N ES odd");
    }
    let t = t_rows! {
        0.001, 3.0, T3_VAR_UNIT_A0001, T3_ES_UNIT_A0001, T3_VAR_ODD_A0001, T3_ES_ODD_A0001;
        0.001, 4.0, T4_VAR_UNIT_A0001, T4_ES_UNIT_A0001, T4_VAR_ODD_A0001, T4_ES_ODD_A0001;
        0.001, 6.0, T6_VAR_UNIT_A0001, T6_ES_UNIT_A0001, T6_VAR_ODD_A0001, T6_ES_ODD_A0001;
        0.001, 10.0, T10_VAR_UNIT_A0001, T10_ES_UNIT_A0001, T10_VAR_ODD_A0001, T10_ES_ODD_A0001;
        0.001, 30.0, T30_VAR_UNIT_A0001, T30_ES_UNIT_A0001, T30_VAR_ODD_A0001, T30_ES_ODD_A0001;
        0.01, 3.0, T3_VAR_UNIT_A001, T3_ES_UNIT_A001, T3_VAR_ODD_A001, T3_ES_ODD_A001;
        0.01, 4.0, T4_VAR_UNIT_A001, T4_ES_UNIT_A001, T4_VAR_ODD_A001, T4_ES_ODD_A001;
        0.01, 6.0, T6_VAR_UNIT_A001, T6_ES_UNIT_A001, T6_VAR_ODD_A001, T6_ES_ODD_A001;
        0.01, 10.0, T10_VAR_UNIT_A001, T10_ES_UNIT_A001, T10_VAR_ODD_A001, T10_ES_ODD_A001;
        0.01, 30.0, T30_VAR_UNIT_A001, T30_ES_UNIT_A001, T30_VAR_ODD_A001, T30_ES_ODD_A001;
        0.025, 3.0, T3_VAR_UNIT_A0025, T3_ES_UNIT_A0025, T3_VAR_ODD_A0025, T3_ES_ODD_A0025;
        0.025, 4.0, T4_VAR_UNIT_A0025, T4_ES_UNIT_A0025, T4_VAR_ODD_A0025, T4_ES_ODD_A0025;
        0.025, 6.0, T6_VAR_UNIT_A0025, T6_ES_UNIT_A0025, T6_VAR_ODD_A0025, T6_ES_ODD_A0025;
        0.025, 10.0, T10_VAR_UNIT_A0025, T10_ES_UNIT_A0025, T10_VAR_ODD_A0025, T10_ES_ODD_A0025;
        0.025, 30.0, T30_VAR_UNIT_A0025, T30_ES_UNIT_A0025, T30_VAR_ODD_A0025, T30_ES_ODD_A0025;
        0.05, 3.0, T3_VAR_UNIT_A005, T3_ES_UNIT_A005, T3_VAR_ODD_A005, T3_ES_ODD_A005;
        0.05, 4.0, T4_VAR_UNIT_A005, T4_ES_UNIT_A005, T4_VAR_ODD_A005, T4_ES_ODD_A005;
        0.05, 6.0, T6_VAR_UNIT_A005, T6_ES_UNIT_A005, T6_VAR_ODD_A005, T6_ES_ODD_A005;
        0.05, 10.0, T10_VAR_UNIT_A005, T10_ES_UNIT_A005, T10_VAR_ODD_A005, T10_ES_ODD_A005;
        0.05, 30.0, T30_VAR_UNIT_A005, T30_ES_UNIT_A005, T30_VAR_ODD_A005, T30_ES_ODD_A005;
    };
    for (a, df, vu, eu, vo, eo) in t {
        let tag = format!("a={a} df={df}");
        close(
            student_t_var(1.0, a, df, 0.0).unwrap(),
            vu,
            &format!("t VaR unit {tag}"),
        );
        close(
            student_t_var(SIG, a, df, MEAN).unwrap(),
            vo,
            &format!("t VaR odd {tag}"),
        );
        close(
            student_t_es(1.0, a, df, 0.0).unwrap(),
            eu,
            &format!("t ES unit {tag}"),
        );
        close(
            student_t_es(SIG, a, df, MEAN).unwrap(),
            eo,
            &format!("t ES odd {tag}"),
        );
    }
}

#[test]
fn cornish_fisher() {
    let rows = [
        (
            -0.5,
            1.0,
            0.01,
            CF_SM05_K1_A001_VALID,
            CF_SM05_K1_A001_VAR_ODD,
        ),
        (
            0.122,
            -0.427,
            0.01,
            CF_S0122_KM0427_A001_VALID,
            CF_S0122_KM0427_A001_VAR_ODD,
        ),
        (
            0.2,
            0.5,
            0.025,
            CF_S02_K05_A0025_VALID,
            CF_S02_K05_A0025_VAR_ODD,
        ),
        (
            -0.3,
            2.0,
            0.05,
            CF_SM03_K2_A005_VALID,
            CF_SM03_K2_A005_VAR_ODD,
        ),
        (0.0, 0.0, 0.01, CF_S0_K0_A001_VALID, CF_S0_K0_A001_VAR_ODD),
    ];
    for (s, k, a, valid, var) in rows {
        assert_eq!(cornish_fisher_valid(s, k, a).unwrap(), valid == 1.0);
        close(
            cornish_fisher_var(SIG, s, k, a, MEAN, 1.0, true).unwrap(),
            var,
            "CF",
        );
    }
}

#[test]
fn case_b_parametric() {
    let (w, cov) = case_b();
    close(portfolio_sigma(&w, &cov).unwrap(), B_SIGMA, "B sigma");
    let rows = [
        (
            0.01,
            B_NORMAL_VAR_A001,
            B_NORMAL_VAR_10D_A001,
            B_NORMAL_ES_A001,
            B_T5_VAR_A001,
            B_T5_ES_A001,
        ),
        (
            0.025,
            B_NORMAL_VAR_A0025,
            B_NORMAL_VAR_10D_A0025,
            B_NORMAL_ES_A0025,
            B_T5_VAR_A0025,
            B_T5_ES_A0025,
        ),
    ];
    for (a, var, var10, es, tv, te) in rows {
        let n1 = var_covar(&w, &cov, a, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
        close(n1.var, var, "B N VaR");
        close(n1.es, es, "B N ES");
        let n10 = var_covar(&w, &cov, a, 10.0, TailDist::Normal, 6.0, 0.0).unwrap();
        close(n10.var, var10, "B N VaR 10d");
        let t5 = var_covar(&w, &cov, a, 1.0, TailDist::StudentT, 5.0, 0.0).unwrap();
        close(t5.var, tv, "B t5 VaR");
        close(t5.es, te, "B t5 ES");
    }
}

#[test]
fn historical_kth_worst_exact() {
    let p = conv_sample();
    let rows = [
        (0.004, HIST_KTH_VAR_A0004, HIST_KTH_ES_A0004),
        (0.01, HIST_KTH_VAR_A001, HIST_KTH_ES_A001),
        (0.025, HIST_KTH_VAR_A0025, HIST_KTH_ES_A0025),
        (0.05, HIST_KTH_VAR_A005, HIST_KTH_ES_A005),
        (0.1, HIST_KTH_VAR_A01, HIST_KTH_ES_A01),
    ];
    for (a, var, es) in rows {
        let (v, e) = empirical_var_es(&p, a, None).unwrap();
        assert_eq!(v.to_bits(), var.to_bits(), "VaR {a}");
        close_rel(e, es, &format!("ES {a}"), 1e-12);
    }
    assert_eq!(HIST_KTH_RANK_A001, 3.0); // MAR99: the 3rd worst of 250
}

macro_rules! rows4 {
    ($($n:expr, $x:expr, $a:expr, $lr:ident, $p:ident;)*) => { [$(($n, $x, $a, $lr, $p)),*] };
}
macro_rules! rows3 {
    ($($n:expr, $x:expr, $a:expr, $c:ident;)*) => { [$(($n, $x, $a, $c)),*] };
}

#[test]
fn kupiec_christoffersen_basel() {
    let ks = rows4! {
        250, 0, 0.01, KUPIEC_LR_N250_X0_A001, KUPIEC_P_N250_X0_A001;
        250, 1, 0.01, KUPIEC_LR_N250_X1_A001, KUPIEC_P_N250_X1_A001;
        250, 2, 0.01, KUPIEC_LR_N250_X2_A001, KUPIEC_P_N250_X2_A001;
        250, 3, 0.01, KUPIEC_LR_N250_X3_A001, KUPIEC_P_N250_X3_A001;
        250, 4, 0.01, KUPIEC_LR_N250_X4_A001, KUPIEC_P_N250_X4_A001;
        250, 5, 0.01, KUPIEC_LR_N250_X5_A001, KUPIEC_P_N250_X5_A001;
        250, 6, 0.01, KUPIEC_LR_N250_X6_A001, KUPIEC_P_N250_X6_A001;
        250, 7, 0.01, KUPIEC_LR_N250_X7_A001, KUPIEC_P_N250_X7_A001;
        250, 8, 0.01, KUPIEC_LR_N250_X8_A001, KUPIEC_P_N250_X8_A001;
        250, 9, 0.01, KUPIEC_LR_N250_X9_A001, KUPIEC_P_N250_X9_A001;
        250, 10, 0.01, KUPIEC_LR_N250_X10_A001, KUPIEC_P_N250_X10_A001;
        250, 11, 0.01, KUPIEC_LR_N250_X11_A001, KUPIEC_P_N250_X11_A001;
        250, 12, 0.01, KUPIEC_LR_N250_X12_A001, KUPIEC_P_N250_X12_A001;
        500, 9, 0.01, KUPIEC_LR_N500_X9_A001, KUPIEC_P_N500_X9_A001;
        1000, 15, 0.01, KUPIEC_LR_N1000_X15_A001, KUPIEC_P_N1000_X15_A001;
        250, 12, 0.025, KUPIEC_LR_N250_X12_A0025, KUPIEC_P_N250_X12_A0025;
        250, 5, 0.05, KUPIEC_LR_N250_X5_A005, KUPIEC_P_N250_X5_A005;
        1000, 50, 0.05, KUPIEC_LR_N1000_X50_A005, KUPIEC_P_N1000_X50_A005;
    };
    for (n, x, a, lr, p) in ks {
        let r = kupiec_pof(&KupiecInput {
            n_obs: n,
            n_exceptions: x,
            alpha: a,
        })
        .unwrap();
        if lr < 1e-300 {
            assert!(r.lr.abs() < 1e-12, "{n}/{x}");
        } else {
            close(r.lr, lr, &format!("kupiec LR {n}/{x}"));
        }
        close(r.p, p, &format!("kupiec p {n}/{x}"));
    }
    let pats = [
        (
            40,
            0,
            CHR_EQ_IND_LR,
            CHR_EQ_IND_P,
            CHR_EQ_CC_LR,
            CHR_EQ_CC_P,
        ),
        (
            37,
            5,
            CHR_FX_IND_LR,
            CHR_FX_IND_P,
            CHR_FX_CC_LR,
            CHR_FX_CC_P,
        ),
    ];
    for (m, rem, il, ip, cl, cp) in pats {
        let ex: Vec<i32> = (0..250)
            .map(|t| i32::from(t % m == rem || t == 100 || t == 101))
            .collect();
        let ind = christoffersen_independence(&ex).unwrap();
        let cc = conditional_coverage(&ex, 0.01).unwrap();
        close(ind.lr, il, "chr ind LR");
        close(ind.p, ip, "chr ind p");
        close(cc.lr, cl, "chr cc LR");
        close(cc.p, cp, "chr cc p");
    }
    let bs = rows3! {
        250, 0, 0.01, BASEL_CDF_N250_X0_A001; 250, 1, 0.01, BASEL_CDF_N250_X1_A001;
        250, 2, 0.01, BASEL_CDF_N250_X2_A001; 250, 3, 0.01, BASEL_CDF_N250_X3_A001;
        250, 4, 0.01, BASEL_CDF_N250_X4_A001; 250, 5, 0.01, BASEL_CDF_N250_X5_A001;
        250, 6, 0.01, BASEL_CDF_N250_X6_A001; 250, 7, 0.01, BASEL_CDF_N250_X7_A001;
        250, 8, 0.01, BASEL_CDF_N250_X8_A001; 250, 9, 0.01, BASEL_CDF_N250_X9_A001;
        250, 10, 0.01, BASEL_CDF_N250_X10_A001; 500, 8, 0.01, BASEL_CDF_N500_X8_A001;
        500, 12, 0.01, BASEL_CDF_N500_X12_A001; 1000, 15, 0.01, BASEL_CDF_N1000_X15_A001;
        250, 12, 0.025, BASEL_CDF_N250_X12_A0025;
    };
    for (n, x, a, c) in bs {
        let r = basel_traffic_light(&BaselInput {
            n_obs: n,
            n_exceptions: x,
            alpha: a,
        })
        .unwrap();
        close(r.cumulative_prob, c, &format!("basel {n}/{x}"));
    }
}

/// Monte Carlo VaR of the Gaussian case-B book (linear P&L `w . r` on the
/// simulated factor returns) converges to the exact quantile:
/// |MC - exact| <= 3 SE, with the bootstrap SE of the same scenarios.
#[test]
fn monte_carlo_gaussian_book_within_3_se() {
    let (w, cov) = case_b();
    let fc = FactorCov {
        factors: vec!["FX:EUR".into(), "FX:JPY".into(), "IR:USD".into()],
        cov,
    };
    for seed in 1..=3u64 {
        for (dist, exact) in [
            (McDist::Normal, B_NORMAL_VAR_A001),
            (McDist::StudentT, B_T5_VAR_A001),
        ] {
            let (r, _) =
                simulate_factor_returns(&fc, 200_000, dist, 5.0, &JumpSpec::default(), seed, 1.0)
                    .unwrap();
            let pnl: Vec<f64> = (0..r.data.rows())
                .map(|s| (0..3).map(|j| w[j] * r.data.get(s, j)).sum())
                .collect();
            let var = empirical_var(&pnl, 0.01, None).unwrap();
            let se = var_standard_error_bootstrap(&pnl, 0.01, 200, seed + 99).unwrap();
            assert!(
                (var - exact).abs() <= 3.0 * se,
                "seed {seed} {dist:?}: mc {var} exact {exact} se {se}"
            );
        }
    }
}
