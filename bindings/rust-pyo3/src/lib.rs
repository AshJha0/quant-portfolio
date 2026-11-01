//! PyO3 bindings over the quant-portfolio Rust engines.
//!
//! Functions are prefixed by engine: `eq_` (equity options, equity VaR) and
//! `fx_` (FX options, FX VaR). Batch functions read C-contiguous float64
//! arrays in place, write new arrays, and release the GIL while the engine
//! runs. Every engine error becomes a Python `ValueError` carrying the
//! engine's message.

use std::borrow::Cow;
use std::fmt::Display;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use eq_options_engine as eqo;
use eq_var_engine as eqv;
use fx_options_engine as fxo;
use fx_var_engine as fxv;

fn verr<E: Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn slice<'a>(a: &'a PyReadonlyArray1<'_, f64>) -> Cow<'a, [f64]> {
    match a.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(a.as_array().to_vec()),
    }
}

fn bools(a: &PyReadonlyArray1<'_, bool>) -> Vec<bool> {
    a.as_array().to_vec()
}

fn check_len(n: usize, m: usize, name: &str) -> PyResult<()> {
    if n == m {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "length mismatch: {name} has {m} elements, expected {n}"
        )))
    }
}

fn parse_call(s: &str) -> PyResult<bool> {
    match s {
        "call" => Ok(true),
        "put" => Ok(false),
        _ => Err(PyValueError::new_err(format!(
            "option_type must be 'call' or 'put', got '{s}'"
        ))),
    }
}

fn matrix_rows(a: &PyReadonlyArray2<'_, f64>) -> (usize, usize, Vec<f64>) {
    let v = a.as_array();
    let (r, c) = v.dim();
    (r, c, v.iter().copied().collect())
}

// ------------------------------------------------------------- equity options

fn eq_ot(call: bool) -> eqo::OptionType {
    if call {
        eqo::OptionType::Call
    } else {
        eqo::OptionType::Put
    }
}

fn eq_batch_err(e: eqo::fast::BatchError) -> PyErr {
    PyValueError::new_err(format!("element {}: {}", e.index, e.code))
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r, sigma, q=0.0, option_type="call"))]
#[allow(non_snake_case)]
fn eq_bs_price(
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
) -> PyResult<f64> {
    let ot = eq_ot(parse_call(option_type)?);
    eqo::fast::try_bs_price(S, K, T, r, sigma, q, ot).map_err(verr)
}

fn eq_greeks_dict<'py>(py: Python<'py>, g: &eqo::BsGreeks) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("price", g.price)?;
    d.set_item("delta", g.delta)?;
    d.set_item("gamma", g.gamma)?;
    d.set_item("vega", g.vega)?;
    d.set_item("theta", g.theta)?;
    d.set_item("rho", g.rho)?;
    d.set_item("vanna", g.vanna)?;
    d.set_item("volga", g.volga)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r, sigma, q=0.0, option_type="call"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn eq_bs_greeks<'py>(
    py: Python<'py>,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ot = eq_ot(parse_call(option_type)?);
    let g = eqo::fast::try_bs_greeks(S, K, T, r, sigma, q, ot).map_err(verr)?;
    eq_greeks_dict(py, &g)
}

struct EqCols<'a> {
    s: Cow<'a, [f64]>,
    k: Cow<'a, [f64]>,
    t: Cow<'a, [f64]>,
    r: Cow<'a, [f64]>,
    sigma: Cow<'a, [f64]>,
    q: Cow<'a, [f64]>,
    types: Vec<eqo::OptionType>,
}

#[allow(clippy::too_many_arguments)]
fn eq_cols<'a>(
    s: &'a PyReadonlyArray1<'_, f64>,
    k: &'a PyReadonlyArray1<'_, f64>,
    t: &'a PyReadonlyArray1<'_, f64>,
    r: &'a PyReadonlyArray1<'_, f64>,
    sigma: &'a PyReadonlyArray1<'_, f64>,
    q: &'a PyReadonlyArray1<'_, f64>,
    is_call: &PyReadonlyArray1<'_, bool>,
) -> PyResult<EqCols<'a>> {
    let n = s.len();
    check_len(n, k.len(), "K")?;
    check_len(n, t.len(), "T")?;
    check_len(n, r.len(), "r")?;
    check_len(n, sigma.len(), "sigma")?;
    check_len(n, q.len(), "q")?;
    check_len(n, is_call.len(), "is_call")?;
    Ok(EqCols {
        s: slice(s),
        k: slice(k),
        t: slice(t),
        r: slice(r),
        sigma: slice(sigma),
        q: slice(q),
        types: bools(is_call).into_iter().map(eq_ot).collect(),
    })
}

impl EqCols<'_> {
    fn batch(&self) -> eqo::fast::BsBatch<'_> {
        eqo::fast::BsBatch {
            s: &self.s,
            k: &self.k,
            t: &self.t,
            r: &self.r,
            sigma: &self.sigma,
            q: &self.q,
            option_type: &self.types,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r, sigma, q, is_call))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn eq_bs_price_batch<'py>(
    py: Python<'py>,
    S: PyReadonlyArray1<'py, f64>,
    K: PyReadonlyArray1<'py, f64>,
    T: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cols = eq_cols(&S, &K, &T, &r, &sigma, &q, &is_call)?;
    let mut out = vec![0.0; S.len()];
    py.detach(|| eqo::fast::bs_price_batch(&cols.batch(), &mut out))
        .map_err(eq_batch_err)?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r, sigma, q, is_call))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn eq_bs_greeks_batch<'py>(
    py: Python<'py>,
    S: PyReadonlyArray1<'py, f64>,
    K: PyReadonlyArray1<'py, f64>,
    T: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyDict>> {
    let cols = eq_cols(&S, &K, &T, &r, &sigma, &q, &is_call)?;
    let n = S.len();
    let mut o: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; n]);
    py.detach(|| {
        let [p, de, ga, ve, th, rh, va, vo] = &mut o;
        let mut out = eqo::fast::GreeksOut {
            price: p,
            delta: de,
            gamma: ga,
            vega: ve,
            theta: th,
            rho: rh,
            vanna: va,
            volga: vo,
        };
        eqo::fast::bs_greeks_batch(&cols.batch(), &mut out)
    })
    .map_err(eq_batch_err)?;
    let d = PyDict::new(py);
    let names = [
        "price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga",
    ];
    for (name, v) in names.iter().zip(o) {
        d.set_item(name, PyArray1::from_vec(py, v))?;
    }
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (price, S, K, T, r, q=0.0, option_type="call"))]
#[allow(non_snake_case)]
fn eq_implied_vol(
    price: f64,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    q: f64,
    option_type: &str,
) -> PyResult<f64> {
    let ot = eq_ot(parse_call(option_type)?);
    eqo::fast::try_implied_vol(price, S, K, T, r, q, ot).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (price, S, K, T, r, q, is_call))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn eq_implied_vol_batch<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    S: PyReadonlyArray1<'py, f64>,
    K: PyReadonlyArray1<'py, f64>,
    T: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // The price column takes the place of sigma in the shared column check.
    let cols = eq_cols(&S, &K, &T, &r, &price, &q, &is_call)?;
    let mut out = vec![0.0; S.len()];
    py.detach(|| -> Result<(), eqo::fast::BatchError> {
        for (index, o) in out.iter_mut().enumerate() {
            *o = eqo::fast::try_implied_vol(
                cols.sigma[index],
                cols.s[index],
                cols.k[index],
                cols.t[index],
                cols.r[index],
                cols.q[index],
                cols.types[index],
            )
            .map_err(|code| eqo::fast::BatchError { index, code })?;
        }
        Ok(())
    })
    .map_err(eq_batch_err)?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r, sigma, q=0.0, option_type="call", *, american=true, n_steps=500))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn eq_crr_price(
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
    american: bool,
    n_steps: usize,
) -> PyResult<f64> {
    let ot = eq_ot(parse_call(option_type)?);
    let ex = if american {
        eqo::Exercise::American
    } else {
        eqo::Exercise::European
    };
    let mut ws = eqo::CrrWorkspace::new();
    eqo::fast::try_crr_price(S, K, T, r, sigma, q, ot, ex, n_steps, &mut ws).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r, sigma, q=0.0, option_type="call", *, n_paths=100_000,
                    antithetic=true, control_variate=true, seed=42, threads=1))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn eq_mc_price<'py>(
    py: Python<'py>,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
    n_paths: usize,
    antithetic: bool,
    control_variate: bool,
    seed: u64,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ot = eq_ot(parse_call(option_type)?);
    if threads == 0 {
        // The Rust engine treats 0 as 1; the C++ engine rejects it. The
        // bindings expose one rule: threads must be >= 1.
        return Err(PyValueError::new_err("threads must be >= 1, got 0"));
    }
    let res = py
        .detach(|| {
            let mut ws = eqo::McWorkspace::new();
            eqo::fast::try_mc_price(
                S,
                K,
                T,
                r,
                sigma,
                q,
                ot,
                n_paths,
                antithetic,
                control_variate,
                seed,
                threads,
                &mut ws,
            )
        })
        .map_err(verr)?;
    let d = PyDict::new(py);
    d.set_item("price", res.price)?;
    d.set_item("std_error", res.std_error)?;
    d.set_item("ci_low", res.ci_low)?;
    d.set_item("ci_high", res.ci_high)?;
    d.set_item("n_paths", res.n_paths)?;
    Ok(d)
}

// ----------------------------------------------------------------- FX options

fn fx_ot(call: bool) -> fxo::OptionType {
    if call {
        fxo::OptionType::Call
    } else {
        fxo::OptionType::Put
    }
}

fn fx_batch_err(e: fxo::fast::BatchError) -> PyErr {
    PyValueError::new_err(format!("element {}: {}", e.index, e.error))
}

const FX_GREEKS: [&str; 10] = [
    "price",
    "delta_spot",
    "delta_forward",
    "gamma",
    "vega",
    "theta",
    "rho_domestic",
    "rho_foreign",
    "vanna",
    "volga",
];

#[pyfunction]
#[pyo3(signature = (S, K, T, r_d, r_f, sigma, option_type))]
#[allow(non_snake_case)]
fn fx_gk_price(
    S: f64,
    K: f64,
    T: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: &str,
) -> PyResult<f64> {
    let ot = fx_ot(parse_call(option_type)?);
    fxo::fast::gk_price_k(S, K, T, r_d, r_f, sigma, ot).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r_d, r_f, sigma, option_type))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn fx_gk_greeks<'py>(
    py: Python<'py>,
    S: f64,
    K: f64,
    T: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ot = fx_ot(parse_call(option_type)?);
    let g = fxo::fast::greeks_k(S, K, T, r_d, r_f, sigma, ot).map_err(verr)?;
    let v = [
        g.price,
        g.delta_spot,
        g.delta_forward,
        g.gamma,
        g.vega,
        g.theta,
        g.rho_domestic,
        g.rho_foreign,
        g.vanna,
        g.volga,
    ];
    let d = PyDict::new(py);
    for (name, x) in FX_GREEKS.iter().zip(v) {
        d.set_item(name, x)?;
    }
    Ok(d)
}

struct FxCols<'a> {
    s: Cow<'a, [f64]>,
    k: Cow<'a, [f64]>,
    t: Cow<'a, [f64]>,
    r_d: Cow<'a, [f64]>,
    r_f: Cow<'a, [f64]>,
    sigma: Cow<'a, [f64]>,
    types: Vec<fxo::OptionType>,
}

#[allow(clippy::too_many_arguments)]
fn fx_cols<'a>(
    s: &'a PyReadonlyArray1<'_, f64>,
    k: &'a PyReadonlyArray1<'_, f64>,
    t: &'a PyReadonlyArray1<'_, f64>,
    r_d: &'a PyReadonlyArray1<'_, f64>,
    r_f: &'a PyReadonlyArray1<'_, f64>,
    sigma: &'a PyReadonlyArray1<'_, f64>,
    is_call: &PyReadonlyArray1<'_, bool>,
) -> PyResult<FxCols<'a>> {
    let n = s.len();
    check_len(n, k.len(), "K")?;
    check_len(n, t.len(), "T")?;
    check_len(n, r_d.len(), "r_d")?;
    check_len(n, r_f.len(), "r_f")?;
    check_len(n, sigma.len(), "sigma")?;
    check_len(n, is_call.len(), "is_call")?;
    Ok(FxCols {
        s: slice(s),
        k: slice(k),
        t: slice(t),
        r_d: slice(r_d),
        r_f: slice(r_f),
        sigma: slice(sigma),
        types: bools(is_call).into_iter().map(fx_ot).collect(),
    })
}

impl FxCols<'_> {
    fn batch(&self) -> fxo::fast::GkBatch<'_> {
        fxo::fast::GkBatch {
            s: &self.s,
            k: &self.k,
            t: &self.t,
            r_d: &self.r_d,
            r_f: &self.r_f,
            sigma: &self.sigma,
            option_type: &self.types,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r_d, r_f, sigma, is_call))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn fx_gk_price_batch<'py>(
    py: Python<'py>,
    S: PyReadonlyArray1<'py, f64>,
    K: PyReadonlyArray1<'py, f64>,
    T: PyReadonlyArray1<'py, f64>,
    r_d: PyReadonlyArray1<'py, f64>,
    r_f: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cols = fx_cols(&S, &K, &T, &r_d, &r_f, &sigma, &is_call)?;
    let mut out = vec![0.0; S.len()];
    py.detach(|| fxo::fast::gk_price_batch(&cols.batch(), &mut out))
        .map_err(fx_batch_err)?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r_d, r_f, sigma, is_call))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn fx_gk_greeks_batch<'py>(
    py: Python<'py>,
    S: PyReadonlyArray1<'py, f64>,
    K: PyReadonlyArray1<'py, f64>,
    T: PyReadonlyArray1<'py, f64>,
    r_d: PyReadonlyArray1<'py, f64>,
    r_f: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyDict>> {
    let cols = fx_cols(&S, &K, &T, &r_d, &r_f, &sigma, &is_call)?;
    let n = S.len();
    let mut o: [Vec<f64>; 10] = std::array::from_fn(|_| vec![0.0; n]);
    py.detach(|| {
        let [p, ds, df, ga, ve, th, rd, rf, va, vo] = &mut o;
        let mut out = fxo::fast::GreeksBatchOut {
            price: p,
            delta_spot: ds,
            delta_forward: df,
            gamma: ga,
            vega: ve,
            theta: th,
            rho_domestic: rd,
            rho_foreign: rf,
            vanna: va,
            volga: vo,
        };
        fxo::fast::greeks_batch(&cols.batch(), &mut out)
    })
    .map_err(fx_batch_err)?;
    let d = PyDict::new(py);
    for (name, v) in FX_GREEKS.iter().zip(o) {
        d.set_item(name, PyArray1::from_vec(py, v))?;
    }
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (price, S, K, T, r_d, r_f, option_type))]
#[allow(non_snake_case)]
fn fx_implied_vol(
    price: f64,
    S: f64,
    K: f64,
    T: f64,
    r_d: f64,
    r_f: f64,
    option_type: &str,
) -> PyResult<f64> {
    let ot = fx_ot(parse_call(option_type)?);
    fxo::fast::implied_vol_k(price, S, K, T, r_d, r_f, ot).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (price, S, K, T, r_d, r_f, is_call))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn fx_implied_vol_batch<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    S: PyReadonlyArray1<'py, f64>,
    K: PyReadonlyArray1<'py, f64>,
    T: PyReadonlyArray1<'py, f64>,
    r_d: PyReadonlyArray1<'py, f64>,
    r_f: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // The price column takes the place of sigma in the shared column check.
    let cols = fx_cols(&S, &K, &T, &r_d, &r_f, &price, &is_call)?;
    let mut out = vec![0.0; S.len()];
    py.detach(|| -> Result<(), fxo::fast::BatchError> {
        for (index, o) in out.iter_mut().enumerate() {
            *o = fxo::fast::implied_vol_k(
                cols.sigma[index],
                cols.s[index],
                cols.k[index],
                cols.t[index],
                cols.r_d[index],
                cols.r_f[index],
                cols.types[index],
            )
            .map_err(|error| fxo::fast::BatchError { index, error })?;
        }
        Ok(())
    })
    .map_err(fx_batch_err)?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r_d, r_f, sigma, option_type, *, steps=500, american=true))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn fx_binomial_price(
    S: f64,
    K: f64,
    T: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: &str,
    steps: u32,
    american: bool,
) -> PyResult<f64> {
    let ot = fx_ot(parse_call(option_type)?);
    let ex = if american {
        fxo::Exercise::American
    } else {
        fxo::Exercise::European
    };
    let mut ws = fxo::fast::BinomialWorkspace::new();
    fxo::fast::binomial_price_ws(&mut ws, S, K, T, r_d, r_f, sigma, ot, steps, ex).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (S, K, T, r_d, r_f, sigma, option_type, *, n_paths=100_000, seed=0,
                    antithetic=true, control_variate=true, threads=1))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn fx_mc_price<'py>(
    py: Python<'py>,
    S: f64,
    K: f64,
    T: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: &str,
    n_paths: u64,
    seed: u64,
    antithetic: bool,
    control_variate: bool,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ot = fx_ot(parse_call(option_type)?);
    let res = py
        .detach(|| {
            let mut ws = fxo::fast::McWorkspace::new();
            fxo::fast::mc_price_par(
                &mut ws,
                S,
                K,
                T,
                r_d,
                r_f,
                sigma,
                ot,
                n_paths,
                seed,
                antithetic,
                control_variate,
                threads,
            )
        })
        .map_err(verr)?;
    let d = PyDict::new(py);
    d.set_item("price", res.price)?;
    d.set_item("std_error", res.std_error)?;
    d.set_item("ci_low", res.ci_low)?;
    d.set_item("ci_high", res.ci_high)?;
    d.set_item("n_paths", res.n_paths)?;
    d.set_item("method", res.method)?;
    Ok(d)
}

// ----------------------------------------------------------------- equity VaR

fn eq_tail(dist: &str, df: f64) -> PyResult<eqv::TailModel> {
    match dist {
        "normal" => Ok(eqv::TailModel::Normal),
        "t" => Ok(eqv::TailModel::StudentT { df }),
        _ => Err(PyValueError::new_err(format!(
            "dist must be 'normal' or 't', got '{dist}'"
        ))),
    }
}

fn eq_matrix(cov: &PyReadonlyArray2<'_, f64>) -> PyResult<eqv::matrix::Matrix> {
    let (r, c, data) = matrix_rows(cov);
    eqv::matrix::Matrix::from_vec(r, c, data).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (pnl, *, alpha=0.01))]
fn eq_historical_var(py: Python<'_>, pnl: PyReadonlyArray1<'_, f64>, alpha: f64) -> PyResult<f64> {
    let p = slice(&pnl);
    py.detach(|| eqv::historical::historical_var(&p, alpha))
        .map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (pnl, *, alpha=0.01))]
fn eq_expected_shortfall(
    py: Python<'_>,
    pnl: PyReadonlyArray1<'_, f64>,
    alpha: f64,
) -> PyResult<f64> {
    let p = slice(&pnl);
    py.detach(|| eqv::expected_shortfall::expected_shortfall(&p, alpha))
        .map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (exposures, cov, *, alpha=0.01, dist="normal", df=6.0, mean=0.0, horizon_days=1))]
#[allow(clippy::too_many_arguments)]
fn eq_parametric_var(
    exposures: PyReadonlyArray1<'_, f64>,
    cov: PyReadonlyArray2<'_, f64>,
    alpha: f64,
    dist: &str,
    df: f64,
    mean: f64,
    horizon_days: u32,
) -> PyResult<f64> {
    let m = eq_matrix(&cov)?;
    let tail = eq_tail(dist, df)?;
    eqv::parametric::parametric_var_full(&slice(&exposures), &m, alpha, tail, mean, horizon_days)
        .map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (exposures, cov, *, alpha=0.01, n_paths=100_000, dist="normal", df=6.0, seed=0))]
#[allow(clippy::too_many_arguments)]
fn eq_monte_carlo_var(
    py: Python<'_>,
    exposures: PyReadonlyArray1<'_, f64>,
    cov: PyReadonlyArray2<'_, f64>,
    alpha: f64,
    n_paths: usize,
    dist: &str,
    df: f64,
    seed: u64,
) -> PyResult<f64> {
    let m = eq_matrix(&cov)?;
    let tail = eq_tail(dist, df)?;
    let e = slice(&exposures);
    py.detach(|| eqv::monte_carlo::monte_carlo_var(&e, &m, alpha, n_paths, tail, seed))
        .map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (*, n_obs, n_exceptions, alpha=0.01))]
fn eq_kupiec_pof<'py>(
    py: Python<'py>,
    n_obs: i64,
    n_exceptions: i64,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let k = eqv::backtest::kupiec_pof(&eqv::backtest::KupiecInput {
        n_obs,
        n_exceptions,
        alpha,
    })
    .map_err(verr)?;
    let d = PyDict::new(py);
    d.set_item("lr", k.lr)?;
    d.set_item("pvalue", k.pvalue)?;
    d.set_item("expected", k.expected)?;
    d.set_item("rate", k.rate)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (*, n_obs, n_exceptions, alpha=0.01))]
fn eq_basel_traffic_light<'py>(
    py: Python<'py>,
    n_obs: i64,
    n_exceptions: i64,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let b = eqv::backtest::basel_traffic_light(&eqv::backtest::BaselInput {
        n_obs,
        n_exceptions,
        alpha,
    })
    .map_err(verr)?;
    let d = PyDict::new(py);
    d.set_item("zone", b.zone.to_string())?;
    d.set_item("multiplier", b.multiplier)?;
    d.set_item("plus_factor", b.plus_factor)?;
    d.set_item("cumulative_prob", b.cumulative_prob)?;
    Ok(d)
}

// --------------------------------------------------------------------- FX VaR

fn fx_tail(dist: &str) -> PyResult<fxv::TailDist> {
    match dist {
        "normal" => Ok(fxv::TailDist::Normal),
        "t" => Ok(fxv::TailDist::StudentT),
        _ => Err(PyValueError::new_err(format!(
            "dist must be 'normal' or 't', got '{dist}'"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (pnl, *, alpha=0.01))]
fn fx_empirical_var_es(
    py: Python<'_>,
    pnl: PyReadonlyArray1<'_, f64>,
    alpha: f64,
) -> PyResult<(f64, f64)> {
    let p = slice(&pnl);
    py.detach(|| fxv::expected_shortfall::empirical_var_es(&p, alpha, None))
        .map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (sigma, *, alpha=0.01, mean=0.0))]
fn fx_normal_var(sigma: f64, alpha: f64, mean: f64) -> PyResult<f64> {
    fxv::expected_shortfall::normal_var(sigma, alpha, mean).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (sigma, *, alpha=0.01, mean=0.0))]
fn fx_normal_es(sigma: f64, alpha: f64, mean: f64) -> PyResult<f64> {
    fxv::expected_shortfall::normal_es(sigma, alpha, mean).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (sigma, *, alpha=0.01, df=6.0, mean=0.0))]
fn fx_t_var(sigma: f64, alpha: f64, df: f64, mean: f64) -> PyResult<f64> {
    fxv::expected_shortfall::student_t_var(sigma, alpha, df, mean).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (sigma, *, alpha=0.01, df=6.0, mean=0.0))]
fn fx_t_es(sigma: f64, alpha: f64, df: f64, mean: f64) -> PyResult<f64> {
    fxv::expected_shortfall::student_t_es(sigma, alpha, df, mean).map_err(verr)
}

#[pyfunction]
#[pyo3(signature = (exposures, cov, *, alpha=0.01, horizon_days=1.0, dist="normal", df=6.0, mean=0.0))]
#[allow(clippy::too_many_arguments)]
fn fx_var_covar(
    exposures: PyReadonlyArray1<'_, f64>,
    cov: PyReadonlyArray2<'_, f64>,
    alpha: f64,
    horizon_days: f64,
    dist: &str,
    df: f64,
    mean: f64,
) -> PyResult<(f64, f64)> {
    let (r, c, data) = matrix_rows(&cov);
    let mut m = fxv::matrix::Matrix::zeros(r, c);
    for i in 0..r {
        for j in 0..c {
            m.set(i, j, data[i * c + j]);
        }
    }
    let v = fxv::parametric::var_covar(
        &slice(&exposures),
        &m,
        alpha,
        horizon_days,
        fx_tail(dist)?,
        df,
        mean,
    )
    .map_err(verr)?;
    Ok((v.var, v.es))
}

#[pyfunction]
#[pyo3(signature = (*, n_obs, n_exceptions, alpha=0.01))]
fn fx_kupiec_pof(n_obs: i64, n_exceptions: i64, alpha: f64) -> PyResult<(f64, f64)> {
    let k = fxv::backtest::kupiec_pof(&fxv::backtest::KupiecInput {
        n_obs,
        n_exceptions,
        alpha,
    })
    .map_err(verr)?;
    Ok((k.lr, k.p))
}

#[pyfunction]
#[pyo3(signature = (*, n_obs, n_exceptions, alpha=0.01))]
fn fx_basel_traffic_light<'py>(
    py: Python<'py>,
    n_obs: i64,
    n_exceptions: i64,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let b = fxv::backtest::basel_traffic_light(&fxv::backtest::BaselInput {
        n_obs,
        n_exceptions,
        alpha,
    })
    .map_err(verr)?;
    let zone = match b.zone {
        fxv::backtest::Zone::Green => "green",
        fxv::backtest::Zone::Yellow => "yellow",
        _ => "red",
    };
    let d = PyDict::new(py);
    d.set_item("zone", zone)?;
    d.set_item("n_exceptions", b.n_exceptions)?;
    d.set_item("n_obs", b.n_obs)?;
    d.set_item("cumulative_prob", b.cumulative_prob)?;
    d.set_item("multiplier", b.multiplier)?;
    d.set_item("plus_factor", b.plus_factor)?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "PyO3 bindings over the quant-portfolio Rust engines",
    )?;
    m.add_function(wrap_pyfunction!(eq_bs_price, m)?)?;
    m.add_function(wrap_pyfunction!(eq_bs_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(eq_bs_price_batch, m)?)?;
    m.add_function(wrap_pyfunction!(eq_bs_greeks_batch, m)?)?;
    m.add_function(wrap_pyfunction!(eq_implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(eq_implied_vol_batch, m)?)?;
    m.add_function(wrap_pyfunction!(eq_crr_price, m)?)?;
    m.add_function(wrap_pyfunction!(eq_mc_price, m)?)?;
    m.add_function(wrap_pyfunction!(fx_gk_price, m)?)?;
    m.add_function(wrap_pyfunction!(fx_gk_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(fx_gk_price_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fx_gk_greeks_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fx_implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(fx_implied_vol_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fx_binomial_price, m)?)?;
    m.add_function(wrap_pyfunction!(fx_mc_price, m)?)?;
    m.add_function(wrap_pyfunction!(eq_historical_var, m)?)?;
    m.add_function(wrap_pyfunction!(eq_expected_shortfall, m)?)?;
    m.add_function(wrap_pyfunction!(eq_parametric_var, m)?)?;
    m.add_function(wrap_pyfunction!(eq_monte_carlo_var, m)?)?;
    m.add_function(wrap_pyfunction!(eq_kupiec_pof, m)?)?;
    m.add_function(wrap_pyfunction!(eq_basel_traffic_light, m)?)?;
    m.add_function(wrap_pyfunction!(fx_empirical_var_es, m)?)?;
    m.add_function(wrap_pyfunction!(fx_normal_var, m)?)?;
    m.add_function(wrap_pyfunction!(fx_normal_es, m)?)?;
    m.add_function(wrap_pyfunction!(fx_t_var, m)?)?;
    m.add_function(wrap_pyfunction!(fx_t_es, m)?)?;
    m.add_function(wrap_pyfunction!(fx_var_covar, m)?)?;
    m.add_function(wrap_pyfunction!(fx_kupiec_pof, m)?)?;
    m.add_function(wrap_pyfunction!(fx_basel_traffic_light, m)?)?;
    Ok(())
}
