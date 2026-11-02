// Implied-volatility core: normalised Black solved on the OTM side. The
// implementation is qp::iv_core in cpp/core (qp/iv_core.hpp), shared by
// both options engines; these names keep the eqopt::iv_core API.

#pragma once

#include "qp/iv_core.hpp"

namespace eqopt::iv_core {

using qp::iv_core::black_implied_vol;
using qp::iv_core::black_implied_vol_nx;
using qp::iv_core::IvStatus;
using qp::iv_core::kMaxIter;
using qp::iv_core::normalised_otm_call;
using qp::iv_core::normalised_otm_call_complement;
using qp::iv_core::total_vol;
using qp::iv_core::total_vol_nx;

}  // namespace eqopt::iv_core
