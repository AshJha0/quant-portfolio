// Brent root finder: qp::brentq in cpp/core (qp/brent.hpp). This name keeps
// the internal fxopt::detail::brentq that tests and older code refer to.

#pragma once

#include "qp/brent.hpp"

namespace fxopt::detail {

using qp::brentq;

}  // namespace fxopt::detail
