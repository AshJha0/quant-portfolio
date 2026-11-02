/// \file thread_pool.hpp
/// \brief The fixed-size worker pool, now qp::ThreadPool in cpp/core
///        (qp/thread_pool.hpp). These names keep the eqopt:: API.

#pragma once

#include "qp/thread_pool.hpp"

namespace eqopt {

using qp::default_thread_pool;
using qp::ThreadPool;

}  // namespace eqopt
