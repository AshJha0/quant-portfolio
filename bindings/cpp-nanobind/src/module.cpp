// Module entry point: one extension, four engine submodule-free namespaces
// (functions are prefixed eq_ / fx_ instead).
#include "common.hpp"

namespace qpb {
void bind_eqopt(nb::module_& m);
void bind_fxopt(nb::module_& m);
void bind_eqvar(nb::module_& m);
void bind_fxvar(nb::module_& m);
}  // namespace qpb

NB_MODULE(_core, m) {
    m.doc() = "nanobind bindings over the quant-portfolio C++ engines";
    qpb::bind_eqopt(m);
    qpb::bind_fxopt(m);
    qpb::bind_eqvar(m);
    qpb::bind_fxvar(m);
}
