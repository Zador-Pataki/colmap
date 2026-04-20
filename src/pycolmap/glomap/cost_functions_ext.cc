// §13 — cost-functor Python factory bindings.
//
// Empty by design. Per `research/04-mpsfm-usage/api-contracts.md` (and a
// `grep -rn 'pyglomap\.\(make_\|[A-Z][a-zA-Z]*Error\|[A-Z][a-zA-Z]*Cost\)'
// mpsfm/` sweep, 2026-04-20), mpsfm does NOT instantiate any fork cost
// functor directly from Python. All cost-functor instantiations happen
// inside C++ via the estimator classes (GlobalPositioner, RotationEstimator,
// BundleAdjuster) which are driven from Python through the §12
// `pycolmap.glomap.run_*` pipeline functions.
//
// mpsfm's own Ceres problems use `pyceres.CostFunction` subclasses defined
// in-python (see IntrinsicsRandomWalkCostFunction, ScalePriorCostFunction,
// PinholePriorCostFunction, SimplePinholePriorCostFunction,
// ScaleRegularizerCostFunction in mpsfm/utils/mpsfm.py). These don't need
// any glomap binding.
//
// If future mpsfm code ever constructs a fork functor from Python (e.g.
// to build an auxiliary problem that uses BATAPairwiseDirectionError),
// add an `m.def("make_..._cost", ...)` factory here.

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindGlomapCostFunctions(py::module& /*m*/) {
  // No-op. See comment above.
}
