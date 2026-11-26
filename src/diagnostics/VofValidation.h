#pragma once

#include <iosfwd>

namespace fluidlab::diagnostics {

int runVofGeometryValidation(std::ostream& output);
int runVofConservationValidation(std::ostream& output);
int runVofAdvectionValidation(std::ostream& output);
int runClsvofCouplingValidation(std::ostream& output);
int runClsvofBenchmark(std::ostream& output);
int runVofFluxAudit(std::ostream& output);
int runVofShapeConvergenceValidation(std::ostream& output);
int runVofHeightFunctionValidation(std::ostream& output);
int runClsvofGeometryCouplingValidation(std::ostream& output);
int runPlicGhostFluidValidation(std::ostream& output);
int runVofSplitConservationValidation(std::ostream& output);
int runVofSplitBoundednessValidation(std::ostream& output);
int runVofSplitShapeValidation(std::ostream& output);
int runVofSplitBenchmark(std::ostream& output);

} // namespace fluidlab::diagnostics
