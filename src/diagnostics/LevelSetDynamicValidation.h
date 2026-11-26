#pragma once

#include <iosfwd>

namespace fluidlab::diagnostics {

int runLevelSetDynamicValidation(std::ostream& output);
int runLevelSetRedistanceBenchmark(std::ostream& output);
int runLevelSetParticleConsistencyValidation(std::ostream& output);
int runGhostFluidDynamicCpuValidation(std::ostream& output);

} // namespace fluidlab::diagnostics
