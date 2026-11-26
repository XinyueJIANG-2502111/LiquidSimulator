#pragma once

#include <iosfwd>

namespace fluidlab::diagnostics {

int runInterfaceReconstructionValidation(std::ostream& output);
int runGhostFluidCpuValidation(std::ostream& output);

} // namespace fluidlab::diagnostics
