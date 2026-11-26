#pragma once

#include <filesystem>
#include <iosfwd>

namespace fluidlab::diagnostics {

int runGpuVofFluxValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofPlicValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofSplitValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofRollbackValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofBenchmark(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofMacDiagnostic(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofPrecisionAudit(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofMultistepValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofBoundedFluxValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofMacReplay(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofVelocityCompatibility(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofMultistepBenchmark(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofRoundoffAudit(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofCommitSafetyValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofConservativeRepairValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofBandVelocityValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofLongValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofDynamicCompare(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuVofSafetyBenchmark(std::ostream& output, const std::filesystem::path& shaderDirectory);

} // namespace fluidlab::diagnostics
