#pragma once

#include <filesystem>
#include <iosfwd>

struct GLFWwindow;

namespace fluidlab::diagnostics {

int runGpuLevelSetValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuLevelSetDynamicValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuLevelSetRollbackValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory, GLFWwindow* window);
int runGpuLevelSetBenchmark(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuLevelSetAreaAudit(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuLevelSetAdvectionCompare(
    std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuLevelSetLongValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuLevelSetRedistanceValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuLevelSetExtensionValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuLevelSetConservationCompare(
    std::ostream& output, const std::filesystem::path& shaderDirectory);
int runVofMacReplayValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory);

} // namespace fluidlab::diagnostics
