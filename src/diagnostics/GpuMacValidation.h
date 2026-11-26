#pragma once

#include <filesystem>
#include <iosfwd>

namespace fluidlab::diagnostics {

int runGpuMacValidation(std::ostream& output, const std::filesystem::path& shaderDirectory);
int runGpuPressureOptimizationValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuMultigridValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuDynamicValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuDynamicTopologyValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuResidentPressureValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuMixedPressureValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuSolverBenchmark(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuMgpcgProfile(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuMixedReliabilityValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuMixedLongValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuCoarseFusionRegression(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuInterfaceAccuracyValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);

} // namespace fluidlab::diagnostics
