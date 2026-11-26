#pragma once

#include <filesystem>
#include <iosfwd>

struct GLFWwindow;

namespace fluidlab::diagnostics {

int runContainerMotionValidation(std::ostream& output);
int runExternalForceValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    GLFWwindow* sharingWindow);
int runContainerInertiaValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runGpuFluidInjectionValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runDesktopInputTimingValidation(std::ostream& output);
int runDesktopInteractionBenchmark(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory);
int runDesktopFramePacingValidation(std::ostream& output);
int runDesktopModalLoopValidation(std::ostream& output, GLFWwindow* window);
int runDesktopSettleGenerate(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& outputPath);
int runDesktopSettleLoadValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath);
int runDesktopInertiaFromSettledValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath);
int runDesktopInertiaPipelineAudit(std::ostream& output);
int runDesktopInertiaTimingValidation(std::ostream& output);
int runDesktopInertiaResponseValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath);
int runDesktopInertiaLatencyBenchmark(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath);
int runDesktopSchedulerValidation(std::ostream& output);
int runDesktopAsyncThroughputValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    GLFWwindow* sharingWindow);
int runDesktopBatchInputAudit(std::ostream& output);
int runDesktopThroughputAccountingValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath,
    GLFWwindow* sharingWindow);
int runDesktopSustainedRealtimeBenchmark(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath,
    GLFWwindow* sharingWindow);

} // namespace fluidlab::diagnostics
