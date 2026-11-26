#pragma once

#include "GpuDynamicFreeSurfaceStepper.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <mutex>
#include <thread>

struct GLFWwindow;

namespace fluidlab::simulation {

enum class GpuAsyncCandidateState {
    idle,
    prepared,
    submitted,
    inFlight,
    readyForValidation,
    accepted,
    rejectedRetry,
    failed,
    canceled,
};

struct GpuAsyncCandidateStatistics final {
    std::uint64_t candidateId = 0;
    std::size_t submittedCount = 0;
    std::size_t acceptedCount = 0;
    std::size_t failedCount = 0;
    std::size_t canceledCount = 0;
    std::size_t fencePollCount = 0;
    std::size_t incompleteFencePollCount = 0;
    double fencePollingMilliseconds = 0.0;
    double lastInFlightMilliseconds = 0.0;
    double lastSubmissionMilliseconds = 0.0;
    double lastMailboxLatencyMilliseconds = 0.0;
    double lastWorkerBusyMilliseconds = 0.0;
    double workerBusyMilliseconds = 0.0;
    double workerIdleMilliseconds = 0.0;
    double lastSnapshotPublishMilliseconds = 0.0;
    double lastWorkerCompletionToPublishMilliseconds = 0.0;
    std::size_t lastBatchSize = 1;
    std::size_t lastBatchCompleted = 0;
    std::uint64_t workerThreadStartCount = 0;
    std::uint64_t completedTaskCount = 0;
};

class GpuAsyncDynamicFreeSurfaceStepper final {
public:
    GpuAsyncDynamicFreeSurfaceStepper(
        GLFWwindow* sharingWindow,
        const std::filesystem::path& divergenceShaderPath,
        GpuDynamicFreeSurfaceSettings settings = {});
    ~GpuAsyncDynamicFreeSurfaceStepper();

    GpuAsyncDynamicFreeSurfaceStepper(const GpuAsyncDynamicFreeSurfaceStepper&) = delete;
    GpuAsyncDynamicFreeSurfaceStepper& operator=(
        const GpuAsyncDynamicFreeSurfaceStepper&) = delete;

    [[nodiscard]] GpuDynamicTopologyReport initialize(
        const MacGrid& grid,
        const MarkerParticleSystem& particles,
        const LevelSetField* initialLevelSet = nullptr);
    [[nodiscard]] bool submit();
    [[nodiscard]] bool submit(const SimulationStepInput& input);
    [[nodiscard]] bool submitBatch(
        const SimulationStepInput& input,
        std::size_t outerStepCount);
    [[nodiscard]] bool poll();
    void acknowledge() noexcept;
    void requestCancel() noexcept;
    void waitForShutdown() noexcept;

    [[nodiscard]] GpuAsyncCandidateState state() const noexcept;
    [[nodiscard]] bool inFlight() const noexcept;
    [[nodiscard]] bool hasCompletedResult() const noexcept;
    [[nodiscard]] const GpuDynamicFreeSurfaceReport& report() const noexcept;
    [[nodiscard]] GpuAsyncCandidateStatistics statistics() const noexcept;
    [[nodiscard]] GpuDynamicFreeSurfaceStepper& stepper() noexcept;
    [[nodiscard]] const GpuDynamicFreeSurfaceStepper& stepper() const noexcept;

private:
    void workerLoop() noexcept;
    void releaseCompletedFence() noexcept;

    GLFWwindow* m_workerWindow = nullptr;
    GpuDynamicFreeSurfaceStepper m_stepper;
    GpuDynamicFreeSurfaceReport m_report;
    GpuDynamicFreeSurfaceReport m_workerReport;
    GpuAsyncCandidateStatistics m_statistics;
    GpuAsyncCandidateState m_state = GpuAsyncCandidateState::idle;
    std::thread m_worker;
    mutable std::mutex m_mutex;
    std::condition_variable m_workerCondition;
    void* m_fence = nullptr;
    std::exception_ptr m_workerException;
    bool m_cancelRequested = false;
    bool m_workerFinished = false;
    bool m_taskPending = false;
    bool m_stopRequested = false;
    bool m_workerStopped = false;
    std::chrono::steady_clock::time_point m_submissionTime{};
    std::chrono::steady_clock::time_point m_workerCompletionTime{};
    SimulationStepInput m_pendingInput;
    std::size_t m_pendingOuterStepCount = 1;
};

[[nodiscard]] const char* gpuAsyncCandidateStateName(
    GpuAsyncCandidateState state) noexcept;

} // namespace fluidlab::simulation
