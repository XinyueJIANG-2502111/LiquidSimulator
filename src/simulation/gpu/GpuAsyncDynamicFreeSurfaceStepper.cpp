#include "GpuAsyncDynamicFreeSurfaceStepper.h"

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <utility>

namespace fluidlab::simulation {
namespace {

using Clock = std::chrono::steady_clock;

GpuDynamicFreeSurfaceSettings asynchronousSettings(
    GpuDynamicFreeSurfaceSettings settings) noexcept {
    settings.synchronousGpuTimingEnabled = false;
    settings.asynchronousGpuTimingEnabled = false;
    return settings;
}

} // namespace

GpuAsyncDynamicFreeSurfaceStepper::GpuAsyncDynamicFreeSurfaceStepper(
    GLFWwindow* sharingWindow,
    const std::filesystem::path& divergenceShaderPath,
    GpuDynamicFreeSurfaceSettings settings)
    : m_stepper(divergenceShaderPath, asynchronousSettings(settings)) {
    if (sharingWindow == nullptr) {
        throw std::invalid_argument("GPU asynchronous stepping requires a sharing window.");
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    m_workerWindow = glfwCreateWindow(
        1, 1, "LiquidSimulator GPU worker", nullptr, sharingWindow);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    if (m_workerWindow == nullptr) {
        throw std::runtime_error("Failed to create the shared GPU worker context.");
    }
    m_worker = std::thread(&GpuAsyncDynamicFreeSurfaceStepper::workerLoop, this);
    m_statistics.workerThreadStartCount = 1;
}

GpuAsyncDynamicFreeSurfaceStepper::~GpuAsyncDynamicFreeSurfaceStepper() {
    waitForShutdown();
    if (m_fence != nullptr) {
        glDeleteSync(static_cast<GLsync>(m_fence));
        m_fence = nullptr;
    }
    if (m_workerWindow != nullptr) {
        glfwDestroyWindow(m_workerWindow);
        m_workerWindow = nullptr;
    }
}

GpuDynamicTopologyReport GpuAsyncDynamicFreeSurfaceStepper::initialize(
    const MacGrid& grid,
    const MarkerParticleSystem& particles,
    const LevelSetField* initialLevelSet) {
    if (inFlight()) {
        throw std::logic_error("Cannot initialize while a GPU candidate is in flight.");
    }
    const auto topology = m_stepper.initialize(grid, particles, initialLevelSet);
    m_stepper.grid().dispatchDivergence();
    m_stepper.grid().captureCommittedRenderState();
    m_stepper.grid().setCommittedRenderStateEnabled(true);
    m_report = {};
    m_report.topology = topology;
    m_state = GpuAsyncCandidateState::idle;
    return topology;
}

bool GpuAsyncDynamicFreeSurfaceStepper::submit() {
    return submit({});
}

bool GpuAsyncDynamicFreeSurfaceStepper::submit(const SimulationStepInput& input) {
    return submitBatch(input, 1);
}

bool GpuAsyncDynamicFreeSurfaceStepper::submitBatch(
    const SimulationStepInput& input,
    std::size_t outerStepCount) {
    if (!isFinite(input)) return false;
    if (outerStepCount == 0 || outerStepCount > 4) return false;
    if (inFlight() || m_state == GpuAsyncCandidateState::failed ||
        m_state == GpuAsyncCandidateState::canceled) {
        return false;
    }
    if (m_fence != nullptr) {
        glDeleteSync(static_cast<GLsync>(m_fence));
        m_fence = nullptr;
    }
    m_cancelRequested = false;
    m_workerFinished = false;
    m_workerException = nullptr;
    m_state = GpuAsyncCandidateState::prepared;
    const auto submitStart = Clock::now();
    ++m_statistics.candidateId;
    ++m_statistics.submittedCount;
    m_submissionTime = submitStart;
    m_state = GpuAsyncCandidateState::submitted;
    {
        const std::scoped_lock lock(m_mutex);
        if (m_stopRequested || m_taskPending ||
            (!m_workerFinished && m_fence != nullptr)) {
            return false;
        }

        // Copy the task while holding the mailbox lock. The worker never reads
        // mutable Desktop input or main-thread state directly.
        m_pendingInput = input;
        m_pendingOuterStepCount = outerStepCount;
        m_taskPending = true;
    }
    m_workerCondition.notify_one();
    m_state = GpuAsyncCandidateState::inFlight;
    m_statistics.lastSubmissionMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - submitStart).count();
    m_statistics.lastBatchSize = outerStepCount;
    return true;
}

bool GpuAsyncDynamicFreeSurfaceStepper::poll() {
    if (m_state != GpuAsyncCandidateState::inFlight) return false;
    GLsync fence = nullptr;
    Clock::time_point workerCompletionTime{};
    {
        const std::scoped_lock lock(m_mutex);
        fence = static_cast<GLsync>(m_fence);
        workerCompletionTime = m_workerCompletionTime;
        if (fence == nullptr && m_workerFinished && m_workerException != nullptr) {
            m_state = GpuAsyncCandidateState::failed;
            ++m_statistics.failedCount;
            m_report.failureReason = GpuDynamicFailureReason::pressureFailure;
            m_report.failureMessage = "GPU worker stopped before creating a completion fence.";
            releaseCompletedFence();
            return true;
        }
    }
    ++m_statistics.fencePollCount;
    if (fence == nullptr) {
        ++m_statistics.incompleteFencePollCount;
        return false;
    }
    const auto pollStart = Clock::now();
    const GLenum waitResult = glClientWaitSync(fence, 0, 0);
    m_statistics.fencePollingMilliseconds +=
        std::chrono::duration<double, std::milli>(Clock::now() - pollStart).count();
    if (waitResult == GL_TIMEOUT_EXPIRED) {
        ++m_statistics.incompleteFencePollCount;
        return false;
    }
    if (waitResult == GL_WAIT_FAILED) {
        m_state = GpuAsyncCandidateState::failed;
        ++m_statistics.failedCount;
        m_report.failureReason = GpuDynamicFailureReason::pressureFailure;
        m_report.failureMessage = "OpenGL fence polling failed for the GPU candidate.";
        releaseCompletedFence();
        return true;
    }

    m_state = GpuAsyncCandidateState::readyForValidation;
    releaseCompletedFence();
    m_statistics.lastInFlightMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - m_submissionTime).count();
    if (m_workerException != nullptr) {
        m_state = GpuAsyncCandidateState::failed;
        ++m_statistics.failedCount;
        m_report.failureReason = GpuDynamicFailureReason::pressureFailure;
        try {
            std::rethrow_exception(m_workerException);
        } catch (const std::exception& error) {
            m_report.failureMessage = error.what();
        } catch (...) {
            m_report.failureMessage = "Unknown GPU worker failure.";
        }
        return true;
    }

    m_report = std::move(m_workerReport);
    if (m_cancelRequested) {
        m_state = GpuAsyncCandidateState::canceled;
        ++m_statistics.canceledCount;
        return true;
    }
    const auto publishStart = Clock::now();

    // Rendering consumes only a committed snapshot. Candidate buffers remain
    // private to the worker until validation and the completion fence succeed.
    m_stepper.grid().captureCommittedRenderState();
    m_statistics.lastSnapshotPublishMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - publishStart).count();
    m_statistics.lastWorkerCompletionToPublishMilliseconds = workerCompletionTime == Clock::time_point{}
        ? 0.0 : std::chrono::duration<double, std::milli>(
            publishStart - workerCompletionTime).count();
    if (m_report.failureReason != GpuDynamicFailureReason::none) {
        m_state = GpuAsyncCandidateState::failed;
        ++m_statistics.failedCount;
    } else {
        m_state = GpuAsyncCandidateState::accepted;
        ++m_statistics.acceptedCount;
    }
    return true;
}

void GpuAsyncDynamicFreeSurfaceStepper::acknowledge() noexcept {
    if (m_state == GpuAsyncCandidateState::accepted ||
        m_state == GpuAsyncCandidateState::rejectedRetry) {
        m_state = GpuAsyncCandidateState::idle;
    }
}

void GpuAsyncDynamicFreeSurfaceStepper::requestCancel() noexcept {
    if (inFlight()) m_cancelRequested = true;
    else m_state = GpuAsyncCandidateState::canceled;
}

void GpuAsyncDynamicFreeSurfaceStepper::waitForShutdown() noexcept {
    {
        const std::scoped_lock lock(m_mutex);
        m_stopRequested = true;
    }
    m_workerCondition.notify_one();
    if (m_worker.joinable()) m_worker.join();
    if (m_fence != nullptr) {
        static_cast<void>(glClientWaitSync(
            static_cast<GLsync>(m_fence), GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED));
    }
}

GpuAsyncCandidateState GpuAsyncDynamicFreeSurfaceStepper::state() const noexcept {
    return m_state;
}

bool GpuAsyncDynamicFreeSurfaceStepper::inFlight() const noexcept {
    return m_state == GpuAsyncCandidateState::prepared ||
        m_state == GpuAsyncCandidateState::submitted ||
        m_state == GpuAsyncCandidateState::inFlight ||
        m_state == GpuAsyncCandidateState::readyForValidation;
}

bool GpuAsyncDynamicFreeSurfaceStepper::hasCompletedResult() const noexcept {
    return m_state == GpuAsyncCandidateState::accepted ||
        m_state == GpuAsyncCandidateState::failed ||
        m_state == GpuAsyncCandidateState::canceled;
}

const GpuDynamicFreeSurfaceReport& GpuAsyncDynamicFreeSurfaceStepper::report() const noexcept {
    return m_report;
}

GpuAsyncCandidateStatistics
GpuAsyncDynamicFreeSurfaceStepper::statistics() const noexcept {
    const std::scoped_lock lock(m_mutex);
    return m_statistics;
}

GpuDynamicFreeSurfaceStepper& GpuAsyncDynamicFreeSurfaceStepper::stepper() noexcept {
    return m_stepper;
}

const GpuDynamicFreeSurfaceStepper&
GpuAsyncDynamicFreeSurfaceStepper::stepper() const noexcept {
    return m_stepper;
}

void GpuAsyncDynamicFreeSurfaceStepper::workerLoop() noexcept {
    glfwMakeContextCurrent(m_workerWindow);
    for (;;) {
        SimulationStepInput taskInput;
        std::size_t requestedOuterSteps = 1;
        {
            std::unique_lock lock(m_mutex);
            const auto idleStart = Clock::now();
            m_workerCondition.wait(lock, [this] { return m_taskPending || m_stopRequested; });
            m_statistics.workerIdleMilliseconds +=
                std::chrono::duration<double, std::milli>(Clock::now() - idleStart).count();
            if (m_stopRequested && !m_taskPending) break;
            taskInput = m_pendingInput;
            requestedOuterSteps = m_pendingOuterStepCount;
            m_taskPending = false;
            m_statistics.lastMailboxLatencyMilliseconds =
                std::chrono::duration<double, std::milli>(Clock::now() - m_submissionTime).count();
        }

        GpuDynamicFreeSurfaceReport workerReport;
        std::exception_ptr workerException;
        GLsync fence = nullptr;
        const auto busyStart = Clock::now();
        try {
            const std::uint64_t firstOuterStep = m_stepper.completedOuterSteps();
            std::size_t calls = 0;
            do {
                workerReport = m_stepper.step(taskInput);
                ++calls;
                if (workerReport.failureReason != GpuDynamicFailureReason::none) break;
                if (calls > requestedOuterSteps * 1024) {
                    throw std::runtime_error("GPU asynchronous batch exceeded its continuation budget.");
                }
            } while (m_stepper.completedOuterSteps() - firstOuterStep < requestedOuterSteps);
            workerReport.batchOuterStepsRequested = requestedOuterSteps;
            workerReport.batchOuterStepsCompleted = static_cast<std::size_t>(
                m_stepper.completedOuterSteps() - firstOuterStep);
            fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            if (fence == nullptr) {
                throw std::runtime_error("Failed to create the GPU candidate completion fence.");
            }
        } catch (...) {
            workerException = std::current_exception();
            fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }
        glFlush();
        const auto completionTime = Clock::now();
        const double busyMilliseconds = std::chrono::duration<double, std::milli>(
            completionTime - busyStart).count();
        const std::size_t completedOuterSteps = workerReport.batchOuterStepsCompleted;
        {
            const std::scoped_lock lock(m_mutex);
            m_workerReport = std::move(workerReport);
            m_workerException = workerException;
            m_fence = fence;
            m_workerFinished = true;
            m_workerCompletionTime = completionTime;
            m_statistics.lastWorkerBusyMilliseconds = busyMilliseconds;
            m_statistics.workerBusyMilliseconds += busyMilliseconds;
            m_statistics.lastBatchCompleted = completedOuterSteps;
            ++m_statistics.completedTaskCount;
        }
    }
    glfwMakeContextCurrent(nullptr);
    {
        const std::scoped_lock lock(m_mutex);
        m_workerStopped = true;
    }
}

void GpuAsyncDynamicFreeSurfaceStepper::releaseCompletedFence() noexcept {
    if (m_fence != nullptr) {
        glDeleteSync(static_cast<GLsync>(m_fence));
        m_fence = nullptr;
    }
}

const char* gpuAsyncCandidateStateName(GpuAsyncCandidateState state) noexcept {
    switch (state) {
    case GpuAsyncCandidateState::idle: return "Idle";
    case GpuAsyncCandidateState::prepared: return "Prepared";
    case GpuAsyncCandidateState::submitted: return "Submitted";
    case GpuAsyncCandidateState::inFlight: return "InFlight";
    case GpuAsyncCandidateState::readyForValidation: return "ReadyForValidation";
    case GpuAsyncCandidateState::accepted: return "Accepted";
    case GpuAsyncCandidateState::rejectedRetry: return "RejectedRetry";
    case GpuAsyncCandidateState::failed: return "Failed";
    case GpuAsyncCandidateState::canceled: return "Canceled";
    }
    return "Unknown";
}

} // namespace fluidlab::simulation
