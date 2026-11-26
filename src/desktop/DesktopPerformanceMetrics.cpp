#include "DesktopPerformanceMetrics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fluidlab::desktop {
namespace {

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

} // namespace

DesktopPerformanceMetrics::DesktopPerformanceMetrics(std::size_t capacity)
    : m_samples(capacity) {
    if (capacity == 0) throw std::invalid_argument("Frame metric capacity must be positive.");
}

void DesktopPerformanceMetrics::add(const DesktopFrameTiming& timing) noexcept {
    m_samples[m_next] = timing;
    m_next = (m_next + 1) % m_samples.size();
    m_size = std::min(m_size + 1, m_samples.size());
}

void DesktopPerformanceMetrics::clear() noexcept { m_next = m_size = 0; }

DesktopPerformanceSummary DesktopPerformanceMetrics::summary() const {
    DesktopPerformanceSummary result;
    result.sampleCount = m_size;
    if (m_size == 0) return result;
    std::vector<double> frames;
    frames.reserve(m_size);
    double repeatedRunMilliseconds = 0.0;
    for (std::size_t i = 0; i < m_size; ++i) {
        const std::size_t index = m_size == m_samples.size()
            ? (m_next + i) % m_samples.size() : i;
        const DesktopFrameTiming& sample = m_samples[index];
        frames.push_back(sample.frameMilliseconds);
        result.frameMaximumMilliseconds = std::max(
            result.frameMaximumMilliseconds, sample.frameMilliseconds);
        result.averageEventMilliseconds += sample.eventMilliseconds;
        result.averagePhysicsSubmitMilliseconds += sample.physicsSubmitMilliseconds;
        result.averageFencePollMilliseconds += sample.fencePollMilliseconds;
        result.averageRenderMilliseconds += sample.renderMilliseconds;
        result.averageSwapMilliseconds += sample.swapMilliseconds;
        result.averageCommittedLagSeconds += sample.committedLagSeconds;
        result.averagePhysicsRate += sample.physicsRate;
        result.averagePhysicsBacklogSeconds += sample.physicsBacklogSeconds;
        result.maximumPhysicsBacklogSeconds = std::max(
            result.maximumPhysicsBacklogSeconds, sample.physicsBacklogSeconds);
        result.averageSnapshotPublishMilliseconds += sample.snapshotPublishMilliseconds;
        result.averageSnapshotPresentationMilliseconds +=
            sample.snapshotPresentationMilliseconds;
        result.committedOuterSteps = std::max(
            result.committedOuterSteps, sample.committedOuterSteps);
        result.over33Milliseconds += sample.frameMilliseconds > 33.3 ? 1u : 0u;
        result.over50Milliseconds += sample.frameMilliseconds > 50.0 ? 1u : 0u;
        result.over100Milliseconds += sample.frameMilliseconds > 100.0 ? 1u : 0u;
        result.moveSizeFrames += sample.moveSizeFrame ? 1u : 0u;
        result.repeatedSnapshotFrames += sample.repeatedSnapshot ? 1u : 0u;
        if (sample.repeatedSnapshot) {
            repeatedRunMilliseconds += sample.frameMilliseconds;
            result.maximumRepeatedSnapshotMilliseconds = std::max(
                result.maximumRepeatedSnapshotMilliseconds, repeatedRunMilliseconds);
        } else {
            repeatedRunMilliseconds = 0.0;
        }
    }
    const double denominator = static_cast<double>(m_size);
    result.averageEventMilliseconds /= denominator;
    result.averagePhysicsSubmitMilliseconds /= denominator;
    result.averageFencePollMilliseconds /= denominator;
    result.averageRenderMilliseconds /= denominator;
    result.averageSwapMilliseconds /= denominator;
    result.averageCommittedLagSeconds /= denominator;
    result.averagePhysicsRate /= denominator;
    result.averagePhysicsBacklogSeconds /= denominator;
    result.averageSnapshotPublishMilliseconds /= denominator;
    result.averageSnapshotPresentationMilliseconds /= denominator;
    result.frameP50Milliseconds = percentile(frames, 0.50);
    result.frameP95Milliseconds = percentile(frames, 0.95);
    result.frameP99Milliseconds = percentile(frames, 0.99);
    return result;
}

std::size_t DesktopPerformanceMetrics::size() const noexcept { return m_size; }
std::size_t DesktopPerformanceMetrics::capacity() const noexcept { return m_samples.size(); }

void ModalFrameScheduler::enter() noexcept { m_inMoveSize = true; }
void ModalFrameScheduler::exit() noexcept { m_inMoveSize = false; }
bool ModalFrameScheduler::inMoveSize() const noexcept { return m_inMoveSize; }
bool ModalFrameScheduler::beginFrame() noexcept {
    if (m_frameRunning) {
        ++m_skippedReentrantFrames;
        return false;
    }
    m_frameRunning = true;
    return true;
}
void ModalFrameScheduler::endFrame() noexcept { m_frameRunning = false; }
std::size_t ModalFrameScheduler::skippedReentrantFrames() const noexcept {
    return m_skippedReentrantFrames;
}

DesktopPhysicsClock::DesktopPhysicsClock(double outerTimeStep)
    : m_outerTimeStep(outerTimeStep) {
    if (!std::isfinite(outerTimeStep) || outerTimeStep <= 0.0) {
        throw std::invalid_argument("Desktop physics clock step must be positive and finite.");
    }
}

void DesktopPhysicsClock::reset(double wallTime, double simulationTime) noexcept {
    m_lastWallTime = std::isfinite(wallTime) ? wallTime : 0.0;
    m_wallElapsed = 0.0;
    m_targetSimulationTime = std::isfinite(simulationTime) ? simulationTime : 0.0;
    m_committedSimulationTime = m_targetSimulationTime;
    m_initialSimulationTime = m_targetSimulationTime;
    m_initialized = true;
}

void DesktopPhysicsClock::update(
    double wallTime,
    double committedSimulationTime,
    bool paused) noexcept {
    if (!m_initialized) {
        reset(wallTime, committedSimulationTime);
        return;
    }
    if (!std::isfinite(wallTime) || !std::isfinite(committedSimulationTime)) return;
    const double elapsed = std::max(0.0, wallTime - m_lastWallTime);
    m_lastWallTime = wallTime;
    m_committedSimulationTime = committedSimulationTime;
    if (!paused) {
        m_wallElapsed += elapsed;
        m_targetSimulationTime += elapsed;
    }
}

std::size_t DesktopPhysicsClock::planBatch(
    std::size_t maximumBatch,
    bool candidateInFlight,
    bool continuingOuterStep) const noexcept {
    if (maximumBatch == 0 || candidateInFlight) return 0;
    if (continuingOuterStep) return 1;
    const double backlog = std::max(0.0,
        m_targetSimulationTime - m_committedSimulationTime);
    const auto available = static_cast<std::size_t>(std::floor(
        (backlog + m_outerTimeStep * 1.0e-6) / m_outerTimeStep));
    return std::min(maximumBatch, available);
}

DesktopPhysicsClockSnapshot DesktopPhysicsClock::snapshot() const noexcept {
    DesktopPhysicsClockSnapshot result;
    result.wallElapsed = m_wallElapsed;
    result.targetSimulationTime = m_targetSimulationTime;
    result.committedSimulationTime = m_committedSimulationTime;
    result.physicsBacklog = std::max(0.0,
        m_targetSimulationTime - m_committedSimulationTime);
    result.physicsRate = m_wallElapsed > 0.0
        ? (m_committedSimulationTime - m_initialSimulationTime) / m_wallElapsed : 0.0;
    result.availableOuterSteps = static_cast<std::size_t>(std::floor(
        (result.physicsBacklog + m_outerTimeStep * 1.0e-6) / m_outerTimeStep));
    return result;
}

} // namespace fluidlab::desktop
