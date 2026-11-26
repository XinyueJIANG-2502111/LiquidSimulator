#pragma once

#include <cstddef>
#include <vector>

namespace fluidlab::desktop {

struct DesktopFrameTiming final {
    double frameMilliseconds = 0.0;
    double eventMilliseconds = 0.0;
    double physicsSubmitMilliseconds = 0.0;
    double fencePollMilliseconds = 0.0;
    double renderMilliseconds = 0.0;
    double swapMilliseconds = 0.0;
    double committedLagSeconds = 0.0;
    double physicsRate = 0.0;
    double physicsBacklogSeconds = 0.0;
    double snapshotPublishMilliseconds = 0.0;
    double snapshotPresentationMilliseconds = 0.0;
    std::size_t committedOuterSteps = 0;
    bool moveSizeFrame = false;
    bool repeatedSnapshot = false;
};

struct DesktopPerformanceSummary final {
    std::size_t sampleCount = 0;
    double frameP50Milliseconds = 0.0;
    double frameP95Milliseconds = 0.0;
    double frameP99Milliseconds = 0.0;
    double frameMaximumMilliseconds = 0.0;
    double averageEventMilliseconds = 0.0;
    double averagePhysicsSubmitMilliseconds = 0.0;
    double averageFencePollMilliseconds = 0.0;
    double averageRenderMilliseconds = 0.0;
    double averageSwapMilliseconds = 0.0;
    double averageCommittedLagSeconds = 0.0;
    double averagePhysicsRate = 0.0;
    double averagePhysicsBacklogSeconds = 0.0;
    double maximumPhysicsBacklogSeconds = 0.0;
    double averageSnapshotPublishMilliseconds = 0.0;
    double averageSnapshotPresentationMilliseconds = 0.0;
    double maximumRepeatedSnapshotMilliseconds = 0.0;
    std::size_t committedOuterSteps = 0;
    std::size_t over33Milliseconds = 0;
    std::size_t over50Milliseconds = 0;
    std::size_t over100Milliseconds = 0;
    std::size_t moveSizeFrames = 0;
    std::size_t repeatedSnapshotFrames = 0;
};

class DesktopPerformanceMetrics final {
public:
    explicit DesktopPerformanceMetrics(std::size_t capacity = 600);
    void add(const DesktopFrameTiming& timing) noexcept;
    void clear() noexcept;
    [[nodiscard]] DesktopPerformanceSummary summary() const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;

private:
    std::vector<DesktopFrameTiming> m_samples;
    std::size_t m_next = 0;
    std::size_t m_size = 0;
};

class ModalFrameScheduler final {
public:
    void enter() noexcept;
    void exit() noexcept;
    [[nodiscard]] bool inMoveSize() const noexcept;
    [[nodiscard]] bool beginFrame() noexcept;
    void endFrame() noexcept;
    [[nodiscard]] std::size_t skippedReentrantFrames() const noexcept;

private:
    bool m_inMoveSize = false;
    bool m_frameRunning = false;
    std::size_t m_skippedReentrantFrames = 0;
};

struct DesktopPhysicsClockSnapshot final {
    double wallElapsed = 0.0;
    double targetSimulationTime = 0.0;
    double committedSimulationTime = 0.0;
    double physicsBacklog = 0.0;
    double physicsRate = 0.0;
    std::size_t availableOuterSteps = 0;
};

class DesktopPhysicsClock final {
public:
    explicit DesktopPhysicsClock(double outerTimeStep = 1.0 / 120.0);
    void reset(double wallTime, double simulationTime) noexcept;
    void update(double wallTime, double committedSimulationTime, bool paused) noexcept;
    [[nodiscard]] std::size_t planBatch(
        std::size_t maximumBatch,
        bool candidateInFlight,
        bool continuingOuterStep) const noexcept;
    [[nodiscard]] DesktopPhysicsClockSnapshot snapshot() const noexcept;

private:
    double m_outerTimeStep = 1.0 / 120.0;
    double m_lastWallTime = 0.0;
    double m_wallElapsed = 0.0;
    double m_targetSimulationTime = 0.0;
    double m_committedSimulationTime = 0.0;
    double m_initialSimulationTime = 0.0;
    bool m_initialized = false;
};

} // namespace fluidlab::desktop
