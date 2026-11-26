#pragma once

#include "../simulation/SimulationStepInput.h"

#include <cstdint>
#include <deque>
#include <vector>

namespace fluidlab::desktop {

struct ContainerMotionSettings final {
    double pixelsPerSimulationUnit = 600.0;
    double minimumSampleInterval = 1.0e-4;
    double maximumSampleInterval = 0.125;
    double staleSampleInterval = 0.25;
    double velocitySmoothingTime = 0.018;
    double accelerationSmoothingTime = 0.0;
    double maximumAcceleration = 30.0;
    double maximumPositionJumpPixels = 1600.0;
    double maximumQueuedTime = 0.35;
    bool horizontalOnly = true;
};

struct ContainerMotionState final {
    fluidlab::simulation::Vector2 position{};
    fluidlab::simulation::Vector2 rawVelocity{};
    fluidlab::simulation::Vector2 velocity{};
    fluidlab::simulation::Vector2 rawAcceleration{};
    fluidlab::simulation::Vector2 acceleration{};
    double sampleTime = 0.0;
    std::uint64_t version = 0;
    bool valid = false;
    bool discontinuity = false;
    bool accelerationLimited = false;
};

struct ContainerMotionSegment final {
    double startTime = 0.0;
    double endTime = 0.0;
    fluidlab::simulation::Vector2 startVelocity{};
    fluidlab::simulation::Vector2 endVelocity{};
    fluidlab::simulation::Vector2 externalAcceleration{};
    fluidlab::simulation::Vector2 externalVelocityDelta{};
    std::uint64_t version = 0;
    bool accelerationLimited = false;
};

struct ContainerMotionDiagnostics final {
    std::size_t pendingSegments = 0;
    double pendingTime = 0.0;
    double oldestSampleAge = 0.0;
    std::uint64_t preparedVersion = 0;
    std::uint64_t committedVersion = 0;
    std::size_t generatedSegments = 0;
    std::size_t committedSegments = 0;
    std::size_t droppedSegments = 0;
    std::size_t limitedSegments = 0;
    fluidlab::simulation::Vector2 generatedVelocityDelta{};
    fluidlab::simulation::Vector2 committedVelocityDelta{};
    fluidlab::simulation::Vector2 droppedVelocityDelta{};
    fluidlab::simulation::Vector2 pendingVelocityDelta{};
    fluidlab::simulation::Vector2 preparedVelocityDelta{};
    double preparedSegmentTime = 0.0;
    double frontConsumedTime = 0.0;
};

class ContainerMotionInput final {
public:
    explicit ContainerMotionInput(ContainerMotionSettings settings = {});

    [[nodiscard]] ContainerMotionState addPositionSample(
        double xPixels,
        double yPixels,
        double sampleTimeSeconds) noexcept;
    [[nodiscard]] fluidlab::simulation::SimulationStepInput sample(
        double sampleTimeSeconds,
        bool enabled = true) const noexcept;
    [[nodiscard]] fluidlab::simulation::SimulationStepInput prepareStep(
        double sampleTimeSeconds,
        double physicalDurationSeconds,
        bool enabled = true) noexcept;
    [[nodiscard]] bool commitPreparedStep(std::uint64_t version) noexcept;
    [[nodiscard]] bool commitPreparedFraction(
        std::uint64_t version,
        double completedFraction) noexcept;
    [[nodiscard]] double maximumContinuousPreparedDuration() const noexcept;
    void discardPreparedStep() noexcept;
    void reset() noexcept;

    [[nodiscard]] const ContainerMotionSettings& settings() const noexcept;
    [[nodiscard]] const ContainerMotionState& state() const noexcept;
    [[nodiscard]] ContainerMotionDiagnostics diagnostics(double nowSeconds) const noexcept;
    [[nodiscard]] const std::deque<ContainerMotionSegment>& pendingSegments() const noexcept;

private:
    struct PreparedConsumption final {
        std::uint64_t version = 0;
        double consumedSegmentTime = 0.0;
        fluidlab::simulation::Vector2 velocityDelta{};
        bool active = false;
    };

    void trimQueue() noexcept;

    ContainerMotionSettings m_settings;
    ContainerMotionState m_state;
    std::deque<ContainerMotionSegment> m_segments;
    double m_frontConsumedTime = 0.0;
    PreparedConsumption m_prepared;
    ContainerMotionDiagnostics m_diagnostics;
};

} // namespace fluidlab::desktop
