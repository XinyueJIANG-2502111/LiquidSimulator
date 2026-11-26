#include "ContainerMotionInput.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fluidlab::desktop {
namespace {

using fluidlab::simulation::Vector2;

[[nodiscard]] Vector2 add(Vector2 a, Vector2 b) noexcept {
    return {a.x + b.x, a.y + b.y};
}

[[nodiscard]] Vector2 subtract(Vector2 a, Vector2 b) noexcept {
    return {a.x - b.x, a.y - b.y};
}

[[nodiscard]] Vector2 scale(Vector2 value, double factor) noexcept {
    return {
        static_cast<float>(static_cast<double>(value.x) * factor),
        static_cast<float>(static_cast<double>(value.y) * factor)};
}

[[nodiscard]] Vector2 clampMagnitude(Vector2 value, double maximum) noexcept {
    const double magnitude = std::hypot(
        static_cast<double>(value.x), static_cast<double>(value.y));
    if (magnitude <= maximum || magnitude == 0.0) return value;
    return scale(value, maximum / magnitude);
}

[[nodiscard]] double smoothingWeight(double dt, double timeConstant) noexcept {
    if (timeConstant <= 0.0) return 1.0;
    return 1.0 - std::exp(-dt / timeConstant);
}

} // namespace

ContainerMotionInput::ContainerMotionInput(ContainerMotionSettings settings)
    : m_settings(settings) {
    if (!std::isfinite(settings.pixelsPerSimulationUnit) ||
        settings.pixelsPerSimulationUnit <= 0.0 ||
        !std::isfinite(settings.minimumSampleInterval) ||
        settings.minimumSampleInterval <= 0.0 ||
        !std::isfinite(settings.maximumSampleInterval) ||
        settings.maximumSampleInterval <= settings.minimumSampleInterval ||
        !std::isfinite(settings.staleSampleInterval) ||
        settings.staleSampleInterval < settings.maximumSampleInterval ||
        !std::isfinite(settings.velocitySmoothingTime) ||
        settings.velocitySmoothingTime < 0.0 ||
        !std::isfinite(settings.accelerationSmoothingTime) ||
        settings.accelerationSmoothingTime < 0.0 ||
        !std::isfinite(settings.maximumAcceleration) ||
        settings.maximumAcceleration <= 0.0 ||
        !std::isfinite(settings.maximumPositionJumpPixels) ||
        settings.maximumPositionJumpPixels <= 0.0 ||
        !std::isfinite(settings.maximumQueuedTime) ||
        settings.maximumQueuedTime <= 0.0) {
        throw std::invalid_argument("Invalid container motion settings.");
    }
}

ContainerMotionState ContainerMotionInput::addPositionSample(
    double xPixels,
    double yPixels,
    double sampleTimeSeconds) noexcept {
    if (!std::isfinite(xPixels) || !std::isfinite(yPixels) ||
        !std::isfinite(sampleTimeSeconds)) {
        reset();
        m_state.discontinuity = true;
        return m_state;
    }

    const Vector2 position{
        static_cast<float>(xPixels / m_settings.pixelsPerSimulationUnit),
        static_cast<float>(-yPixels / m_settings.pixelsPerSimulationUnit)};
    if (!m_state.valid) {
        m_state.position = position;
        m_state.sampleTime = sampleTimeSeconds;
        m_state.valid = true;
        m_state.discontinuity = false;
        ++m_state.version;
        return m_state;
    }

    const double dt = sampleTimeSeconds - m_state.sampleTime;
    const Vector2 displacement = subtract(position, m_state.position);
    const double jumpPixels = std::hypot(
        static_cast<double>(displacement.x), static_cast<double>(displacement.y)) *
        m_settings.pixelsPerSimulationUnit;
    if (!std::isfinite(dt) || dt < m_settings.minimumSampleInterval) {
        return m_state;
    }
    if (dt > m_settings.maximumSampleInterval ||
        jumpPixels > m_settings.maximumPositionJumpPixels) {
        // A long gap or position jump is not reliable motion. Start a new sample
        // history instead of converting it into a large artificial impulse.
        discardPreparedStep();
        m_segments.clear();
        m_frontConsumedTime = 0.0;
        m_state.position = position;
        m_state.rawVelocity = {};
        m_state.velocity = {};
        m_state.rawAcceleration = {};
        m_state.acceleration = {};
        m_state.sampleTime = sampleTimeSeconds;
        m_state.discontinuity = true;
        ++m_state.version;
        return m_state;
    }

    const Vector2 previousVelocity = m_state.velocity;
    Vector2 measuredVelocity = scale(displacement, 1.0 / dt);
    const double velocityWeight = smoothingWeight(dt, m_settings.velocitySmoothingTime);
    Vector2 filteredVelocity = add(
        scale(m_state.velocity, 1.0 - velocityWeight),
        scale(measuredVelocity, velocityWeight));
    Vector2 measuredAcceleration = scale(
        subtract(filteredVelocity, m_state.velocity), 1.0 / dt);
    const double accelerationWeight =
        smoothingWeight(dt, m_settings.accelerationSmoothingTime);
    Vector2 filteredAcceleration = add(
        scale(m_state.acceleration, 1.0 - accelerationWeight),
        scale(measuredAcceleration, accelerationWeight));
    if (m_settings.horizontalOnly) {
        measuredVelocity.y = 0.0f;
        filteredVelocity.y = 0.0f;
        measuredAcceleration.y = 0.0f;
        filteredAcceleration.y = 0.0f;
    }
    const Vector2 unclampedAcceleration = filteredAcceleration;
    filteredAcceleration = clampMagnitude(filteredAcceleration, m_settings.maximumAcceleration);
    const bool accelerationLimited =
        std::abs(unclampedAcceleration.x - filteredAcceleration.x) > 1.0e-6f ||
        std::abs(unclampedAcceleration.y - filteredAcceleration.y) > 1.0e-6f;

    ContainerMotionSegment segment;
    segment.startTime = m_state.sampleTime;
    segment.endTime = sampleTimeSeconds;
    segment.startVelocity = previousVelocity;
    segment.endVelocity = filteredVelocity;
    segment.externalAcceleration = {-filteredAcceleration.x, -filteredAcceleration.y};
    segment.externalVelocityDelta = scale(segment.externalAcceleration, dt);
    segment.version = m_state.version + 1;
    segment.accelerationLimited = accelerationLimited;

    // Store ordered velocity changes rather than only the latest acceleration.
    // This keeps short start/stop impulses distinct until physics consumes them.
    const double segmentMagnitude = std::hypot(
        static_cast<double>(segment.externalVelocityDelta.x),
        static_cast<double>(segment.externalVelocityDelta.y));
    if (segmentMagnitude > 1.0e-9) {
        m_segments.push_back(segment);
        ++m_diagnostics.generatedSegments;
        if (accelerationLimited) ++m_diagnostics.limitedSegments;
        m_diagnostics.generatedVelocityDelta = add(
            m_diagnostics.generatedVelocityDelta, segment.externalVelocityDelta);
        trimQueue();
    }

    m_state.position = position;
    m_state.rawVelocity = measuredVelocity;
    m_state.velocity = filteredVelocity;
    m_state.rawAcceleration = measuredAcceleration;
    m_state.acceleration = filteredAcceleration;
    m_state.sampleTime = sampleTimeSeconds;
    m_state.discontinuity = false;
    m_state.accelerationLimited = accelerationLimited;
    ++m_state.version;
    return m_state;
}

fluidlab::simulation::SimulationStepInput ContainerMotionInput::sample(
    double sampleTimeSeconds,
    bool enabled) const noexcept {
    fluidlab::simulation::SimulationStepInput result;
    result.sourceTimeBegin = m_state.sampleTime;
    result.sourceTimeEnd = sampleTimeSeconds;
    result.physicalDuration = std::max(0.0, sampleTimeSeconds - m_state.sampleTime);
    result.version = m_state.version;
    result.valid = std::isfinite(sampleTimeSeconds) &&
        sampleTimeSeconds >= m_state.sampleTime;
    if (!enabled || !m_state.valid || m_state.discontinuity ||
        !result.valid || sampleTimeSeconds - m_state.sampleTime > m_settings.staleSampleInterval) {
        result.externalAcceleration = {};
        return result;
    }
    result.externalAcceleration = {
        -m_state.acceleration.x,
        -m_state.acceleration.y};
    result.externalVelocityDelta = scale(
        result.externalAcceleration, result.physicalDuration);
    return result;
}

fluidlab::simulation::SimulationStepInput ContainerMotionInput::prepareStep(
    double sampleTimeSeconds,
    double physicalDurationSeconds,
    bool enabled) noexcept {
    if (m_prepared.active) {
        // Candidate retries must observe the exact same prepared impulse.
        fluidlab::simulation::SimulationStepInput result;
        result.externalVelocityDelta = m_prepared.velocityDelta;
        result.physicalDuration = physicalDurationSeconds;
        result.externalAcceleration = physicalDurationSeconds > 0.0
            ? scale(m_prepared.velocityDelta, 1.0 / physicalDurationSeconds)
            : Vector2{};
        result.version = m_prepared.version;
        if (!m_segments.empty()) {
            result.sourceTimeBegin = m_segments.front().startTime + m_frontConsumedTime;
            result.sourceTimeEnd = result.sourceTimeBegin + m_prepared.consumedSegmentTime;
        }
        result.valid = std::isfinite(sampleTimeSeconds) &&
            std::isfinite(physicalDurationSeconds) && physicalDurationSeconds > 0.0;
        return result;
    }

    fluidlab::simulation::SimulationStepInput result;
    result.physicalDuration = physicalDurationSeconds;
    result.version = m_state.version;
    result.valid = std::isfinite(sampleTimeSeconds) &&
        std::isfinite(physicalDurationSeconds) && physicalDurationSeconds > 0.0;
    if (!result.valid) return result;

    if (!enabled) {
        while (!m_segments.empty()) {
            m_diagnostics.droppedVelocityDelta = add(
                m_diagnostics.droppedVelocityDelta,
                m_segments.front().externalVelocityDelta);
            ++m_diagnostics.droppedSegments;
            m_segments.pop_front();
        }
        m_frontConsumedTime = 0.0;
        result.sourceTimeBegin = sampleTimeSeconds;
        result.sourceTimeEnd = sampleTimeSeconds;
        m_prepared = {result.version, 0.0, {}, true};
        m_diagnostics.preparedVersion = result.version;
        return result;
    }

    if (m_segments.empty()) {
        result.sourceTimeBegin = sampleTimeSeconds;
        result.sourceTimeEnd = sampleTimeSeconds;
        m_prepared = {result.version, 0.0, {}, true};
        m_diagnostics.preparedVersion = result.version;
        return result;
    }

    const auto& segment = m_segments.front();
    const double segmentDuration = segment.endTime - segment.startTime;
    const double remaining = std::max(0.0, segmentDuration - m_frontConsumedTime);
    const double consumed = std::min(physicalDurationSeconds, remaining);
    const Vector2 delta = scale(segment.externalAcceleration, consumed);

    // Consume at most one segment per transaction. The scheduler may batch
    // outer steps only while they remain inside this continuous time interval.
    result.externalVelocityDelta = delta;
    result.externalAcceleration = scale(delta, 1.0 / physicalDurationSeconds);
    result.sourceTimeBegin = segment.startTime + m_frontConsumedTime;
    result.sourceTimeEnd = result.sourceTimeBegin + consumed;
    result.version = segment.version;
    m_prepared = {segment.version, consumed, delta, true};
    m_diagnostics.preparedVersion = segment.version;
    return result;
}

bool ContainerMotionInput::commitPreparedStep(std::uint64_t version) noexcept {
    return commitPreparedFraction(version, 1.0);
}

bool ContainerMotionInput::commitPreparedFraction(
    std::uint64_t version,
    double completedFraction) noexcept {
    if (!m_prepared.active || m_prepared.version != version) return false;
    if (!std::isfinite(completedFraction) || completedFraction < 0.0 ||
        completedFraction > 1.0) {
        return false;
    }
    const double consumedTime = m_prepared.consumedSegmentTime * completedFraction;
    const Vector2 committedDelta = scale(m_prepared.velocityDelta, completedFraction);

    // Advance the queue only after the matching candidate prefix is committed.
    // A rejected suffix remains available for the next attempt.
    m_diagnostics.committedVelocityDelta = add(
        m_diagnostics.committedVelocityDelta, committedDelta);
    m_diagnostics.committedVersion = version;
    if (!m_segments.empty() && consumedTime > 0.0) {
        m_frontConsumedTime += consumedTime;
        const double duration = m_segments.front().endTime - m_segments.front().startTime;
        if (m_frontConsumedTime + 1.0e-12 >= duration) {
            m_segments.pop_front();
            m_frontConsumedTime = 0.0;
            ++m_diagnostics.committedSegments;
        }
    }
    m_prepared = {};
    trimQueue();
    return true;
}

double ContainerMotionInput::maximumContinuousPreparedDuration() const noexcept {
    if (m_prepared.active) return m_prepared.consumedSegmentTime;
    if (m_segments.empty()) return std::numeric_limits<double>::infinity();
    const double duration = m_segments.front().endTime - m_segments.front().startTime;
    return std::max(0.0, duration - m_frontConsumedTime);
}

void ContainerMotionInput::discardPreparedStep() noexcept {
    m_prepared = {};
}

void ContainerMotionInput::reset() noexcept {
    const std::uint64_t nextVersion = m_state.version + 1;
    m_state = {};
    m_state.version = nextVersion;
    m_segments.clear();
    m_frontConsumedTime = 0.0;
    m_prepared = {};
    m_diagnostics = {};
}

const ContainerMotionSettings& ContainerMotionInput::settings() const noexcept {
    return m_settings;
}

const ContainerMotionState& ContainerMotionInput::state() const noexcept {
    return m_state;
}

ContainerMotionDiagnostics ContainerMotionInput::diagnostics(
    double nowSeconds) const noexcept {
    ContainerMotionDiagnostics result = m_diagnostics;
    result.pendingSegments = m_segments.size();
    for (std::size_t index = 0; index < m_segments.size(); ++index) {
        const double duration = m_segments[index].endTime - m_segments[index].startTime;
        const double remaining = index == 0
            ? std::max(0.0, duration - m_frontConsumedTime) : duration;
        result.pendingTime += remaining;
        result.pendingVelocityDelta = add(result.pendingVelocityDelta,
            scale(m_segments[index].externalAcceleration, remaining));
    }
    if (!m_segments.empty() && std::isfinite(nowSeconds)) {
        result.oldestSampleAge = std::max(0.0,
            nowSeconds - (m_segments.front().startTime + m_frontConsumedTime));
    }
    result.preparedVersion = m_prepared.active ? m_prepared.version : 0;
    result.preparedVelocityDelta = m_prepared.active ? m_prepared.velocityDelta : Vector2{};
    result.preparedSegmentTime = m_prepared.active ? m_prepared.consumedSegmentTime : 0.0;
    result.frontConsumedTime = m_frontConsumedTime;
    return result;
}

const std::deque<ContainerMotionSegment>&
ContainerMotionInput::pendingSegments() const noexcept {
    return m_segments;
}

void ContainerMotionInput::trimQueue() noexcept {
    if (m_prepared.active) return;
    double queued = 0.0;
    for (std::size_t index = 0; index < m_segments.size(); ++index) {
        const double duration = m_segments[index].endTime - m_segments[index].startTime;
        queued += index == 0 ? std::max(0.0, duration - m_frontConsumedTime) : duration;
    }
    while (!m_segments.empty() && queued > m_settings.maximumQueuedTime) {
        const auto segment = m_segments.front();
        const double duration = segment.endTime - segment.startTime;
        const double remaining = std::max(0.0, duration - m_frontConsumedTime);
        const Vector2 dropped = scale(segment.externalAcceleration, remaining);
        m_diagnostics.droppedVelocityDelta = add(
            m_diagnostics.droppedVelocityDelta, dropped);
        ++m_diagnostics.droppedSegments;
        queued -= remaining;
        m_segments.pop_front();
        m_frontConsumedTime = 0.0;
    }
}

} // namespace fluidlab::desktop
