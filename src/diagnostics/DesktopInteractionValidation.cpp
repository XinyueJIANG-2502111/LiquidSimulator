#include "DesktopInteractionValidation.h"

#include "../desktop/ContainerMotionInput.h"
#include "../desktop/DesktopPerformanceMetrics.h"
#include "../desktop/DesktopSettledState.h"
#include "../desktop/WindowsMoveLoopAdapter.h"
#include "../simulation/gpu/GpuDynamicFreeSurfaceStepper.h"
#include "../simulation/gpu/GpuAsyncDynamicFreeSurfaceStepper.h"
#include "../simulation/mac/Scenarios.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

namespace fluidlab::diagnostics {
namespace {

using fluidlab::desktop::ContainerMotionInput;
using fluidlab::desktop::ContainerMotionSettings;
using fluidlab::desktop::DesktopSettledState;
using fluidlab::simulation::GpuDynamicFailureReason;
using fluidlab::simulation::GpuDynamicFreeSurfaceReport;
using fluidlab::simulation::GpuDynamicFreeSurfaceSettings;
using fluidlab::simulation::GpuDynamicFreeSurfaceStepper;
using fluidlab::simulation::MacGrid;
using fluidlab::simulation::MarkerParticle;
using fluidlab::simulation::MarkerParticleSystem;
using fluidlab::simulation::SimulationStepInput;
using fluidlab::simulation::Vector2;

struct ParticleMetrics final {
    double centerX = 0.0;
    double centerY = 0.0;
    std::size_t count = 0;
};

struct SurfaceMetrics final {
    double leftHeight = 0.0;
    double rightHeight = 0.0;
    double difference = 0.0;
};

[[nodiscard]] std::pair<double, float> velocityMetrics(
    const std::vector<float>& horizontal, const std::vector<float>& vertical);

[[nodiscard]] ParticleMetrics particleMetrics(const std::vector<MarkerParticle>& particles) {
    ParticleMetrics result;
    for (const auto& particle : particles) {
        if (!particle.valid) continue;
        result.centerX += particle.position.x;
        result.centerY += particle.position.y;
        ++result.count;
    }
    if (result.count != 0) {
        result.centerX /= static_cast<double>(result.count);
        result.centerY /= static_cast<double>(result.count);
    }
    return result;
}

[[nodiscard]] SurfaceMetrics particleSurfaceMetrics(
    const std::vector<MarkerParticle>& particles,
    double domainMidpoint) {
    std::vector<double> left;
    std::vector<double> right;
    for (const auto& particle : particles) {
        if (!particle.valid) continue;
        (particle.position.x < domainMidpoint ? left : right).push_back(particle.position.y);
    }
    const auto robustHeight = [](std::vector<double>& values) {
        if (values.empty()) return 0.0;
        std::sort(values.begin(), values.end());
        const std::size_t index = static_cast<std::size_t>(
            0.95 * static_cast<double>(values.size() - 1));
        return values[index];
    };
    SurfaceMetrics result;
    result.leftHeight = robustHeight(left);
    result.rightHeight = robustHeight(right);
    result.difference = result.rightHeight - result.leftHeight;
    return result;
}

struct RunResult final {
    GpuDynamicFreeSurfaceReport report;
    ParticleMetrics particles;
    double wallMilliseconds = 0.0;
    std::size_t rejected = 0;
    double velocityRms = 0.0;
    float maximumVelocity = 0.0f;
    std::vector<double> candidateWallMilliseconds;
};

struct SequenceResult final {
    GpuDynamicFreeSurfaceReport report;
    ParticleMetrics particles;
    SurfaceMetrics surface;
    double velocityRms = 0.0;
    double peakVelocityRms = 0.0;
    double kineticEnergyProxy = 0.0;
    double peakKineticEnergyProxy = 0.0;
    double peakSurfaceDifference = 0.0;
    double inputAbsoluteDelta = 0.0;
    double inputSignedDelta = 0.0;
    double committedSignedDelta = 0.0;
    double wallMilliseconds = 0.0;
    std::size_t rejected = 0;
};

[[nodiscard]] RunResult runGpuSteps(
    const std::filesystem::path& shaderDirectory,
    std::size_t width,
    std::size_t height,
    std::size_t outerSteps,
    Vector2 externalAcceleration,
    std::size_t forcedRejections = 0,
    bool useStepInput = true) {
    const float cellSize = 1.0f / static_cast<float>(std::max(width, height));
    MacGrid grid(width, height, cellSize, cellSize);
    fluidlab::simulation::initializeDamBreak(grid);
    MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    GpuDynamicFreeSurfaceSettings settings;
    settings.synchronousGpuTimingEnabled = true;
    settings.forcedCandidateRejectionsForValidation = forcedRejections;
    GpuDynamicFreeSurfaceStepper stepper(shaderDirectory / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles);
    if (!topology.topologyValid || !topology.multigridValid || !stepper.healthy()) {
        throw std::runtime_error("External force validation initialization failed.");
    }

    const auto start = std::chrono::steady_clock::now();
    RunResult result;
    for (std::size_t outer = 0; outer < outerSteps;) {
        SimulationStepInput input;
        input.externalAcceleration = externalAcceleration;
        input.sourceTimeBegin = outer * settings.timeStep;
        input.sourceTimeEnd = (outer + 1) * settings.timeStep;
        input.version = outer + 1;
        const auto candidateStart = std::chrono::steady_clock::now();
        result.report = useStepInput ? stepper.step(input) : stepper.step();
        result.candidateWallMilliseconds.push_back(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - candidateStart).count());
        result.rejected += result.report.rejectedSubsteps;
        if (result.report.failureReason != GpuDynamicFailureReason::none) {
            throw std::runtime_error("External force validation step failed: " +
                result.report.failureMessage);
        }
        if (result.report.outerStepCompleted) ++outer;
    }
    result.wallMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    result.particles = particleMetrics(stepper.grid().readMarkerParticles());
    std::tie(result.velocityRms, result.maximumVelocity) = velocityMetrics(
        stepper.grid().readHorizontalVelocity(), stepper.grid().readVerticalVelocity());
    return result;
}

[[nodiscard]] RunResult runGpuSteps(
    const std::filesystem::path& shaderDirectory,
    std::size_t width,
    std::size_t height,
    std::size_t outerSteps,
    float externalAccelerationX,
    std::size_t forcedRejections = 0,
    bool useStepInput = true) {
    return runGpuSteps(
        shaderDirectory,
        width,
        height,
        outerSteps,
        Vector2{externalAccelerationX, 0.0f},
        forcedRejections,
        useStepInput);
}

[[nodiscard]] bool nearlyZero(double value, double tolerance) noexcept {
    return std::abs(value) <= tolerance;
}

[[nodiscard]] double samplePercentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

[[nodiscard]] std::pair<double, float> velocityMetrics(
    const std::vector<float>& horizontal, const std::vector<float>& vertical) {
    double squareSum = 0.0;
    float maximum = 0.0f;
    for (float value : horizontal) {
        squareSum += static_cast<double>(value) * value;
        maximum = std::max(maximum, std::abs(value));
    }
    for (float value : vertical) {
        squareSum += static_cast<double>(value) * value;
        maximum = std::max(maximum, std::abs(value));
    }
    const std::size_t count = horizontal.size() + vertical.size();
    return {count == 0 ? 0.0 : std::sqrt(squareSum / static_cast<double>(count)), maximum};
}

[[nodiscard]] DesktopSettledState captureSettledState(
    const GpuDynamicFreeSurfaceStepper& stepper,
    double preEvolutionSeconds,
    double velocityRms,
    float maximumVelocity,
    double centroidDrift,
    std::uint32_t stableWindows) {
    DesktopSettledState state;
    state.metadata.width = static_cast<std::uint32_t>(stepper.grid().width());
    state.metadata.height = static_cast<std::uint32_t>(stepper.grid().height());
    state.metadata.cellWidth = stepper.grid().cellWidth();
    state.metadata.cellHeight = stepper.grid().cellHeight();
    state.metadata.preEvolutionSeconds = preEvolutionSeconds;
    state.metadata.velocityRms = velocityRms;
    state.metadata.maximumVelocity = maximumVelocity;
    state.metadata.centroidDriftPerSecond = centroidDrift;
    state.metadata.stableWindowCount = stableWindows;
    state.metadata.configurationId = fluidlab::desktop::desktopSettledConfigurationId(
        state.metadata.width, state.metadata.height,
        state.metadata.cellWidth, state.metadata.cellHeight);
    state.horizontalVelocity = stepper.grid().readHorizontalVelocity();
    state.verticalVelocity = stepper.grid().readVerticalVelocity();
    state.pressure = stepper.grid().readPressure();
    state.cellTypes = stepper.grid().readCellTypes();
    state.particles = stepper.grid().readMarkerParticles();
    return state;
}

[[nodiscard]] RunResult runSettledSteps(
    const std::filesystem::path& shaderDirectory,
    const DesktopSettledState& state,
    std::size_t outerSteps,
    float accelerationX) {
    MacGrid grid(state.metadata.width, state.metadata.height,
        state.metadata.cellWidth, state.metadata.cellHeight);
    MarkerParticleSystem particles;
    fluidlab::desktop::applyDesktopSettledState(state, grid, particles);
    GpuDynamicFreeSurfaceSettings settings;
    settings.synchronousGpuTimingEnabled = true;
    GpuDynamicFreeSurfaceStepper stepper(shaderDirectory / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles);
    if (!topology.topologyValid || !topology.multigridValid || !stepper.healthy()) {
        throw std::runtime_error("Settled GPU state topology initialization failed.");
    }
    RunResult result;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t outer = 0; outer < outerSteps;) {
        SimulationStepInput input;
        input.externalAcceleration = {accelerationX, 0.0f};
        input.sourceTimeBegin = outer * settings.timeStep;
        input.sourceTimeEnd = (outer + 1) * settings.timeStep;
        input.version = outer + 1;
        result.report = stepper.step(input);
        result.rejected += result.report.rejectedSubsteps;
        if (result.report.failureReason != GpuDynamicFailureReason::none) {
            throw std::runtime_error("Settled state step failed: " + result.report.failureMessage);
        }
        if (result.report.outerStepCompleted) ++outer;
    }
    result.wallMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    result.particles = particleMetrics(stepper.grid().readMarkerParticles());
    std::tie(result.velocityRms, result.maximumVelocity) = velocityMetrics(
        stepper.grid().readHorizontalVelocity(), stepper.grid().readVerticalVelocity());
    return result;
}

[[nodiscard]] SequenceResult runSettledSequence(
    const std::filesystem::path& shaderDirectory,
    const DesktopSettledState& state,
    const std::vector<SimulationStepInput>& inputs) {
    MacGrid grid(state.metadata.width, state.metadata.height,
        state.metadata.cellWidth, state.metadata.cellHeight);
    MarkerParticleSystem particles;
    fluidlab::desktop::applyDesktopSettledState(state, grid, particles);
    GpuDynamicFreeSurfaceSettings settings;
    settings.synchronousGpuTimingEnabled = false;
    GpuDynamicFreeSurfaceStepper stepper(shaderDirectory / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles);
    if (!topology.topologyValid || !topology.multigridValid || !stepper.healthy()) {
        throw std::runtime_error("Settled input sequence topology initialization failed.");
    }

    SequenceResult result;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& input : inputs) {
        result.inputAbsoluteDelta += std::abs(input.externalVelocityDelta.x);
        result.inputSignedDelta += input.externalVelocityDelta.x;
        do {
            result.report = stepper.step(input);
            result.rejected += result.report.rejectedSubsteps;
            result.committedSignedDelta +=
                result.report.appliedExternalVelocityDelta.x;
            if (result.report.failureReason != GpuDynamicFailureReason::none) {
                throw std::runtime_error("Settled input sequence failed: " +
                    result.report.failureMessage);
            }
        } while (!result.report.outerStepCompleted);

        const auto horizontal = stepper.grid().readHorizontalVelocity();
        const auto vertical = stepper.grid().readVerticalVelocity();
        float maximum = 0.0f;
        std::tie(result.velocityRms, maximum) = velocityMetrics(horizontal, vertical);
        double squareSum = 0.0;
        for (float value : horizontal) squareSum += static_cast<double>(value) * value;
        for (float value : vertical) squareSum += static_cast<double>(value) * value;
        result.kineticEnergyProxy = 0.5 * squareSum;
        result.peakVelocityRms = std::max(result.peakVelocityRms, result.velocityRms);
        result.peakKineticEnergyProxy = std::max(
            result.peakKineticEnergyProxy, result.kineticEnergyProxy);
        const auto currentSurface = particleSurfaceMetrics(
            stepper.grid().readMarkerParticles(),
            0.5 * state.metadata.width * state.metadata.cellWidth);
        result.peakSurfaceDifference = std::max(
            result.peakSurfaceDifference, std::abs(currentSurface.difference));
    }
    const auto finalParticles = stepper.grid().readMarkerParticles();
    result.particles = particleMetrics(finalParticles);
    result.surface = particleSurfaceMetrics(finalParticles,
        0.5 * state.metadata.width * state.metadata.cellWidth);
    result.wallMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

[[nodiscard]] std::vector<SimulationStepInput> buildQueuedMotionInputs(
    const ContainerMotionSettings& settings,
    const std::vector<std::tuple<double, double, double>>& samples,
    std::size_t totalSteps) {
    ContainerMotionInput motion(settings);
    for (const auto& [time, x, y] : samples) {
        static_cast<void>(motion.addPositionSample(x, y, time));
    }
    std::vector<SimulationStepInput> result;
    result.reserve(totalSteps);
    constexpr double physicalStep = 1.0 / 120.0;
    const double finalTime = samples.empty() ? 0.0 : std::get<0>(samples.back());
    for (std::size_t step = 0; step < totalSteps; ++step) {
        auto input = motion.prepareStep(finalTime, physicalStep, true);
        result.push_back(input);
        if (!motion.commitPreparedStep(input.version)) {
            throw std::runtime_error("Deterministic motion input commit failed.");
        }
    }
    return result;
}

[[nodiscard]] std::vector<SimulationStepInput> buildLegacyLatestInputs(
    const ContainerMotionSettings& settings,
    const std::vector<std::tuple<double, double, double>>& samples,
    std::size_t totalSteps) {
    ContainerMotionInput motion(settings);
    for (const auto& [time, x, y] : samples) {
        static_cast<void>(motion.addPositionSample(x, y, time));
    }
    std::vector<SimulationStepInput> result(totalSteps);
    if (!samples.empty() && !result.empty()) {
        auto latest = motion.sample(std::get<0>(samples.back()), true);
        latest.physicalDuration = 1.0 / 120.0;
        latest.externalVelocityDelta = {
            latest.externalAcceleration.x * static_cast<float>(latest.physicalDuration),
            latest.externalAcceleration.y * static_cast<float>(latest.physicalDuration)};
        result.front() = latest;
    }
    for (std::size_t index = 1; index < result.size(); ++index) {
        result[index].sourceTimeBegin = std::get<0>(samples.back());
        result[index].sourceTimeEnd = std::get<0>(samples.back());
        result[index].physicalDuration = 1.0 / 120.0;
        result[index].version = result.front().version + index;
    }
    return result;
}

} // namespace

int runContainerMotionValidation(std::ostream& output) {
    ContainerMotionSettings settings;
    settings.pixelsPerSimulationUnit = 100.0;
    settings.velocitySmoothingTime = 0.0;
    settings.accelerationSmoothingTime = 0.0;
    settings.maximumAcceleration = 100.0;
    settings.horizontalOnly = false;
    ContainerMotionInput motion(settings);

    static_cast<void>(motion.addPositionSample(0.0, 0.0, 0.0));
    auto input = motion.sample(0.0);
    if (!nearlyZero(input.externalAcceleration.x, 1.0e-6) ||
        !nearlyZero(input.externalAcceleration.y, 1.0e-6)) {
        throw std::runtime_error("Stationary container produced inertia.");
    }

    for (int i = 1; i <= 6; ++i) {
        static_cast<void>(motion.addPositionSample(10.0 * i, -5.0 * i, 0.1 * i));
    }
    input = motion.sample(0.6);
    if (!nearlyZero(input.externalAcceleration.x, 1.0e-4) ||
        !nearlyZero(input.externalAcceleration.y, 1.0e-4)) {
        throw std::runtime_error("Constant container velocity produced sustained inertia.");
    }

    motion.reset();
    static_cast<void>(motion.addPositionSample(0.0, 0.0, 0.0));
    static_cast<void>(motion.addPositionSample(1.0, 0.0, 0.1));
    input = motion.sample(0.1);
    if (!(input.externalAcceleration.x < 0.0f)) {
        throw std::runtime_error("Rightward acceleration did not produce leftward inertia.");
    }
    static_cast<void>(motion.addPositionSample(1.0, 0.0, 0.2));
    input = motion.sample(0.2);
    if (!(input.externalAcceleration.x > 0.0f)) {
        throw std::runtime_error("Rightward deceleration did not produce rightward inertia.");
    }

    motion.reset();
    static_cast<void>(motion.addPositionSample(0.0, 0.0, 0.0));
    static_cast<void>(motion.addPositionSample(0.0, -1.0, 0.1));
    input = motion.sample(0.1);
    if (!(input.externalAcceleration.y < 0.0f) ||
        !nearlyZero(input.externalAcceleration.x, 1.0e-6)) {
        throw std::runtime_error("Upward acceleration did not produce downward inertia.");
    }
    static_cast<void>(motion.addPositionSample(0.0, -1.0, 0.2));
    input = motion.sample(0.2);
    if (!(input.externalAcceleration.y > 0.0f) ||
        !nearlyZero(input.externalAcceleration.x, 1.0e-6)) {
        throw std::runtime_error("Upward deceleration did not produce upward inertia.");
    }

    motion.reset();
    static_cast<void>(motion.addPositionSample(0.0, 0.0, 0.0));
    static_cast<void>(motion.addPositionSample(1.0, -1.0, 0.1));
    input = motion.sample(0.1);
    if (!(input.externalAcceleration.x < 0.0f &&
          input.externalAcceleration.y < 0.0f)) {
        throw std::runtime_error("Diagonal acceleration did not preserve both inertia axes.");
    }

    output << "Container motion validation passed: constant velocity inertia="
           << 0.0 << ", horizontal, vertical, and diagonal signs are correct.\n";
    return 0;
}

int runDesktopInputTimingValidation(std::ostream& output) {
    ContainerMotionInput motion;
    static_cast<void>(motion.addPositionSample(100.0, 100.0, 1.0));
    const auto duplicate = motion.addPositionSample(101.0, 100.0, 1.0);
    if (duplicate.discontinuity) {
        throw std::runtime_error("Duplicate timestamp should be ignored, not marked discontinuous.");
    }
    const auto gap = motion.addPositionSample(120.0, 100.0, 1.5);
    if (!gap.discontinuity || motion.sample(1.5).externalAcceleration.x != 0.0f) {
        throw std::runtime_error("Long input gap did not reset inertial estimation.");
    }
    const auto stale = motion.sample(2.0);
    if (stale.externalAcceleration.x != 0.0f || stale.externalAcceleration.y != 0.0f) {
        throw std::runtime_error("Stale container input did not decay to zero.");
    }
    const auto invalid = motion.addPositionSample(
        std::numeric_limits<double>::quiet_NaN(), 0.0, 2.1);
    if (invalid.valid) {
        throw std::runtime_error("Non-finite desktop input was not rejected.");
    }
    output << "Desktop input timing validation passed: duplicate, gap, stale, and non-finite samples are safe.\n";
    return 0;
}

int runExternalForceValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    GLFWwindow* sharingWindow) {
    const auto zero = runGpuSteps(shaderDirectory, 48, 32, 4, 0.0f);
    const Vector2 forcedAcceleration{6.0f, -4.0f};
    const auto forced = runGpuSteps(
        shaderDirectory, 48, 32, 4, forcedAcceleration, 1);
    if (forced.report.inputVersion != 4 ||
        std::abs(forced.report.appliedExternalAcceleration.x -
            forcedAcceleration.x) > 1.0e-6f ||
        std::abs(forced.report.appliedExternalAcceleration.y -
            forcedAcceleration.y) > 1.0e-6f ||
        std::abs(forced.report.appliedExternalVelocityDelta.x -
            forcedAcceleration.x / 120.0f) > 2.0e-6f ||
        std::abs(forced.report.appliedExternalVelocityDelta.y -
            forcedAcceleration.y / 120.0f) > 2.0e-6f ||
        forced.rejected == 0 || forced.report.maximumAcceptedCfl > 0.751f ||
        !forced.report.pressureConverged) {
        throw std::runtime_error("GPU external acceleration transaction validation failed.");
    }
    if (!(forced.particles.centerX > zero.particles.centerX + 1.0e-5 &&
          forced.particles.centerY < zero.particles.centerY - 1.0e-5)) {
        throw std::runtime_error(
            "Two-axis external acceleration did not move the liquid in both directions.");
    }
    MacGrid invalidGrid(48, 32, 1.0f / 48.0f, 1.0f / 48.0f);
    fluidlab::simulation::initializeDamBreak(invalidGrid);
    MarkerParticleSystem invalidParticles;
    invalidParticles.seedFluidCellsExact(invalidGrid, 8);
    GpuDynamicFreeSurfaceStepper invalidStepper(
        shaderDirectory / "mac_divergence.comp");
    static_cast<void>(invalidStepper.initialize(invalidGrid, invalidParticles));
    SimulationStepInput invalidInput;
    invalidInput.externalAcceleration.x = std::numeric_limits<float>::infinity();
    const auto invalidReport = invalidStepper.step(invalidInput);
    if (invalidReport.failureReason != GpuDynamicFailureReason::invalidStepInput ||
        invalidStepper.simulationTime() != 0.0 || !invalidStepper.healthy()) {
        throw std::runtime_error("Invalid external input did not fail without state mutation.");
    }

    const float cellSize = 1.0f / 48.0f;
    MacGrid grid(48, 32, cellSize, cellSize);
    fluidlab::simulation::initializeDamBreak(grid);
    MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    GpuDynamicFreeSurfaceSettings continuationSettings;
    continuationSettings.maximumSubsteps = 1;
    continuationSettings.maximumCfl = 0.05f;
    GpuDynamicFreeSurfaceStepper continuation(
        shaderDirectory / "mac_divergence.comp", continuationSettings);
    static_cast<void>(continuation.initialize(grid, particles));
    SimulationStepInput firstInput;
    firstInput.externalAcceleration = {20.0f, 0.0f};
    firstInput.version = 41;
    const auto firstReport = continuation.step(firstInput);
    if (firstReport.outerStepCompleted || continuation.pendingPhysicalTime() <= 0.0) {
        throw std::runtime_error("External input continuation test did not span calls.");
    }
    SimulationStepInput replacementInput;
    replacementInput.externalAcceleration = {-20.0f, 0.0f};
    replacementInput.version = 42;
    const auto continuationReport = continuation.step(replacementInput);
    if (continuationReport.inputVersion != 41 ||
        continuationReport.appliedExternalAcceleration.x != 20.0f) {
        throw std::runtime_error("Pending physical time did not preserve the frozen input snapshot.");
    }

    fluidlab::simulation::GpuAsyncDynamicFreeSurfaceStepper asynchronous(
        sharingWindow, shaderDirectory / "mac_divergence.comp");
    static_cast<void>(asynchronous.initialize(grid, particles));
    SimulationStepInput asyncInput;
    asyncInput.externalAcceleration = {3.0f, -2.0f};
    asyncInput.version = 77;
    if (!asynchronous.submit(asyncInput)) {
        throw std::runtime_error("Asynchronous external input submission failed.");
    }
    while (!asynchronous.poll()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (asynchronous.report().inputVersion != 77 ||
        asynchronous.report().appliedExternalAcceleration.x != 3.0f ||
        asynchronous.report().appliedExternalAcceleration.y != -2.0f) {
        throw std::runtime_error("Asynchronous worker did not preserve its input snapshot.");
    }
    asynchronous.acknowledge();
    output << std::fixed << std::setprecision(6)
           << "External force validation passed: zero centroid=(" << zero.particles.centerX
           << ',' << zero.particles.centerY << "), forced centroid=("
           << forced.particles.centerX << ',' << forced.particles.centerY
           << "), rejected candidates=" << forced.rejected
           << ", frozen/async input versions=41/77"
           << ", CFL=" << forced.report.maximumAcceptedCfl
           << ", residual=" << forced.report.projection.pressure.finalResidual.rootMeanSquare
           << ".\n";
    return 0;
}

int runContainerInertiaValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    const auto leftInertia = runGpuSteps(shaderDirectory, 48, 32, 8, -8.0f);
    const auto rightInertia = runGpuSteps(shaderDirectory, 48, 32, 8, 8.0f);
    const double separation = rightInertia.particles.centerX - leftInertia.particles.centerX;
    if (!(separation > 1.0e-5)) {
        throw std::runtime_error("Opposite container accelerations did not produce opposite liquid response.");
    }
    output << std::fixed << std::setprecision(7)
           << "Container inertia validation passed: left/right liquid centroids="
           << leftInertia.particles.centerX << '/' << rightInertia.particles.centerX
           << ", separation=" << separation << ".\n";
    return 0;
}

int runGpuFluidInjectionValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    constexpr std::size_t width = 48;
    constexpr std::size_t height = 32;
    constexpr float cellSize = 1.0f / 48.0f;

    const auto run = [&](std::size_t forcedRejections) {
        MacGrid grid(width, height, cellSize, cellSize);
        fluidlab::simulation::initializeDamBreak(grid);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        const std::size_t initialParticles = particles.validParticleCount();
        GpuDynamicFreeSurfaceSettings settings;
        settings.forcedCandidateRejectionsForValidation = forcedRejections;
        GpuDynamicFreeSurfaceStepper stepper(
            shaderDirectory / "mac_divergence.comp", settings);
        static_cast<void>(stepper.initialize(grid, particles));

        SimulationStepInput input;
        input.physicalDuration = settings.timeStep;
        input.sourceTimeEnd = settings.timeStep;
        input.version = 1;
        input.fluidInjection.active = true;
        input.fluidInjection.position = {0.72f, 0.48f};
        input.fluidInjection.radius = 3.0f * cellSize;
        input.fluidInjection.requestedArea = 2.0 * cellSize * cellSize;
        input.fluidInjection.version = 1;
        const auto report = stepper.step(input);
        if (report.failureReason != GpuDynamicFailureReason::none ||
            !report.outerStepCompleted || report.injectionCellsAccepted != 2 ||
            report.injectionParticlesAdded != 16 ||
            report.particleCount != initialParticles + 16 ||
            report.fluidCellCount <= 0 || !report.pressureConverged) {
            throw std::runtime_error("GPU fluid injection did not enter formal physics.");
        }

        SimulationStepInput releasedInput;
        releasedInput.physicalDuration = settings.timeStep;
        releasedInput.sourceTimeEnd = 2.0 * settings.timeStep;
        releasedInput.version = 2;
        const auto releasedReport = stepper.step(releasedInput);
        if (releasedReport.failureReason != GpuDynamicFailureReason::none ||
            !releasedReport.outerStepCompleted ||
            releasedReport.injectionCellsAccepted != 0 ||
            releasedReport.injectionParticlesAdded != 0 ||
            releasedReport.particleCount != initialParticles + 16) {
            throw std::runtime_error("Released fluid emitter continued injecting particles.");
        }
        return std::pair{report, initialParticles};
    };

    const auto [normal, initialParticles] = run(0);
    const auto [rollback, ignoredInitialParticles] = run(1);
    static_cast<void>(ignoredInitialParticles);
    if (rollback.injectionParticlesAdded != 16 || rollback.rejectedSubsteps == 0 ||
        rollback.particleCount != initialParticles + 16) {
        throw std::runtime_error("Rejected injection candidate duplicated committed particles.");
    }

    struct VolumeResult final {
        std::size_t initialParticles = 0;
        std::size_t finalParticles = 0;
        std::size_t finalFluidCells = 0;
        double acceptedArea = 0.0;
    };
    const auto runVolumeSequence = [&](bool inject) {
        constexpr std::size_t sequenceWidth = 64;
        constexpr std::size_t sequenceHeight = 48;
        constexpr float sequenceCellSize = 1.0f / 64.0f;
        MacGrid grid(sequenceWidth, sequenceHeight,
            sequenceCellSize, sequenceCellSize);
        fluidlab::simulation::initializeDamBreak(grid);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        VolumeResult result;
        result.initialParticles = particles.validParticleCount();
        GpuDynamicFreeSurfaceSettings settings;
        GpuDynamicFreeSurfaceStepper stepper(
            shaderDirectory / "mac_divergence.comp", settings);
        static_cast<void>(stepper.initialize(grid, particles));
        for (std::size_t outer = 0; outer < 48; ++outer) {
            SimulationStepInput input;
            input.physicalDuration = settings.timeStep;
            input.sourceTimeBegin = outer * settings.timeStep;
            input.sourceTimeEnd = (outer + 1) * settings.timeStep;
            input.version = outer + 1;
            input.fluidInjection.active = inject && outer < 24;
            input.fluidInjection.position = {0.72f, 0.62f};
            input.fluidInjection.radius = 3.0f * sequenceCellSize;
            input.fluidInjection.requestedArea =
                sequenceCellSize * sequenceCellSize;
            input.fluidInjection.version = outer + 1;
            const auto report = stepper.step(input);
            if (report.failureReason != GpuDynamicFailureReason::none ||
                !report.outerStepCompleted || !report.pressureConverged) {
                throw std::runtime_error(
                    "GPU fluid injection volume sequence failed formal physics.");
            }
            result.acceptedArea += report.injectionAreaAccepted;
            result.finalParticles = report.particleCount;
            result.finalFluidCells = report.fluidCellCount;
        }
        return result;
    };

    const VolumeResult baseline = runVolumeSequence(false);
    const VolumeResult injected = runVolumeSequence(true);
    if (injected.acceptedArea <= 0.0 ||
        injected.finalParticles <= baseline.finalParticles ||
        injected.finalFluidCells <= baseline.finalFluidCells) {
        throw std::runtime_error(
            "Injected water was not retained by the formal fluid classification.");
    }

    output << "GPU fluid injection validation passed: accepted cells="
           << normal.injectionCellsAccepted
           << ", particles=" << initialParticles << "->" << normal.particleCount
           << ", accepted area=" << normal.injectionAreaAccepted
           << ", rollback retries=" << rollback.rejectedSubsteps
           << ", 64x48 retained particles=" << baseline.finalParticles
           << "->" << injected.finalParticles
           << ", fluid cells=" << baseline.finalFluidCells
           << "->" << injected.finalFluidCells << ".\n";
    return 0;
}

int runDesktopInteractionBenchmark(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    output << "Desktop interaction hidden-context benchmark (6 outer steps)\n"
              "Grid  Baseline wall  Zero-input wall  Pulse wall  Candidate P50/P95/P99  Physical rate\n";
    for (const auto [width, height] :
        std::array<std::pair<std::size_t, std::size_t>, 4>{
            std::pair{48U, 32U}, std::pair{64U, 48U},
            std::pair{64U, 64U}, std::pair{128U, 128U}}) {
        const auto baseline = runGpuSteps(
            shaderDirectory, width, height, 6, 0.0f, 0, false);
        const auto zero = runGpuSteps(shaderDirectory, width, height, 6, 0.0f);
        const auto pulse = runGpuSteps(shaderDirectory, width, height, 6, 5.0f);
        const double physicalRate = pulse.wallMilliseconds > 0.0
            ? (6.0 / 120.0) / (pulse.wallMilliseconds * 1.0e-3) : 0.0;
        output << std::setw(3) << width << 'x' << std::setw(3) << height << "  "
               << std::fixed << std::setprecision(3)
               << std::setw(13) << baseline.wallMilliseconds << "  "
               << std::setw(15) << zero.wallMilliseconds << "  "
               << std::setw(10) << pulse.wallMilliseconds << "  "
               << samplePercentile(pulse.candidateWallMilliseconds, 0.50) << '/'
               << samplePercentile(pulse.candidateWallMilliseconds, 0.95) << '/'
               << samplePercentile(pulse.candidateWallMilliseconds, 0.99) << "  "
               << std::setw(13) << physicalRate << '\n';
    }
    output << "Benchmark passed. Render FPS is intentionally not inferred from hidden physics timing.\n";
    return 0;
}

int runDesktopFramePacingValidation(std::ostream& output) {
    fluidlab::desktop::DesktopPerformanceMetrics metrics(8);
    for (double frame : {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0}) {
        fluidlab::desktop::DesktopFrameTiming timing;
        timing.frameMilliseconds = frame;
        timing.eventMilliseconds = 1.0;
        timing.physicsSubmitMilliseconds = 2.0;
        timing.fencePollMilliseconds = 0.5;
        timing.renderMilliseconds = 3.0;
        timing.swapMilliseconds = 4.0;
        timing.committedLagSeconds = 0.01;
        timing.moveSizeFrame = frame >= 70.0;
        timing.repeatedSnapshot = frame == 80.0;
        metrics.add(timing);
    }
    const auto summary = metrics.summary();
    if (summary.sampleCount != 8 || summary.frameP50Milliseconds != 40.0 ||
        summary.frameP95Milliseconds != 70.0 || summary.frameP99Milliseconds != 70.0 ||
        summary.frameMaximumMilliseconds != 80.0 || summary.over50Milliseconds != 3 ||
        summary.moveSizeFrames != 2 || summary.repeatedSnapshotFrames != 1) {
        throw std::runtime_error("Desktop frame percentile validation failed.");
    }
    output << "Desktop frame pacing validation passed: P50/P95/P99/max="
           << summary.frameP50Milliseconds << '/' << summary.frameP95Milliseconds << '/'
           << summary.frameP99Milliseconds << '/' << summary.frameMaximumMilliseconds
           << " ms, >50 ms=" << summary.over50Milliseconds << ".\n";
    return 0;
}

int runDesktopModalLoopValidation(std::ostream& output, GLFWwindow* window) {
    fluidlab::desktop::WindowsMoveLoopAdapter adapter;
    std::size_t ticks = 0;
    if (!adapter.install(window, [&ticks]() { ++ticks; }, 16)) {
        throw std::runtime_error("Windows move-loop subclass installation failed.");
    }
    // Synthetic calls exercise only our callback path and never send desktop input.
    adapter.syntheticEnterForValidation();
    adapter.syntheticTimerForValidation();
    adapter.syntheticExitForValidation();
    if (!adapter.installed() || adapter.inMoveSize() || adapter.timerFrameCount() != 1 || ticks != 1) {
        throw std::runtime_error("Windows move-loop adapter lifecycle failed.");
    }
    adapter.uninstall();
    fluidlab::desktop::ModalFrameScheduler scheduler;
    scheduler.enter();
    if (!scheduler.inMoveSize() || !scheduler.beginFrame() || scheduler.beginFrame()) {
        throw std::runtime_error("Modal frame scheduler reentry protection failed.");
    }
    scheduler.endFrame();
    scheduler.exit();
    if (scheduler.inMoveSize() || scheduler.skippedReentrantFrames() != 1) {
        throw std::runtime_error("Modal frame scheduler lifecycle failed.");
    }
    output << "Desktop modal-loop state validation passed: reentrant frames skipped="
           << scheduler.skippedReentrantFrames()
           << ", synthetic timer ticks=" << ticks
           << ". Real title-bar dragging remains a manual test.\n";
    return 0;
}

int runDesktopSettleGenerate(
    std::ostream& output, const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& outputPath) {
    constexpr std::size_t width = 64;
    constexpr std::size_t height = 48;
    constexpr float cellSize = 1.0f / 64.0f;
    constexpr double observationSeconds = 0.5;
    constexpr std::uint32_t requiredStableWindows = 4;
    constexpr double minimumEvolutionSeconds = 10.0;
    constexpr double maximumEvolutionSeconds = 30.0;
    constexpr double maximumRms = 0.035;
    constexpr float maximumPeak = 0.40f;
    constexpr double maximumCentroidDrift = 0.04;

    MacGrid grid(width, height, cellSize, cellSize);
    fluidlab::simulation::initializeDamBreak(grid);
    MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    GpuDynamicFreeSurfaceSettings settings;
    settings.synchronousGpuTimingEnabled = false;
    GpuDynamicFreeSurfaceStepper stepper(shaderDirectory / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles);
    if (!topology.topologyValid || !topology.multigridValid || !stepper.healthy()) {
        throw std::runtime_error("Desktop settle generation topology is invalid.");
    }

    std::uint32_t stableWindows = 0;
    double nextObservation = observationSeconds;
    ParticleMetrics previousCentroid = particleMetrics(stepper.grid().readMarkerParticles());
    double velocityRms = 0.0;
    float maximumVelocity = 0.0f;
    double centroidDrift = std::numeric_limits<double>::infinity();
    GpuDynamicFreeSurfaceReport report;
    while (stepper.simulationTime() < maximumEvolutionSeconds &&
           stableWindows < requiredStableWindows) {
        report = stepper.step();
        if (report.failureReason != GpuDynamicFailureReason::none) {
            throw std::runtime_error("Desktop settle generation failed: " + report.failureMessage);
        }
        if (!report.outerStepCompleted || stepper.simulationTime() + 1.0e-9 < nextObservation) continue;
        const auto horizontal = stepper.grid().readHorizontalVelocity();
        const auto vertical = stepper.grid().readVerticalVelocity();
        std::tie(velocityRms, maximumVelocity) = velocityMetrics(horizontal, vertical);
        const ParticleMetrics centroid = particleMetrics(stepper.grid().readMarkerParticles());
        centroidDrift = std::hypot(centroid.centerX - previousCentroid.centerX,
            centroid.centerY - previousCentroid.centerY) / observationSeconds;
        previousCentroid = centroid;
        const bool stable = stepper.simulationTime() >= minimumEvolutionSeconds &&
            velocityRms <= maximumRms && maximumVelocity <= maximumPeak &&
            centroidDrift <= maximumCentroidDrift && report.pressureConverged &&
            report.failureReason == GpuDynamicFailureReason::none;
        stableWindows = stable ? stableWindows + 1 : 0;
        output << std::fixed << std::setprecision(6)
               << "Settle observation t=" << stepper.simulationTime()
               << " rms=" << velocityRms << " max=" << maximumVelocity
               << " centroidRate=" << centroidDrift
               << " stableWindows=" << stableWindows << '\n';
        nextObservation += observationSeconds;
    }
    if (stableWindows < requiredStableWindows) {
        throw std::runtime_error("Dam-break pre-evolution did not satisfy settled-state criteria.");
    }
    auto state = captureSettledState(stepper, stepper.simulationTime(), velocityRms,
        maximumVelocity, centroidDrift, stableWindows);
    fluidlab::desktop::saveDesktopSettledState(outputPath, state);
    output << "Desktop settled state generated: path=" << outputPath.string()
           << ", sourceTime=" << stepper.simulationTime() << ", particles="
           << state.particles.size() << ".\n";
    return 0;
}

int runDesktopSettleLoadValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath) {
    constexpr std::size_t width = 64, height = 48;
    constexpr float cellSize = 1.0f / 64.0f;
    const auto loaded = fluidlab::desktop::loadDesktopSettledState(
        statePath, width, height, cellSize, cellSize);
    if (!loaded.valid) throw std::runtime_error("Settled load validation failed: " + loaded.message);
    const auto first = runSettledSteps(shaderDirectory, loaded.state, 1, 0.0f);
    const auto repeat = runSettledSteps(shaderDirectory, loaded.state, 1, 0.0f);
    const auto continuation = runSettledSteps(shaderDirectory, loaded.state, 60, 0.0f);
    if (!first.report.pressureConverged || first.report.failureReason != GpuDynamicFailureReason::none ||
        std::abs(first.particles.centerX - repeat.particles.centerX) > 1.0e-7 ||
        std::abs(first.particles.centerY - repeat.particles.centerY) > 1.0e-7) {
        throw std::runtime_error("Settled state first step or deterministic reset failed.");
    }
    if (continuation.velocityRms > 0.08 || continuation.maximumVelocity > 0.75f ||
        continuation.report.failureReason != GpuDynamicFailureReason::none) {
        throw std::runtime_error("Settled state did not remain dynamically quiet after loading.");
    }
    const auto mismatch = fluidlab::desktop::loadDesktopSettledState(
        statePath, width + 1, height, cellSize, cellSize);
    if (mismatch.valid) throw std::runtime_error("Settled state layout mismatch was accepted.");

    const auto damagedPath = std::filesystem::temp_directory_path() / "liquidsim_settled_damaged.bin";
    std::filesystem::copy_file(statePath, damagedPath,
        std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream damaged(damagedPath, std::ios::binary | std::ios::in | std::ios::out);
        char byte = 0;
        damaged.seekg(-1, std::ios::end);
        damaged.read(&byte, 1);
        byte = static_cast<char>(byte ^ 0x5A);
        damaged.seekp(-1, std::ios::end);
        damaged.write(&byte, 1);
    }
    const auto damaged = fluidlab::desktop::loadDesktopSettledState(
        damagedPath, width, height, cellSize, cellSize);
    std::error_code removeError;
    std::filesystem::remove(damagedPath, removeError);
    if (damaged.valid) throw std::runtime_error("Settled state checksum damage was accepted.");
    output << std::fixed << std::setprecision(6)
           << "Desktop settled load validation passed: sourceTime="
           << loaded.state.metadata.preEvolutionSeconds << ", initialRms="
           << loaded.state.metadata.velocityRms << ", first-step CFL="
           << first.report.maximumAcceptedCfl << ", 0.5s RMS/max="
           << continuation.velocityRms << '/' << continuation.maximumVelocity << ".\n";
    return 0;
}

int runDesktopInertiaFromSettledValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath) {
    constexpr std::size_t width = 64, height = 48;
    constexpr float cellSize = 1.0f / 64.0f;
    const auto loaded = fluidlab::desktop::loadDesktopSettledState(
        statePath, width, height, cellSize, cellSize);
    if (!loaded.valid) throw std::runtime_error("Settled inertia state is unavailable: " + loaded.message);
    const auto zero = runSettledSteps(shaderDirectory, loaded.state, 12, 0.0f);
    const auto left = runSettledSteps(shaderDirectory, loaded.state, 12, -8.0f);
    const auto right = runSettledSteps(shaderDirectory, loaded.state, 12, 8.0f);
    if (!(left.particles.centerX < zero.particles.centerX &&
          right.particles.centerX > zero.particles.centerX) ||
        !left.report.pressureConverged || !right.report.pressureConverged) {
        throw std::runtime_error("Settled water did not respond correctly to opposite inertia.");
    }
    output << std::fixed << std::setprecision(7)
           << "Settled inertia validation passed: left/zero/right centroid="
           << left.particles.centerX << '/' << zero.particles.centerX << '/'
           << right.particles.centerX << ", rejected="
           << left.rejected + right.rejected << ".\n";
    return 0;
}

int runDesktopInertiaPipelineAudit(std::ostream& output) {
    ContainerMotionSettings currentSettings;
    currentSettings.velocitySmoothingTime = 0.035;
    currentSettings.accelerationSmoothingTime = 0.055;
    ContainerMotionInput current(currentSettings);
    ContainerMotionInput revised;
    const std::array<std::tuple<double, double, double>, 5> samples{{
        {0.000, 0.0, 0.0},
        {0.016, 96.0, 0.0},
        {0.032, 96.0, 0.0},
        {0.048, 32.0, 0.0},
        {0.064, 32.0, 0.0}}};
    output << "Desktop inertia pipeline audit\n"
              "Time Position RawV FilteredV RawA FilteredA Limited Queue\n";
    for (const auto& [time, x, y] : samples) {
        static_cast<void>(current.addPositionSample(x, y, time));
        const auto state = revised.addPositionSample(x, y, time);
        output << std::fixed << std::setprecision(6)
               << time << ' ' << x << ' ' << state.rawVelocity.x << ' '
               << state.velocity.x << ' ' << state.rawAcceleration.x << ' '
               << state.acceleration.x << ' ' << (state.accelerationLimited ? 1 : 0)
               << ' ' << revised.diagnostics(time).pendingSegments << '\n';
    }

    const auto latest = current.sample(0.064, true);
    std::vector<float> orderedAcceleration;
    double orderedAbsoluteDelta = 0.0;
    double orderedSignedDelta = 0.0;
    constexpr double physicalStep = 1.0 / 120.0;
    while (revised.diagnostics(0.064).pendingSegments != 0) {
        const auto input = revised.prepareStep(0.064, physicalStep, true);
        orderedAcceleration.push_back(input.externalAcceleration.x);
        orderedAbsoluteDelta += std::abs(input.externalVelocityDelta.x);
        orderedSignedDelta += input.externalVelocityDelta.x;
        if (!revised.commitPreparedStep(input.version)) {
            throw std::runtime_error("Pipeline audit could not commit an ordered segment.");
        }
    }
    const bool hasNegative = std::any_of(orderedAcceleration.begin(),
        orderedAcceleration.end(), [](float value) { return value < -1.0e-4f; });
    const bool hasPositive = std::any_of(orderedAcceleration.begin(),
        orderedAcceleration.end(), [](float value) { return value > 1.0e-4f; });
    if (!hasNegative || !hasPositive || orderedAcceleration.size() < 4 ||
        orderedAbsoluteDelta <= std::abs(latest.externalAcceleration.x) * physicalStep) {
        throw std::runtime_error("Ordered inertia history did not preserve start/stop/reversal pulses.");
    }
    output << "Latest-only acceleration=" << latest.externalAcceleration.x
           << ", ordered steps=" << orderedAcceleration.size()
           << ", ordered abs/signed delta=" << orderedAbsoluteDelta << '/'
           << orderedSignedDelta << ".\n"
           << "Pipeline audit passed: latest-only sampling loses ordered impulse history.\n";
    return 0;
}

int runDesktopInertiaTimingValidation(std::ostream& output) {
    ContainerMotionInput motion;
    static_cast<void>(motion.addPositionSample(0.0, 0.0, 0.0));
    static_cast<void>(motion.addPositionSample(90.0, 0.0, 0.015));
    static_cast<void>(motion.addPositionSample(90.0, 0.0, 0.030));
    const auto first = motion.prepareStep(0.030, 1.0 / 120.0, true);
    motion.discardPreparedStep();
    const auto retry = motion.prepareStep(0.030, 1.0 / 120.0, true);
    if (first.version != retry.version ||
        std::abs(first.externalVelocityDelta.x - retry.externalVelocityDelta.x) > 1.0e-8f ||
        !motion.commitPreparedStep(retry.version)) {
        throw std::runtime_error("Inertia rollback changed or lost the prepared impulse.");
    }
    auto second = motion.prepareStep(0.030, 1.0 / 120.0, true);
    while (second.version == first.version) {
        if (!motion.commitPreparedStep(second.version)) {
            throw std::runtime_error("The first impulse could not be drained deterministically.");
        }
        second = motion.prepareStep(0.030, 1.0 / 120.0, true);
    }
    if (first.externalAcceleration.x * second.externalAcceleration.x >= 0.0f ||
        !motion.commitPreparedStep(second.version)) {
        throw std::runtime_error("Start and stop impulses were merged or reordered.");
    }

    ContainerMotionSettings boundedSettings;
    boundedSettings.maximumQueuedTime = 0.04;
    ContainerMotionInput bounded(boundedSettings);
    static_cast<void>(bounded.addPositionSample(0.0, 0.0, 0.0));
    for (int index = 1; index <= 8; ++index) {
        static_cast<void>(bounded.addPositionSample(
            index % 2 == 0 ? 0.0 : 50.0, 0.0, 0.016 * index));
    }
    const auto boundedDiagnostics = bounded.diagnostics(0.128);
    if (boundedDiagnostics.pendingTime > boundedSettings.maximumQueuedTime + 0.0161 ||
        boundedDiagnostics.droppedSegments == 0) {
        throw std::runtime_error("Inertia history capacity policy was not explicit or bounded.");
    }

    motion.reset();
    if (motion.diagnostics(1.0).pendingSegments != 0) {
        throw std::runtime_error("Reset retained desktop inertia history.");
    }
    output << std::fixed << std::setprecision(7)
           << "Desktop inertia timing validation passed: retry delta="
           << retry.externalVelocityDelta.x << ", next delta="
           << second.externalVelocityDelta.x << ", dropped segments="
           << boundedDiagnostics.droppedSegments << ".\n";
    return 0;
}

int runDesktopInertiaResponseValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath) {
    constexpr std::size_t width = 64, height = 48;
    constexpr float cellSize = 1.0f / 64.0f;
    const auto loaded = fluidlab::desktop::loadDesktopSettledState(
        statePath, width, height, cellSize, cellSize);
    if (!loaded.valid) {
        throw std::runtime_error("Inertia response state is unavailable: " + loaded.message);
    }
    const std::vector<std::tuple<double, double, double>> samples{
        {0.000, 0.0, 0.0},
        {0.016, 96.0, 0.0},
        {0.032, 96.0, 0.0}};
    ContainerMotionSettings legacySettings;
    legacySettings.velocitySmoothingTime = 0.035;
    legacySettings.accelerationSmoothingTime = 0.055;
    constexpr std::size_t steps = 60;
    const auto legacyInputs = buildLegacyLatestInputs(legacySettings, samples, steps);
    const auto queuedInputs = buildQueuedMotionInputs({}, samples, steps);
    std::vector<SimulationStepInput> zeroInputs(steps);
    for (std::size_t index = 0; index < zeroInputs.size(); ++index) {
        zeroInputs[index].physicalDuration = 1.0 / 120.0;
        zeroInputs[index].version = index + 1;
    }
    const auto zero = runSettledSequence(shaderDirectory, loaded.state, zeroInputs);
    const auto legacy = runSettledSequence(shaderDirectory, loaded.state, legacyInputs);
    const auto queued = runSettledSequence(shaderDirectory, loaded.state, queuedInputs);
    if (!(queued.inputAbsoluteDelta > legacy.inputAbsoluteDelta * 2.0) ||
        !(queued.peakKineticEnergyProxy > zero.peakKineticEnergyProxy * 1.05) ||
        queued.report.maximumAcceptedCfl > 0.751f || !queued.report.pressureConverged ||
        std::abs(queued.committedSignedDelta - queued.inputSignedDelta) > 2.0e-5) {
        throw std::runtime_error("Queued inertia did not improve input fidelity safely.");
    }
    output << std::fixed << std::setprecision(7)
           << "Desktop settled response comparison\n"
              "Mode AbsInput SignedInput Committed PeakRms PeakEnergy PeakSurface FinalSurface Rejected WallMs\n"
           << "zero " << zero.inputAbsoluteDelta << ' ' << zero.inputSignedDelta << ' '
           << zero.committedSignedDelta << ' ' << zero.peakVelocityRms << ' '
           << zero.peakKineticEnergyProxy << ' ' << zero.peakSurfaceDifference << ' '
           << zero.surface.difference << ' ' << zero.rejected << ' ' << zero.wallMilliseconds << '\n'
           << "latest " << legacy.inputAbsoluteDelta << ' ' << legacy.inputSignedDelta << ' '
           << legacy.committedSignedDelta << ' ' << legacy.peakVelocityRms << ' '
           << legacy.peakKineticEnergyProxy << ' ' << legacy.peakSurfaceDifference << ' '
           << legacy.surface.difference << ' ' << legacy.rejected << ' ' << legacy.wallMilliseconds << '\n'
           << "queued " << queued.inputAbsoluteDelta << ' ' << queued.inputSignedDelta << ' '
           << queued.committedSignedDelta << ' ' << queued.peakVelocityRms << ' '
           << queued.peakKineticEnergyProxy << ' ' << queued.peakSurfaceDifference << ' '
           << queued.surface.difference << ' ' << queued.rejected << ' ' << queued.wallMilliseconds << '\n'
           << "Desktop inertia response validation passed.\n";
    return 0;
}

int runDesktopInertiaLatencyBenchmark(
    std::ostream& output, const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath) {
    constexpr std::size_t width = 64, height = 48;
    constexpr float cellSize = 1.0f / 64.0f;
    const auto loaded = fluidlab::desktop::loadDesktopSettledState(
        statePath, width, height, cellSize, cellSize);
    if (!loaded.valid) {
        throw std::runtime_error("Inertia benchmark state is unavailable: " + loaded.message);
    }
    const std::vector<std::tuple<double, double, double>> samples{
        {0.000, 0.0, 0.0}, {0.016, 80.0, 0.0},
        {0.032, 80.0, 0.0}, {0.048, 0.0, 0.0}, {0.064, 0.0, 0.0}};
    const auto inputs = buildQueuedMotionInputs({}, samples, 120);
    std::vector<double> timings;
    for (int sample = 0; sample < 5; ++sample) {
        const auto result = runSettledSequence(shaderDirectory, loaded.state, inputs);
        timings.push_back(result.wallMilliseconds);
    }
    ContainerMotionInput motion;
    static_cast<void>(motion.addPositionSample(0.0, 0.0, 0.0));
    constexpr std::size_t preparationSamples = 100000;
    const auto preparationStart = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < preparationSamples; ++index) {
        const auto input = motion.prepareStep(
            static_cast<double>(index) / 120.0, 1.0 / 120.0, true);
        static_cast<void>(motion.commitPreparedStep(input.version));
    }
    const double preparationMicroseconds = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - preparationStart).count() /
        static_cast<double>(preparationSamples);
    output << std::fixed << std::setprecision(3)
           << "Desktop inertia latency benchmark: samples=5, 120 steps median/P95="
           << samplePercentile(timings, 0.50) << '/'
           << samplePercentile(timings, 0.95)
           << " ms, input prepare/commit=" << preparationMicroseconds
           << " us. Input preparation is CPU-only and adds no GPU dispatch.\n";
    return 0;
}

int runDesktopBatchInputAudit(std::ostream& output) {
    constexpr double outerStep = 1.0 / 120.0;
    constexpr double tolerance = 2.0e-6;
    const auto ledgerError = [](const fluidlab::desktop::ContainerMotionDiagnostics& d) {
        const double accountedX = static_cast<double>(d.committedVelocityDelta.x) +
            static_cast<double>(d.pendingVelocityDelta.x) +
            static_cast<double>(d.droppedVelocityDelta.x);
        const double accountedY = static_cast<double>(d.committedVelocityDelta.y) +
            static_cast<double>(d.pendingVelocityDelta.y) +
            static_cast<double>(d.droppedVelocityDelta.y);
        return std::max(
            std::abs(static_cast<double>(d.generatedVelocityDelta.x) - accountedX),
            std::abs(static_cast<double>(d.generatedVelocityDelta.y) - accountedY));
    };
    const auto makeSingleSegment = [](double duration) {
        ContainerMotionSettings settings;
        settings.pixelsPerSimulationUnit = 1.0;
        settings.velocitySmoothingTime = 0.0;
        settings.accelerationSmoothingTime = 0.0;
        settings.maximumAcceleration = 1.0e6;
        settings.maximumQueuedTime = 1.0;
        ContainerMotionInput motion(settings);
        static_cast<void>(motion.addPositionSample(0.0, 0.0, 0.0));
        static_cast<void>(motion.addPositionSample(duration, 0.0, duration));
        return motion;
    };

    output << "Desktop batch input prefix audit\n"
              "LengthSteps Completed PreparedDelta CommittedDelta PendingDelta LedgerError\n";
    for (const double lengthSteps : {0.5, 1.0, 1.5, 2.25, 4.0}) {
        for (std::size_t completed = 0; completed <= 4; ++completed) {
            auto motion = makeSingleSegment(lengthSteps * outerStep);
            const auto input = motion.prepareStep(0.1, 4.0 * outerStep, true);
            const double fraction = static_cast<double>(completed) / 4.0;
            if (!motion.commitPreparedFraction(input.version, fraction)) {
                throw std::runtime_error("Batch prefix input commit was rejected.");
            }
            const auto diagnostics = motion.diagnostics(0.1);
            const double expected = static_cast<double>(input.externalVelocityDelta.x) * fraction;
            const double error = ledgerError(diagnostics);
            if (std::abs(static_cast<double>(diagnostics.committedVelocityDelta.x) - expected) >
                    tolerance || error > tolerance) {
                throw std::runtime_error("Batch prefix input ledger did not conserve impulse.");
            }
            output << std::fixed << std::setprecision(5)
                   << lengthSteps << ' ' << completed << ' '
                   << input.externalVelocityDelta.x << ' '
                   << diagnostics.committedVelocityDelta.x << ' '
                   << diagnostics.pendingVelocityDelta.x << ' '
                   << std::scientific << error << std::fixed << '\n';
        }
    }

    auto retry = makeSingleSegment(4.0 * outerStep);
    const auto first = retry.prepareStep(0.1, 4.0 * outerStep, true);
    if (!retry.commitPreparedFraction(first.version, 0.75)) {
        throw std::runtime_error("Three-step prefix commit failed.");
    }
    const auto retryInput = retry.prepareStep(0.2, outerStep, true);
    if (std::abs(static_cast<double>(retryInput.externalVelocityDelta.x) -
            static_cast<double>(first.externalVelocityDelta.x) * 0.25) > tolerance ||
        !retry.commitPreparedStep(retryInput.version) ||
        ledgerError(retry.diagnostics(0.2)) > tolerance) {
        throw std::runtime_error("Partial batch retry did not consume the exact suffix.");
    }

    auto canceled = makeSingleSegment(1.5 * outerStep);
    const auto canceledInput = canceled.prepareStep(0.1, 2.0 * outerStep, true);
    canceled.discardPreparedStep();
    const auto repeatedInput = canceled.prepareStep(0.2, 2.0 * outerStep, true);
    if (canceledInput.version != repeatedInput.version ||
        std::abs(canceledInput.externalVelocityDelta.x -
            repeatedInput.externalVelocityDelta.x) > tolerance) {
        throw std::runtime_error("Canceled batch did not preserve its uncommitted input.");
    }
    canceled.discardPreparedStep();

    ContainerMotionSettings sequenceSettings;
    sequenceSettings.pixelsPerSimulationUnit = 1.0;
    sequenceSettings.velocitySmoothingTime = 0.0;
    sequenceSettings.accelerationSmoothingTime = 0.0;
    sequenceSettings.maximumAcceleration = 1.0e6;
    sequenceSettings.maximumQueuedTime = 1.0;
    ContainerMotionInput sequence(sequenceSettings);
    static_cast<void>(sequence.addPositionSample(0.0, 0.0, 0.0));
    static_cast<void>(sequence.addPositionSample(outerStep, 0.0, outerStep));
    static_cast<void>(sequence.addPositionSample(outerStep, 0.0, 2.0 * outerStep));
    static_cast<void>(sequence.addPositionSample(2.0 * outerStep, 0.0, 3.0 * outerStep));
    std::array<float, 3> orderedDelta{};
    for (std::size_t index = 0; index < orderedDelta.size(); ++index) {
        const auto input = sequence.prepareStep(0.1 + index * outerStep, outerStep, true);
        orderedDelta[index] = input.externalVelocityDelta.x;
        if (!sequence.commitPreparedStep(input.version)) {
            throw std::runtime_error("Ordered pulse commit failed.");
        }
    }
    if (!(orderedDelta[0] < 0.0f && orderedDelta[1] > 0.0f &&
          orderedDelta[2] < 0.0f) || ledgerError(sequence.diagnostics(0.2)) > tolerance) {
        throw std::runtime_error("Opposite input pulses were reordered or merged.");
    }

    ContainerMotionSettings boundedSettings = sequenceSettings;
    boundedSettings.maximumQueuedTime = 0.035;
    ContainerMotionInput bounded(boundedSettings);
    static_cast<void>(bounded.addPositionSample(0.0, 0.0, 0.0));
    for (std::size_t index = 1; index <= 12; ++index) {
        const double time = static_cast<double>(index) * 0.01;
        const double position = index % 2 == 0 ? 0.0 : 0.01;
        static_cast<void>(bounded.addPositionSample(position, 0.0, time));
    }
    const auto boundedDiagnostics = bounded.diagnostics(0.12);
    if (boundedDiagnostics.droppedSegments == 0 ||
        boundedDiagnostics.pendingTime > boundedSettings.maximumQueuedTime + 0.0101 ||
        ledgerError(boundedDiagnostics) > tolerance) {
        throw std::runtime_error("Bounded input history did not preserve its drop ledger.");
    }
    bounded.reset();
    const auto resetDiagnostics = bounded.diagnostics(1.0);
    if (resetDiagnostics.pendingSegments != 0 ||
        resetDiagnostics.preparedVersion != 0 ||
        resetDiagnostics.committedVelocityDelta.x != 0.0f) {
        throw std::runtime_error("Reset retained a batch input transaction.");
    }

    output << "Ordered pulse deltas=" << orderedDelta[0] << ',' << orderedDelta[1]
           << ',' << orderedDelta[2] << ", overload dropped segments="
           << boundedDiagnostics.droppedSegments << ".\n"
           << "Desktop batch input audit passed: committed impulse equals the exact "
              "successful uniform-acceleration prefix; cancel, retry, reset, and overload "
              "preserve the ledger.\n";
    return 0;
}

int runDesktopSchedulerValidation(std::ostream& output) {
    constexpr double outerStep = 1.0 / 120.0;
    const auto simulate = [&](const std::vector<double>& frameDurations) {
        fluidlab::desktop::DesktopPhysicsClock clock(outerStep);
        double wallTime = 0.0;
        double committedTime = 0.0;
        std::size_t maximumBatch = 0;
        clock.reset(wallTime, committedTime);
        for (const double frameDuration : frameDurations) {
            wallTime += frameDuration;
            clock.update(wallTime, committedTime, false);
            const std::size_t batch = clock.planBatch(4, false, false);
            maximumBatch = std::max(maximumBatch, batch);
            committedTime += static_cast<double>(batch) * outerStep;
            clock.update(wallTime, committedTime, false);
        }
        return std::pair{clock.snapshot(), maximumBatch};
    };

    for (const double fps : {30.0, 60.0, 120.0, 144.0}) {
        std::vector<double> frames(static_cast<std::size_t>(fps * 4.0), 1.0 / fps);
        const auto [snapshot, maximumBatch] = simulate(frames);
        if (std::abs(snapshot.physicsRate - 1.0) > 0.012 ||
            snapshot.physicsBacklog > outerStep * 1.01 || maximumBatch > 4) {
            throw std::runtime_error("Desktop physics clock remained coupled to render cadence.");
        }
        const double legacyRate = std::min(1.0, fps / 120.0);
        output << std::fixed << std::setprecision(4)
               << "Scheduler cadence " << fps << " FPS: legacy async rate="
               << legacyRate << ", budgeted rate=" << snapshot.physicsRate
               << ", backlog=" << snapshot.physicsBacklog
               << ", max batch=" << maximumBatch << ".\n";
    }

    std::vector<double> irregular;
    irregular.reserve(240);
    for (std::size_t index = 0; index < 240; ++index) {
        irregular.push_back(index == 80 ? 0.100 :
            (index % 3 == 0 ? 1.0 / 30.0 : 1.0 / 120.0));
    }
    const auto [irregularSnapshot, irregularBatch] = simulate(irregular);
    if (std::abs(irregularSnapshot.physicsRate - 1.0) > 0.02 || irregularBatch > 4) {
        throw std::runtime_error("Irregular frame pacing changed the physical clock rate.");
    }

    fluidlab::desktop::DesktopPhysicsClock pausedClock(outerStep);
    pausedClock.reset(0.0, 0.0);
    pausedClock.update(1.0, 0.0, true);
    if (pausedClock.snapshot().physicsBacklog != 0.0) {
        throw std::runtime_error("Paused wall time accumulated a physics backlog.");
    }
    pausedClock.update(1.5, 0.0, false);
    if (std::abs(pausedClock.snapshot().physicsBacklog - 0.5) > 1.0e-12) {
        throw std::runtime_error("Physics clock did not resume from a clean wall-time baseline.");
    }
    pausedClock.reset(4.0, 2.0);
    if (pausedClock.snapshot().physicsBacklog != 0.0) {
        throw std::runtime_error("Physics clock reset retained stale backlog.");
    }

    ContainerMotionInput motion;
    static_cast<void>(motion.addPositionSample(0.0, 0.0, 0.0));
    static_cast<void>(motion.addPositionSample(48.0, 0.0, 0.016));
    const auto prepared = motion.prepareStep(0.016, 2.0 * outerStep, true);
    if (!motion.commitPreparedFraction(prepared.version, 0.5)) {
        throw std::runtime_error("Partial batch input commit failed.");
    }
    const auto remaining = motion.prepareStep(0.024, outerStep, true);
    if (!std::isfinite(remaining.externalVelocityDelta.x)) {
        throw std::runtime_error("Partial batch commit corrupted the motion queue.");
    }
    motion.discardPreparedStep();

    output << "Desktop scheduler validation passed: wall-clock budgeting is render-rate "
              "independent, bounded to four steps, and pause/reset/input transactions are safe.\n";
    return 0;
}

int runDesktopAsyncThroughputValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    GLFWwindow* sharingWindow) {
    constexpr std::size_t width = 48, height = 32, outerSteps = 24;
    constexpr float cellSize = 1.0f / 48.0f;
    const auto run = [&](std::size_t batchSize) {
        MacGrid grid(width, height, cellSize, cellSize);
        MarkerParticleSystem particles;
        fluidlab::simulation::initializeDamBreak(grid);
        particles.seedFluidCellsExact(grid, 8);
        fluidlab::simulation::GpuAsyncDynamicFreeSurfaceStepper runner(
            sharingWindow, shaderDirectory / "mac_divergence.comp");
        static_cast<void>(runner.initialize(grid, particles));
        const auto start = std::chrono::steady_clock::now();
        std::size_t completed = 0;
        while (completed < outerSteps) {
            const std::size_t requested = std::min(batchSize, outerSteps - completed);
            if (!runner.submitBatch({}, requested)) {
                throw std::runtime_error("Asynchronous batch submission failed.");
            }
            while (!runner.poll()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            const auto report = runner.report();
            if (report.failureReason != GpuDynamicFailureReason::none ||
                report.batchOuterStepsCompleted != requested) {
                throw std::runtime_error("Asynchronous batch did not complete its requested steps.");
            }
            completed += report.batchOuterStepsCompleted;
            runner.acknowledge();
        }
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return std::tuple{milliseconds,
            runner.stepper().grid().readHorizontalVelocity(),
            runner.stepper().grid().readVerticalVelocity(),
            runner.statistics()};
    };

    const auto [singleMilliseconds, singleU, singleV, singleStatistics] = run(1);
    const auto [batchMilliseconds, batchU, batchV, batchStatistics] = run(2);
    double maximumDifference = 0.0;
    for (std::size_t index = 0; index < singleU.size(); ++index) {
        maximumDifference = std::max(maximumDifference,
            std::abs(static_cast<double>(singleU[index] - batchU[index])));
    }
    for (std::size_t index = 0; index < singleV.size(); ++index) {
        maximumDifference = std::max(maximumDifference,
            std::abs(static_cast<double>(singleV[index] - batchV[index])));
    }
    if (maximumDifference > 1.0e-6 || batchStatistics.lastBatchCompleted != 2 ||
        !std::isfinite(batchStatistics.lastSnapshotPublishMilliseconds) ||
        !std::isfinite(batchStatistics.lastWorkerCompletionToPublishMilliseconds)) {
        throw std::runtime_error("Batched asynchronous stepping changed the formal GPU result.");
    }
    output << std::fixed << std::setprecision(3)
           << "Desktop async throughput validation: 24 steps single/batch2 wall="
           << singleMilliseconds << '/' << batchMilliseconds
           << " ms, task count=" << singleStatistics.completedTaskCount << '/'
           << batchStatistics.completedTaskCount << ", max velocity difference="
           << std::scientific << maximumDifference << std::fixed
           << ", snapshot publish=" << batchStatistics.lastSnapshotPublishMilliseconds
           << " ms, worker-to-publish="
           << batchStatistics.lastWorkerCompletionToPublishMilliseconds << " ms.\n"
           << "Desktop async throughput validation passed.\n";
    return 0;
}

int runDesktopThroughputAccountingValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath,
    GLFWwindow* sharingWindow) {
    constexpr std::size_t width = 64, height = 48, outerSteps = 60;
    constexpr float cellSize = 1.0f / 64.0f;
    constexpr double physicalTime = static_cast<double>(outerSteps) / 120.0;
    const auto loaded = fluidlab::desktop::loadDesktopSettledState(
        statePath, width, height, cellSize, cellSize);
    if (!loaded.valid) {
        throw std::runtime_error("Throughput accounting state is unavailable: " + loaded.message);
    }

    struct Result final {
        double wallMilliseconds = 0.0;
        double stateReadbackMilliseconds = 0.0;
        std::size_t tasks = 0;
        std::size_t rejected = 0;
        double workerBusyMilliseconds = 0.0;
        double workerIdleMilliseconds = 0.0;
        double publishMilliseconds = 0.0;
        double workerToPublishMilliseconds = 0.0;
        std::vector<float> horizontal;
        std::vector<float> vertical;
        std::vector<MarkerParticle> particles;
        std::vector<fluidlab::simulation::CellType> cellTypes;
        GpuDynamicFreeSurfaceReport report;
    };
    const auto initialState = [&]() {
        MacGrid grid(width, height, cellSize, cellSize);
        MarkerParticleSystem particles;
        fluidlab::desktop::applyDesktopSettledState(loaded.state, grid, particles);
        return std::pair{std::move(grid), std::move(particles)};
    };
    const auto capture = [](auto& stepper, Result& result) {
        const auto readbackStart = std::chrono::steady_clock::now();
        result.horizontal = stepper.grid().readHorizontalVelocity();
        result.vertical = stepper.grid().readVerticalVelocity();
        result.particles = stepper.grid().readMarkerParticles();
        result.cellTypes = stepper.grid().readCellTypes();
        result.stateReadbackMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - readbackStart).count();
    };
    const auto runDirect = [&]() {
        auto [grid, particles] = initialState();
        GpuDynamicFreeSurfaceSettings settings;
        settings.synchronousGpuTimingEnabled = false;
        settings.asynchronousGpuTimingEnabled = false;
        GpuDynamicFreeSurfaceStepper runner(
            shaderDirectory / "mac_divergence.comp", settings);
        static_cast<void>(runner.initialize(grid, particles));
        Result result;
        const auto start = std::chrono::steady_clock::now();
        while (runner.completedOuterSteps() < outerSteps) {
            result.report = runner.step();
            if (result.report.failureReason != GpuDynamicFailureReason::none) {
                throw std::runtime_error("Direct throughput reference failed.");
            }
            result.rejected += result.report.rejectedSubsteps;
        }
        result.wallMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        result.tasks = outerSteps;
        capture(runner, result);
        return result;
    };
    const auto runAsync = [&](std::size_t batchSize) {
        auto [grid, particles] = initialState();
        fluidlab::simulation::GpuAsyncDynamicFreeSurfaceStepper runner(
            sharingWindow, shaderDirectory / "mac_divergence.comp");
        static_cast<void>(runner.initialize(grid, particles));
        Result result;
        const auto start = std::chrono::steady_clock::now();
        std::size_t completed = 0;
        while (completed < outerSteps) {
            const std::size_t requested = std::min(batchSize, outerSteps - completed);
            if (!runner.submitBatch({}, requested)) {
                throw std::runtime_error("Unified asynchronous throughput submission failed.");
            }
            while (!runner.poll()) std::this_thread::yield();
            result.report = runner.report();
            if (result.report.failureReason != GpuDynamicFailureReason::none ||
                result.report.batchOuterStepsCompleted != requested) {
                throw std::runtime_error("Unified asynchronous throughput batch failed.");
            }
            completed += result.report.batchOuterStepsCompleted;
            result.rejected += result.report.rejectedSubsteps;
            runner.acknowledge();
        }
        result.wallMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        const auto statistics = runner.statistics();
        result.tasks = statistics.completedTaskCount;
        result.workerBusyMilliseconds = statistics.workerBusyMilliseconds;
        result.workerIdleMilliseconds = statistics.workerIdleMilliseconds;
        result.publishMilliseconds = statistics.lastSnapshotPublishMilliseconds;
        result.workerToPublishMilliseconds =
            statistics.lastWorkerCompletionToPublishMilliseconds;
        capture(runner.stepper(), result);
        return result;
    };
    const auto direct = runDirect();
    const auto single = runAsync(1);
    const auto batch2 = runAsync(2);
    const auto batch4 = runAsync(4);

    const auto compare = [&](const Result& candidate) {
        double maximum = 0.0;
        for (std::size_t index = 0; index < direct.horizontal.size(); ++index) {
            maximum = std::max(maximum, std::abs(static_cast<double>(
                direct.horizontal[index] - candidate.horizontal[index])));
        }
        for (std::size_t index = 0; index < direct.vertical.size(); ++index) {
            maximum = std::max(maximum, std::abs(static_cast<double>(
                direct.vertical[index] - candidate.vertical[index])));
        }
        if (direct.cellTypes != candidate.cellTypes ||
            direct.particles.size() != candidate.particles.size()) {
            return std::numeric_limits<double>::infinity();
        }
        for (std::size_t index = 0; index < direct.particles.size(); ++index) {
            maximum = std::max(maximum, std::abs(static_cast<double>(
                direct.particles[index].position.x - candidate.particles[index].position.x)));
            maximum = std::max(maximum, std::abs(static_cast<double>(
                direct.particles[index].position.y - candidate.particles[index].position.y)));
        }
        return maximum;
    };
    const double singleError = compare(single);
    const double batch2Error = compare(batch2);
    const double batch4Error = compare(batch4);
    if (singleError > 1.0e-6 || batch2Error > 1.0e-6 || batch4Error > 1.0e-6) {
        throw std::runtime_error("Unified throughput modes produced different physical states.");
    }

    output << "Desktop unified throughput accounting (64x48 settled, 60 steps, 0.5 s)\n"
              "Mode WallMs PhysicsRate Tasks WorkerBusyMs WorkerIdleMs ReadbackMs Rejected StateMax\n";
    const auto print = [&](const char* name, const Result& result, double error) {
        output << std::fixed << std::setprecision(3) << name << ' '
               << result.wallMilliseconds << ' '
               << physicalTime / (result.wallMilliseconds * 1.0e-3) << ' '
               << result.tasks << ' ' << result.workerBusyMilliseconds << ' '
               << result.workerIdleMilliseconds << ' ' << result.stateReadbackMilliseconds << ' '
               << result.rejected << ' ' << std::scientific << error << std::fixed << '\n';
    };
    print("direct", direct, 0.0);
    print("async1", single, singleError);
    print("batch2", batch2, batch2Error);
    print("batch4", batch4, batch4Error);
    output << "Last batch2 publish/worker-to-publish=" << batch2.publishMilliseconds << '/'
           << batch2.workerToPublishMilliseconds << " ms; initialization, state load, shader "
              "construction, and final readback are excluded from WallMs and reported separately.\n"
              "Desktop throughput accounting validation passed.\n";
    return 0;
}

int runDesktopSustainedRealtimeBenchmark(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory,
    const std::filesystem::path& statePath,
    GLFWwindow* sharingWindow) {
    constexpr std::size_t width = 64, height = 48;
    constexpr float cellSize = 1.0f / 64.0f;
    constexpr double outerStep = 1.0 / 120.0;
    constexpr double sampleSeconds = 10.0;
    const auto loaded = fluidlab::desktop::loadDesktopSettledState(
        statePath, width, height, cellSize, cellSize);
    if (!loaded.valid) {
        throw std::runtime_error("Sustained benchmark state is unavailable: " + loaded.message);
    }
    struct Result final {
        const char* name = "";
        double wallSeconds = 0.0;
        double committedSeconds = 0.0;
        double physicsRate = 0.0;
        double finalBacklog = 0.0;
        double maximumBacklog = 0.0;
        double averageBacklog = 0.0;
        double firstHalfBacklog = 0.0;
        double secondHalfBacklog = 0.0;
        double workerBusyMilliseconds = 0.0;
        double workerIdleMilliseconds = 0.0;
        double maximumRepeatedMilliseconds = 0.0;
        std::size_t ticks = 0;
        std::size_t publications = 0;
        std::size_t repeatedTicks = 0;
        std::array<std::size_t, 5> batches{};
    };
    const auto run = [&](const char* name, double tickSeconds, std::size_t maximumBatch) {
        MacGrid grid(width, height, cellSize, cellSize);
        MarkerParticleSystem particles;
        fluidlab::desktop::applyDesktopSettledState(loaded.state, grid, particles);
        fluidlab::simulation::GpuAsyncDynamicFreeSurfaceStepper runner(
            sharingWindow, shaderDirectory / "mac_divergence.comp");
        static_cast<void>(runner.initialize(grid, particles));
        fluidlab::desktop::DesktopPhysicsClock clock(outerStep);
        Result result;
        result.name = name;
        const auto start = std::chrono::steady_clock::now();
        auto nextTick = start;
        clock.reset(0.0, 0.0);
        std::size_t repeatedRun = 0;
        double backlogSum = 0.0;
        double firstHalfBacklogSum = 0.0;
        double secondHalfBacklogSum = 0.0;
        std::size_t firstHalfSamples = 0;
        std::size_t secondHalfSamples = 0;
        bool candidateInFlight = false;
        while (result.wallSeconds < sampleSeconds) {
            nextTick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(tickSeconds));
            std::this_thread::sleep_until(nextTick);
            const auto now = std::chrono::steady_clock::now();
            result.wallSeconds = std::chrono::duration<double>(now - start).count();
            bool published = false;
            if (candidateInFlight && runner.poll()) {
                const auto report = runner.report();
                if (report.failureReason != GpuDynamicFailureReason::none) {
                    throw std::runtime_error("Sustained asynchronous candidate failed.");
                }
                result.committedSeconds = report.simulationTime;
                ++result.publications;
                published = true;
                runner.acknowledge();
                candidateInFlight = false;
            }
            clock.update(result.wallSeconds, result.committedSeconds, false);
            const auto beforeSubmit = clock.snapshot();
            result.maximumBacklog = std::max(
                result.maximumBacklog, beforeSubmit.physicsBacklog);
            backlogSum += beforeSubmit.physicsBacklog;
            if (result.wallSeconds <= sampleSeconds * 0.5) {
                firstHalfBacklogSum += beforeSubmit.physicsBacklog;
                ++firstHalfSamples;
            } else {
                secondHalfBacklogSum += beforeSubmit.physicsBacklog;
                ++secondHalfSamples;
            }
            const std::size_t batch = clock.planBatch(
                maximumBatch, candidateInFlight,
                runner.stepper().pendingPhysicalTime() > 0.0);
            if (batch > 0 && !candidateInFlight) {
                if (!runner.submitBatch({}, batch)) {
                    throw std::runtime_error("Sustained asynchronous submission failed.");
                }
                ++result.batches[batch];
                candidateInFlight = true;
            }
            if (published) {
                repeatedRun = 0;
            } else {
                ++result.repeatedTicks;
                ++repeatedRun;
                result.maximumRepeatedMilliseconds = std::max(
                    result.maximumRepeatedMilliseconds,
                    static_cast<double>(repeatedRun) * tickSeconds * 1000.0);
            }
            ++result.ticks;
        }
        clock.update(result.wallSeconds, result.committedSeconds, false);
        const auto finalClock = clock.snapshot();
        result.finalBacklog = finalClock.physicsBacklog;
        result.physicsRate = result.wallSeconds > 0.0
            ? result.committedSeconds / result.wallSeconds : 0.0;
        result.averageBacklog = result.ticks > 0
            ? backlogSum / static_cast<double>(result.ticks) : 0.0;
        result.firstHalfBacklog = firstHalfSamples > 0
            ? firstHalfBacklogSum / static_cast<double>(firstHalfSamples) : 0.0;
        result.secondHalfBacklog = secondHalfSamples > 0
            ? secondHalfBacklogSum / static_cast<double>(secondHalfSamples) : 0.0;
        const auto statistics = runner.statistics();
        result.workerBusyMilliseconds = statistics.workerBusyMilliseconds;
        result.workerIdleMilliseconds = statistics.workerIdleMilliseconds;
        if (candidateInFlight) {
            runner.requestCancel();
            while (!runner.poll()) std::this_thread::yield();
        }
        return result;
    };

    const std::array<Result, 3> results{
        run("60Hz-max1", 1.0 / 60.0, 1),
        run("60Hz-max2", 1.0 / 60.0, 2),
        run("30Hz-max4", 1.0 / 30.0, 4)};
    output << "Desktop sustained hidden-context benchmark (64x48 settled, 10 s wall each)\n"
              "Mode Wall Physical Rate Backlog Avg/First/Second/Max Publications Repeated MaxRepeatMs "
              "Batch1/2/3/4 WorkerBusyMs WorkerIdleMs\n";
    for (const auto& result : results) {
        output << std::fixed << std::setprecision(3) << result.name << ' '
               << result.wallSeconds << ' ' << result.committedSeconds << ' '
               << result.physicsRate << ' ' << result.finalBacklog << ' '
               << result.averageBacklog << '/' << result.firstHalfBacklog << '/'
               << result.secondHalfBacklog << '/' << result.maximumBacklog << ' '
               << result.publications << ' '
               << result.repeatedTicks << ' ' << result.maximumRepeatedMilliseconds << ' '
               << result.batches[1] << '/' << result.batches[2] << '/'
               << result.batches[3] << '/' << result.batches[4] << ' '
               << result.workerBusyMilliseconds << ' ' << result.workerIdleMilliseconds << '\n';
    }
    if (results[1].physicsRate < 0.95 || results[1].finalBacklog > 0.15 ||
        results[2].physicsRate < 0.95 || results[2].finalBacklog > 0.15) {
        throw std::runtime_error("Default sustained scheduling did not maintain realtime physics.");
    }
    output << "Sustained benchmark passed for hidden-context 60 Hz max2 and simulated "
              "30 Hz modal pacing. Visible rendering and real title-bar dragging remain manual tests.\n";
    return 0;
}

} // namespace fluidlab::diagnostics
