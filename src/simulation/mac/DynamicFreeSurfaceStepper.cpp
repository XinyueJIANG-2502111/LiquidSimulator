#include "DynamicFreeSurfaceStepper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace fluidlab::simulation {

DynamicFreeSurfaceStepper::DynamicFreeSurfaceStepper(DynamicFreeSurfaceSettings settings)
    : m_settings(settings) {
    if (!std::isfinite(settings.timeStep) || settings.timeStep <= 0.0f ||
        !std::isfinite(settings.density) || settings.density <= 0.0f ||
        !std::isfinite(settings.gravity.x) || !std::isfinite(settings.gravity.y) ||
        !std::isfinite(settings.maximumCfl) || settings.maximumCfl <= 0.0f ||
        settings.maximumSubsteps == 0 || settings.extrapolationLayers == 0) {
        throw std::invalid_argument("Invalid dynamic free-surface settings.");
    }
}

DynamicFreeSurfaceReport DynamicFreeSurfaceStepper::step(
    MacGrid& grid,
    MarkerParticleSystem& particles) {
    return step(grid, particles, {});
}

DynamicFreeSurfaceReport DynamicFreeSurfaceStepper::step(
    MacGrid& grid,
    MarkerParticleSystem& particles,
    const SimulationStepInput& input) {
    if (!isFinite(input)) {
        throw std::invalid_argument("The external acceleration input is invalid.");
    }
    using Clock = std::chrono::steady_clock;
    const auto totalStart = Clock::now();
    DynamicFreeSurfaceReport report;
    report.particleCountBefore = particles.validParticleCount();
    const float minimumCellSize = std::min(grid.cellWidth(), grid.cellHeight());
    const Vector2 acceleration{
        m_settings.gravity.x + input.externalAcceleration.x,
        m_settings.gravity.y + input.externalAcceleration.y};
    const float accelerationMagnitude = std::hypot(acceleration.x, acceleration.y);
    const float estimatedSpeed = maximumSpeed(grid) + accelerationMagnitude * m_settings.timeStep;
    const float estimatedCfl = estimatedSpeed * m_settings.timeStep / minimumCellSize;
    const std::size_t requiredSubsteps = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(estimatedCfl / m_settings.maximumCfl)));
    report.substeps = std::min(requiredSubsteps, m_settings.maximumSubsteps);
    report.cflLimitReached = requiredSubsteps > m_settings.maximumSubsteps;
    const float substepTime = m_settings.timeStep / static_cast<float>(report.substeps);

    for (std::size_t substep = 0; substep < report.substeps; ++substep) {
        advanceSubstep(grid, particles, substepTime, acceleration, report);
        m_simulationTime += substepTime;
    }
    report.particleCountAfter = particles.validParticleCount();
    report.fluidCellCount = fluidCellCount(grid);
    report.isolatedFluidCellCount = isolatedFluidCellCount(grid);
    report.markedArea = static_cast<float>(report.fluidCellCount) *
        grid.cellWidth() * grid.cellHeight();
    report.maximumSpeed = maximumSpeed(grid);
    report.simulationTime = m_simulationTime;
    report.timings.totalMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - totalStart).count();
    return report;
}

const DynamicFreeSurfaceSettings& DynamicFreeSurfaceStepper::settings() const noexcept {
    return m_settings;
}

double DynamicFreeSurfaceStepper::simulationTime() const noexcept { return m_simulationTime; }
void DynamicFreeSurfaceStepper::resetTime() noexcept { m_simulationTime = 0.0; }

float DynamicFreeSurfaceStepper::maximumSpeed(const MacGrid& grid) noexcept {
    float maximum = 0.0f;
    for (std::size_t y = 0; y < grid.height(); ++y) for (std::size_t x = 0; x < grid.width(); ++x) {
        if (grid.cellType(x, y) != CellType::fluid) continue;
        const Vector2 velocity = grid.cellCenteredVelocity(x, y);
        maximum = std::max(maximum, std::hypot(velocity.x, velocity.y));
    }
    return maximum;
}

std::size_t DynamicFreeSurfaceStepper::fluidCellCount(const MacGrid& grid) noexcept {
    return static_cast<std::size_t>(std::count(
        grid.cellTypes().begin(), grid.cellTypes().end(), CellType::fluid));
}

std::size_t DynamicFreeSurfaceStepper::isolatedFluidCellCount(const MacGrid& grid) noexcept {
    std::size_t isolated = 0;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (grid.cellType(x, y) != CellType::fluid) continue;
            bool connected = false;
            if (x > 0) connected = connected || grid.cellType(x - 1, y) == CellType::fluid;
            if (x + 1 < grid.width()) connected = connected || grid.cellType(x + 1, y) == CellType::fluid;
            if (y > 0) connected = connected || grid.cellType(x, y - 1) == CellType::fluid;
            if (y + 1 < grid.height()) connected = connected || grid.cellType(x, y + 1) == CellType::fluid;
            isolated += connected ? 0U : 1U;
        }
    }
    return isolated;
}

void DynamicFreeSurfaceStepper::advanceSubstep(
    MacGrid& grid,
    MarkerParticleSystem& particles,
    float timeStep,
    Vector2 acceleration,
    DynamicFreeSurfaceReport& report) {
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    particles.rebuildFluidClassification(grid);
    auto end = Clock::now();
    report.timings.classificationMilliseconds +=
        std::chrono::duration<double, std::milli>(end - start).count();

    start = end;
    m_extrapolator.extrapolate(grid, m_settings.extrapolationLayers);
    end = Clock::now();
    report.timings.velocityExtrapolationMilliseconds +=
        std::chrono::duration<double, std::milli>(end - start).count();

    start = end;
    const bool workspaceNeeded = m_workspace == nullptr || !m_workspace->hasSameLayout(grid);
    MacGrid& advected = prepareWorkspace(grid);
    report.timings.workspaceAllocations += workspaceNeeded ? 1U : 0U;
    m_advector.advect(grid, advected, timeStep);
    grid.swapVelocityFields(advected);
    end = Clock::now();
    report.timings.velocityAdvectionMilliseconds +=
        std::chrono::duration<double, std::milli>(end - start).count();

    start = end;
    applyAcceleration(grid, acceleration, timeStep);
    grid.enforceSolidBoundaries();
    end = Clock::now();
    report.timings.accelerationAndBoundaryMilliseconds +=
        std::chrono::duration<double, std::milli>(end - start).count();

    FreeSurfacePressureProjector projector(PressureSolverSettings{
        .density = m_settings.density,
        .timeStep = timeStep,
        .residualTolerance = m_settings.pressureResidualTolerance,
        .maximumIterations = m_settings.pressureMaximumIterations,
        .warmStartEnabled = m_settings.pressureWarmStartEnabled,
        .pressureMeanRemovalInterval = 0,
    });
    start = end;
    report.projection = projector.project(grid);
    ++report.pressureSolveCount;
    report.pressureIterationsTotal += report.projection.projection.pressureSolve.iterations;
    report.pressureIterationsMaximum = std::max(
        report.pressureIterationsMaximum, report.projection.projection.pressureSolve.iterations);
    report.pressureConvergedCount +=
        report.projection.projection.pressureSolve.converged ? 1U : 0U;
    report.maximumFinalResidualRms = std::max(
        report.maximumFinalResidualRms,
        report.projection.projection.pressureSolve.finalResidual.rootMeanSquare);
    report.maximumFinalResidualAbsolute = std::max(
        report.maximumFinalResidualAbsolute,
        report.projection.projection.pressureSolve.finalResidual.maximumAbsolute);
    report.maximumPostProjectionDivergence = std::max(
        report.maximumPostProjectionDivergence,
        report.projection.projection.divergenceAfter.maximumAbsolute);
    end = Clock::now();
    report.timings.divergenceMilliseconds +=
        report.projection.projection.timings.divergenceBeforeMilliseconds;
    report.timings.pressureSolveMilliseconds +=
        report.projection.projection.timings.pressureSolveMilliseconds;
    report.timings.pressureApplicationMilliseconds +=
        report.projection.projection.timings.pressureApplicationMilliseconds;

    start = end;
    m_extrapolator.extrapolate(grid, m_settings.extrapolationLayers);
    end = Clock::now();
    report.timings.velocityExtrapolationMilliseconds +=
        std::chrono::duration<double, std::milli>(end - start).count();

    start = end;
    const ParticleAdvectionReport particleReport = particles.advectRk2(grid, timeStep);
    end = Clock::now();
    report.timings.particleAdvectionAndCollisionMilliseconds +=
        std::chrono::duration<double, std::milli>(end - start).count();
    report.particleAdvection.particleCount = particleReport.particleCount;
    report.particleAdvection.collisionCount += particleReport.collisionCount;
    report.particleAdvection.invalidPositionCount += particleReport.invalidPositionCount;
    start = end;
    particles.rebuildFluidClassification(grid);
    end = Clock::now();
    report.timings.classificationMilliseconds +=
        std::chrono::duration<double, std::milli>(end - start).count();
    report.maximumObservedCfl = std::max(
        report.maximumObservedCfl,
        maximumSpeed(grid) * timeStep / std::min(grid.cellWidth(), grid.cellHeight()));
}

void DynamicFreeSurfaceStepper::applyAcceleration(
    MacGrid& grid,
    Vector2 acceleration,
    float timeStep) const noexcept {
    const float du = acceleration.x * timeStep;
    const float dv = acceleration.y * timeStep;
    for (std::size_t y = 0; y < grid.height(); ++y) for (std::size_t x = 1; x < grid.width(); ++x) {
        const CellType left = grid.cellType(x - 1, y), right = grid.cellType(x, y);
        if (left != CellType::solid && right != CellType::solid &&
            (left == CellType::fluid || right == CellType::fluid)) {
            grid.horizontalVelocity(x, y) += du;
        }
    }
    for (std::size_t y = 1; y < grid.height(); ++y) for (std::size_t x = 0; x < grid.width(); ++x) {
        const CellType below = grid.cellType(x, y - 1), above = grid.cellType(x, y);
        if (below != CellType::solid && above != CellType::solid &&
            (below == CellType::fluid || above == CellType::fluid)) {
            grid.verticalVelocity(x, y) += dv;
        }
    }
}

MacGrid& DynamicFreeSurfaceStepper::prepareWorkspace(const MacGrid& grid) {
    if (m_workspace == nullptr || !m_workspace->hasSameLayout(grid)) {
        m_workspace = std::make_unique<MacGrid>(
            grid.width(), grid.height(), grid.cellWidth(), grid.cellHeight());
    }
    return *m_workspace;
}

} // namespace fluidlab::simulation
