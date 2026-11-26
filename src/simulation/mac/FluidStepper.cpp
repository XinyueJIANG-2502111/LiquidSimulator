#include "FluidStepper.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace fluidlab::simulation {

CpuFluidStepper::CpuFluidStepper(FluidStepSettings settings)
    : m_settings(settings),
      m_projector(PressureSolverSettings{
          .density = settings.density,
          .timeStep = settings.timeStep,
          .residualTolerance = settings.pressureResidualTolerance,
          .relaxationFactor = settings.pressureRelaxationFactor,
          .maximumIterations = settings.pressureMaximumIterations,
          .residualCheckInterval = settings.pressureResidualCheckInterval,
          .warmStartEnabled = settings.pressureWarmStartEnabled,
          .pressureMeanRemovalInterval = settings.pressureMeanRemovalInterval,
      }) {
    if (!std::isfinite(settings.timeStep) || settings.timeStep <= 0.0f) {
        throw std::invalid_argument("Fluid step timestep must be a positive finite value.");
    }
    if (!std::isfinite(settings.density) || settings.density <= 0.0f) {
        throw std::invalid_argument("Fluid density must be a positive finite value.");
    }
    if (!std::isfinite(settings.externalAcceleration.x) ||
        !std::isfinite(settings.externalAcceleration.y)) {
        throw std::invalid_argument("External acceleration must contain finite values.");
    }
}

FluidStepReport CpuFluidStepper::step(MacGrid& grid) {
    using Clock = std::chrono::steady_clock;
    const auto totalStart = Clock::now();

    FluidStepReport report;
    report.kineticEnergyBefore = meanKineticEnergy(grid);

    // External acceleration uses an explicit Euler update before advection.
    // The fixed dt keeps this force integration independent of render timing.
    applyExternalAcceleration(grid);
    grid.enforceClosedBoundary();
    const auto forcesEnd = Clock::now();

    MacGrid& advected = prepareAdvectionWorkspace(grid);
    m_advector.advect(grid, advected, m_settings.timeStep);
    const auto advectionEnd = Clock::now();
    advected.enforceClosedBoundary();
    advected.computeDivergence();
    report.divergenceAfterAdvection = advected.divergenceStatistics();
    m_divergenceAfterAdvection.assign(advected.divergences().begin(), advected.divergences().end());
    report.kineticEnergyAfterAdvection = meanKineticEnergy(advected);
    const auto diagnosticsEnd = Clock::now();

    grid.swapVelocityFields(advected);
    report.projection = m_projector.project(grid);
    grid.enforceClosedBoundary();
    grid.computeDivergence();
    report.kineticEnergyAfterProjection = meanKineticEnergy(grid);
    const auto totalEnd = Clock::now();

    report.timings.externalForcesMilliseconds =
        std::chrono::duration<double, std::milli>(forcesEnd - totalStart).count();
    report.timings.velocityAdvectionMilliseconds =
        std::chrono::duration<double, std::milli>(advectionEnd - forcesEnd).count();
    report.timings.advectionDiagnosticsMilliseconds =
        std::chrono::duration<double, std::milli>(diagnosticsEnd - advectionEnd).count();
    report.timings.pressureProjectionMilliseconds =
        std::chrono::duration<double, std::milli>(totalEnd - diagnosticsEnd).count();
    report.timings.totalMilliseconds =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    return report;
}

const FluidStepSettings& CpuFluidStepper::settings() const noexcept {
    return m_settings;
}

const PressureSolverSettings& CpuFluidStepper::pressureSettings() const noexcept {
    return m_projector.settings();
}

const std::vector<float>& CpuFluidStepper::divergenceAfterAdvectionField() const noexcept {
    return m_divergenceAfterAdvection;
}

std::size_t CpuFluidStepper::workspaceAllocationCount() const noexcept {
    return m_workspaceAllocationCount;
}

std::size_t CpuFluidStepper::workspaceBytes() const noexcept {
    const std::size_t gridBytes = m_advectionWorkspace != nullptr
        ? m_advectionWorkspace->storageBytes()
        : 0;
    return gridBytes + m_divergenceAfterAdvection.capacity() * sizeof(float);
}

void CpuFluidStepper::invalidatePressureWarmStart(MacGrid& grid) const noexcept {
    // Call this after a discontinuous timestep, density, or boundary change.
    // Grid-size changes naturally arrive in a newly constructed grid.
    grid.clearPressure();
}

float CpuFluidStepper::meanKineticEnergy(const MacGrid& grid) noexcept {
    double energySum = 0.0;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const Vector2 velocity = grid.cellCenteredVelocity(x, y);
            energySum += 0.5 * (static_cast<double>(velocity.x) * velocity.x +
                                static_cast<double>(velocity.y) * velocity.y);
        }
    }
    return static_cast<float>(energySum / static_cast<double>(grid.width() * grid.height()));
}

void CpuFluidStepper::applyExternalAcceleration(MacGrid& grid) const noexcept {
    const float horizontalDelta = m_settings.timeStep * m_settings.externalAcceleration.x;
    const float verticalDelta = m_settings.timeStep * m_settings.externalAcceleration.y;

    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t faceX = 1; faceX < grid.width(); ++faceX) {
            grid.horizontalVelocity(faceX, y) += horizontalDelta;
        }
    }
    for (std::size_t faceY = 1; faceY < grid.height(); ++faceY) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            grid.verticalVelocity(x, faceY) += verticalDelta;
        }
    }
}

MacGrid& CpuFluidStepper::prepareAdvectionWorkspace(const MacGrid& grid) {
    if (m_advectionWorkspace == nullptr || !m_advectionWorkspace->hasSameLayout(grid)) {
        m_advectionWorkspace = std::make_unique<MacGrid>(
            grid.width(),
            grid.height(),
            grid.cellWidth(),
            grid.cellHeight());
        m_divergenceAfterAdvection.reserve(grid.width() * grid.height());
        ++m_workspaceAllocationCount;
    }
    return *m_advectionWorkspace;
}

} // namespace fluidlab::simulation
