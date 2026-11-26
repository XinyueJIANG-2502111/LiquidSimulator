#pragma once

#include "PressureProjection.h"
#include "VelocityAdvection.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace fluidlab::simulation {

struct FluidStepSettings final {
    float timeStep = 1.0f / 60.0f;
    float density = 1.0f;
    Vector2 externalAcceleration{};
    bool pressureWarmStartEnabled = true;
    std::size_t pressureMeanRemovalInterval = 0;
    float pressureResidualTolerance = 5.0e-3f;
    float pressureRelaxationFactor = 1.7f;
    std::size_t pressureMaximumIterations = 4000;
    std::size_t pressureResidualCheckInterval = 10;
};

struct FluidStepTimings final {
    double externalForcesMilliseconds = 0.0;
    double velocityAdvectionMilliseconds = 0.0;
    double advectionDiagnosticsMilliseconds = 0.0;
    double pressureProjectionMilliseconds = 0.0;
    double totalMilliseconds = 0.0;
};

struct FluidStepReport final {
    float kineticEnergyBefore = 0.0f;
    float kineticEnergyAfterAdvection = 0.0f;
    float kineticEnergyAfterProjection = 0.0f;
    DivergenceStatistics divergenceAfterAdvection;
    ProjectionReport projection;
    FluidStepTimings timings;
};

class CpuFluidStepper final {
public:
    explicit CpuFluidStepper(FluidStepSettings settings = {});

    [[nodiscard]] FluidStepReport step(MacGrid& grid);
    [[nodiscard]] const FluidStepSettings& settings() const noexcept;
    [[nodiscard]] const PressureSolverSettings& pressureSettings() const noexcept;
    [[nodiscard]] const std::vector<float>& divergenceAfterAdvectionField() const noexcept;
    [[nodiscard]] std::size_t workspaceAllocationCount() const noexcept;
    [[nodiscard]] std::size_t workspaceBytes() const noexcept;
    void invalidatePressureWarmStart(MacGrid& grid) const noexcept;

    [[nodiscard]] static float meanKineticEnergy(const MacGrid& grid) noexcept;

private:
    void applyExternalAcceleration(MacGrid& grid) const noexcept;
    MacGrid& prepareAdvectionWorkspace(const MacGrid& grid);

    FluidStepSettings m_settings;
    VelocityAdvector m_advector;
    PressureProjector m_projector;
    std::unique_ptr<MacGrid> m_advectionWorkspace;
    std::vector<float> m_divergenceAfterAdvection;
    std::size_t m_workspaceAllocationCount = 0;
};

} // namespace fluidlab::simulation
