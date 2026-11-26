#pragma once

#include "FreeSurfacePressureProjection.h"
#include "MarkerParticles.h"
#include "VelocityAdvection.h"
#include "VelocityExtrapolation.h"
#include "../SimulationStepInput.h"

#include <cstddef>
#include <memory>

namespace fluidlab::simulation {

struct DynamicFreeSurfaceSettings final {
    float timeStep = 1.0f / 120.0f;
    float density = 1.0f;
    Vector2 gravity{0.0f, -9.81f};
    float maximumCfl = 0.75f;
    std::size_t maximumSubsteps = 8;
    std::size_t extrapolationLayers = 3;
    float pressureResidualTolerance = 5.0e-3f;
    std::size_t pressureMaximumIterations = 4000;
    bool pressureWarmStartEnabled = true;
};

struct DynamicFreeSurfaceReport final {
    struct Timings final {
        double totalMilliseconds = 0.0;
        double classificationMilliseconds = 0.0;
        double velocityExtrapolationMilliseconds = 0.0;
        double velocityAdvectionMilliseconds = 0.0;
        double accelerationAndBoundaryMilliseconds = 0.0;
        double divergenceMilliseconds = 0.0;
        double pressureSolveMilliseconds = 0.0;
        double pressureApplicationMilliseconds = 0.0;
        double particleAdvectionAndCollisionMilliseconds = 0.0;
        std::size_t workspaceAllocations = 0;
    } timings;
    FreeSurfaceProjectionReport projection;
    ParticleAdvectionReport particleAdvection;
    std::size_t substeps = 0;
    std::size_t particleCountBefore = 0;
    std::size_t particleCountAfter = 0;
    std::size_t fluidCellCount = 0;
    std::size_t isolatedFluidCellCount = 0;
    std::size_t pressureSolveCount = 0;
    std::size_t pressureIterationsTotal = 0;
    std::size_t pressureIterationsMaximum = 0;
    std::size_t pressureConvergedCount = 0;
    float maximumFinalResidualRms = 0.0f;
    float maximumFinalResidualAbsolute = 0.0f;
    float maximumPostProjectionDivergence = 0.0f;
    float markedArea = 0.0f;
    float maximumSpeed = 0.0f;
    float maximumObservedCfl = 0.0f;
    double simulationTime = 0.0;
    bool cflLimitReached = false;
};

class DynamicFreeSurfaceStepper final {
public:
    explicit DynamicFreeSurfaceStepper(DynamicFreeSurfaceSettings settings = {});

    [[nodiscard]] DynamicFreeSurfaceReport step(MacGrid& grid, MarkerParticleSystem& particles);
    [[nodiscard]] DynamicFreeSurfaceReport step(
        MacGrid& grid,
        MarkerParticleSystem& particles,
        const SimulationStepInput& input);
    [[nodiscard]] const DynamicFreeSurfaceSettings& settings() const noexcept;
    [[nodiscard]] double simulationTime() const noexcept;
    void resetTime() noexcept;

    [[nodiscard]] static float maximumSpeed(const MacGrid& grid) noexcept;
    [[nodiscard]] static std::size_t fluidCellCount(const MacGrid& grid) noexcept;
    [[nodiscard]] static std::size_t isolatedFluidCellCount(const MacGrid& grid) noexcept;

private:
    void advanceSubstep(
        MacGrid& grid,
        MarkerParticleSystem& particles,
        float timeStep,
        Vector2 acceleration,
        DynamicFreeSurfaceReport& report);
    void applyAcceleration(MacGrid& grid, Vector2 acceleration, float timeStep) const noexcept;
    MacGrid& prepareWorkspace(const MacGrid& grid);

    DynamicFreeSurfaceSettings m_settings;
    VelocityAdvector m_advector;
    VelocityExtrapolator m_extrapolator;
    std::unique_ptr<MacGrid> m_workspace;
    double m_simulationTime = 0.0;
};

} // namespace fluidlab::simulation
