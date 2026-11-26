#pragma once

#include "MacGrid.h"

#include <cstddef>

namespace fluidlab::simulation {

struct PressureSolverSettings final {
    float density = 1.0f;
    float timeStep = 1.0f / 60.0f;
    float residualTolerance = 5.0e-3f;
    float relaxationFactor = 1.7f;
    std::size_t maximumIterations = 4000;
    std::size_t residualCheckInterval = 10;
    bool warmStartEnabled = false;
    // Zero removes the pressure mean only at solve boundaries.
    std::size_t pressureMeanRemovalInterval = 1;
};

struct PressureResidual final {
    float maximumAbsolute = 0.0f;
    float rootMeanSquare = 0.0f;
};

enum class PressureSolveTermination {
    initialResidualWithinTolerance,
    iterativeResidualWithinTolerance,
    maximumIterationsReached,
};

struct PressureSolveReport final {
    PressureResidual initialResidual;
    PressureResidual finalResidual;
    float residualTolerance = 0.0f;
    float initialMaximumAbsolutePressure = 0.0f;
    float finalMaximumAbsolutePressure = 0.0f;
    std::size_t iterations = 0;
    std::size_t pressureMeanRemovalCount = 0;
    bool warmStartUsed = false;
    bool converged = false;
    PressureSolveTermination termination = PressureSolveTermination::maximumIterationsReached;
};

struct ProjectionTimings final {
    double divergenceBeforeMilliseconds = 0.0;
    double pressureSolveMilliseconds = 0.0;
    double pressureApplicationMilliseconds = 0.0;
    double totalMilliseconds = 0.0;
};

struct ProjectionReport final {
    DivergenceStatistics divergenceBefore;
    DivergenceStatistics divergenceAfter;
    PressureSolveReport pressureSolve;
    float maximumVelocityChange = 0.0f;
    ProjectionTimings timings;
};

class PressureProjector final {
public:
    explicit PressureProjector(PressureSolverSettings settings = {});

    [[nodiscard]] ProjectionReport project(MacGrid& grid) const;
    [[nodiscard]] const PressureSolverSettings& settings() const noexcept;

private:
    [[nodiscard]] PressureSolveReport solvePressure(MacGrid& grid) const;
    [[nodiscard]] PressureResidual computeResidual(const MacGrid& grid) const noexcept;
    [[nodiscard]] float removePressureMean(MacGrid& grid) const noexcept;
    [[nodiscard]] float applyPressureGradient(MacGrid& grid) const noexcept;

    PressureSolverSettings m_settings;
};

} // namespace fluidlab::simulation
