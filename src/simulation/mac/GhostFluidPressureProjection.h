#pragma once

#include "LevelSetInterface.h"
#include "PressureProjection.h"

#include <cstddef>
#include <vector>

namespace fluidlab::simulation {

enum class GhostThetaState : unsigned char {
    unused,
    valid,
    clamped,
    fallback,
};

struct GhostThetaStatistics final {
    std::size_t validFaceCount = 0;
    std::size_t invalidFaceCount = 0;
    std::size_t clampedFaceCount = 0;
    std::size_t fallbackFaceCount = 0;
    std::size_t solidAdjacentFaceCount = 0;
};

struct GhostFluidThetaData final {
    std::size_t width = 0;
    std::size_t height = 0;
    double minimumTheta = 0.1;
    std::size_t interfaceVersion = 0;
    std::vector<double> horizontalTheta;
    std::vector<double> verticalTheta;
    std::vector<GhostThetaState> horizontalState;
    std::vector<GhostThetaState> verticalState;
    GhostThetaStatistics statistics;
};

struct GhostFluidLinearSystem final {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<int> unknownIndex;
    std::vector<std::size_t> cellIndices;
    std::vector<double> matrix;
    std::vector<double> rightHandSide;
    std::vector<std::size_t> pureNeumannUnknowns;
    std::size_t pureNeumannComponentCount = 0;
    bool hasDirichletBoundary = false;
    bool supported = true;
};

struct GhostFluidSolveResult final {
    std::vector<double> pressure;
    PressureResidual residual;
    bool solved = false;
    bool compatibleRightHandSide = true;
};

[[nodiscard]] GhostFluidThetaData buildGhostFluidTheta(
    const MacGrid& grid,
    const LevelSetField& interfaceField,
    double minimumTheta,
    bool fallbackToCellBoundary = true);

[[nodiscard]] GhostFluidLinearSystem assembleGhostFluidSystem(
    const MacGrid& grid,
    const GhostFluidThetaData& theta,
    double density,
    double timeStep);

[[nodiscard]] std::vector<double> applyGhostFluidMatrix(
    const GhostFluidLinearSystem& system,
    const std::vector<double>& pressure);

[[nodiscard]] GhostFluidSolveResult solveGhostFluidSystemDirect(
    const GhostFluidLinearSystem& system);

void writeGhostFluidPressure(
    MacGrid& grid,
    const GhostFluidLinearSystem& system,
    const std::vector<double>& pressure) noexcept;

[[nodiscard]] float applyGhostFluidPressureGradient(
    MacGrid& grid,
    const GhostFluidThetaData& theta,
    double density,
    double timeStep) noexcept;

} // namespace fluidlab::simulation
