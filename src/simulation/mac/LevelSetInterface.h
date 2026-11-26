#pragma once

#include "MacGrid.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace fluidlab::simulation {

struct LevelSetField final {
    std::size_t width = 0;
    std::size_t height = 0;
    double cellWidth = 0.0;
    double cellHeight = 0.0;
    std::vector<double> values;
    std::vector<unsigned char> confidence;
    std::size_t sourceVersion = 0;

    [[nodiscard]] double value(std::size_t x, std::size_t y) const noexcept;
    [[nodiscard]] double sample(double x, double y) const noexcept;
    [[nodiscard]] Vector2 normal(double x, double y) const noexcept;
};

struct LevelSetAreaEstimate final {
    double area = 0.0;
    std::size_t validSamples = 0;
    std::size_t invalidSamples = 0;
};

struct LevelSetReinitializationReport final {
    std::size_t crossingCount = 0;
    std::size_t segmentCount = 0;
    std::size_t processedCellCount = 0;
    std::size_t validCellCount = 0;
    std::size_t invalidCellCount = 0;
    double maximumInterfaceDisplacement = 0.0;
    double rmsInterfaceDisplacement = 0.0;
    double areaBefore = 0.0;
    double areaAfter = 0.0;
    double meanGradientError = 0.0;
    double maximumGradientError = 0.0;
    std::size_t negativeComponentCountBefore = 0;
    std::size_t negativeComponentCountAfter = 0;
};

struct LevelSetRedistanceSettings final {
    double narrowBandWidth = 0.0;
    std::size_t areaSamplesPerAxis = 2;
    bool preserveInterfaceCrossings = false;
};

using ScalarFieldFunction = std::function<double(double, double)>;
using VelocityFieldFunction = std::function<Vector2(double, double)>;

[[nodiscard]] LevelSetField sampleAnalyticLevelSet(
    std::size_t width,
    std::size_t height,
    double cellWidth,
    double cellHeight,
    const ScalarFieldFunction& field,
    std::size_t sourceVersion = 1);

// Builds a cell-centered half-cell proxy from binary classification. It is a
// deterministic baseline and does not recover subcell particle geometry.
[[nodiscard]] LevelSetField buildClassificationLevelSet(const MacGrid& grid);

[[nodiscard]] LevelSetField advectLevelSetRk2(
    const LevelSetField& source,
    const VelocityFieldFunction& velocity,
    double timeStep);

struct LevelSetMacCormackReport final {
    std::size_t correctedCellCount = 0;
    std::size_t limiterClampCount = 0;
    std::size_t fallbackCellCount = 0;
    std::size_t nonFiniteCellCount = 0;
};

[[nodiscard]] LevelSetField advectLevelSetLimitedMacCormack(
    const LevelSetField& source,
    const VelocityFieldFunction& velocity,
    double timeStep,
    LevelSetMacCormackReport* report = nullptr);

[[nodiscard]] LevelSetReinitializationReport reinitializeLevelSet(
    LevelSetField& field,
    double narrowBandWidth);

// Rebuilds distance only near marching-squares zero segments. Segment indices
// are binned into grid cells so distance queries do not scan the full contour.
[[nodiscard]] LevelSetReinitializationReport redistanceLevelSetNarrowBand(
    LevelSetField& field,
    const LevelSetRedistanceSettings& settings);

[[nodiscard]] LevelSetAreaEstimate estimateLevelSetArea(
    const LevelSetField& field,
    std::size_t samplesPerAxis);

} // namespace fluidlab::simulation
