#pragma once

#include "MacGrid.h"

namespace fluidlab::simulation {

class MacVelocitySampler final {
public:
    [[nodiscard]] static float sampleHorizontal(const MacGrid& grid, Vector2 position) noexcept;
    [[nodiscard]] static float sampleVertical(const MacGrid& grid, Vector2 position) noexcept;
    [[nodiscard]] static Vector2 sampleVelocity(const MacGrid& grid, Vector2 position) noexcept;
    [[nodiscard]] static Vector2 clampToDomain(const MacGrid& grid, Vector2 position) noexcept;

    [[nodiscard]] static Vector2 traceBackwardRk2(
        const MacGrid& grid,
        Vector2 targetPosition,
        float timeStep) noexcept;
};

class VelocityAdvector final {
public:
    // Source and destination must be distinct grids with identical layouts.
    // All samples come from source so face updates cannot contaminate later reads.
    void advect(const MacGrid& source, MacGrid& destination, float timeStep) const;
};

} // namespace fluidlab::simulation
