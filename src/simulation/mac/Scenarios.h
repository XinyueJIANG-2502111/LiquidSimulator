#pragma once

#include "MacGrid.h"

namespace fluidlab::simulation {

void initializeDeterministicDivergentVortex(MacGrid& grid) noexcept;
void initializeClosedVortex(MacGrid& grid, float strength = 0.25f) noexcept;
void initializeLiquidRectangle(
    MacGrid& grid,
    std::size_t beginX,
    std::size_t beginY,
    std::size_t endX,
    std::size_t endY);
void initializeDamBreak(MacGrid& grid);

} // namespace fluidlab::simulation
