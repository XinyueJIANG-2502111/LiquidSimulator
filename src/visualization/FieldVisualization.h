#pragma once

#include "../simulation/mac/MacGrid.h"

#include <cstddef>
#include <span>
#include <vector>

namespace fluidlab::visualization {

[[nodiscard]] std::vector<float> buildSpeedTexture(
    const simulation::MacGrid& grid);

void buildSpeedTexture(const simulation::MacGrid& grid, std::vector<float>& pixels);

[[nodiscard]] std::vector<float> buildSignedScalarTexture(
    std::span<const float> values);

void buildSignedScalarTexture(std::span<const float> values, std::vector<float>& pixels);

void buildCellTypeTexture(const simulation::MacGrid& grid, std::vector<float>& pixels);

[[nodiscard]] std::vector<float> buildVelocityArrowVertices(
    const simulation::MacGrid& grid,
    std::size_t samplingStride);

void buildVelocityArrowVertices(
    const simulation::MacGrid& grid,
    std::size_t samplingStride,
    std::vector<float>& vertices);

} // namespace fluidlab::visualization
