#include "Scenarios.h"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace fluidlab::simulation {

void initializeClosedVortex(MacGrid& grid, float strength) noexcept {
    grid.clear();

    constexpr float pi = std::numbers::pi_v<float>;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        const float positionY = (static_cast<float>(y) + 0.5f) * grid.cellHeight();
        for (std::size_t faceX = 0; faceX <= grid.width(); ++faceX) {
            const float positionX = static_cast<float>(faceX) * grid.cellWidth();
            grid.horizontalVelocity(faceX, y) =
                strength * pi * std::sin(pi * positionX) * std::cos(pi * positionY);
        }
    }

    for (std::size_t faceY = 0; faceY <= grid.height(); ++faceY) {
        const float positionY = static_cast<float>(faceY) * grid.cellHeight();
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const float positionX = (static_cast<float>(x) + 0.5f) * grid.cellWidth();
            grid.verticalVelocity(x, faceY) =
                -strength * pi * std::cos(pi * positionX) * std::sin(pi * positionY);
        }
    }

    grid.enforceClosedBoundary();
    grid.computeDivergence();
}

void initializeDeterministicDivergentVortex(MacGrid& grid) noexcept {
    grid.clear();

    constexpr float pi = std::numbers::pi_v<float>;
    constexpr float vortexStrength = 0.12f;
    constexpr float gradientStrength = 0.08f;

    // The stream-function component is analytically divergence-free and remains
    // after projection. The potential-gradient component deliberately adds
    // divergence while retaining zero normal velocity on every outer wall.
    for (std::size_t y = 0; y < grid.height(); ++y) {
        const float positionY = (static_cast<float>(y) + 0.5f) * grid.cellHeight();
        for (std::size_t faceX = 0; faceX <= grid.width(); ++faceX) {
            const float positionX = static_cast<float>(faceX) * grid.cellWidth();
            const float vortex = vortexStrength * pi * std::sin(pi * positionX) * std::cos(pi * positionY);
            const float gradient = -gradientStrength * 2.0f * pi *
                                   std::sin(2.0f * pi * positionX) * std::cos(2.0f * pi * positionY);
            grid.horizontalVelocity(faceX, y) = vortex + gradient;
        }
    }

    for (std::size_t faceY = 0; faceY <= grid.height(); ++faceY) {
        const float positionY = static_cast<float>(faceY) * grid.cellHeight();
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const float positionX = (static_cast<float>(x) + 0.5f) * grid.cellWidth();
            const float vortex = -vortexStrength * pi * std::cos(pi * positionX) * std::sin(pi * positionY);
            const float gradient = -gradientStrength * 2.0f * pi *
                                   std::cos(2.0f * pi * positionX) * std::sin(2.0f * pi * positionY);
            grid.verticalVelocity(x, faceY) = vortex + gradient;
        }
    }

    grid.enforceClosedBoundary();
    grid.computeDivergence();
}

void initializeLiquidRectangle(
    MacGrid& grid,
    std::size_t beginX,
    std::size_t beginY,
    std::size_t endX,
    std::size_t endY) {
    if (beginX >= endX || beginY >= endY || endX > grid.width() || endY > grid.height()) {
        throw std::invalid_argument("Liquid rectangle must be non-empty and inside the grid.");
    }
    grid.fillCellTypes(CellType::air);
    for (std::size_t y = beginY; y < endY; ++y) {
        for (std::size_t x = beginX; x < endX; ++x) {
            grid.setCellType(x, y, CellType::fluid);
        }
    }
}

void initializeDamBreak(MacGrid& grid) {
    if (grid.width() < 8 || grid.height() < 8) {
        throw std::invalid_argument("Dam-break grid must be at least 8 x 8 cells.");
    }
    grid.clear();
    grid.fillCellTypes(CellType::air);
    for (std::size_t y = 0; y < grid.height(); ++y) {
        grid.setCellType(0, y, CellType::solid);
        grid.setCellType(grid.width() - 1, y, CellType::solid);
    }
    for (std::size_t x = 0; x < grid.width(); ++x) {
        grid.setCellType(x, 0, CellType::solid);
        grid.setCellType(x, grid.height() - 1, CellType::solid);
    }
    const std::size_t liquidEndX = 1 + (grid.width() - 2) / 3;
    const std::size_t liquidEndY = 1 + 2 * (grid.height() - 2) / 3;
    for (std::size_t y = 1; y < liquidEndY; ++y) {
        for (std::size_t x = 1; x < liquidEndX; ++x) {
            grid.setCellType(x, y, CellType::fluid);
        }
    }
    grid.enforceSolidBoundaries();
    grid.computeDivergence();
}

} // namespace fluidlab::simulation
