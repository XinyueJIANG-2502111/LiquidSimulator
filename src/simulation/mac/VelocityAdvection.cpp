#include "VelocityAdvection.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fluidlab::simulation {
namespace {

[[nodiscard]] float bilinearSample(
    const MacGrid& grid,
    bool horizontalComponent,
    float gridX,
    float gridY) noexcept {
    const std::size_t maximumX = horizontalComponent ? grid.width() : grid.width() - 1;
    const std::size_t maximumY = horizontalComponent ? grid.height() - 1 : grid.height();
    const float clampedX = std::clamp(gridX, 0.0f, static_cast<float>(maximumX));
    const float clampedY = std::clamp(gridY, 0.0f, static_cast<float>(maximumY));
    const std::size_t x0 = static_cast<std::size_t>(std::floor(clampedX));
    const std::size_t y0 = static_cast<std::size_t>(std::floor(clampedY));
    const std::size_t x1 = std::min(x0 + 1, maximumX);
    const std::size_t y1 = std::min(y0 + 1, maximumY);
    const float fractionX = clampedX - static_cast<float>(x0);
    const float fractionY = clampedY - static_cast<float>(y0);

    const auto value = [&](std::size_t x, std::size_t y) {
        return horizontalComponent
            ? grid.horizontalVelocity(x, y)
            : grid.verticalVelocity(x, y);
    };
    const float lower = std::lerp(value(x0, y0), value(x1, y0), fractionX);
    const float upper = std::lerp(value(x0, y1), value(x1, y1), fractionX);
    return std::lerp(lower, upper, fractionY);
}

void validateMatchingLayouts(const MacGrid& source, const MacGrid& destination) {
    if (&source == &destination) {
        throw std::invalid_argument("Velocity advection requires distinct source and destination grids.");
    }
    if (source.width() != destination.width() || source.height() != destination.height() ||
        source.cellWidth() != destination.cellWidth() || source.cellHeight() != destination.cellHeight()) {
        throw std::invalid_argument("Velocity advection grids must have identical layouts.");
    }
}

} // namespace

float MacVelocitySampler::sampleHorizontal(const MacGrid& grid, Vector2 position) noexcept {
    const Vector2 clamped = clampToDomain(grid, position);
    const float gridX = clamped.x / grid.cellWidth();
    const float gridY = clamped.y / grid.cellHeight() - 0.5f;
    return bilinearSample(grid, true, gridX, gridY);
}

float MacVelocitySampler::sampleVertical(const MacGrid& grid, Vector2 position) noexcept {
    const Vector2 clamped = clampToDomain(grid, position);
    const float gridX = clamped.x / grid.cellWidth() - 0.5f;
    const float gridY = clamped.y / grid.cellHeight();
    return bilinearSample(grid, false, gridX, gridY);
}

Vector2 MacVelocitySampler::sampleVelocity(const MacGrid& grid, Vector2 position) noexcept {
    return {
        sampleHorizontal(grid, position),
        sampleVertical(grid, position),
    };
}

Vector2 MacVelocitySampler::clampToDomain(const MacGrid& grid, Vector2 position) noexcept {
    const float domainWidth = static_cast<float>(grid.width()) * grid.cellWidth();
    const float domainHeight = static_cast<float>(grid.height()) * grid.cellHeight();
    const auto clampCoordinate = [](float value, float maximum) {
        if (std::isnan(value)) {
            return 0.0f;
        }
        if (value <= 0.0f) {
            return 0.0f;
        }
        if (value >= maximum) {
            return maximum;
        }
        return value;
    };
    return {
        clampCoordinate(position.x, domainWidth),
        clampCoordinate(position.y, domainHeight),
    };
}

Vector2 MacVelocitySampler::traceBackwardRk2(
    const MacGrid& grid,
    Vector2 targetPosition,
    float timeStep) noexcept {
    const Vector2 target = clampToDomain(grid, targetPosition);
    const Vector2 initialVelocity = sampleVelocity(grid, target);
    const Vector2 midpoint = clampToDomain(grid, {
        target.x - 0.5f * timeStep * initialVelocity.x,
        target.y - 0.5f * timeStep * initialVelocity.y,
    });
    const Vector2 midpointVelocity = sampleVelocity(grid, midpoint);
    return clampToDomain(grid, {
        target.x - timeStep * midpointVelocity.x,
        target.y - timeStep * midpointVelocity.y,
    });
}

void VelocityAdvector::advect(const MacGrid& source, MacGrid& destination, float timeStep) const {
    validateMatchingLayouts(source, destination);
    if (!std::isfinite(timeStep) || timeStep < 0.0f) {
        throw std::invalid_argument("Velocity advection timestep must be finite and non-negative.");
    }
    if (!source.allFinite()) {
        throw std::invalid_argument("Velocity advection source grid must contain only finite values.");
    }

    // Every staggered velocity entry and every divergence entry is overwritten
    // below. Preserving unrelated pressure storage avoids redundant full-grid
    // clears and keeps velocity advection independent of pressure ownership.
    if (timeStep == 0.0f) {
        for (std::size_t y = 0; y < source.height(); ++y) {
            for (std::size_t faceX = 0; faceX <= source.width(); ++faceX) {
                destination.horizontalVelocity(faceX, y) = source.horizontalVelocity(faceX, y);
            }
        }
        for (std::size_t faceY = 0; faceY <= source.height(); ++faceY) {
            for (std::size_t x = 0; x < source.width(); ++x) {
                destination.verticalVelocity(x, faceY) = source.verticalVelocity(x, faceY);
            }
        }
        destination.computeDivergence();
        return;
    }

    for (std::size_t y = 0; y < source.height(); ++y) {
        for (std::size_t faceX = 0; faceX <= source.width(); ++faceX) {
            const Vector2 targetPosition{
                static_cast<float>(faceX) * source.cellWidth(),
                (static_cast<float>(y) + 0.5f) * source.cellHeight(),
            };
            const Vector2 sourcePosition =
                MacVelocitySampler::traceBackwardRk2(source, targetPosition, timeStep);
            destination.horizontalVelocity(faceX, y) =
                MacVelocitySampler::sampleHorizontal(source, sourcePosition);
        }
    }

    for (std::size_t faceY = 0; faceY <= source.height(); ++faceY) {
        for (std::size_t x = 0; x < source.width(); ++x) {
            const Vector2 targetPosition{
                (static_cast<float>(x) + 0.5f) * source.cellWidth(),
                static_cast<float>(faceY) * source.cellHeight(),
            };
            const Vector2 sourcePosition =
                MacVelocitySampler::traceBackwardRk2(source, targetPosition, timeStep);
            destination.verticalVelocity(x, faceY) =
                MacVelocitySampler::sampleVertical(source, sourcePosition);
        }
    }
    destination.computeDivergence();
}

} // namespace fluidlab::simulation
