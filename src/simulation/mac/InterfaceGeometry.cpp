#include "InterfaceGeometry.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fluidlab::simulation {
namespace {

[[nodiscard]] float distanceToParticleUnion(
    Vector2 point,
    const std::vector<MarkerParticle>& particles,
    float supportRadius) noexcept {
    float minimumSquared = std::numeric_limits<float>::infinity();
    for (const MarkerParticle& particle : particles) {
        if (!particle.valid || !std::isfinite(particle.position.x) ||
            !std::isfinite(particle.position.y)) continue;
        const float dx = point.x - particle.position.x;
        const float dy = point.y - particle.position.y;
        minimumSquared = std::min(minimumSquared, dx * dx + dy * dy);
    }
    return std::isfinite(minimumSquared)
        ? std::sqrt(minimumSquared) - supportRadius
        : std::numeric_limits<float>::infinity();
}

[[nodiscard]] float crossingFraction(float first, float second) noexcept {
    if (!std::isfinite(first) || !std::isfinite(second) ||
        (first < 0.0f) == (second < 0.0f)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float denominator = first - second;
    if (std::abs(denominator) <= std::numeric_limits<float>::epsilon()) return 0.5f;
    return std::clamp(first / denominator, 0.0f, 1.0f);
}

} // namespace

InterfaceGeometryResult reconstructParticleInterface(
    std::size_t width,
    std::size_t height,
    float cellWidth,
    float cellHeight,
    const std::vector<MarkerParticle>& particles,
    const InterfaceGeometrySettings& settings) {
    if (width == 0 || height == 0 || !(cellWidth > 0.0f) || !(cellHeight > 0.0f)) {
        throw std::invalid_argument("Interface geometry requires a positive grid layout.");
    }
    if (!(settings.particleSupportRadius > 0.0f) ||
        settings.areaSamplesPerAxis == 0) {
        throw std::invalid_argument("Interface geometry settings are invalid.");
    }

    InterfaceGeometryResult result;
    result.width = width;
    result.height = height;
    result.cellWidth = cellWidth;
    result.cellHeight = cellHeight;
    result.particleSupportRadius = settings.particleSupportRadius;
    const std::size_t cellCount = width * height;
    result.signedDistanceProxy.resize(cellCount);
    result.normals.resize(cellCount);
    result.areaFractions.resize(cellCount);
    result.horizontalInterfaceFractions.assign(
        (width > 1 ? width - 1 : 0) * height,
        std::numeric_limits<float>::quiet_NaN());
    result.verticalInterfaceFractions.assign(
        width * (height > 1 ? height - 1 : 0),
        std::numeric_limits<float>::quiet_NaN());
    result.validParticleCount = static_cast<std::size_t>(std::count_if(
        particles.begin(), particles.end(), [](const MarkerParticle& particle) {
            return particle.valid && std::isfinite(particle.position.x) &&
                std::isfinite(particle.position.y);
        }));

    const auto sample = [&](float x, float y) {
        return distanceToParticleUnion(
            Vector2{x, y}, particles, settings.particleSupportRadius);
    };
    const auto indexOf = [width](std::size_t x, std::size_t y) {
        return y * width + x;
    };
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t index = indexOf(x, y);
            const float centerX = (static_cast<float>(x) + 0.5f) * cellWidth;
            const float centerY = (static_cast<float>(y) + 0.5f) * cellHeight;
            result.signedDistanceProxy[index] = sample(centerX, centerY);
            const float left = sample(centerX - 0.5f * cellWidth, centerY);
            const float right = sample(centerX + 0.5f * cellWidth, centerY);
            const float below = sample(centerX, centerY - 0.5f * cellHeight);
            const float above = sample(centerX, centerY + 0.5f * cellHeight);
            float nx = (right - left) / cellWidth;
            float ny = (above - below) / cellHeight;
            const float length = std::sqrt(nx * nx + ny * ny);
            if (std::isfinite(length) && length > 1.0e-6f) {
                nx /= length;
                ny /= length;
            } else {
                nx = 0.0f;
                ny = 0.0f;
            }
            result.normals[index] = {nx, ny};

            std::size_t inside = 0;
            const std::size_t samples = settings.areaSamplesPerAxis;
            for (std::size_t sy = 0; sy < samples; ++sy) {
                for (std::size_t sx = 0; sx < samples; ++sx) {
                    const float px = (static_cast<float>(x) +
                        (static_cast<float>(sx) + 0.5f) / static_cast<float>(samples)) *
                        cellWidth;
                    const float py = (static_cast<float>(y) +
                        (static_cast<float>(sy) + 0.5f) / static_cast<float>(samples)) *
                        cellHeight;
                    if (sample(px, py) <= 0.0f) ++inside;
                }
            }
            result.areaFractions[index] = static_cast<float>(inside) /
                static_cast<float>(samples * samples);
            result.estimatedArea += static_cast<double>(result.areaFractions[index]) *
                static_cast<double>(cellWidth) * cellHeight;
        }
    }

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x + 1 < width; ++x) {
            result.horizontalInterfaceFractions[y * (width - 1) + x] = crossingFraction(
                result.signedDistanceProxy[indexOf(x, y)],
                result.signedDistanceProxy[indexOf(x + 1, y)]);
        }
    }
    for (std::size_t y = 0; y + 1 < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            result.verticalInterfaceFractions[y * width + x] = crossingFraction(
                result.signedDistanceProxy[indexOf(x, y)],
                result.signedDistanceProxy[indexOf(x, y + 1)]);
        }
    }
    return result;
}

} // namespace fluidlab::simulation
