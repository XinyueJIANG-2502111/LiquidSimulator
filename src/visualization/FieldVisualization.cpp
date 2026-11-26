#include "FieldVisualization.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fluidlab::visualization {
namespace {

void appendLine(std::vector<float>& vertices, float x0, float y0, float x1, float y1) {
    vertices.push_back(x0);
    vertices.push_back(y0);
    vertices.push_back(x1);
    vertices.push_back(y1);
}

} // namespace

std::vector<float> buildSpeedTexture(const simulation::MacGrid& grid) {
    std::vector<float> pixels;
    buildSpeedTexture(grid, pixels);
    return pixels;
}

void buildSpeedTexture(const simulation::MacGrid& grid, std::vector<float>& pixels) {
    float maximumSpeed = 0.0f;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const simulation::Vector2 velocity = grid.cellCenteredVelocity(x, y);
            const float speed = std::hypot(velocity.x, velocity.y);
            maximumSpeed = std::max(maximumSpeed, speed);
        }
    }

    const float inverseMaximum = maximumSpeed > 0.0f ? 1.0f / maximumSpeed : 0.0f;
    pixels.assign(grid.width() * grid.height() * 4, 1.0f);
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const simulation::Vector2 velocity = grid.cellCenteredVelocity(x, y);
            const float value = std::clamp(
                std::hypot(velocity.x, velocity.y) * inverseMaximum,
                0.0f,
                1.0f);
            const std::size_t index = y * grid.width() + x;
            pixels[index * 4 + 0] = 0.015f + 0.08f * value;
            pixels[index * 4 + 1] = 0.025f + 0.65f * value;
            pixels[index * 4 + 2] = 0.06f + 0.90f * value;
        }
    }
}

std::vector<float> buildSignedScalarTexture(std::span<const float> values) {
    std::vector<float> pixels;
    buildSignedScalarTexture(values, pixels);
    return pixels;
}

void buildSignedScalarTexture(std::span<const float> values, std::vector<float>& pixels) {
    if (values.empty()) {
        throw std::invalid_argument("A scalar visualization requires at least one value.");
    }

    float maximumAbsolute = 0.0f;
    for (const float value : values) {
        maximumAbsolute = std::max(maximumAbsolute, std::abs(value));
    }
    const float inverseMaximum = maximumAbsolute > 0.0f ? 1.0f / maximumAbsolute : 0.0f;

    pixels.assign(values.size() * 4, 1.0f);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const float normalized = std::clamp(values[index] * inverseMaximum, -1.0f, 1.0f);
        const float magnitude = std::abs(normalized);
        const float neutral = 0.035f * (1.0f - magnitude);
        pixels[index * 4 + 0] = neutral + (normalized > 0.0f ? 0.95f * magnitude : 0.02f * magnitude);
        pixels[index * 4 + 1] = neutral + 0.10f * magnitude;
        pixels[index * 4 + 2] = neutral + (normalized < 0.0f ? 0.95f * magnitude : 0.02f * magnitude);
    }
}

void buildCellTypeTexture(const simulation::MacGrid& grid, std::vector<float>& pixels) {
    pixels.assign(grid.width() * grid.height() * 4, 1.0f);
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const std::size_t index = (y * grid.width() + x) * 4;
            switch (grid.cellType(x, y)) {
            case simulation::CellType::fluid:
                pixels[index + 0] = 0.04f;
                pixels[index + 1] = 0.30f;
                pixels[index + 2] = 0.90f;
                break;
            case simulation::CellType::air:
                pixels[index + 0] = 0.015f;
                pixels[index + 1] = 0.020f;
                pixels[index + 2] = 0.035f;
                break;
            case simulation::CellType::solid:
                pixels[index + 0] = 0.38f;
                pixels[index + 1] = 0.40f;
                pixels[index + 2] = 0.44f;
                break;
            }
        }
    }
}

std::vector<float> buildVelocityArrowVertices(
    const simulation::MacGrid& grid,
    std::size_t samplingStride) {
    std::vector<float> vertices;
    buildVelocityArrowVertices(grid, samplingStride, vertices);
    return vertices;
}

void buildVelocityArrowVertices(
    const simulation::MacGrid& grid,
    std::size_t samplingStride,
    std::vector<float>& vertices) {
    if (samplingStride == 0) {
        throw std::invalid_argument("Velocity arrow sampling stride must be greater than zero.");
    }

    float maximumSpeed = 0.0f;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const simulation::Vector2 velocity = grid.cellCenteredVelocity(x, y);
            maximumSpeed = std::max(maximumSpeed, std::hypot(velocity.x, velocity.y));
        }
    }

    vertices.clear();
    if (maximumSpeed == 0.0f) {
        return;
    }

    const std::size_t sampledRows = (grid.height() + samplingStride - 1) / samplingStride;
    const std::size_t sampledColumns = (grid.width() + samplingStride - 1) / samplingStride;
    vertices.reserve(sampledRows * sampledColumns * 12);

    constexpr float arrowLength = 0.045f;
    constexpr float headLength = 0.010f;
    constexpr float headWidth = 0.006f;
    for (std::size_t y = samplingStride / 2; y < grid.height(); y += samplingStride) {
        for (std::size_t x = samplingStride / 2; x < grid.width(); x += samplingStride) {
            const simulation::Vector2 velocity = grid.cellCenteredVelocity(x, y);
            const float speed = std::hypot(velocity.x, velocity.y);
            if (speed < maximumSpeed * 1.0e-4f) {
                continue;
            }

            const float directionX = velocity.x / speed;
            const float directionY = velocity.y / speed;
            const float length = arrowLength * speed / maximumSpeed;
            const float startX = (static_cast<float>(x) + 0.5f) / static_cast<float>(grid.width());
            const float startY = (static_cast<float>(y) + 0.5f) / static_cast<float>(grid.height());
            const float endX = startX + directionX * length;
            const float endY = startY + directionY * length;
            const float baseX = endX - directionX * headLength;
            const float baseY = endY - directionY * headLength;
            const float perpendicularX = -directionY * headWidth;
            const float perpendicularY = directionX * headWidth;

            appendLine(vertices, startX, startY, endX, endY);
            appendLine(vertices, endX, endY, baseX + perpendicularX, baseY + perpendicularY);
            appendLine(vertices, endX, endY, baseX - perpendicularX, baseY - perpendicularY);
        }
    }
}

} // namespace fluidlab::visualization
