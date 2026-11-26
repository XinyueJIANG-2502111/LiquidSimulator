#include "MacGrid.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace fluidlab::simulation {

MacGrid::MacGrid(std::size_t width, std::size_t height, float cellWidth, float cellHeight)
    : m_width(width),
      m_height(height),
      m_cellWidth(cellWidth),
      m_cellHeight(cellHeight),
      m_horizontalVelocity((width + 1) * height, 0.0f),
      m_verticalVelocity(width * (height + 1), 0.0f),
      m_pressure(width * height, 0.0f),
      m_divergence(width * height, 0.0f),
      m_cellTypes(width * height, CellType::fluid) {
    if (width < 2 || height < 2) {
        throw std::invalid_argument("A MAC grid requires at least 2 x 2 cells.");
    }
    if (!std::isfinite(cellWidth) || !std::isfinite(cellHeight) || cellWidth <= 0.0f || cellHeight <= 0.0f) {
        throw std::invalid_argument("MAC grid cell dimensions must be positive finite values.");
    }
}

std::size_t MacGrid::width() const noexcept { return m_width; }
std::size_t MacGrid::height() const noexcept { return m_height; }
float MacGrid::cellWidth() const noexcept { return m_cellWidth; }
float MacGrid::cellHeight() const noexcept { return m_cellHeight; }

bool MacGrid::hasSameLayout(const MacGrid& other) const noexcept {
    return m_width == other.m_width && m_height == other.m_height &&
           m_cellWidth == other.m_cellWidth && m_cellHeight == other.m_cellHeight;
}

std::size_t MacGrid::storageBytes() const noexcept {
    return sizeof(float) * (m_horizontalVelocity.capacity() + m_verticalVelocity.capacity() +
                            m_pressure.capacity() + m_divergence.capacity()) +
           sizeof(CellType) * m_cellTypes.capacity();
}

float& MacGrid::horizontalVelocity(std::size_t faceX, std::size_t cellY) noexcept {
    assert(faceX <= m_width && cellY < m_height);
    return m_horizontalVelocity[horizontalIndex(faceX, cellY)];
}

float MacGrid::horizontalVelocity(std::size_t faceX, std::size_t cellY) const noexcept {
    assert(faceX <= m_width && cellY < m_height);
    return m_horizontalVelocity[horizontalIndex(faceX, cellY)];
}

float& MacGrid::verticalVelocity(std::size_t cellX, std::size_t faceY) noexcept {
    assert(cellX < m_width && faceY <= m_height);
    return m_verticalVelocity[verticalIndex(cellX, faceY)];
}

float MacGrid::verticalVelocity(std::size_t cellX, std::size_t faceY) const noexcept {
    assert(cellX < m_width && faceY <= m_height);
    return m_verticalVelocity[verticalIndex(cellX, faceY)];
}

float& MacGrid::pressure(std::size_t cellX, std::size_t cellY) noexcept {
    assert(cellX < m_width && cellY < m_height);
    return m_pressure[cellIndex(cellX, cellY)];
}

float MacGrid::pressure(std::size_t cellX, std::size_t cellY) const noexcept {
    assert(cellX < m_width && cellY < m_height);
    return m_pressure[cellIndex(cellX, cellY)];
}

float MacGrid::divergence(std::size_t cellX, std::size_t cellY) const noexcept {
    assert(cellX < m_width && cellY < m_height);
    return m_divergence[cellIndex(cellX, cellY)];
}

CellType MacGrid::cellType(std::size_t cellX, std::size_t cellY) const noexcept {
    assert(cellX < m_width && cellY < m_height);
    return m_cellTypes[cellIndex(cellX, cellY)];
}

void MacGrid::setCellType(std::size_t cellX, std::size_t cellY, CellType type) noexcept {
    assert(cellX < m_width && cellY < m_height);
    m_cellTypes[cellIndex(cellX, cellY)] = type;
}

void MacGrid::fillCellTypes(CellType type) noexcept {
    std::fill(m_cellTypes.begin(), m_cellTypes.end(), type);
}

void MacGrid::clear() noexcept {
    std::fill(m_horizontalVelocity.begin(), m_horizontalVelocity.end(), 0.0f);
    std::fill(m_verticalVelocity.begin(), m_verticalVelocity.end(), 0.0f);
    std::fill(m_pressure.begin(), m_pressure.end(), 0.0f);
    std::fill(m_divergence.begin(), m_divergence.end(), 0.0f);
}

void MacGrid::clearPressure() noexcept {
    std::fill(m_pressure.begin(), m_pressure.end(), 0.0f);
}

void MacGrid::swapVelocityFields(MacGrid& other) {
    if (!hasSameLayout(other)) {
        throw std::invalid_argument("Velocity fields can only be swapped between matching MAC grids.");
    }

    // Pressure stays on the primary grid so it remains a valid warm-start guess.
    // Only the two advected face fields change ownership without allocation.
    m_horizontalVelocity.swap(other.m_horizontalVelocity);
    m_verticalVelocity.swap(other.m_verticalVelocity);
}

void MacGrid::enforceClosedBoundary() noexcept {
    // A stationary solid wall only constrains the normal face velocity here.
    // Tangential no-slip treatment belongs to the later viscosity step.
    for (std::size_t y = 0; y < m_height; ++y) {
        horizontalVelocity(0, y) = 0.0f;
        horizontalVelocity(m_width, y) = 0.0f;
    }
    for (std::size_t x = 0; x < m_width; ++x) {
        verticalVelocity(x, 0) = 0.0f;
        verticalVelocity(x, m_height) = 0.0f;
    }
}

void MacGrid::enforceSolidBoundaries() noexcept {
    enforceClosedBoundary();
    for (std::size_t y = 0; y < m_height; ++y) {
        for (std::size_t faceX = 1; faceX < m_width; ++faceX) {
            if (cellType(faceX - 1, y) == CellType::solid ||
                cellType(faceX, y) == CellType::solid) {
                horizontalVelocity(faceX, y) = 0.0f;
            }
        }
    }
    for (std::size_t faceY = 1; faceY < m_height; ++faceY) {
        for (std::size_t x = 0; x < m_width; ++x) {
            if (cellType(x, faceY - 1) == CellType::solid ||
                cellType(x, faceY) == CellType::solid) {
                verticalVelocity(x, faceY) = 0.0f;
            }
        }
    }
}

void MacGrid::computeDivergence() noexcept {
    const float inverseDx = 1.0f / m_cellWidth;
    const float inverseDy = 1.0f / m_cellHeight;
    for (std::size_t y = 0; y < m_height; ++y) {
        for (std::size_t x = 0; x < m_width; ++x) {
            m_divergence[cellIndex(x, y)] =
                (horizontalVelocity(x + 1, y) - horizontalVelocity(x, y)) * inverseDx +
                (verticalVelocity(x, y + 1) - verticalVelocity(x, y)) * inverseDy;
        }
    }
}

DivergenceStatistics MacGrid::divergenceStatistics() const noexcept {
    DivergenceStatistics statistics;
    double absoluteSum = 0.0;
    for (const float value : m_divergence) {
        const float absoluteValue = std::abs(value);
        statistics.maximumAbsolute = std::max(statistics.maximumAbsolute, absoluteValue);
        absoluteSum += absoluteValue;
    }
    statistics.meanAbsolute = static_cast<float>(absoluteSum / static_cast<double>(m_divergence.size()));
    return statistics;
}

float MacGrid::maximumBoundaryNormalSpeed() const noexcept {
    float maximum = 0.0f;
    for (std::size_t y = 0; y < m_height; ++y) {
        maximum = std::max(maximum, std::abs(horizontalVelocity(0, y)));
        maximum = std::max(maximum, std::abs(horizontalVelocity(m_width, y)));
    }
    for (std::size_t x = 0; x < m_width; ++x) {
        maximum = std::max(maximum, std::abs(verticalVelocity(x, 0)));
        maximum = std::max(maximum, std::abs(verticalVelocity(x, m_height)));
    }
    return maximum;
}

Vector2 MacGrid::cellCenteredVelocity(std::size_t cellX, std::size_t cellY) const noexcept {
    assert(cellX < m_width && cellY < m_height);
    return {
        0.5f * (horizontalVelocity(cellX, cellY) + horizontalVelocity(cellX + 1, cellY)),
        0.5f * (verticalVelocity(cellX, cellY) + verticalVelocity(cellX, cellY + 1)),
    };
}

bool MacGrid::allFinite() const noexcept {
    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
    };
    return finite(m_horizontalVelocity) && finite(m_verticalVelocity) &&
           finite(m_pressure) && finite(m_divergence);
}

const std::vector<float>& MacGrid::horizontalVelocities() const noexcept { return m_horizontalVelocity; }
const std::vector<float>& MacGrid::verticalVelocities() const noexcept { return m_verticalVelocity; }
const std::vector<float>& MacGrid::pressures() const noexcept { return m_pressure; }
const std::vector<float>& MacGrid::divergences() const noexcept { return m_divergence; }
const std::vector<CellType>& MacGrid::cellTypes() const noexcept { return m_cellTypes; }

void MacGrid::loadState(
    const std::vector<float>& horizontalVelocity,
    const std::vector<float>& verticalVelocity,
    const std::vector<float>& pressure,
    const std::vector<CellType>& cellTypes) {
    if (horizontalVelocity.size() != m_horizontalVelocity.size() ||
        verticalVelocity.size() != m_verticalVelocity.size() ||
        pressure.size() != m_pressure.size() ||
        cellTypes.size() != m_cellTypes.size()) {
        throw std::invalid_argument("MAC state layout does not match the destination grid.");
    }
    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(),
            [](float value) { return std::isfinite(value); });
    };
    if (!finite(horizontalVelocity) || !finite(verticalVelocity) || !finite(pressure)) {
        throw std::invalid_argument("MAC state contains a non-finite value.");
    }
    m_horizontalVelocity = horizontalVelocity;
    m_verticalVelocity = verticalVelocity;
    m_pressure = pressure;
    m_cellTypes = cellTypes;
    std::fill(m_divergence.begin(), m_divergence.end(), 0.0f);
    enforceSolidBoundaries();
    computeDivergence();
}

std::size_t MacGrid::horizontalIndex(std::size_t faceX, std::size_t cellY) const noexcept {
    return cellY * (m_width + 1) + faceX;
}

std::size_t MacGrid::verticalIndex(std::size_t cellX, std::size_t faceY) const noexcept {
    return faceY * m_width + cellX;
}

std::size_t MacGrid::cellIndex(std::size_t cellX, std::size_t cellY) const noexcept {
    return cellY * m_width + cellX;
}

} // namespace fluidlab::simulation
