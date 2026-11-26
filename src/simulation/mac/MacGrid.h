#pragma once

#include <cstddef>
#include <vector>

namespace fluidlab::simulation {

struct Vector2 final {
    float x = 0.0f;
    float y = 0.0f;
};

struct DivergenceStatistics final {
    float maximumAbsolute = 0.0f;
    float meanAbsolute = 0.0f;
};

enum class CellType : unsigned char {
    fluid,
    air,
    solid,
};

class MacGrid final {
public:
    MacGrid(std::size_t width, std::size_t height, float cellWidth, float cellHeight);

    [[nodiscard]] std::size_t width() const noexcept;
    [[nodiscard]] std::size_t height() const noexcept;
    [[nodiscard]] float cellWidth() const noexcept;
    [[nodiscard]] float cellHeight() const noexcept;
    [[nodiscard]] bool hasSameLayout(const MacGrid& other) const noexcept;
    [[nodiscard]] std::size_t storageBytes() const noexcept;

    // u lies on vertical cell faces and has dimensions (Nx + 1) x Ny.
    [[nodiscard]] float& horizontalVelocity(std::size_t faceX, std::size_t cellY) noexcept;
    [[nodiscard]] float horizontalVelocity(std::size_t faceX, std::size_t cellY) const noexcept;

    // v lies on horizontal cell faces and has dimensions Nx x (Ny + 1).
    [[nodiscard]] float& verticalVelocity(std::size_t cellX, std::size_t faceY) noexcept;
    [[nodiscard]] float verticalVelocity(std::size_t cellX, std::size_t faceY) const noexcept;

    // Pressure and divergence lie at cell centers and both have dimensions Nx x Ny.
    [[nodiscard]] float& pressure(std::size_t cellX, std::size_t cellY) noexcept;
    [[nodiscard]] float pressure(std::size_t cellX, std::size_t cellY) const noexcept;
    [[nodiscard]] float divergence(std::size_t cellX, std::size_t cellY) const noexcept;
    [[nodiscard]] CellType cellType(std::size_t cellX, std::size_t cellY) const noexcept;
    void setCellType(std::size_t cellX, std::size_t cellY, CellType type) noexcept;
    void fillCellTypes(CellType type) noexcept;

    void clear() noexcept;
    void clearPressure() noexcept;
    void swapVelocityFields(MacGrid& other);
    void enforceClosedBoundary() noexcept;
    void enforceSolidBoundaries() noexcept;
    void computeDivergence() noexcept;

    [[nodiscard]] DivergenceStatistics divergenceStatistics() const noexcept;
    [[nodiscard]] float maximumBoundaryNormalSpeed() const noexcept;
    [[nodiscard]] Vector2 cellCenteredVelocity(std::size_t cellX, std::size_t cellY) const noexcept;
    [[nodiscard]] bool allFinite() const noexcept;

    [[nodiscard]] const std::vector<float>& horizontalVelocities() const noexcept;
    [[nodiscard]] const std::vector<float>& verticalVelocities() const noexcept;
    [[nodiscard]] const std::vector<float>& pressures() const noexcept;
    [[nodiscard]] const std::vector<float>& divergences() const noexcept;
    [[nodiscard]] const std::vector<CellType>& cellTypes() const noexcept;

    void loadState(
        const std::vector<float>& horizontalVelocity,
        const std::vector<float>& verticalVelocity,
        const std::vector<float>& pressure,
        const std::vector<CellType>& cellTypes);

private:
    [[nodiscard]] std::size_t horizontalIndex(std::size_t faceX, std::size_t cellY) const noexcept;
    [[nodiscard]] std::size_t verticalIndex(std::size_t cellX, std::size_t faceY) const noexcept;
    [[nodiscard]] std::size_t cellIndex(std::size_t cellX, std::size_t cellY) const noexcept;

    std::size_t m_width;
    std::size_t m_height;
    float m_cellWidth;
    float m_cellHeight;
    std::vector<float> m_horizontalVelocity;
    std::vector<float> m_verticalVelocity;
    std::vector<float> m_pressure;
    std::vector<float> m_divergence;
    std::vector<CellType> m_cellTypes;
};

} // namespace fluidlab::simulation
