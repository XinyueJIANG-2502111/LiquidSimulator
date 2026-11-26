#pragma once

#include "MacGrid.h"

#include <cstddef>
#include <vector>

namespace fluidlab::simulation {

struct MarkerParticle final {
    Vector2 position;
    bool valid = true;
};

struct ParticleAdvectionReport final {
    std::size_t particleCount = 0;
    std::size_t collisionCount = 0;
    std::size_t invalidPositionCount = 0;
};

class MarkerParticleSystem final {
public:
    void seedFluidCells(const MacGrid& grid, std::size_t particlesPerAxis = 2);
    void seedFluidCellsExact(const MacGrid& grid, std::size_t particlesPerCell);
    void rebuildFluidClassification(MacGrid& grid) noexcept;
    [[nodiscard]] ParticleAdvectionReport advectRk2(const MacGrid& grid, float timeStep) noexcept;

    [[nodiscard]] const std::vector<MarkerParticle>& particles() const noexcept;
    [[nodiscard]] std::size_t validParticleCount() const noexcept;
    [[nodiscard]] std::size_t invalidPositionCount() const noexcept;
    [[nodiscard]] bool allFiniteAndInside(const MacGrid& grid) const noexcept;
    void loadState(const MacGrid& grid, const std::vector<MarkerParticle>& particles);

private:
    [[nodiscard]] static bool positionInsideSolid(const MacGrid& grid, Vector2 position) noexcept;
    [[nodiscard]] static Vector2 clampInsideDomain(const MacGrid& grid, Vector2 position) noexcept;
    [[nodiscard]] static Vector2 resolveCollision(
        const MacGrid& grid,
        Vector2 previous,
        Vector2 candidate,
        bool& collided) noexcept;

    std::vector<MarkerParticle> m_particles;
    std::vector<unsigned char> m_occupiedCells;
    std::size_t m_invalidPositionCount = 0;
};

} // namespace fluidlab::simulation
