#include "MarkerParticles.h"

#include "VelocityAdvection.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fluidlab::simulation {

void MarkerParticleSystem::seedFluidCells(const MacGrid& grid, std::size_t particlesPerAxis) {
    if (particlesPerAxis == 0) {
        throw std::invalid_argument("Marker particle sampling must be greater than zero.");
    }
    m_particles.clear();
    m_invalidPositionCount = 0;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (grid.cellType(x, y) != CellType::fluid) continue;
            for (std::size_t py = 0; py < particlesPerAxis; ++py) {
                for (std::size_t px = 0; px < particlesPerAxis; ++px) {
                    const float fractionX = (static_cast<float>(px) + 0.5f) /
                        static_cast<float>(particlesPerAxis);
                    const float fractionY = (static_cast<float>(py) + 0.5f) /
                        static_cast<float>(particlesPerAxis);
                    m_particles.push_back({{
                        (static_cast<float>(x) + fractionX) * grid.cellWidth(),
                        (static_cast<float>(y) + fractionY) * grid.cellHeight(),
                    }, true});
                }
            }
        }
    }
}

void MarkerParticleSystem::seedFluidCellsExact(
    const MacGrid& grid,
    std::size_t particlesPerCell) {
    if (particlesPerCell == 0) {
        throw std::invalid_argument("Marker particle count per cell must be greater than zero.");
    }
    const std::size_t columns = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<double>(particlesPerCell))));
    const std::size_t rows = (particlesPerCell + columns - 1) / columns;
    m_particles.clear();
    m_invalidPositionCount = 0;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (grid.cellType(x, y) != CellType::fluid) continue;
            for (std::size_t particleIndex = 0; particleIndex < particlesPerCell;
                ++particleIndex) {
                const std::size_t px = particleIndex % columns;
                const std::size_t py = particleIndex / columns;
                const float fractionX = (static_cast<float>(px) + 0.5f) /
                    static_cast<float>(columns);
                const float fractionY = (static_cast<float>(py) + 0.5f) /
                    static_cast<float>(rows);
                m_particles.push_back({{
                    (static_cast<float>(x) + fractionX) * grid.cellWidth(),
                    (static_cast<float>(y) + fractionY) * grid.cellHeight(),
                }, true});
            }
        }
    }
}

void MarkerParticleSystem::rebuildFluidClassification(MacGrid& grid) noexcept {
    m_occupiedCells.assign(grid.width() * grid.height(), 0);
    for (const MarkerParticle& particle : m_particles) {
        if (!particle.valid || !std::isfinite(particle.position.x) ||
            !std::isfinite(particle.position.y)) continue;
        const Vector2 position = clampInsideDomain(grid, particle.position);
        const std::size_t x = std::min(
            static_cast<std::size_t>(position.x / grid.cellWidth()), grid.width() - 1);
        const std::size_t y = std::min(
            static_cast<std::size_t>(position.y / grid.cellHeight()), grid.height() - 1);
        if (grid.cellType(x, y) != CellType::solid) m_occupiedCells[y * grid.width() + x] = 1;
    }
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (grid.cellType(x, y) == CellType::solid) continue;
            const CellType previous = grid.cellType(x, y);
            const CellType updated = m_occupiedCells[y * grid.width() + x]
                ? CellType::fluid : CellType::air;
            if (updated == CellType::air || previous == CellType::air) grid.pressure(x, y) = 0.0f;
            grid.setCellType(x, y, updated);
        }
    }
}

ParticleAdvectionReport MarkerParticleSystem::advectRk2(const MacGrid& grid, float timeStep) noexcept {
    ParticleAdvectionReport report;
    report.particleCount = validParticleCount();
    for (MarkerParticle& particle : m_particles) {
        if (!particle.valid) continue;
        const Vector2 previous = particle.position;
        const Vector2 firstVelocity = MacVelocitySampler::sampleVelocity(grid, previous);
        const Vector2 midpoint = clampInsideDomain(grid, {
            previous.x + 0.5f * timeStep * firstVelocity.x,
            previous.y + 0.5f * timeStep * firstVelocity.y,
        });
        const Vector2 midpointVelocity = MacVelocitySampler::sampleVelocity(grid, midpoint);
        Vector2 candidate{
            previous.x + timeStep * midpointVelocity.x,
            previous.y + timeStep * midpointVelocity.y,
        };
        if (!std::isfinite(candidate.x) || !std::isfinite(candidate.y)) {
            ++m_invalidPositionCount;
            ++report.invalidPositionCount;
            continue;
        }
        bool collided = false;
        candidate = resolveCollision(grid, previous, candidate, collided);
        particle.position = candidate;
        report.collisionCount += collided ? 1U : 0U;
    }
    return report;
}

const std::vector<MarkerParticle>& MarkerParticleSystem::particles() const noexcept {
    return m_particles;
}

std::size_t MarkerParticleSystem::validParticleCount() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        m_particles.begin(), m_particles.end(), [](const MarkerParticle& particle) { return particle.valid; }));
}

std::size_t MarkerParticleSystem::invalidPositionCount() const noexcept {
    return m_invalidPositionCount;
}

bool MarkerParticleSystem::allFiniteAndInside(const MacGrid& grid) const noexcept {
    const float width = static_cast<float>(grid.width()) * grid.cellWidth();
    const float height = static_cast<float>(grid.height()) * grid.cellHeight();
    return std::all_of(m_particles.begin(), m_particles.end(), [&](const MarkerParticle& particle) {
        return !particle.valid ||
            (std::isfinite(particle.position.x) && std::isfinite(particle.position.y) &&
             particle.position.x > 0.0f && particle.position.x < width &&
             particle.position.y > 0.0f && particle.position.y < height &&
             !positionInsideSolid(grid, particle.position));
    });
}

void MarkerParticleSystem::loadState(
    const MacGrid& grid,
    const std::vector<MarkerParticle>& particles) {
    m_particles = particles;
    m_invalidPositionCount = 0;
    if (!allFiniteAndInside(grid)) {
        m_particles.clear();
        throw std::invalid_argument("Marker state is not finite and inside the grid.");
    }
    m_occupiedCells.assign(grid.width() * grid.height(), 0);
}

bool MarkerParticleSystem::positionInsideSolid(const MacGrid& grid, Vector2 position) noexcept {
    const Vector2 clamped = clampInsideDomain(grid, position);
    const std::size_t x = std::min(
        static_cast<std::size_t>(clamped.x / grid.cellWidth()), grid.width() - 1);
    const std::size_t y = std::min(
        static_cast<std::size_t>(clamped.y / grid.cellHeight()), grid.height() - 1);
    return grid.cellType(x, y) == CellType::solid;
}

Vector2 MarkerParticleSystem::clampInsideDomain(const MacGrid& grid, Vector2 position) noexcept {
    const float epsilonX = grid.cellWidth() * 1.0e-3f;
    const float epsilonY = grid.cellHeight() * 1.0e-3f;
    return {
        std::clamp(position.x, epsilonX,
            static_cast<float>(grid.width()) * grid.cellWidth() - epsilonX),
        std::clamp(position.y, epsilonY,
            static_cast<float>(grid.height()) * grid.cellHeight() - epsilonY),
    };
}

Vector2 MarkerParticleSystem::resolveCollision(
    const MacGrid& grid,
    Vector2 previous,
    Vector2 candidate,
    bool& collided) noexcept {
    candidate = clampInsideDomain(grid, candidate);
    if (!positionInsideSolid(grid, candidate)) return candidate;
    collided = true;
    const Vector2 xOnly = clampInsideDomain(grid, {candidate.x, previous.y});
    if (!positionInsideSolid(grid, xOnly)) return xOnly;
    const Vector2 yOnly = clampInsideDomain(grid, {previous.x, candidate.y});
    if (!positionInsideSolid(grid, yOnly)) return yOnly;
    return previous;
}

} // namespace fluidlab::simulation
