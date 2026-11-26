#pragma once

#include "../simulation/mac/MacGrid.h"
#include "../simulation/mac/MarkerParticles.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fluidlab::desktop {

struct DesktopSettledMetadata final {
    std::uint32_t formatVersion = 1;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    float cellWidth = 0.0f;
    float cellHeight = 0.0f;
    double preEvolutionSeconds = 0.0;
    double velocityRms = 0.0;
    float maximumVelocity = 0.0f;
    double centroidDriftPerSecond = 0.0;
    std::uint32_t stableWindowCount = 0;
    std::uint64_t configurationId = 0;
};

struct DesktopSettledState final {
    DesktopSettledMetadata metadata;
    std::vector<float> horizontalVelocity;
    std::vector<float> verticalVelocity;
    std::vector<float> pressure;
    std::vector<fluidlab::simulation::CellType> cellTypes;
    std::vector<fluidlab::simulation::MarkerParticle> particles;
};

struct DesktopSettledLoadResult final {
    bool valid = false;
    std::string message;
    DesktopSettledState state;
};

[[nodiscard]] std::uint64_t desktopSettledConfigurationId(
    std::size_t width, std::size_t height, float dx, float dy) noexcept;
void saveDesktopSettledState(
    const std::filesystem::path& path, const DesktopSettledState& state);
[[nodiscard]] DesktopSettledLoadResult loadDesktopSettledState(
    const std::filesystem::path& path,
    std::size_t expectedWidth,
    std::size_t expectedHeight,
    float expectedDx,
    float expectedDy) noexcept;
void applyDesktopSettledState(
    const DesktopSettledState& state,
    fluidlab::simulation::MacGrid& grid,
    fluidlab::simulation::MarkerParticleSystem& particles);

} // namespace fluidlab::desktop
