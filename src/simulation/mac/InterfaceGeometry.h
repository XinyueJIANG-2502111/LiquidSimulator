#pragma once

#include "MarkerParticles.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace fluidlab::simulation {

struct InterfaceGeometrySettings final {
    float particleSupportRadius = 0.0f;
    std::size_t areaSamplesPerAxis = 4;
};

struct InterfaceGeometryResult final {
    std::size_t width = 0;
    std::size_t height = 0;
    float cellWidth = 0.0f;
    float cellHeight = 0.0f;
    float particleSupportRadius = 0.0f;
    std::vector<float> signedDistanceProxy;
    std::vector<Vector2> normals;
    std::vector<float> areaFractions;
    std::vector<float> horizontalInterfaceFractions;
    std::vector<float> verticalInterfaceFractions;
    double estimatedArea = 0.0;
    std::size_t validParticleCount = 0;
};

// Reconstructs the union of equal-radius particle support disks. The returned
// distance is exact for that proxy geometry, not for the unknown physical surface.
[[nodiscard]] InterfaceGeometryResult reconstructParticleInterface(
    std::size_t width,
    std::size_t height,
    float cellWidth,
    float cellHeight,
    const std::vector<MarkerParticle>& particles,
    const InterfaceGeometrySettings& settings);

} // namespace fluidlab::simulation
