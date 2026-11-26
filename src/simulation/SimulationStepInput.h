#pragma once

#include "mac/MacGrid.h"

#include <cstdint>

namespace fluidlab::simulation {

struct FluidInjectionInput final {
    Vector2 position{};
    float radius = 0.0f;
    double requestedArea = 0.0;
    std::uint64_t version = 0;
    bool active = false;
};

struct SimulationStepInput final {
    Vector2 externalAcceleration{};
    Vector2 externalVelocityDelta{};
    double sourceTimeBegin = 0.0;
    double sourceTimeEnd = 0.0;
    double physicalDuration = 0.0;
    std::uint64_t version = 0;
    FluidInjectionInput fluidInjection{};
    bool valid = true;
};

[[nodiscard]] bool isFinite(const SimulationStepInput& input) noexcept;

} // namespace fluidlab::simulation
