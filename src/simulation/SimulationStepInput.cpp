#include "SimulationStepInput.h"

#include <cmath>

namespace fluidlab::simulation {

bool isFinite(const SimulationStepInput& input) noexcept {
    return input.valid &&
        std::isfinite(input.externalAcceleration.x) &&
        std::isfinite(input.externalAcceleration.y) &&
        std::isfinite(input.externalVelocityDelta.x) &&
        std::isfinite(input.externalVelocityDelta.y) &&
        std::isfinite(input.sourceTimeBegin) &&
        std::isfinite(input.sourceTimeEnd) &&
        std::isfinite(input.physicalDuration) &&
        std::isfinite(input.fluidInjection.position.x) &&
        std::isfinite(input.fluidInjection.position.y) &&
        std::isfinite(input.fluidInjection.radius) &&
        std::isfinite(input.fluidInjection.requestedArea) &&
        input.fluidInjection.radius >= 0.0f &&
        input.fluidInjection.requestedArea >= 0.0 &&
        input.physicalDuration >= 0.0 &&
        input.sourceTimeEnd >= input.sourceTimeBegin;
}

} // namespace fluidlab::simulation
