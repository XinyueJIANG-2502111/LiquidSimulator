#pragma once

#include "PressureProjection.h"

#include <cstddef>

namespace fluidlab::simulation {

struct FreeSurfaceProjectionReport final {
    ProjectionReport projection;
    std::size_t fluidCellCount = 0;
    std::size_t fluidAirFaceCount = 0;
    bool pressureReferenceFixedByAir = false;
};

class FreeSurfacePressureProjector final {
public:
    explicit FreeSurfacePressureProjector(PressureSolverSettings settings = {});

    [[nodiscard]] FreeSurfaceProjectionReport project(MacGrid& grid) const;
    [[nodiscard]] const PressureSolverSettings& settings() const noexcept;

private:
    struct DomainInfo final {
        std::size_t fluidCellCount = 0;
        std::size_t fluidAirFaceCount = 0;
    };

    [[nodiscard]] DomainInfo inspectDomain(const MacGrid& grid) const noexcept;
    [[nodiscard]] DivergenceStatistics fluidDivergenceStatistics(const MacGrid& grid) const noexcept;
    [[nodiscard]] PressureResidual computeResidual(const MacGrid& grid) const noexcept;
    [[nodiscard]] float normalizeClosedFluidPressure(MacGrid& grid) const noexcept;
    [[nodiscard]] float maximumAbsoluteFluidPressure(const MacGrid& grid) const noexcept;
    [[nodiscard]] float applyPressureGradient(MacGrid& grid) const noexcept;
    void clearNonFluidPressure(MacGrid& grid) const noexcept;

    PressureSolverSettings m_settings;
};

} // namespace fluidlab::simulation
