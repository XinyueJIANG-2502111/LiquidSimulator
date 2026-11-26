#pragma once

#include "LevelSetInterface.h"
#include "VolumeFractionField.h"
#include "GhostFluidPressureProjection.h"

#include <cstddef>

namespace fluidlab::simulation {

struct ClsvofCouplingSettings final {
    double maximumCorrectionDistance = 0.0;
    double minimumInterfaceFraction = 1.0e-6;
    double maximumInterfaceFraction = 1.0 - 1.0e-6;
    double narrowBandWidth = 0.0;
    bool redistanceAfterCorrection = true;
};

struct ClsvofCouplingReport final {
    std::size_t interfaceCellCount = 0;
    std::size_t correctedCellCount = 0;
    std::size_t plicFailureCount = 0;
    std::size_t lowConfidenceCount = 0;
    std::size_t rejectedLargeCorrectionCount = 0;
    std::size_t validSegmentCount = 0;
    std::size_t incompatibleSegmentCount = 0;
    std::size_t geometryFallbackCount = 0;
    double vofArea = 0.0;
    double levelSetAreaBefore = 0.0;
    double levelSetAreaAfter = 0.0;
    double maximumInterfaceCorrection = 0.0;
    double rmsInterfaceCorrection = 0.0;
    LevelSetReinitializationReport redistance;
};

enum class ClsvofCouplingMethod : unsigned char {
    cellCenter,
    plicSegmentDistance,
};

struct CoupledInterfaceState final {
    VolumeFractionField volumeFraction;
    LevelSetField levelSet;
    std::size_t geometryVersion = 0;
    double physicalTime = 0.0;
    bool valid = false;
};

class CoupledLevelSetVOF final {
public:
    void initialize(
        VolumeFractionField volumeFraction,
        LevelSetField levelSet,
        double physicalTime = 0.0);

    [[nodiscard]] ClsvofCouplingReport prepareCandidate(
        const VolumeFractionField& advectedVolumeFraction,
        const LevelSetField& advectedLevelSet,
        const ClsvofCouplingSettings& settings);

    void acceptCandidate();
    void rejectCandidate() noexcept;
    void reset();

    [[nodiscard]] const CoupledInterfaceState& committed() const noexcept;
    [[nodiscard]] const CoupledInterfaceState& candidate() const noexcept;
    [[nodiscard]] bool hasCandidate() const noexcept;

private:
    CoupledInterfaceState m_committed;
    CoupledInterfaceState m_candidate;
    bool m_hasCandidate = false;
};

[[nodiscard]] ClsvofCouplingReport constrainLevelSetToVolumeFraction(
    const VolumeFractionField& volumeFraction,
    LevelSetField& levelSet,
    const ClsvofCouplingSettings& settings);

[[nodiscard]] ClsvofCouplingReport constrainLevelSetToPlicGeometry(
    const VolumeFractionField& volumeFraction,
    LevelSetField& levelSet,
    const ClsvofCouplingSettings& settings,
    PlicNormalMethod normalMethod = PlicNormalMethod::heightFunction);

[[nodiscard]] GhostFluidThetaData buildPlicGhostFluidTheta(
    const MacGrid& grid,
    const VolumeFractionField& volumeFraction,
    PlicNormalMethod normalMethod,
    double minimumTheta,
    bool fallbackToCellBoundary = true);

} // namespace fluidlab::simulation
