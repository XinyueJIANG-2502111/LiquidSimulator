#pragma once

#include "GhostFluidPressureProjection.h"
#include "MarkerParticles.h"

#include <cstddef>

namespace fluidlab::simulation {

struct DynamicInterfaceVersions final {
    std::size_t classification = 0;
    std::size_t geometry = 0;
    std::size_t theta = 0;
    std::size_t fineOperator = 0;
    std::size_t multigridHierarchy = 0;
};

struct DynamicInterfaceSnapshot final {
    LevelSetField field;
    DynamicInterfaceVersions versions;
    double simulationTime = 0.0;
    bool valid = false;
};

struct ParticleLevelSetConsistency final {
    std::size_t validParticleCount = 0;
    std::size_t particleInPositivePhiCount = 0;
    std::size_t farPositiveParticleCount = 0;
    std::size_t negativeCellWithoutParticleCount = 0;
    std::size_t classificationSignMismatchCount = 0;
    std::size_t lowConfidenceInterfaceCellCount = 0;
    std::size_t solidAdjacentConflictCount = 0;
    double maximumParticlePositiveDistance = 0.0;
    double meanClassificationInterfaceDistance = 0.0;
};

struct ParticleCorrectionSettings final {
    bool enabled = false;
    double interfaceTolerance = 0.0;
    double maximumCorrectionDistance = 0.0;
    double correctionDepth = 0.0;
};

struct ParticleCorrectionReport final {
    std::size_t proposedCount = 0;
    std::size_t appliedCount = 0;
    std::size_t rejectedFarCount = 0;
    std::size_t rejectedSolidCount = 0;
    double areaBefore = 0.0;
    double areaAfter = 0.0;
    double maximumValueChange = 0.0;
};

struct DynamicLevelSetAdvanceReport final {
    LevelSetReinitializationReport redistance;
    ParticleLevelSetConsistency consistency;
    ParticleCorrectionReport correction;
    GhostThetaStatistics theta;
    double advectionMilliseconds = 0.0;
    double redistanceMilliseconds = 0.0;
    double consistencyMilliseconds = 0.0;
    double correctionMilliseconds = 0.0;
    double thetaMilliseconds = 0.0;
};

class DynamicLevelSetTracker final {
public:
    void initialize(LevelSetField initialField, double simulationTime = 0.0);

    [[nodiscard]] DynamicLevelSetAdvanceReport prepareCandidate(
        const MacGrid& velocityGrid,
        const MarkerParticleSystem& particles,
        double timeStep,
        const LevelSetRedistanceSettings& redistanceSettings,
        const ParticleCorrectionSettings& correctionSettings,
        double thetaMinimum = 0.1);

    void acceptCandidate();
    void rejectCandidate() noexcept;

    [[nodiscard]] const DynamicInterfaceSnapshot& committed() const noexcept;
    [[nodiscard]] const DynamicInterfaceSnapshot& candidate() const noexcept;
    [[nodiscard]] bool hasCandidate() const noexcept;

private:
    DynamicInterfaceSnapshot m_committed;
    DynamicInterfaceSnapshot m_candidate;
    bool m_hasCandidate = false;
};

[[nodiscard]] ParticleLevelSetConsistency diagnoseParticleLevelSetConsistency(
    const MacGrid& grid,
    const MarkerParticleSystem& particles,
    const LevelSetField& field,
    double interfaceTolerance);

[[nodiscard]] ParticleCorrectionReport applyConstrainedParticleCorrection(
    const MacGrid& grid,
    const MarkerParticleSystem& particles,
    LevelSetField& field,
    const ParticleCorrectionSettings& settings);

} // namespace fluidlab::simulation
