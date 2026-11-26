#pragma once

#include "GpuMacGrid.h"
#include "../SimulationStepInput.h"

#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace fluidlab::simulation {

struct GpuDynamicFreeSurfaceSettings final {
    float timeStep = 1.0f / 120.0f;
    float density = 1.0f;
    Vector2 gravity{0.0f, -9.81f};
    float maximumCfl = 0.75f;
    float cflPlanningSafetyFactor = 0.8f;
    std::size_t maximumSubsteps = 32;
    std::size_t maximumCandidateRetries = 8;
    float minimumSubstep = 1.0e-7f;
    float candidateRetryScale = 0.5f;
    std::size_t extrapolationLayers = 3;
    float pressureResidualTolerance = 5.0e-3f;
    std::size_t pressureMaximumIterations = 4000;
    std::size_t pressureResidualCheckInterval = 20;
    GpuPressureControlMode pressureControlMode = GpuPressureControlMode::cpuSynchronized;
    GpuPressurePrecisionMode pressurePrecisionMode = GpuPressurePrecisionMode::fp64;
    bool pressureFusedCoarseSmootherEnabled = true;
    std::size_t pressureGpuResidentBatchCount = 1;
    bool pressureWarmStartEnabled = true;
    bool particleRedistributionEnabled = false;
    std::size_t particleRedistributionInterval = 8;
    std::size_t particleRedistributionTargetMinimum = 2;
    bool synchronousGpuTimingEnabled = true;
    bool asynchronousGpuTimingEnabled = true;
    bool detailedVelocityDiagnostics = false;
    bool adaptiveCflSchedulingEnabled = true;
    std::size_t forcedCandidateRejectionsForValidation = 0;
    std::size_t projectionSnapshotFirstOuterStep = std::numeric_limits<std::size_t>::max();
    std::size_t projectionSnapshotLastOuterStep = 0;
    bool experimentalLevelSetEnabled = false;
    GpuLevelSetSettings experimentalLevelSetSettings{};
    std::function<void(const MacGrid&, float)> acceptedSubstepValidationObserver;
};

enum class GpuVelocityStage {
    stepStart,
    preAdvectionExtrapolation,
    advection,
    forceAndBoundary,
    projection,
    postProjectionExtrapolation,
};

enum class GpuDynamicFailureReason {
    none,
    notInitialized,
    invalidStepInput,
    cflSubstepLimit,
    invalidParticle,
    pressureFailure,
    pressureNotConverged,
    topologyAnalysisFailure,
    unsupportedPressureTopology,
};

struct GpuProjectionDiagnosticSnapshot final {
    std::size_t outerStep = 0;
    std::size_t substep = 0;
    float timeStep = 0.0f;
    std::vector<float> horizontalBefore;
    std::vector<float> verticalBefore;
    std::vector<float> horizontalAfter;
    std::vector<float> verticalAfter;
    std::vector<float> horizontalAfterExtrapolation;
    std::vector<float> verticalAfterExtrapolation;
    std::vector<float> pressureBefore;
    std::vector<float> pressureAfter;
    std::vector<double> residualAfter;
    std::vector<CellType> cellTypes;
    std::vector<CellType> previousCellTypes;
    std::vector<CellType> cellTypesAfterClassification;
    std::vector<GLuint> particleCounts;
    std::vector<GLuint> particleCountsAfterClassification;
    std::vector<GLuint> horizontalValidityAfterExtrapolation;
    std::vector<GLuint> verticalValidityAfterExtrapolation;
    GpuPressureOperatorSnapshot pressureOperator;
    GpuProjectionReport projection;
    GpuParticleAdvectionReport particleAdvection;
    GpuClassificationReport classification;
    float maximumPostProjectionSpeed = 0.0f;
    float maximumVelocityChange = 0.0f;
    double kineticEnergyBefore = 0.0;
    double kineticEnergyAfter = 0.0;
    char maximumComponent = 'u';
    std::size_t maximumIndex = 0;
};

struct GpuDynamicFreeSurfaceReport final {
    struct Timings final {
        double totalCpuMilliseconds = 0.0;
        double velocityReductionGpuMilliseconds = 0.0;
        double velocityReductionReadbackMilliseconds = 0.0;
        double velocityExtrapolationGpuMilliseconds = 0.0;
        double velocityAdvectionGpuMilliseconds = 0.0;
        double forceBoundaryGpuMilliseconds = 0.0;
        double divergenceGpuMilliseconds = 0.0;
        double pressureGpuMilliseconds = 0.0;
        double pressureReductionGpuMilliseconds = 0.0;
        double pressureReadbackMilliseconds = 0.0;
        double projectionGpuMilliseconds = 0.0;
        double particleGpuMilliseconds = 0.0;
        double classificationGpuMilliseconds = 0.0;
        double particleRedistributionGpuMilliseconds = 0.0;
        double topologyFineGpuMilliseconds = 0.0;
        double topologyAnalysisGpuMilliseconds = 0.0;
        double topologyCoarseGpuMilliseconds = 0.0;
        double topologyEndToEndMilliseconds = 0.0;
        double smallStatusReadbackMilliseconds = 0.0;
        double asynchronousStepGpuMilliseconds = 0.0;
        std::size_t asynchronousStepTimingLatency = 0;
        bool asynchronousStepTimingAvailable = false;
    } timings;
    GpuProjectionReport projection;
    GpuParticleAdvectionReport particleAdvection;
    GpuClassificationReport classification;
    GpuParticleRedistributionReport redistribution;
    GpuDynamicTopologyReport topology;
    GpuVelocityExtrema velocity;
    GpuLevelSetReport levelSet;
    GpuVelocityStage peakVelocityStage = GpuVelocityStage::stepStart;
    GpuDynamicFailureReason failureReason = GpuDynamicFailureReason::none;
    std::string failureMessage;
    std::size_t substeps = 0;
    std::size_t candidateAttempts = 0;
    std::size_t rejectedSubsteps = 0;
    std::size_t rollbackCount = 0;
    std::size_t batchOuterStepsRequested = 1;
    std::size_t batchOuterStepsCompleted = 0;
    std::size_t maximumRetriesForOneSubstep = 0;
    std::size_t particleCount = 0;
    std::size_t fluidCellCount = 0;
    std::size_t pressureIterationsTotal = 0;
    std::size_t pressureIterationsMaximum = 0;
    std::size_t pressureSubmittedIterationsTotal = 0;
    std::size_t pressureSkippedIterationsTotal = 0;
    std::size_t pressureStateReadbackCount = 0;
    std::size_t pressureDispatchCount = 0;
    std::size_t particlesAdded = 0;
    std::size_t injectionCellsRequested = 0;
    std::size_t injectionCellsAccepted = 0;
    std::size_t injectionParticlesAdded = 0;
    std::size_t topologyRebuildCount = 0;
    std::size_t topologyReuseCount = 0;
    std::size_t topologyLabelPropagationIterations = 0;
    float maximumObservedCfl = 0.0f;
    float maximumAcceptedCfl = 0.0f;
    float lastSubmittedTimeStep = 0.0f;
    float markedArea = 0.0f;
    double simulationTime = 0.0;
    double physicalTimeRequested = 0.0;
    double physicalTimeAdvanced = 0.0;
    double remainingPhysicalTime = 0.0;
    double injectionAreaRequested = 0.0;
    double injectionAreaAccepted = 0.0;
    Vector2 appliedExternalAcceleration{};
    Vector2 appliedExternalVelocityDelta{};
    Vector2 effectiveAcceleration{};
    double inputSourceTimeBegin = 0.0;
    double inputSourceTimeEnd = 0.0;
    std::uint64_t inputVersion = 0;
    bool topologyValid = false;
    bool pressureConverged = false;
    bool advanced = false;
    bool outerStepCompleted = false;
    bool computeBudgetExhausted = false;
    bool pressureTopologyReused = false;
    std::optional<GpuProjectionDiagnosticSnapshot> projectionSnapshot;
};

class GpuDynamicFreeSurfaceStepper final {
public:
    explicit GpuDynamicFreeSurfaceStepper(
        const std::filesystem::path& divergenceShaderPath,
        GpuDynamicFreeSurfaceSettings settings = {});

    [[nodiscard]] GpuDynamicTopologyReport initialize(
        const MacGrid& grid,
        const MarkerParticleSystem& particles,
        const LevelSetField* initialLevelSet = nullptr);
    [[nodiscard]] GpuDynamicFreeSurfaceReport step();
    [[nodiscard]] GpuDynamicFreeSurfaceReport step(const SimulationStepInput& input);
    [[nodiscard]] GpuMacGrid& grid() noexcept;
    [[nodiscard]] const GpuMacGrid& grid() const noexcept;
    [[nodiscard]] const GpuDynamicFreeSurfaceSettings& settings() const noexcept;
    [[nodiscard]] double simulationTime() const noexcept;
    [[nodiscard]] double pendingPhysicalTime() const noexcept;
    [[nodiscard]] std::uint64_t completedOuterSteps() const noexcept;
    [[nodiscard]] bool healthy() const noexcept;

private:
    void fail(
        GpuDynamicFreeSurfaceReport& report,
        GpuDynamicFailureReason reason,
        std::string message) noexcept;
    [[nodiscard]] GpuPressureSettings pressureSettings(float timeStep) const noexcept;

    GpuMacGrid m_grid;
    GpuDynamicFreeSurfaceSettings m_settings;
    std::size_t m_width = 0;
    std::size_t m_height = 0;
    float m_dx = 0.0f;
    float m_dy = 0.0f;
    double m_simulationTime = 0.0;
    bool m_initialized = false;
    bool m_healthy = false;
    GpuDynamicTopologyReport m_topology;
    GpuClassificationReport m_classification;
    GpuProjectionReport m_lastProjection;
    GpuVelocityStage m_lastPeakVelocityStage = GpuVelocityStage::stepStart;
    std::uint64_t m_completedOuterSteps = 0;
    double m_pendingPhysicalTime = 0.0;
    std::size_t m_forcedCandidateRejectionsRemaining = 0;
    SimulationStepInput m_activeStepInput;
    bool m_activeInjectionApplied = false;
};

[[nodiscard]] const char* gpuDynamicFailureName(GpuDynamicFailureReason reason) noexcept;
[[nodiscard]] const char* gpuVelocityStageName(GpuVelocityStage stage) noexcept;

} // namespace fluidlab::simulation
