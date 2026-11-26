#pragma once

#include "../mac/MacGrid.h"
#include "../mac/MarkerParticles.h"
#include "../mac/LevelSetInterface.h"
#include "../mac/PressureProjection.h"
#include "../../graphics/ShaderProgram.h"

#include <filesystem>
#include <chrono>
#include <cstdint>
#include <array>
#include <vector>

#include <glad/gl.h>

namespace fluidlab::simulation {

enum class GpuPressureSolver {
    weightedJacobi,
    preconditionedConjugateGradient,
    multigridPreconditionedConjugateGradient,
};

enum class GpuPressureControlMode {
    cpuSynchronized,
    gpuResidentBatches,
};

enum class GpuPressurePrecisionMode {
    fp64,
    fp64OuterFp32Multigrid,
};

enum class GpuPressureSolveState : std::uint32_t {
    running = 0,
    converged = 1,
    maximumIterationsReached = 2,
    numericalBreakdown = 3,
    invalidTopology = 4,
    invalidResidual = 5,
};

enum class GpuPressureFailureReason {
    none,
    numericalBreakdown,
    unsupportedMultiplePureNeumannComponents,
    isolatedFluidCell,
    dynamicPressureTopologyStale,
    dynamicPressureTopologyAnalysisFailed,
};

struct GpuDynamicTopologyReport final {
    bool fineOperatorValid = false;
    bool topologyValid = false;
    bool multigridValid = false;
    bool usedFullClassificationReadback = false;
    bool reusedExistingHierarchy = false;
    std::size_t fluidCellCount = 0;
    std::size_t connectedComponentCount = 0;
    std::size_t pureNeumannComponentCount = 0;
    std::size_t isolatedFluidCellCount = 0;
    std::size_t labelPropagationIterations = 0;
    std::size_t activeMultigridLevelCount = 0;
    std::size_t statusBytesRead = 0;
    std::uint64_t classificationVersion = 0;
    std::uint64_t fineOperatorVersion = 0;
    std::uint64_t multigridVersion = 0;
    double fineOperatorGpuMilliseconds = 0.0;
    double topologyGpuMilliseconds = 0.0;
    double coarseOperatorGpuMilliseconds = 0.0;
    double statusReadbackMilliseconds = 0.0;
    double totalCpuMilliseconds = 0.0;
};

struct GpuAsyncTimingSample final {
    double milliseconds = 0.0;
    std::size_t latencySteps = 0;
    bool available = false;
};

enum class GpuLevelSetThetaState : GLuint {
    unused = 0,
    valid = 1,
    clamped = 2,
    fallback = 3,
};

enum class GpuLevelSetAdvectionMethod : GLuint {
    rk2SemiLagrangian = 0,
    limitedMacCormack = 1,
};

enum class GpuLevelSetRedistanceMethod : GLuint {
    marchingSquares = 0,
    interfacePreserving = 1,
};

enum class GpuLevelSetVelocityMode : GLuint {
    formalExtrapolated = 0,
    normalExtended = 1,
};

struct GpuLevelSetSettings final {
    float narrowBandWidth = 0.0f;
    float thetaMinimum = 0.1f;
    float minimumConfidence = 0.5f;
    std::size_t redistanceInterval = 4;
    double redistancePhysicalInterval = 0.0;
    std::size_t diagnosticReadbackInterval = 8;
    GpuLevelSetAdvectionMethod advectionMethod =
        GpuLevelSetAdvectionMethod::rk2SemiLagrangian;
    GpuLevelSetRedistanceMethod redistanceMethod =
        GpuLevelSetRedistanceMethod::marchingSquares;
    GpuLevelSetVelocityMode velocityMode =
        GpuLevelSetVelocityMode::formalExtrapolated;
    std::size_t velocityExtensionLayers = 6;
    bool stageAreaAuditEnabled = false;
};

struct GpuLevelSetReport final {
    std::uint64_t geometryVersion = 0;
    std::uint64_t thetaVersion = 0;
    std::size_t advectionCount = 0;
    std::size_t redistanceCount = 0;
    std::size_t invalidSampleCount = 0;
    std::size_t invalidPhiCount = 0;
    std::size_t lowConfidenceCellCount = 0;
    std::size_t signConflictCount = 0;
    std::size_t positivePhiParticleCount = 0;
    std::size_t validThetaCount = 0;
    std::size_t clampedThetaCount = 0;
    std::size_t fallbackThetaCount = 0;
    std::size_t classificationStableThetaChangeCount = 0;
    std::size_t negativeCellCount = 0;
    std::size_t negativeComponentCount = 0;
    std::size_t topologyPropagationIterations = 0;
    std::size_t highOrderCorrectionCount = 0;
    std::size_t limiterClampCount = 0;
    std::size_t highOrderFallbackCount = 0;
    std::size_t highOrderNonFiniteCount = 0;
    std::size_t domainClampedBacktraceCount = 0;
    std::size_t solidBacktraceCount = 0;
    std::size_t lowConfidenceBacktraceCount = 0;
    std::size_t stageAreaAuditCount = 0;
    std::size_t interfacePreservedCellCount = 0;
    std::size_t velocityExtensionSeedCount = 0;
    std::size_t velocityExtensionFilledCount = 0;
    std::size_t velocityExtensionUnresolvedCount = 0;
    float maximumThetaChange = 0.0f;
    double geometricArea = 0.0;
    double initialArea = 0.0;
    double areaBeforeAdvection = 0.0;
    double areaAfterAdvection = 0.0;
    double areaAfterRedistance = 0.0;
    double lastAdvectionAreaChange = 0.0;
    double lastRedistanceAreaChange = 0.0;
    double cumulativeAdvectionAreaChange = 0.0;
    double cumulativeRedistanceAreaChange = 0.0;
    double fluidMeanAbsoluteDivergence = 0.0;
    double interfaceMeanAbsoluteDivergence = 0.0;
    double airBandMeanAbsoluteDivergence = 0.0;
    float fluidMaximumAbsoluteDivergence = 0.0f;
    float interfaceMaximumAbsoluteDivergence = 0.0f;
    float airBandMaximumAbsoluteDivergence = 0.0f;
    double physicalTime = 0.0;
    double lastRedistancePhysicalTime = 0.0;
    double advectionCpuMilliseconds = 0.0;
    double redistanceCpuMilliseconds = 0.0;
    double velocityExtensionCpuMilliseconds = 0.0;
    double thetaCpuMilliseconds = 0.0;
    bool initialized = false;
    bool candidateValid = false;
};

struct GpuLevelSetSnapshot final {
    std::vector<float> phi;
    std::vector<GLuint> confidence;
    std::vector<float> horizontalTheta;
    std::vector<float> verticalTheta;
    std::vector<GLuint> horizontalThetaState;
    std::vector<GLuint> verticalThetaState;
    GpuLevelSetReport report;
};

struct GpuPressureOperatorSnapshot final {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<double> diagonal;
    std::vector<std::array<double, 4>> weights;
    std::vector<GLuint> active;
    std::vector<GLuint> component;
};

struct GpuClassificationReport final {
    std::size_t particleCount = 0;
    std::size_t validParticleCount = 0;
    std::size_t invalidParticleCount = 0;
    std::size_t fluidCellCount = 0;
    std::size_t changedCellCount = 0;
    std::size_t enclosedAirCellCount = 0;
    std::size_t surfaceHoleCandidateCount = 0;
    std::size_t phaseToggleCount = 0;
    std::size_t repeatedFlickerCellCount = 0;
    std::size_t airCellWithParticleCount = 0;
    std::size_t maximumParticlesPerCell = 0;
    std::size_t bytesRead = 0;
    double gpuMilliseconds = 0.0;
    double readbackMilliseconds = 0.0;
};

struct GpuParticleRedistributionReport final {
    std::size_t particleCountBefore = 0;
    std::size_t particleCountAfter = 0;
    std::size_t addedParticleCount = 0;
    std::size_t capacityOverflowCount = 0;
    double gpuMilliseconds = 0.0;
    double readbackMilliseconds = 0.0;
};

struct GpuFluidInjectionReport final {
    std::size_t requestedCellCount = 0;
    std::size_t acceptedCellCount = 0;
    std::size_t particlesAdded = 0;
    std::size_t rejectedSolidOrFluidCells = 0;
    bool capacityReached = false;
};

struct GpuParticleAdvectionReport final {
    ParticleAdvectionReport particles;
    std::size_t bytesRead = 0;
    double gpuMilliseconds = 0.0;
    double readbackMilliseconds = 0.0;
};

struct GpuPressureSettings final {
    float density = 1.0f;
    float timeStep = 1.0f / 60.0f;
    float residualTolerance = 5.0e-3f;
    float relaxationFactor = 0.8f;
    std::size_t maximumIterations = 12000;
    std::size_t residualCheckInterval = 100;
    std::size_t multigridPreSmoothIterations = 2;
    std::size_t multigridPostSmoothIterations = 2;
    std::size_t multigridCoarseIterations = 40;
    float multigridRelaxationFactor = 0.8f;
    GpuPressureSolver solver = GpuPressureSolver::weightedJacobi;
    bool warmStartEnabled = false;
    bool collectVelocityChange = false;
    GpuPressureControlMode controlMode = GpuPressureControlMode::gpuResidentBatches;
    GpuPressurePrecisionMode precisionMode = GpuPressurePrecisionMode::fp64;
    std::size_t gpuResidentBatchCount = 1;
    bool fusedCoarseSmootherEnabled = true;
};

struct GpuPressureReport final {
    PressureResidual initialResidual;
    PressureResidual finalResidual;
    std::size_t iterations = 0;
    bool converged = false;
    bool warmStartUsed = false;
    PressureSolveTermination termination = PressureSolveTermination::maximumIterationsReached;
    GpuPressureSolver solver = GpuPressureSolver::weightedJacobi;
    GpuPressureFailureReason failureReason = GpuPressureFailureReason::none;
    std::size_t validEquationCount = 0;
    std::size_t residualBytesRead = 0;
    double gpuMilliseconds = 0.0;
    double reductionGpuMilliseconds = 0.0;
    double preconditionerGpuMilliseconds = 0.0;
    double residualReadbackMilliseconds = 0.0;
    std::size_t multigridLevelCount = 0;
    std::size_t multigridMemoryBytes = 0;
    GpuPressureSolveState solveState = GpuPressureSolveState::running;
    std::size_t stateReadbackCount = 0;
    std::size_t submittedIterations = 0;
    std::size_t skippedIterations = 0;
    bool gpuResidentControlUsed = false;
    GpuPressurePrecisionMode precisionMode = GpuPressurePrecisionMode::fp64;
    std::size_t dispatchCount = 0;
};

struct GpuProjectionReport final {
    DivergenceStatistics divergenceBefore;
    DivergenceStatistics divergenceAfter;
    GpuPressureReport pressure;
    double divergenceGpuMilliseconds = 0.0;
    double projectionGpuMilliseconds = 0.0;
    float maximumVelocityChange = 0.0f;
};

struct GpuVelocityExtrema final {
    float maximumAbsoluteU = 0.0f;
    float maximumAbsoluteV = 0.0f;
    std::size_t maximumUIndex = 0;
    std::size_t maximumVIndex = 0;
    double gpuMilliseconds = 0.0;
    double readbackMilliseconds = 0.0;
    std::size_t bytesRead = 0;
};

struct GpuStepReport final {
    GpuProjectionReport projection;
    double advectionGpuMilliseconds = 0.0;
    double forceBoundaryGpuMilliseconds = 0.0;
    double totalCpuSubmissionMilliseconds = 0.0;
};

struct GpuMultigridCycleReport final {
    float initialRootMeanSquare = 0.0f;
    float correctedRootMeanSquare = 0.0f;
    double gpuMilliseconds = 0.0;
    std::size_t validEquationCount = 0;
    std::size_t levelCount = 0;
};

enum class GpuPressureProfileCategory : std::size_t {
    matrixVector,
    typeConversion,
    smoother,
    multigridResidual,
    restriction,
    prolongation,
    coarseSolve,
    vectorUpdate,
    dotReduction,
    trueResidual,
    vCycle,
    count,
};

struct GpuPressureProfileCategoryReport final {
    double gpuMilliseconds = 0.0;
    double cpuSubmissionMilliseconds = 0.0;
    std::size_t dispatchCount = 0;
    std::size_t sampleCount = 0;
};

struct GpuPressureProfileLevelReport final {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t activeCount = 0;
    std::size_t smootherIterations = 0;
    double smootherGpuMilliseconds = 0.0;
    std::size_t smootherDispatchCount = 0;
};

struct GpuPressureProfileReport final {
    GpuPressureReport pressure;
    std::array<GpuPressureProfileCategoryReport,
        static_cast<std::size_t>(GpuPressureProfileCategory::count)> categories{};
    std::vector<GpuPressureProfileLevelReport> levels;
    double endToEndMilliseconds = 0.0;
    double instrumentationOverheadMilliseconds = 0.0;
};

class GpuMacGrid final {
public:
    explicit GpuMacGrid(const std::filesystem::path& divergenceShaderPath);
    ~GpuMacGrid();

    GpuMacGrid(const GpuMacGrid&) = delete;
    GpuMacGrid& operator=(const GpuMacGrid&) = delete;

    void upload(const MacGrid& grid);
    void uploadVelocity(const MacGrid& grid);
    void download(MacGrid& grid) const;
    void dispatchDivergence();
    [[nodiscard]] std::vector<float> readDivergence();
    [[nodiscard]] GpuPressureReport solvePressure(const GpuPressureSettings& settings);
    [[nodiscard]] GpuProjectionReport project(const GpuPressureSettings& settings);
    [[nodiscard]] double advectVelocity(float timeStep);
    [[nodiscard]] double applyAccelerationAndBoundaries(Vector2 acceleration, float timeStep);
    [[nodiscard]] GpuStepReport stepFixedClassification(
        float timeStep,
        Vector2 acceleration,
        const GpuPressureSettings& pressureSettings);
    [[nodiscard]] GpuMultigridCycleReport testMultigridCycle(
        const GpuPressureSettings& settings);
    [[nodiscard]] GpuPressureProfileReport profilePressureSolve(
        const GpuPressureSettings& settings);
    [[nodiscard]] std::vector<double> applyPressurePreconditionerForValidation(
        const std::vector<double>& residual,
        const GpuPressureSettings& settings);
    [[nodiscard]] std::vector<double> readStoredPressureResidualField() const;
    void uploadMarkerParticles(const MarkerParticleSystem& particles);
    [[nodiscard]] GpuFluidInjectionReport injectFluidCells(
        Vector2 position,
        float radius,
        std::size_t requestedCellCount,
        std::size_t particlesPerCell = 8);
    [[nodiscard]] GpuClassificationReport rebuildClassificationFromParticles(
        bool preserveFluidPressure);
    [[nodiscard]] GpuParticleAdvectionReport advectMarkerParticles(float timeStep);
    [[nodiscard]] GpuParticleRedistributionReport redistributeMarkerParticles(
        std::size_t targetMinimum);
    [[nodiscard]] double extrapolateVelocity(std::size_t layers);
    [[nodiscard]] GpuDynamicTopologyReport rebuildDynamicPressureTopology();
    [[nodiscard]] GpuDynamicTopologyReport reuseDynamicPressureTopology();
    void synchronizeDynamicPressureTopology();
    [[nodiscard]] GpuPressureOperatorSnapshot readPressureOperatorLevel(
        std::size_t levelIndex) const;
    [[nodiscard]] GpuVelocityExtrema velocityExtrema();
    void initializeExperimentalLevelSet(
        const LevelSetField& field,
        const GpuLevelSetSettings& settings);
    [[nodiscard]] GpuLevelSetReport advectExperimentalLevelSet(float timeStep);
    [[nodiscard]] GpuLevelSetReport updateExperimentalLevelSetDiagnostics();
    [[nodiscard]] GpuLevelSetSnapshot readExperimentalLevelSet() const;
    [[nodiscard]] bool experimentalLevelSetEnabled() const noexcept;
    [[nodiscard]] const GpuLevelSetReport& experimentalLevelSetReport() const noexcept;
    void saveCandidateState();
    [[nodiscard]] bool restoreCandidateState();
    void commitCandidateState() noexcept;
    [[nodiscard]] bool hasCandidateState() const noexcept;
    void captureCommittedRenderState();
    void setCommittedRenderStateEnabled(bool enabled) noexcept;
    [[nodiscard]] bool committedRenderStateEnabled() const noexcept;
    void dispatchVisualization(GLuint texture, std::size_t mode);
    void setSynchronousTimingEnabled(bool enabled) noexcept;
    [[nodiscard]] GLuint particleBufferHandle() const noexcept;
    [[nodiscard]] std::size_t particleCount() const noexcept;
    [[nodiscard]] std::size_t particleCapacity() const noexcept;

    [[nodiscard]] std::vector<float> readHorizontalVelocity() const;
    [[nodiscard]] std::vector<float> readVerticalVelocity() const;
    [[nodiscard]] std::vector<float> readPressure() const;
    [[nodiscard]] std::vector<double> readResidualField() const;
    [[nodiscard]] std::vector<CellType> readCellTypes() const;
    [[nodiscard]] std::vector<CellType> readPreviousCellTypes() const;
    [[nodiscard]] std::vector<MarkerParticle> readMarkerParticles() const;
    [[nodiscard]] std::vector<GLuint> readParticleCellCounts() const;
    [[nodiscard]] std::vector<GLuint> readClassificationDiagnostics() const;
    [[nodiscard]] std::vector<GLuint> readHorizontalValidity() const;
    [[nodiscard]] std::vector<GLuint> readVerticalValidity() const;
    void beginAsynchronousStepTiming();
    void endAsynchronousStepTiming();
    [[nodiscard]] GpuAsyncTimingSample asynchronousStepTiming() const noexcept;
    [[nodiscard]] double lastUploadMilliseconds() const noexcept;
    [[nodiscard]] double lastGpuMilliseconds() const noexcept;
    [[nodiscard]] std::size_t width() const noexcept;
    [[nodiscard]] std::size_t height() const noexcept;
    [[nodiscard]] float cellWidth() const noexcept;
    [[nodiscard]] float cellHeight() const noexcept;

private:
    struct MultigridLevel final {
        std::size_t width = 0;
        std::size_t height = 0;
        std::size_t activeCount = 0;
        GLuint correction = 0;
        GLuint scratch = 0;
        GLuint rightHandSide = 0;
        GLuint residual = 0;
        GLuint diagonal = 0;
        GLuint weights = 0;
        GLuint active = 0;
        GLuint component = 0;
        GLuint correctionFloat = 0;
        GLuint scratchFloat = 0;
        GLuint rightHandSideFloat = 0;
        GLuint residualFloat = 0;
    };

    void ensureLayout(const MacGrid& grid);
    void bindCommonBuffers() const noexcept;
    void clearPressureBuffers();
    [[nodiscard]] GpuPressureReport solveJacobi(const GpuPressureSettings& settings);
    [[nodiscard]] GpuPressureReport solvePcg(const GpuPressureSettings& settings);
    [[nodiscard]] GpuPressureReport solvePcgGpuResident(const GpuPressureSettings& settings);
    void rebuildMultigridHierarchy(const MacGrid& grid);
    void allocateDynamicMultigridHierarchy();
    void releaseMultigrid() noexcept;
    void applyMultigridPreconditioner(
        GLuint inputResidual,
        GLuint outputPreconditioned,
        const GpuPressureSettings& settings);
    void runMultigridCycle(std::size_t levelIndex, const GpuPressureSettings& settings);
    void runMultigridCycleFloat(std::size_t levelIndex, const GpuPressureSettings& settings);
    void smoothMultigridLevel(
        MultigridLevel& level,
        std::size_t iterations,
        float relaxationFactor);
    void smoothMultigridLevelFloat(
        MultigridLevel& level,
        std::size_t iterations,
        float relaxationFactor);
    void smoothCoarseLevelFused(
        MultigridLevel& level, std::size_t iterations, float relaxationFactor);
    void smoothCoarseLevelFusedFloat(
        MultigridLevel& level, std::size_t iterations, float relaxationFactor);
    [[nodiscard]] bool canUseFusedCoarseSmoother(const MultigridLevel& level) const noexcept;
    void convertDoubleToFloat(GLuint inputBuffer, GLuint outputBuffer, std::size_t elementCount);
    void convertFloatToDouble(GLuint inputBuffer, GLuint outputBuffer, std::size_t elementCount);
    void copyVector(GLuint inputBuffer, GLuint outputBuffer, std::size_t elementCount);
    void clearVector(GLuint outputBuffer, std::size_t elementCount);
    [[nodiscard]] PressureResidual computeResidual(
        float density,
        float timeStep,
        double& reductionGpuMs,
        double& readbackMs,
        std::size_t& validEquationCount,
        std::size_t& bytesRead);
    [[nodiscard]] DivergenceStatistics computeDivergenceStatisticsGpu(
        double& gpuMilliseconds,
        double& readbackMilliseconds,
        std::size_t& bytesRead);
    [[nodiscard]] GLuint reduceDot(GLuint leftBuffer, GLuint rightBuffer);
    [[nodiscard]] GLuint reducePartials(GLuint inputBuffer, std::size_t inputCount);
    void dispatchPcgOperator(GLuint inputBuffer, GLuint outputBuffer);
    void dispatchPcgScalar(GLuint reductionBuffer, GLuint operation);
    [[nodiscard]] GLuint dispatchResidualReduction(
        float density, float timeStep, double& reductionGpuMs);
    void dispatchGpuPressureControl(
        GLuint residualSummaryBuffer,
        float tolerance,
        std::size_t completedIterations,
        std::size_t maximumIterations,
        std::size_t submittedIterations);
    void initializeGpuPressureControl();
    void readGpuPressureControl(GpuPressureReport& report);
    void configureGpuPressureControl(bool enabled);
    void inspectPressureTopology(const MacGrid& grid);
    [[nodiscard]] DivergenceStatistics divergenceStatistics(const std::vector<float>& values) const noexcept;
    [[nodiscard]] double finishTimedQuery();
    void beginTimedQuery();
    void endTimedQuery();
    void release() noexcept;
    void checkOpenGlError(const char* operation) const;
    void pollAsynchronousStepTimings();
    int beginPressureProfileEvent(GpuPressureProfileCategory category, int level = -1);
    void endPressureProfileEvent(int eventIndex, std::size_t dispatchCount);
    void resolvePressureProfile(GpuPressureProfileReport& report);

    graphics::ShaderProgram m_divergenceProgram;
    graphics::ShaderProgram m_jacobiProgram;
    graphics::ShaderProgram m_residualProgram;
    graphics::ShaderProgram m_reduceResidualProgram;
    graphics::ShaderProgram m_reducePartialProgram;
    graphics::ShaderProgram m_reduceDotProgram;
    graphics::ShaderProgram m_pcgOperatorProgram;
    graphics::ShaderProgram m_pcgInitializeProgram;
    graphics::ShaderProgram m_pcgScalarProgram;
    graphics::ShaderProgram m_pcgUpdateProgram;
    graphics::ShaderProgram m_pcgDirectionProgram;
    graphics::ShaderProgram m_mgSmoothProgram;
    graphics::ShaderProgram m_mgCoarseSmoothProgram;
    graphics::ShaderProgram m_mgResidualProgram;
    graphics::ShaderProgram m_mgRestrictProgram;
    graphics::ShaderProgram m_mgProlongProgram;
    graphics::ShaderProgram m_mgSmoothFloatProgram;
    graphics::ShaderProgram m_mgCoarseSmoothFloatProgram;
    graphics::ShaderProgram m_mgResidualFloatProgram;
    graphics::ShaderProgram m_mgRestrictFloatProgram;
    graphics::ShaderProgram m_mgProlongFloatProgram;
    graphics::ShaderProgram m_mgConvertProgram;
    graphics::ShaderProgram m_vectorCopyProgram;
    graphics::ShaderProgram m_projectionProgram;
    graphics::ShaderProgram m_advectionProgram;
    graphics::ShaderProgram m_forceBoundaryProgram;
    graphics::ShaderProgram m_classificationResetProgram;
    graphics::ShaderProgram m_classificationMarkProgram;
    graphics::ShaderProgram m_pressureSanitizeProgram;
    graphics::ShaderProgram m_particleAdvectionProgram;
    graphics::ShaderProgram m_extrapolationInitializeProgram;
    graphics::ShaderProgram m_extrapolationLayerProgram;
    graphics::ShaderProgram m_dynamicFineOperatorProgram;
    graphics::ShaderProgram m_componentInitializeProgram;
    graphics::ShaderProgram m_componentPropagateProgram;
    graphics::ShaderProgram m_componentFlagsProgram;
    graphics::ShaderProgram m_componentSummaryProgram;
    graphics::ShaderProgram m_dynamicCoarseOperatorProgram;
    graphics::ShaderProgram m_classificationDiagnosticsProgram;
    graphics::ShaderProgram m_particleRedistributionProgram;
    graphics::ShaderProgram m_reduceDivergenceProgram;
    graphics::ShaderProgram m_reduceVelocityProgram;
    graphics::ShaderProgram m_reduceVelocityPartialProgram;
    graphics::ShaderProgram m_visualizationProgram;
    graphics::ShaderProgram m_levelSetAdvectionProgram;
    graphics::ShaderProgram m_levelSetCorrectionProgram;
    graphics::ShaderProgram m_levelSetAreaProgram;
    graphics::ShaderProgram m_levelSetRedistanceProgram;
    graphics::ShaderProgram m_levelSetVelocityExtensionInitializeProgram;
    graphics::ShaderProgram m_levelSetVelocityExtensionLayerProgram;
    graphics::ShaderProgram m_levelSetThetaProgram;
    graphics::ShaderProgram m_levelSetParticleDiagnosticProgram;
    graphics::ShaderProgram m_levelSetVelocityDiagnosticProgram;
    graphics::ShaderProgram m_levelSetComponentInitializeProgram;
    graphics::ShaderProgram m_levelSetComponentPropagateProgram;
    graphics::ShaderProgram m_levelSetComponentSummaryProgram;
    GLuint m_uBuffer = 0;
    GLuint m_vBuffer = 0;
    GLuint m_uScratchBuffer = 0;
    GLuint m_vScratchBuffer = 0;
    GLuint m_pressureBuffer = 0;
    GLuint m_pressureScratchBuffer = 0;
    GLuint m_divergenceBuffer = 0;
    GLuint m_cellTypeBuffer = 0;
    GLuint m_residualBuffer = 0;
    GLuint m_reductionBufferA = 0;
    GLuint m_reductionBufferB = 0;
    GLuint m_pcgResidualBuffer = 0;
    GLuint m_pcgPreconditionedBuffer = 0;
    GLuint m_pcgDirectionBuffer = 0;
    GLuint m_pcgApBuffer = 0;
    GLuint m_pcgScalarBuffer = 0;
    GLuint m_pressureControlBuffer = 0;
    GLuint m_solidMaskBuffer = 0;
    GLuint m_previousCellTypeBuffer = 0;
    GLuint m_particleBuffer = 0;
    GLuint m_classificationCounterBuffer = 0;
    GLuint m_particleCellCountBuffer = 0;
    GLuint m_classificationDiagnosticBuffer = 0;
    GLuint m_classificationFlipBuffer = 0;
    GLuint m_redistributionCounterBuffer = 0;
    GLuint m_uValidBuffer = 0;
    GLuint m_vValidBuffer = 0;
    GLuint m_uValidScratchBuffer = 0;
    GLuint m_vValidScratchBuffer = 0;
    GLuint m_componentLabelBufferA = 0;
    GLuint m_componentLabelBufferB = 0;
    GLuint m_componentFlagsBuffer = 0;
    GLuint m_topologyStatusBuffer = 0;
    GLuint m_topologyChangedBuffer = 0;
    GLuint m_multigridBuildStatusBuffer = 0;
    GLuint m_candidateUBuffer = 0;
    GLuint m_candidateVBuffer = 0;
    GLuint m_candidatePressureBuffer = 0;
    GLuint m_candidateCellTypeBuffer = 0;
    GLuint m_candidatePreviousCellTypeBuffer = 0;
    GLuint m_candidateParticleBuffer = 0;
    GLuint m_candidateParticleCellCountBuffer = 0;
    GLuint m_candidateClassificationDiagnosticBuffer = 0;
    GLuint m_candidateClassificationFlipBuffer = 0;
    GLuint m_committedRenderUBuffer = 0;
    GLuint m_committedRenderVBuffer = 0;
    GLuint m_committedRenderPressureBuffer = 0;
    GLuint m_committedRenderDivergenceBuffer = 0;
    GLuint m_committedRenderCellTypeBuffer = 0;
    GLuint m_committedRenderParticleBuffer = 0;
    GLuint m_committedRenderParticleCellCountBuffer = 0;
    GLuint m_committedRenderClassificationDiagnosticBuffer = 0;
    GLuint m_levelSetPhiBuffer = 0;
    GLuint m_levelSetPhiScratchBuffer = 0;
    GLuint m_levelSetPhiAuxBuffer = 0;
    GLuint m_levelSetConfidenceBuffer = 0;
    GLuint m_levelSetConfidenceScratchBuffer = 0;
    GLuint m_levelSetConfidenceAuxBuffer = 0;
    GLuint m_levelSetNarrowBandBuffer = 0;
    GLuint m_levelSetUBuffer = 0;
    GLuint m_levelSetVBuffer = 0;
    GLuint m_levelSetUScratchBuffer = 0;
    GLuint m_levelSetVScratchBuffer = 0;
    GLuint m_levelSetUValidBuffer = 0;
    GLuint m_levelSetVValidBuffer = 0;
    GLuint m_levelSetUValidScratchBuffer = 0;
    GLuint m_levelSetVValidScratchBuffer = 0;
    GLuint m_levelSetHorizontalThetaBuffer = 0;
    GLuint m_levelSetVerticalThetaBuffer = 0;
    GLuint m_levelSetHorizontalThetaStateBuffer = 0;
    GLuint m_levelSetVerticalThetaStateBuffer = 0;
    GLuint m_levelSetPreviousHorizontalThetaBuffer = 0;
    GLuint m_levelSetPreviousVerticalThetaBuffer = 0;
    GLuint m_levelSetDiagnosticBuffer = 0;
    GLuint m_levelSetAreaBuffer = 0;
    GLuint m_levelSetComponentLabelBufferA = 0;
    GLuint m_levelSetComponentLabelBufferB = 0;
    GLuint m_levelSetComponentChangedBuffer = 0;
    GLuint m_candidateLevelSetPhiBuffer = 0;
    GLuint m_candidateLevelSetConfidenceBuffer = 0;
    GLuint m_candidateLevelSetHorizontalThetaBuffer = 0;
    GLuint m_candidateLevelSetVerticalThetaBuffer = 0;
    GLuint m_candidateLevelSetHorizontalThetaStateBuffer = 0;
    GLuint m_candidateLevelSetVerticalThetaStateBuffer = 0;
    GLuint m_committedRenderLevelSetPhiBuffer = 0;
    GLuint m_committedRenderLevelSetConfidenceBuffer = 0;
    GLuint m_committedRenderLevelSetHorizontalThetaBuffer = 0;
    GLuint m_committedRenderLevelSetVerticalThetaBuffer = 0;
    GLuint m_committedRenderLevelSetHorizontalThetaStateBuffer = 0;
    GLuint m_committedRenderLevelSetVerticalThetaStateBuffer = 0;
    GLuint m_timerQuery = 0;
    std::array<GLuint, 2> m_pressureTimestampQueries{};
    struct PressureProfileEvent final {
        GpuPressureProfileCategory category = GpuPressureProfileCategory::matrixVector;
        int level = -1;
        GLuint beginQuery = 0;
        GLuint endQuery = 0;
        std::chrono::steady_clock::time_point cpuBegin{};
        double cpuSubmissionMilliseconds = 0.0;
        std::size_t dispatchCount = 0;
    };
    std::array<GLuint, 256> m_profileTimestampQueries{};
    std::vector<PressureProfileEvent> m_pressureProfileEvents;
    std::array<std::size_t, static_cast<std::size_t>(GpuPressureProfileCategory::count)>
        m_pressureProfileCategorySamples{};
    bool m_pressureProfilingActive = false;
    static constexpr std::size_t asynchronousQueryCount = 4;
    std::array<GLuint, asynchronousQueryCount> m_asynchronousStepQueries{};
    std::array<bool, asynchronousQueryCount> m_asynchronousStepQueryPending{};
    std::array<std::uint64_t, asynchronousQueryCount> m_asynchronousStepQuerySerial{};
    std::size_t m_asynchronousStepQueryCursor = 0;
    std::uint64_t m_asynchronousStepSerial = 0;
    bool m_asynchronousStepQueryActive = false;
    GpuAsyncTimingSample m_lastAsynchronousStepTiming;
    std::size_t m_width = 0;
    std::size_t m_height = 0;
    float m_dx = 0.0f;
    float m_dy = 0.0f;
    bool m_queryPending = false;
    bool m_synchronousTimingEnabled = true;
    bool m_gpuPressureControlActive = false;
    std::size_t m_pressureDispatchCount = 0;
    bool m_fusedCoarseSmootherSupported = false;
    double m_lastUploadMilliseconds = 0.0;
    double m_lastGpuMilliseconds = 0.0;
    std::vector<GLuint> m_cellTypes;
    std::vector<GLuint> m_componentIds;
    std::vector<double> m_pressureUpload;
    std::vector<MultigridLevel> m_multigridLevels;
    std::size_t m_multigridMemoryBytes = 0;
    std::size_t m_multigridTopologyHash = 0;
    std::size_t m_particleCount = 0;
    std::size_t m_particleCapacity = 0;
    std::size_t m_candidateParticleCount = 0;
    bool m_candidateStateSaved = false;
    std::size_t m_committedRenderParticleCount = 0;
    bool m_committedRenderStateEnabled = false;
    bool m_committedRenderStateValid = false;
    std::size_t m_candidateFluidCellCount = 0;
    std::size_t m_candidateConnectedComponentCount = 0;
    std::size_t m_candidateActiveMultigridLevelCount = 0;
    std::size_t m_candidatePureNeumannComponentCount = 0;
    bool m_candidateHasIsolatedFluidCell = false;
    std::uint64_t m_candidateClassificationVersion = 0;
    std::size_t m_fluidCellCount = 0;
    std::size_t m_connectedComponentCount = 0;
    std::size_t m_activeMultigridLevelCount = 0;
    std::size_t m_pureNeumannComponentCount = 0;
    bool m_hasIsolatedFluidCell = false;
    bool m_dynamicPressureTopologyReady = true;
    bool m_dynamicPressureTopologyAnalysisFailed = false;
    std::uint64_t m_classificationVersion = 1;
    std::uint64_t m_fineOperatorVersion = 1;
    std::uint64_t m_multigridVersion = 1;
    GpuLevelSetSettings m_levelSetSettings;
    GpuLevelSetReport m_levelSetReport;
    GpuLevelSetReport m_candidateLevelSetReport;
    GpuLevelSetReport m_committedRenderLevelSetReport;
    std::size_t m_levelSetAcceptedSubsteps = 0;

    [[nodiscard]] double measureExperimentalLevelSetArea(GLuint phiBuffer);
};

} // namespace fluidlab::simulation
