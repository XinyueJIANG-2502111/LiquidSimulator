#include "GpuDynamicFreeSurfaceStepper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fluidlab::simulation {
namespace {

using Clock = std::chrono::steady_clock;

void validateSettings(const GpuDynamicFreeSurfaceSettings& settings) {
    if (!std::isfinite(settings.timeStep) || settings.timeStep <= 0.0f ||
        !std::isfinite(settings.density) || settings.density <= 0.0f ||
        !std::isfinite(settings.gravity.x) || !std::isfinite(settings.gravity.y) ||
        !std::isfinite(settings.maximumCfl) || settings.maximumCfl <= 0.0f ||
        !std::isfinite(settings.cflPlanningSafetyFactor) ||
        settings.cflPlanningSafetyFactor <= 0.0f ||
        settings.cflPlanningSafetyFactor > 1.0f ||
        settings.maximumSubsteps == 0 || settings.maximumCandidateRetries == 0 ||
        !std::isfinite(settings.minimumSubstep) || settings.minimumSubstep <= 0.0f ||
        !std::isfinite(settings.candidateRetryScale) ||
        settings.candidateRetryScale <= 0.0f || settings.candidateRetryScale >= 1.0f ||
        settings.extrapolationLayers == 0 ||
        !std::isfinite(settings.pressureResidualTolerance) ||
        settings.pressureResidualTolerance < 0.0f ||
        settings.pressureMaximumIterations == 0 ||
        settings.pressureResidualCheckInterval == 0 ||
        settings.particleRedistributionInterval == 0 ||
        settings.particleRedistributionTargetMinimum == 0 ||
        settings.particleRedistributionTargetMinimum > 8 ||
        (settings.experimentalLevelSetEnabled &&
            (settings.experimentalLevelSetSettings.redistanceInterval == 0 ||
             settings.experimentalLevelSetSettings.diagnosticReadbackInterval == 0 ||
             !std::isfinite(settings.experimentalLevelSetSettings.thetaMinimum) ||
             settings.experimentalLevelSetSettings.thetaMinimum <= 0.0f ||
             settings.experimentalLevelSetSettings.thetaMinimum > 1.0f))) {
        throw std::invalid_argument("Invalid GPU dynamic free-surface settings.");
    }
}

[[nodiscard]] float maximumVelocity(
    const std::vector<float>& horizontal,
    const std::vector<float>& vertical,
    char& component,
    std::size_t& index) noexcept {
    float maximum = 0.0f;
    component = 'u';
    index = 0;
    for (std::size_t i = 0; i < horizontal.size(); ++i) {
        if (std::abs(horizontal[i]) > maximum) {
            maximum = std::abs(horizontal[i]);
            component = 'u';
            index = i;
        }
    }
    for (std::size_t i = 0; i < vertical.size(); ++i) {
        if (std::abs(vertical[i]) > maximum) {
            maximum = std::abs(vertical[i]);
            component = 'v';
            index = i;
        }
    }
    return maximum;
}

[[nodiscard]] double kineticEnergy(
    const std::vector<float>& horizontal,
    const std::vector<float>& vertical) noexcept {
    double result = 0.0;
    for (const float value : horizontal) result += 0.5 * static_cast<double>(value) * value;
    for (const float value : vertical) result += 0.5 * static_cast<double>(value) * value;
    return result;
}

[[nodiscard]] float maximumVelocityChange(
    const std::vector<float>& horizontalBefore,
    const std::vector<float>& verticalBefore,
    const std::vector<float>& horizontalAfter,
    const std::vector<float>& verticalAfter) noexcept {
    float result = 0.0f;
    for (std::size_t i = 0; i < horizontalBefore.size(); ++i) {
        result = std::max(result, std::abs(horizontalAfter[i] - horizontalBefore[i]));
    }
    for (std::size_t i = 0; i < verticalBefore.size(); ++i) {
        result = std::max(result, std::abs(verticalAfter[i] - verticalBefore[i]));
    }
    return result;
}

} // namespace

GpuDynamicFreeSurfaceStepper::GpuDynamicFreeSurfaceStepper(
    const std::filesystem::path& divergenceShaderPath,
    GpuDynamicFreeSurfaceSettings settings)
    : m_grid(divergenceShaderPath), m_settings(settings) {
    validateSettings(m_settings);
}

GpuDynamicTopologyReport GpuDynamicFreeSurfaceStepper::initialize(
    const MacGrid& grid,
    const MarkerParticleSystem& particles,
    const LevelSetField* initialLevelSet) {
    m_grid.upload(grid);
    m_grid.setSynchronousTimingEnabled(m_settings.synchronousGpuTimingEnabled);
    m_grid.uploadMarkerParticles(particles);
    m_classification = m_grid.rebuildClassificationFromParticles(false);
    if (m_settings.experimentalLevelSetEnabled) {
        if (initialLevelSet == nullptr) {
            throw std::invalid_argument(
                "Experimental GPU level-set mode requires an analytic initial field.");
        }
        m_grid.initializeExperimentalLevelSet(
            *initialLevelSet, m_settings.experimentalLevelSetSettings);
    }
    m_topology = m_grid.rebuildDynamicPressureTopology();
    m_width = grid.width();
    m_height = grid.height();
    m_dx = grid.cellWidth();
    m_dy = grid.cellHeight();
    m_simulationTime = 0.0;
    m_pendingPhysicalTime = 0.0;
    m_activeStepInput = {};
    m_activeInjectionApplied = false;
    m_forcedCandidateRejectionsRemaining =
        m_settings.forcedCandidateRejectionsForValidation;
    m_completedOuterSteps = 0;
    m_lastProjection = {};
    m_lastPeakVelocityStage = GpuVelocityStage::stepStart;
    m_initialized = true;
    m_healthy = m_topology.fineOperatorValid && m_topology.topologyValid &&
        m_topology.multigridValid && m_topology.isolatedFluidCellCount == 0 &&
        m_topology.pureNeumannComponentCount <= 1;
    return m_topology;
}

GpuDynamicFreeSurfaceReport GpuDynamicFreeSurfaceStepper::step() {
    return step({});
}

GpuDynamicFreeSurfaceReport GpuDynamicFreeSurfaceStepper::step(
    const SimulationStepInput& input) {
    const auto totalStart = Clock::now();
    GpuDynamicFreeSurfaceReport report;
    report.simulationTime = m_simulationTime;
    report.particleCount = m_grid.particleCount();
    report.classification = m_classification;
    report.topology = m_topology;
    report.projection = m_lastProjection;
    report.peakVelocityStage = m_lastPeakVelocityStage;
    report.levelSet = m_grid.experimentalLevelSetReport();
    if (!m_initialized) {
        fail(report, GpuDynamicFailureReason::notInitialized,
            "GPU dynamic simulation has not been initialized.");
        return report;
    }
    if (!m_healthy) {
        fail(report, GpuDynamicFailureReason::unsupportedPressureTopology,
            "The current GPU pressure topology is not safe to solve.");
        return report;
    }
    if (!isFinite(input)) {
        report.failureReason = GpuDynamicFailureReason::invalidStepInput;
        report.failureMessage = "The external acceleration input is invalid.";
        return report;
    }

    const double completionTolerance =
        static_cast<double>(std::numeric_limits<float>::epsilon()) * m_settings.timeStep;
    if (m_pendingPhysicalTime <= completionTolerance) {
        m_pendingPhysicalTime = m_settings.timeStep;
        m_activeStepInput = input;
        report.physicalTimeRequested = m_settings.timeStep;
    }
    const Vector2 effectiveAcceleration{
        m_settings.gravity.x + m_activeStepInput.externalAcceleration.x,
        m_settings.gravity.y + m_activeStepInput.externalAcceleration.y};
    report.appliedExternalAcceleration = m_activeStepInput.externalAcceleration;
    report.inputSourceTimeBegin = m_activeStepInput.sourceTimeBegin;
    report.inputSourceTimeEnd = m_activeStepInput.sourceTimeEnd;
    report.effectiveAcceleration = effectiveAcceleration;
    report.inputVersion = m_activeStepInput.version;
    report.injectionAreaRequested = m_activeStepInput.fluidInjection.active
        ? m_activeStepInput.fluidInjection.requestedArea : 0.0;

    report.velocity = m_grid.velocityExtrema();
    report.timings.velocityReductionGpuMilliseconds += report.velocity.gpuMilliseconds;
    report.timings.velocityReductionReadbackMilliseconds += report.velocity.readbackMilliseconds;
    bool detailedVelocityActive = false;
    const auto recordVelocityStage = [&](GpuVelocityStage stage) {
        if (!detailedVelocityActive) return;
        const auto extrema = m_grid.velocityExtrema();
        report.timings.velocityReductionGpuMilliseconds += extrema.gpuMilliseconds;
        report.timings.velocityReductionReadbackMilliseconds += extrema.readbackMilliseconds;
        const float currentMaximum = std::max(
            extrema.maximumAbsoluteU, extrema.maximumAbsoluteV);
        const float recordedMaximum = std::max(
            report.velocity.maximumAbsoluteU, report.velocity.maximumAbsoluteV);
        if (currentMaximum > recordedMaximum) {
            report.velocity = extrema;
            report.peakVelocityStage = stage;
        }
    };
    const float estimatedMaximumU = report.velocity.maximumAbsoluteU +
        std::abs(effectiveAcceleration.x) * static_cast<float>(m_pendingPhysicalTime);
    const float estimatedMaximumV = report.velocity.maximumAbsoluteV +
        std::abs(effectiveAcceleration.y) * static_cast<float>(m_pendingPhysicalTime);
    const float outerCfl = static_cast<float>(m_pendingPhysicalTime) *
        (estimatedMaximumU / m_dx + estimatedMaximumV / m_dy);
    detailedVelocityActive = m_settings.detailedVelocityDiagnostics &&
        outerCfl > 4.0f * m_settings.maximumCfl;
    const float planningCfl = m_settings.maximumCfl *
        (m_settings.adaptiveCflSchedulingEnabled
            ? m_settings.cflPlanningSafetyFactor : 1.0f);
    if (m_settings.asynchronousGpuTimingEnabled) m_grid.beginAsynchronousStepTiming();
    GpuVelocityExtrema currentVelocity = report.velocity;
    float retryTimeStepLimit = std::numeric_limits<float>::infinity();
    std::size_t retriesForCurrentState = 0;
    while (m_pendingPhysicalTime > completionTolerance &&
           report.candidateAttempts < m_settings.maximumSubsteps) {
        const float remainingTime = static_cast<float>(m_pendingPhysicalTime);
        const float predictedMaximumU = currentVelocity.maximumAbsoluteU +
            std::abs(effectiveAcceleration.x) * remainingTime;
        const float predictedMaximumV = currentVelocity.maximumAbsoluteV +
            std::abs(effectiveAcceleration.y) * remainingTime;
        const float currentRate = predictedMaximumU / m_dx +
            predictedMaximumV / m_dy;
        float substepTime = remainingTime;
        if (m_settings.adaptiveCflSchedulingEnabled && currentRate > 0.0f) {
            substepTime = std::min(substepTime, planningCfl / currentRate);
        }
        substepTime = std::min(substepTime, retryTimeStepLimit);
        if (!std::isfinite(substepTime) || substepTime < m_settings.minimumSubstep) {
            fail(report, GpuDynamicFailureReason::cflSubstepLimit,
                "The recoverable GPU CFL step fell below the configured minimum time step.");
            break;
        }
        const std::size_t substep = report.substeps;
        const std::size_t outerStep = static_cast<std::size_t>(m_completedOuterSteps + 1);
        const bool captureSnapshot = outerStep >= m_settings.projectionSnapshotFirstOuterStep &&
            outerStep <= m_settings.projectionSnapshotLastOuterStep;
        GpuProjectionDiagnosticSnapshot snapshot;
        const GpuProjectionReport committedProjection = m_lastProjection;
        const GpuClassificationReport committedClassification = m_classification;
        const GpuDynamicTopologyReport committedTopology = m_topology;
        m_grid.saveCandidateState();
        ++report.candidateAttempts;
        report.lastSubmittedTimeStep = substepTime;

        GpuFluidInjectionReport injection;
        const bool injectionAttempt = m_activeStepInput.fluidInjection.active &&
            !m_activeInjectionApplied;
        if (injectionAttempt) {
            const double cellArea = static_cast<double>(m_dx) * m_dy;
            const std::size_t requestedCells = static_cast<std::size_t>(
                std::floor(m_activeStepInput.fluidInjection.requestedArea / cellArea + 1.0e-9));
            injection = m_grid.injectFluidCells(
                m_activeStepInput.fluidInjection.position,
                m_activeStepInput.fluidInjection.radius,
                requestedCells);
            if (injection.acceptedCellCount != 0) {
                m_classification = m_grid.rebuildClassificationFromParticles(true);
                m_topology = m_grid.rebuildDynamicPressureTopology();
                if (!m_topology.topologyValid || !m_topology.multigridValid) {
                    static_cast<void>(m_grid.restoreCandidateState());
                    m_classification = committedClassification;
                    m_topology = committedTopology;
                    ++report.rollbackCount;
                    fail(report, GpuDynamicFailureReason::topologyAnalysisFailure,
                        "Injected fluid produced an unsupported pressure topology.");
                    break;
                }
            }
        }
        report.timings.velocityExtrapolationGpuMilliseconds +=
            m_grid.extrapolateVelocity(m_settings.extrapolationLayers);
        recordVelocityStage(GpuVelocityStage::preAdvectionExtrapolation);
        report.timings.velocityAdvectionGpuMilliseconds += m_grid.advectVelocity(substepTime);
        recordVelocityStage(GpuVelocityStage::advection);
        report.timings.forceBoundaryGpuMilliseconds +=
            m_grid.applyAccelerationAndBoundaries(effectiveAcceleration, substepTime);
        recordVelocityStage(GpuVelocityStage::forceAndBoundary);

        if (captureSnapshot) {
            snapshot.outerStep = outerStep;
            snapshot.substep = substep;
            snapshot.timeStep = substepTime;
            snapshot.horizontalBefore = m_grid.readHorizontalVelocity();
            snapshot.verticalBefore = m_grid.readVerticalVelocity();
            snapshot.pressureBefore = m_grid.readPressure();
            snapshot.cellTypes = m_grid.readCellTypes();
            snapshot.previousCellTypes = m_grid.readPreviousCellTypes();
            snapshot.particleCounts = m_grid.readParticleCellCounts();
            snapshot.pressureOperator = m_grid.readPressureOperatorLevel(0);
            snapshot.kineticEnergyBefore = kineticEnergy(
                snapshot.horizontalBefore, snapshot.verticalBefore);
        }

        report.projection = m_grid.project(pressureSettings(substepTime));
        m_lastProjection = report.projection;
        recordVelocityStage(GpuVelocityStage::projection);
        report.timings.divergenceGpuMilliseconds += report.projection.divergenceGpuMilliseconds;
        report.timings.pressureGpuMilliseconds += report.projection.pressure.gpuMilliseconds;
        report.timings.pressureReductionGpuMilliseconds +=
            report.projection.pressure.reductionGpuMilliseconds;
        report.timings.pressureReadbackMilliseconds +=
            report.projection.pressure.residualReadbackMilliseconds;
        report.timings.projectionGpuMilliseconds += report.projection.projectionGpuMilliseconds;
        report.pressureIterationsTotal += report.projection.pressure.iterations;
        report.pressureIterationsMaximum = std::max(
            report.pressureIterationsMaximum, report.projection.pressure.iterations);
        report.pressureSubmittedIterationsTotal +=
            report.projection.pressure.submittedIterations;
        report.pressureSkippedIterationsTotal +=
            report.projection.pressure.skippedIterations;
        report.pressureStateReadbackCount +=
            report.projection.pressure.stateReadbackCount;
        report.pressureDispatchCount += report.projection.pressure.dispatchCount;
        report.pressureConverged = report.projection.pressure.converged;
        if (report.projection.pressure.failureReason != GpuPressureFailureReason::none) {
            const bool topologyChanged = m_grid.restoreCandidateState();
            m_lastProjection = committedProjection;
            m_classification = committedClassification;
            m_topology = topologyChanged
                ? m_grid.rebuildDynamicPressureTopology() : committedTopology;
            ++report.rollbackCount;
            fail(report, GpuDynamicFailureReason::pressureFailure,
                "GPU pressure solve rejected the current topology or encountered a numerical failure.");
            break;
        }
        if (!report.projection.pressure.converged) {
            const bool topologyChanged = m_grid.restoreCandidateState();
            m_lastProjection = committedProjection;
            m_classification = committedClassification;
            m_topology = topologyChanged
                ? m_grid.rebuildDynamicPressureTopology() : committedTopology;
            ++report.rollbackCount;
            fail(report, GpuDynamicFailureReason::pressureNotConverged,
                "GPU pressure solve reached its iteration limit without convergence.");
            break;
        }

        if (captureSnapshot) {
            snapshot.horizontalAfter = m_grid.readHorizontalVelocity();
            snapshot.verticalAfter = m_grid.readVerticalVelocity();
            snapshot.pressureAfter = m_grid.readPressure();
            snapshot.residualAfter = m_grid.readResidualField();
            snapshot.projection = report.projection;
            snapshot.maximumPostProjectionSpeed = maximumVelocity(
                snapshot.horizontalAfter, snapshot.verticalAfter,
                snapshot.maximumComponent, snapshot.maximumIndex);
            snapshot.maximumVelocityChange = maximumVelocityChange(
                snapshot.horizontalBefore, snapshot.verticalBefore,
                snapshot.horizontalAfter, snapshot.verticalAfter);
            snapshot.kineticEnergyAfter = kineticEnergy(
                snapshot.horizontalAfter, snapshot.verticalAfter);
        }

        report.timings.velocityExtrapolationGpuMilliseconds +=
            m_grid.extrapolateVelocity(m_settings.extrapolationLayers);
        recordVelocityStage(GpuVelocityStage::postProjectionExtrapolation);
        if (captureSnapshot) {
            snapshot.horizontalAfterExtrapolation = m_grid.readHorizontalVelocity();
            snapshot.verticalAfterExtrapolation = m_grid.readVerticalVelocity();
            snapshot.horizontalValidityAfterExtrapolation = m_grid.readHorizontalValidity();
            snapshot.verticalValidityAfterExtrapolation = m_grid.readVerticalValidity();
        }
        currentVelocity = m_grid.velocityExtrema();
        report.timings.velocityReductionGpuMilliseconds += currentVelocity.gpuMilliseconds;
        report.timings.velocityReductionReadbackMilliseconds +=
            currentVelocity.readbackMilliseconds;
        const float observedSubstepCfl = substepTime *
            (currentVelocity.maximumAbsoluteU / m_dx +
             currentVelocity.maximumAbsoluteV / m_dy);
        report.maximumObservedCfl = std::max(
            report.maximumObservedCfl, observedSubstepCfl);
        const float currentMaximum = std::max(
            currentVelocity.maximumAbsoluteU, currentVelocity.maximumAbsoluteV);
        const float recordedMaximum = std::max(
            report.velocity.maximumAbsoluteU, report.velocity.maximumAbsoluteV);
        if (currentMaximum > recordedMaximum) report.velocity = currentVelocity;
        const bool forceCandidateRejection = m_forcedCandidateRejectionsRemaining != 0;
        if (forceCandidateRejection) --m_forcedCandidateRejectionsRemaining;
        if (m_settings.adaptiveCflSchedulingEnabled &&
            (forceCandidateRejection ||
             observedSubstepCfl > m_settings.maximumCfl * 1.001f)) {
            static_cast<void>(m_grid.restoreCandidateState());
            m_lastProjection = committedProjection;
            m_classification = committedClassification;
            m_topology = committedTopology;
            ++report.rejectedSubsteps;
            ++report.rollbackCount;
            ++retriesForCurrentState;
            report.maximumRetriesForOneSubstep = std::max(
                report.maximumRetriesForOneSubstep, retriesForCurrentState);
            retryTimeStepLimit = substepTime * m_settings.candidateRetryScale;
            if (retriesForCurrentState >= m_settings.maximumCandidateRetries ||
                retryTimeStepLimit < m_settings.minimumSubstep) {
                fail(report, GpuDynamicFailureReason::cflSubstepLimit,
                    "Projected CFL remained unsafe after the configured candidate retries.");
                break;
            }
            currentVelocity = m_grid.velocityExtrema();
            report.timings.velocityReductionGpuMilliseconds += currentVelocity.gpuMilliseconds;
            report.timings.velocityReductionReadbackMilliseconds +=
                currentVelocity.readbackMilliseconds;
            continue;
        }
        report.maximumAcceptedCfl = std::max(
            report.maximumAcceptedCfl, observedSubstepCfl);
        if (m_grid.experimentalLevelSetEnabled()) {
            report.levelSet = m_grid.advectExperimentalLevelSet(substepTime);
        }
        report.particleAdvection = m_grid.advectMarkerParticles(substepTime);
        report.timings.particleGpuMilliseconds += report.particleAdvection.gpuMilliseconds;
        report.timings.smallStatusReadbackMilliseconds +=
            report.particleAdvection.readbackMilliseconds;
        if (report.particleAdvection.particles.invalidPositionCount != 0) {
            const bool topologyChanged = m_grid.restoreCandidateState();
            m_lastProjection = committedProjection;
            m_classification = committedClassification;
            m_topology = topologyChanged
                ? m_grid.rebuildDynamicPressureTopology() : committedTopology;
            ++report.rollbackCount;
            fail(report, GpuDynamicFailureReason::invalidParticle,
                "GPU marker advection produced an invalid particle position.");
            break;
        }
        if (captureSnapshot) snapshot.particleAdvection = report.particleAdvection;

        report.classification = m_grid.rebuildClassificationFromParticles(true);
        m_classification = report.classification;
        report.timings.classificationGpuMilliseconds += report.classification.gpuMilliseconds;
        report.timings.smallStatusReadbackMilliseconds += report.classification.readbackMilliseconds;
        if (m_settings.particleRedistributionEnabled &&
            (m_completedOuterSteps + 1) % m_settings.particleRedistributionInterval == 0) {
            report.redistribution = m_grid.redistributeMarkerParticles(
                m_settings.particleRedistributionTargetMinimum);
            report.particlesAdded += report.redistribution.addedParticleCount;
            report.timings.particleRedistributionGpuMilliseconds +=
                report.redistribution.gpuMilliseconds;
            report.timings.smallStatusReadbackMilliseconds +=
                report.redistribution.readbackMilliseconds;
        }
        if (m_grid.experimentalLevelSetEnabled()) {
            report.levelSet = m_grid.updateExperimentalLevelSetDiagnostics();
        }
        if (captureSnapshot) {
            snapshot.classification = report.classification;
            snapshot.cellTypesAfterClassification = m_grid.readCellTypes();
            snapshot.particleCountsAfterClassification = m_grid.readParticleCellCounts();
            if (!report.projectionSnapshot ||
                snapshot.maximumPostProjectionSpeed >
                    report.projectionSnapshot->maximumPostProjectionSpeed) {
                report.projectionSnapshot = std::move(snapshot);
            }
        }
        report.topology = report.classification.changedCellCount == 0
            ? m_grid.reuseDynamicPressureTopology()
            : m_grid.rebuildDynamicPressureTopology();
        m_topology = report.topology;
        report.pressureTopologyReused = report.pressureTopologyReused ||
            report.topology.reusedExistingHierarchy;
        report.topologyReuseCount += report.topology.reusedExistingHierarchy ? 1 : 0;
        report.topologyRebuildCount += report.topology.reusedExistingHierarchy ? 0 : 1;
        report.topologyLabelPropagationIterations += report.topology.labelPropagationIterations;
        report.timings.topologyFineGpuMilliseconds += report.topology.fineOperatorGpuMilliseconds;
        report.timings.topologyAnalysisGpuMilliseconds += report.topology.topologyGpuMilliseconds;
        report.timings.topologyCoarseGpuMilliseconds += report.topology.coarseOperatorGpuMilliseconds;
        report.timings.topologyEndToEndMilliseconds += report.topology.totalCpuMilliseconds;
        report.timings.smallStatusReadbackMilliseconds += report.topology.statusReadbackMilliseconds;
        if (!report.topology.topologyValid || !report.topology.multigridValid) {
            static_cast<void>(m_grid.restoreCandidateState());
            m_lastProjection = committedProjection;
            m_classification = committedClassification;
            m_topology = m_grid.rebuildDynamicPressureTopology();
            ++report.rollbackCount;
            fail(report, GpuDynamicFailureReason::topologyAnalysisFailure,
                "GPU pressure topology analysis or multigrid construction failed.");
            break;
        }
        if (report.topology.isolatedFluidCellCount != 0 ||
            report.topology.pureNeumannComponentCount > 1) {
            static_cast<void>(m_grid.restoreCandidateState());
            m_lastProjection = committedProjection;
            m_classification = committedClassification;
            m_topology = m_grid.rebuildDynamicPressureTopology();
            ++report.rollbackCount;
            fail(report, GpuDynamicFailureReason::unsupportedPressureTopology,
                "GPU pressure topology contains an isolated cell or multiple pure-Neumann components.");
            break;
        }

        m_grid.commitCandidateState();
        if (injectionAttempt) {
            m_activeInjectionApplied = true;
            report.injectionCellsRequested += injection.requestedCellCount;
            report.injectionCellsAccepted += injection.acceptedCellCount;
            report.injectionParticlesAdded += injection.particlesAdded;
            report.injectionAreaAccepted += static_cast<double>(
                injection.acceptedCellCount) * m_dx * m_dy;
        }
        if (m_settings.acceptedSubstepValidationObserver) {
            MacGrid validationSnapshot(m_width, m_height, m_dx, m_dy);
            m_grid.download(validationSnapshot);
            m_settings.acceptedSubstepValidationObserver(validationSnapshot, substepTime);
        }
        m_simulationTime += substepTime;
        report.physicalTimeAdvanced += substepTime;
        report.appliedExternalVelocityDelta.x +=
            m_activeStepInput.externalAcceleration.x * substepTime;
        report.appliedExternalVelocityDelta.y +=
            m_activeStepInput.externalAcceleration.y * substepTime;
        m_pendingPhysicalTime = std::max(0.0, m_pendingPhysicalTime - substepTime);
        ++report.substeps;
        report.advanced = true;
        retriesForCurrentState = 0;
        retryTimeStepLimit = std::numeric_limits<float>::infinity();
    }

    if (m_healthy && m_pendingPhysicalTime <= completionTolerance) {
        m_pendingPhysicalTime = 0.0;
        report.outerStepCompleted = true;
        m_activeInjectionApplied = false;
        ++m_completedOuterSteps;
    } else if (m_healthy && report.candidateAttempts >= m_settings.maximumSubsteps) {
        report.computeBudgetExhausted = true;
    }

    if (m_settings.asynchronousGpuTimingEnabled) {
        m_grid.endAsynchronousStepTiming();
        const auto asynchronousTiming = m_grid.asynchronousStepTiming();
        report.timings.asynchronousStepGpuMilliseconds = asynchronousTiming.milliseconds;
        report.timings.asynchronousStepTimingLatency = asynchronousTiming.latencySteps;
        report.timings.asynchronousStepTimingAvailable = asynchronousTiming.available;
    }

    report.fluidCellCount = m_classification.fluidCellCount;
    report.particleCount = m_grid.particleCount();
    report.markedArea = static_cast<float>(report.fluidCellCount) * m_dx * m_dy;
    report.topologyValid = m_healthy && m_topology.topologyValid && m_topology.multigridValid;
    report.simulationTime = m_simulationTime;
    report.remainingPhysicalTime = m_pendingPhysicalTime;
    report.timings.totalCpuMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - totalStart).count();
    if (report.advanced) {
        m_lastPeakVelocityStage = report.peakVelocityStage;
    }
    return report;
}

GpuMacGrid& GpuDynamicFreeSurfaceStepper::grid() noexcept { return m_grid; }
const GpuMacGrid& GpuDynamicFreeSurfaceStepper::grid() const noexcept { return m_grid; }
const GpuDynamicFreeSurfaceSettings& GpuDynamicFreeSurfaceStepper::settings() const noexcept {
    return m_settings;
}
double GpuDynamicFreeSurfaceStepper::simulationTime() const noexcept { return m_simulationTime; }
double GpuDynamicFreeSurfaceStepper::pendingPhysicalTime() const noexcept {
    return m_pendingPhysicalTime;
}
std::uint64_t GpuDynamicFreeSurfaceStepper::completedOuterSteps() const noexcept {
    return m_completedOuterSteps;
}
bool GpuDynamicFreeSurfaceStepper::healthy() const noexcept { return m_healthy; }

void GpuDynamicFreeSurfaceStepper::fail(
    GpuDynamicFreeSurfaceReport& report,
    GpuDynamicFailureReason reason,
    std::string message) noexcept {
    report.failureReason = reason;
    report.failureMessage = std::move(message);
    m_healthy = false;
    report.topologyValid = false;
    report.timings.totalCpuMilliseconds = 0.0;
}

GpuPressureSettings GpuDynamicFreeSurfaceStepper::pressureSettings(float timeStep) const noexcept {
    GpuPressureSettings settings;
    settings.density = m_settings.density;
    settings.timeStep = timeStep;
    settings.residualTolerance = m_settings.pressureResidualTolerance;
    settings.maximumIterations = m_settings.pressureMaximumIterations;
    settings.residualCheckInterval = m_settings.pressureResidualCheckInterval;
    settings.controlMode = m_settings.pressureControlMode;
    settings.precisionMode = m_settings.pressurePrecisionMode;
    settings.fusedCoarseSmootherEnabled = m_settings.pressureFusedCoarseSmootherEnabled;
    settings.gpuResidentBatchCount = m_settings.pressureGpuResidentBatchCount;
    settings.solver = GpuPressureSolver::multigridPreconditionedConjugateGradient;
    settings.warmStartEnabled = m_settings.pressureWarmStartEnabled;
    settings.collectVelocityChange = false;
    return settings;
}

const char* gpuDynamicFailureName(GpuDynamicFailureReason reason) noexcept {
    switch (reason) {
    case GpuDynamicFailureReason::none: return "none";
    case GpuDynamicFailureReason::notInitialized: return "not initialized";
    case GpuDynamicFailureReason::invalidStepInput: return "invalid step input";
    case GpuDynamicFailureReason::cflSubstepLimit: return "CFL substep limit";
    case GpuDynamicFailureReason::invalidParticle: return "invalid marker particle";
    case GpuDynamicFailureReason::pressureFailure: return "pressure solver failure";
    case GpuDynamicFailureReason::pressureNotConverged: return "pressure not converged";
    case GpuDynamicFailureReason::topologyAnalysisFailure: return "topology analysis failure";
    case GpuDynamicFailureReason::unsupportedPressureTopology: return "unsupported pressure topology";
    }
    return "unknown";
}

const char* gpuVelocityStageName(GpuVelocityStage stage) noexcept {
    switch (stage) {
    case GpuVelocityStage::stepStart: return "step start";
    case GpuVelocityStage::preAdvectionExtrapolation: return "pre-advection extrapolation";
    case GpuVelocityStage::advection: return "advection";
    case GpuVelocityStage::forceAndBoundary: return "force and boundary";
    case GpuVelocityStage::projection: return "projection";
    case GpuVelocityStage::postProjectionExtrapolation: return "post-projection extrapolation";
    }
    return "unknown";
}

} // namespace fluidlab::simulation
