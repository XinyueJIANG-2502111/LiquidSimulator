#include "GpuLevelSetValidation.h"

#include "../simulation/gpu/GpuDynamicFreeSurfaceStepper.h"
#include "../simulation/gpu/GpuAsyncDynamicFreeSurfaceStepper.h"
#include "../simulation/gpu/GpuMacGrid.h"
#include "../simulation/mac/LevelSetInterface.h"
#include "../simulation/mac/GhostFluidPressureProjection.h"
#include "../simulation/mac/MarkerParticles.h"
#include "../simulation/mac/Scenarios.h"
#include "../simulation/mac/VelocityAdvection.h"
#include "../simulation/mac/VolumeFractionField.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <queue>
#include <vector>
#include <utility>
#include <thread>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace fluidlab::diagnostics {
namespace {

using Clock = std::chrono::steady_clock;
using simulation::LevelSetField;
using simulation::MacGrid;

struct Difference final {
    double rms = 0.0;
    double maximum = 0.0;
};

struct CrossingDifference final {
    double rms = 0.0;
    double maximum = 0.0;
    std::size_t matched = 0;
    std::size_t changedTopology = 0;
};

[[nodiscard]] CrossingDifference compareEdgeCrossings(
    const LevelSetField& before, const LevelSetField& after) {
    CrossingDifference result;
    double squared = 0.0;
    const auto add = [&](double a0, double a1, double b0, double b1, double spacing) {
        const bool crossingA = (a0 <= 0.0 && a1 > 0.0) || (a0 > 0.0 && a1 <= 0.0);
        const bool crossingB = (b0 <= 0.0 && b1 > 0.0) || (b0 > 0.0 && b1 <= 0.0);
        if (crossingA != crossingB) {
            ++result.changedTopology;
            return;
        }
        if (!crossingA) return;
        const double thetaA = std::abs(a0) / std::max(std::abs(a0) + std::abs(a1), 1.0e-30);
        const double thetaB = std::abs(b0) / std::max(std::abs(b0) + std::abs(b1), 1.0e-30);
        const double displacement = std::abs(thetaA - thetaB) * spacing;
        squared += displacement * displacement;
        result.maximum = std::max(result.maximum, displacement);
        ++result.matched;
    };
    for (std::size_t y = 0; y < before.height; ++y) {
        for (std::size_t x = 0; x + 1 < before.width; ++x) {
            add(before.value(x, y), before.value(x + 1, y),
                after.value(x, y), after.value(x + 1, y), before.cellWidth);
        }
    }
    for (std::size_t y = 0; y + 1 < before.height; ++y) {
        for (std::size_t x = 0; x < before.width; ++x) {
            add(before.value(x, y), before.value(x, y + 1),
                after.value(x, y), after.value(x, y + 1), before.cellHeight);
        }
    }
    result.rms = result.matched == 0 ? 0.0 :
        std::sqrt(squared / static_cast<double>(result.matched));
    return result;
}

[[nodiscard]] double meanGradientError(const LevelSetField& field) {
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t y = 1; y + 1 < field.height; ++y) {
        for (std::size_t x = 1; x + 1 < field.width; ++x) {
            const std::size_t index = y * field.width + x;
            if (field.confidence[index] == 0u) continue;
            const double gx = (field.value(x + 1, y) - field.value(x - 1, y)) /
                (2.0 * field.cellWidth);
            const double gy = (field.value(x, y + 1) - field.value(x, y - 1)) /
                (2.0 * field.cellHeight);
            sum += std::abs(std::hypot(gx, gy) - 1.0);
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

[[nodiscard]] LevelSetField fieldFromSnapshot(
    const simulation::GpuLevelSetSnapshot& snapshot, const LevelSetField& layout) {
    LevelSetField result = layout;
    for (std::size_t i = 0; i < result.values.size(); ++i) {
        result.values[i] = snapshot.phi[i];
        result.confidence[i] = snapshot.confidence[i] == 0u ? 0u : 1u;
    }
    return result;
}

[[nodiscard]] Difference compare(
    const std::vector<float>& gpu,
    const std::vector<double>& cpu) {
    Difference result;
    double squared = 0.0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const double error = std::abs(static_cast<double>(gpu[i]) - cpu[i]);
        result.maximum = std::max(result.maximum, error);
        squared += error * error;
    }
    result.rms = gpu.empty() ? 0.0 : std::sqrt(squared / static_cast<double>(gpu.size()));
    return result;
}

[[nodiscard]] Difference compareConfident(
    const std::vector<float>& gpu,
    const std::vector<GLuint>& gpuConfidence,
    const LevelSetField& cpu) {
    Difference result;
    double squared = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        if (gpuConfidence[i] == 0u || cpu.confidence[i] == 0u) continue;
        const double error = std::abs(static_cast<double>(gpu[i]) - cpu.values[i]);
        result.maximum = std::max(result.maximum, error);
        squared += error * error;
        ++count;
    }
    result.rms = count == 0 ? 0.0 : std::sqrt(squared / static_cast<double>(count));
    return result;
}

[[nodiscard]] LevelSetField circleField(std::size_t width, std::size_t height) {
    const double dx = 1.0 / static_cast<double>(width);
    const double dy = 1.0 / static_cast<double>(height);
    return simulation::sampleAnalyticLevelSet(width, height, dx, dy,
        [](double x, double y) {
            return std::hypot(x - 0.37, y - 0.53) - 0.19;
        });
}

void initializeFromLevelSet(MacGrid& grid, const LevelSetField& field, float u, float v) {
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            grid.setCellType(x, y, field.value(x, y) < 0.0
                ? simulation::CellType::fluid : simulation::CellType::air);
        }
    }
    for (std::size_t y = 0; y < grid.height(); ++y)
        for (std::size_t x = 0; x <= grid.width(); ++x) grid.horizontalVelocity(x, y) = u;
    for (std::size_t y = 0; y <= grid.height(); ++y)
        for (std::size_t x = 0; x < grid.width(); ++x) grid.verticalVelocity(x, y) = v;
}

[[nodiscard]] simulation::GpuLevelSetSettings levelSetSettings(std::size_t redistanceInterval) {
    simulation::GpuLevelSetSettings settings;
    settings.thetaMinimum = 0.1f;
    settings.redistanceInterval = redistanceInterval;
    settings.diagnosticReadbackInterval = 1;
    return settings;
}

[[nodiscard]] LevelSetField damField(const MacGrid& grid) {
    const std::size_t liquidEndX = 1 + (grid.width() - 2) / 3;
    const std::size_t liquidEndY = 1 + 2 * (grid.height() - 2) / 3;
    const double minimumX = grid.cellWidth();
    const double minimumY = grid.cellHeight();
    const double maximumX = liquidEndX * grid.cellWidth();
    const double maximumY = liquidEndY * grid.cellHeight();
    const double centerX = 0.5 * (minimumX + maximumX);
    const double centerY = 0.5 * (minimumY + maximumY);
    const double halfX = 0.5 * (maximumX - minimumX);
    const double halfY = 0.5 * (maximumY - minimumY);
    return simulation::sampleAnalyticLevelSet(grid.width(), grid.height(),
        grid.cellWidth(), grid.cellHeight(), [=](double x, double y) {
            const double qx = std::abs(x - centerX) - halfX;
            const double qy = std::abs(y - centerY) - halfY;
            return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
                std::min(std::max(qx, qy), 0.0);
        });
}

} // namespace

int runGpuLevelSetValidation(std::ostream& output, const std::filesystem::path& shaderDirectory) {
    constexpr std::size_t width = 80;
    constexpr std::size_t height = 60;
    constexpr float timeStep = 1.0f / 240.0f;
    const LevelSetField initial = circleField(width, height);
    MacGrid grid(width, height, 1.0f / width, 1.0f / height);
    initializeFromLevelSet(grid, initial, 0.17f, -0.08f);
    simulation::MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    gpu.upload(grid);
    gpu.uploadMarkerParticles(particles);
    gpu.initializeExperimentalLevelSet(initial, levelSetSettings(64));
    gpu.saveCandidateState();
    static_cast<void>(gpu.advectExperimentalLevelSet(timeStep));
    static_cast<void>(gpu.updateExperimentalLevelSetDiagnostics());
    gpu.commitCandidateState();
    const auto snapshot = gpu.readExperimentalLevelSet();
    const LevelSetField expected = simulation::advectLevelSetRk2(initial,
        [](double, double) { return simulation::Vector2{0.17f, -0.08f}; }, timeStep);
    const Difference difference = compare(snapshot.phi, expected.values);
    LevelSetField gpuField = expected;
    for (std::size_t i = 0; i < gpuField.values.size(); ++i) {
        gpuField.values[i] = snapshot.phi[i];
        gpuField.confidence[i] = snapshot.confidence[i] == 0u ? 0u : 1u;
    }
    const auto cpuTheta = simulation::buildGhostFluidTheta(grid, gpuField, 0.1, true);
    double thetaMaximumError = 0.0;
    for (std::size_t i = 0; i < snapshot.horizontalTheta.size(); ++i) {
        thetaMaximumError = std::max(thetaMaximumError,
            std::abs(static_cast<double>(snapshot.horizontalTheta[i]) -
                cpuTheta.horizontalTheta[i]));
    }
    for (std::size_t i = 0; i < snapshot.verticalTheta.size(); ++i) {
        thetaMaximumError = std::max(thetaMaximumError,
            std::abs(static_cast<double>(snapshot.verticalTheta[i]) -
                cpuTheta.verticalTheta[i]));
    }
    bool pass = difference.rms < 2.0e-5 && difference.maximum < 2.0e-4 &&
        thetaMaximumError < 2.0e-6 &&
        snapshot.report.geometryVersion == initial.sourceVersion + 1 &&
        snapshot.report.advectionCount == 1;
    output << "GPU Level Set analytic transport validation\n"
           << "  phi RMS=" << std::scientific << difference.rms
           << " max=" << difference.maximum
           << " theta max=" << thetaMaximumError
           << " sign conflicts=" << snapshot.report.signConflictCount
           << " theta valid/clamped/fallback=" << snapshot.report.validThetaCount << '/'
           << snapshot.report.clampedThetaCount << '/' << snapshot.report.fallbackThetaCount
           << " result=" << (pass ? "PASS" : "FAIL") << '\n';

    output << "GPU redistance analytic narrow-band checks\n";
    output << "Case          Grid       RMS/h      Max/h   Invalid  Result\n";
    struct GeometryCase final {
        const char* name;
        simulation::ScalarFieldFunction field;
    };
    const std::vector<GeometryCase> cases{
        {"horizontal", [](double, double y) { return y - 0.47; }},
        {"tilted", [](double x, double y) {
            return (y - 0.25 * x - 0.37) / std::sqrt(1.0625);
        }},
        {"two-drops", [](double x, double y) {
            return std::min(std::hypot(x - 0.32, y - 0.5) - 0.14,
                std::hypot(x - 0.69, y - 0.5) - 0.12);
        }},
    };
    simulation::GpuMacGrid redistanceGpu(shaderDirectory / "mac_divergence.comp");
    for (const auto& geometry : cases) {
        for (const auto dimensions : {std::pair{40u, 30u}, std::pair{80u, 60u},
                 std::pair{160u, 120u}}) {
            const std::size_t w = dimensions.first;
            const std::size_t h = dimensions.second;
            const LevelSetField analytic = simulation::sampleAnalyticLevelSet(
                w, h, 1.0 / w, 1.0 / h, geometry.field);
            MacGrid redistanceGrid(w, h, 1.0f / w, 1.0f / h);
            initializeFromLevelSet(redistanceGrid, analytic, 0.0f, 0.0f);
            simulation::MarkerParticleSystem redistanceParticles;
            redistanceParticles.seedFluidCellsExact(redistanceGrid, 4);
            redistanceGpu.upload(redistanceGrid);
            redistanceGpu.uploadMarkerParticles(redistanceParticles);
            redistanceGpu.initializeExperimentalLevelSet(analytic, levelSetSettings(1));
            redistanceGpu.saveCandidateState();
            static_cast<void>(redistanceGpu.advectExperimentalLevelSet(1.0e-6f));
            static_cast<void>(redistanceGpu.updateExperimentalLevelSetDiagnostics());
            redistanceGpu.commitCandidateState();
            const auto redistanced = redistanceGpu.readExperimentalLevelSet();
            const Difference geometryError = compareConfident(
                redistanced.phi, redistanced.confidence, analytic);
            const double localH = std::max(1.0 / w, 1.0 / h);
            const bool casePass = geometryError.rms / localH < 0.35 &&
                geometryError.maximum / localH < 1.0 &&
                redistanced.report.invalidPhiCount == 0;
            pass = pass && casePass;
            output << std::left << std::setw(13) << geometry.name << std::right
                   << std::setw(4) << w << 'x' << std::setw(3) << h
                   << std::scientific << std::setprecision(3)
                   << std::setw(12) << geometryError.rms / localH
                   << std::setw(11) << geometryError.maximum / localH
                   << std::setw(10) << redistanced.report.invalidSampleCount
                   << std::setw(8) << (casePass ? "PASS" : "FAIL") << '\n';
        }
    }
    return pass ? 0 : 1;
}

int runGpuLevelSetRollbackValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory, GLFWwindow* window) {
    const LevelSetField initial = circleField(64, 64);
    MacGrid grid(64, 64, 1.0f / 64.0f, 1.0f / 64.0f);
    initializeFromLevelSet(grid, initial, 0.4f, 0.1f);
    simulation::MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    gpu.upload(grid);
    gpu.uploadMarkerParticles(particles);
    gpu.initializeExperimentalLevelSet(initial, levelSetSettings(1));
    const auto before = gpu.readExperimentalLevelSet();
    gpu.saveCandidateState();
    static_cast<void>(gpu.advectExperimentalLevelSet(1.0f / 60.0f));
    static_cast<void>(gpu.updateExperimentalLevelSetDiagnostics());
    static_cast<void>(gpu.restoreCandidateState());
    const auto after = gpu.readExperimentalLevelSet();
    const bool exact = before.phi == after.phi && before.confidence == after.confidence &&
        before.horizontalTheta == after.horizontalTheta &&
        before.verticalTheta == after.verticalTheta;
    const bool versions = before.report.geometryVersion == after.report.geometryVersion &&
        before.report.thetaVersion == after.report.thetaVersion &&
        before.report.physicalTime == after.report.physicalTime;
    bool asyncPass = window != nullptr;
    if (window != nullptr) {
        MacGrid asyncGrid(32, 32, 1.0f / 32.0f, 1.0f / 32.0f);
        simulation::initializeDamBreak(asyncGrid);
        simulation::MarkerParticleSystem asyncParticles;
        asyncParticles.seedFluidCellsExact(asyncGrid, 8);
        const LevelSetField asyncInitial = damField(asyncGrid);
        simulation::GpuDynamicFreeSurfaceSettings asyncSettings;
        asyncSettings.experimentalLevelSetEnabled = true;
        asyncSettings.experimentalLevelSetSettings.diagnosticReadbackInterval = 1;
        asyncSettings.forcedCandidateRejectionsForValidation = 1;
        simulation::GpuAsyncDynamicFreeSurfaceStepper async(
            window, shaderDirectory / "mac_divergence.comp", asyncSettings);
        static_cast<void>(async.initialize(asyncGrid, asyncParticles, &asyncInitial));
        const auto committedBefore = async.stepper().grid().readExperimentalLevelSet();
        asyncPass = async.submit();
        const auto deadline = Clock::now() + std::chrono::seconds(10);
        while (asyncPass && !async.hasCompletedResult() && Clock::now() < deadline) {
            static_cast<void>(async.poll());
            glfwPollEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        asyncPass = asyncPass && async.hasCompletedResult() && async.report().rollbackCount >= 1;
        const auto committedAfter = async.stepper().grid().readExperimentalLevelSet();
        asyncPass = asyncPass && committedAfter.report.physicalTime >
            committedBefore.report.physicalTime &&
            std::abs(committedAfter.report.physicalTime -
                async.stepper().simulationTime()) < 2.0e-6;
        async.acknowledge();
        async.requestCancel();
        async.waitForShutdown();
    }
    output << "GPU Level Set candidate rollback validation: buffers="
           << (exact ? "exact" : "changed") << " versions="
           << (versions ? "exact" : "changed") << " async="
           << (asyncPass ? "committed" : "failed") << " result="
           << (exact && versions && asyncPass ? "PASS" : "FAIL") << '\n';
    return exact && versions && asyncPass ? 0 : 1;
}

int runGpuLevelSetDynamicValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory) {
    MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
    simulation::initializeDamBreak(grid);
    simulation::MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    simulation::GpuDynamicFreeSurfaceSettings settings;
    settings.experimentalLevelSetEnabled = true;
    settings.experimentalLevelSetSettings.redistanceInterval = 4;
    settings.experimentalLevelSetSettings.diagnosticReadbackInterval = 1;
    settings.forcedCandidateRejectionsForValidation = 1;
    settings.maximumSubsteps = 32;
    const LevelSetField initial = damField(grid);
    LevelSetField substepCpu = initial;
    LevelSetField outerEndCpu = initial;
    MacGrid lastAcceptedVelocity = grid;
    std::size_t acceptedByObserver = 0;
    simulation::LevelSetRedistanceSettings redistance;
    redistance.narrowBandWidth = 6.0 * std::max(grid.cellWidth(), grid.cellHeight());
    settings.acceptedSubstepValidationObserver = [&](const MacGrid& snapshot, float dt) {
        substepCpu = simulation::advectLevelSetRk2(substepCpu,
            [&](double x, double y) {
                return simulation::MacVelocitySampler::sampleVelocity(snapshot,
                    {static_cast<float>(x), static_cast<float>(y)});
            }, dt);
        ++acceptedByObserver;
        if (acceptedByObserver % settings.experimentalLevelSetSettings.redistanceInterval == 0) {
            static_cast<void>(simulation::redistanceLevelSetNarrowBand(substepCpu, redistance));
        }
        lastAcceptedVelocity = snapshot;
    };
    simulation::GpuDynamicFreeSurfaceStepper stepper(
        shaderDirectory / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles, &initial);
    bool pass = topology.topologyValid;
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    std::size_t outerEndRedistanceGroup = 0;
    for (std::size_t outer = 0; outer < 60 && stepper.healthy(); ++outer) {
        const auto report = stepper.step();
        accepted += report.substeps;
        rejected += report.rejectedSubsteps;
        pass = pass && report.levelSet.initialized &&
            std::abs(report.levelSet.physicalTime - report.simulationTime) < 2.0e-6;
        if (report.physicalTimeAdvanced > 0.0) {
            outerEndCpu = simulation::advectLevelSetRk2(outerEndCpu,
                [&](double x, double y) {
                    return simulation::MacVelocitySampler::sampleVelocity(lastAcceptedVelocity,
                        {static_cast<float>(x), static_cast<float>(y)});
                }, report.physicalTimeAdvanced);
            const std::size_t group = accepted /
                settings.experimentalLevelSetSettings.redistanceInterval;
            if (group != outerEndRedistanceGroup) {
                static_cast<void>(simulation::redistanceLevelSetNarrowBand(
                    outerEndCpu, redistance));
                outerEndRedistanceGroup = group;
            }
        }
    }
    const auto levelSet = stepper.grid().readExperimentalLevelSet();
    pass = pass && accepted == levelSet.report.advectionCount && rejected >= 1 &&
        std::abs(levelSet.report.physicalTime - stepper.simulationTime()) < 2.0e-6;
    const Difference substepDifference = compareConfident(
        levelSet.phi, levelSet.confidence, substepCpu);
    const Difference outerEndDifference = compareConfident(
        levelSet.phi, levelSet.confidence, outerEndCpu);
    pass = pass && substepDifference.rms < 2.0 * std::max(grid.cellWidth(), grid.cellHeight());
    output << "GPU Level Set dynamic substep validation\n"
           << "  simulation time=" << std::fixed << std::setprecision(6)
           << stepper.simulationTime() << " interface time=" << levelSet.report.physicalTime
           << " accepted=" << accepted << " rejected=" << rejected
           << " advections=" << levelSet.report.advectionCount
           << " conflicts=" << levelSet.report.signConflictCount
           << " positive particles=" << levelSet.report.positivePhiParticleCount
           << " level area=" << levelSet.report.geometricArea
           << " components=" << levelSet.report.negativeComponentCount
           << "\n  A/C substep phi RMS=" << std::scientific << substepDifference.rms
           << " max=" << substepDifference.maximum
           << " B/C outer-end phi RMS=" << outerEndDifference.rms
           << " max=" << outerEndDifference.maximum
           << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runVofMacReplayValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory) {
    MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
    simulation::initializeDamBreak(grid);
    simulation::MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    const LevelSetField initial = damField(grid);
    simulation::VolumeFractionField vof(
        grid.width(), grid.height(), grid.cellWidth(), grid.cellHeight());
    vof.initializeFromSignedDistance(
        [&](double x, double y) { return initial.sample(x, y); }, 32);
    vof.applyWholeCellSolidMask(grid);
    simulation::VolumeFractionField correctedVof = vof;
    simulation::VolumeFractionField refinedCorrectedVof = vof;
    const double initialVofArea = vof.totalArea();

    simulation::GpuDynamicFreeSurfaceSettings settings;
    settings.experimentalLevelSetEnabled = true;
    settings.experimentalLevelSetSettings.advectionMethod =
        simulation::GpuLevelSetAdvectionMethod::limitedMacCormack;
    settings.experimentalLevelSetSettings.redistanceMethod =
        simulation::GpuLevelSetRedistanceMethod::interfacePreserving;
    settings.experimentalLevelSetSettings.redistanceInterval = 4;
    settings.experimentalLevelSetSettings.diagnosticReadbackInterval = 1;
    settings.maximumSubsteps = 32;
    std::size_t observedSubsteps = 0;
    std::size_t rejectedVofSteps = 0;
    double maximumLedgerResidual = 0.0;
    std::size_t limitedFaces = 0;
    std::size_t evaluatedFaces = 0;
    std::size_t nonZeroFluxFaces = 0;
    std::size_t uniqueLimitedFaces = 0;
    std::size_t repeatedLimitEvents = 0;
    std::size_t limiterIterations = 0;
    std::size_t positiveLimitedFaces = 0;
    std::size_t negativeLimitedFaces = 0;
    std::size_t xLimitedFaces = 0;
    std::size_t yLimitedFaces = 0;
    std::size_t donorInsufficient = 0;
    std::size_t receiverCapacity = 0;
    std::size_t coupledInOut = 0;
    std::size_t highCourant = 0;
    std::size_t divergentDonor = 0;
    std::size_t nearBlocked = 0;
    std::size_t plicFailures = 0;
    double rawAbsoluteFlux = 0.0;
    double finalAbsoluteFlux = 0.0;
    std::vector<double> limiterScales;
    std::vector<simulation::LimitedFluxEvent> strongestLimitedFaces;
    double minimumFraction = 1.0;
    double maximumFraction = 0.0;
    std::size_t correctedRejected = 0;
    std::size_t correctedUniqueLimited = 0;
    std::size_t correctedRepeated = 0;
    std::size_t correctedIterations = 0;
    std::size_t correctedNearBlocked = 0;
    std::size_t correctedPlicFailures = 0;
    double correctedRawFlux = 0.0;
    double correctedFinalFlux = 0.0;
    double correctedCompression = 0.0;
    double correctedMaximumResidual = 0.0;
    std::size_t refinedRejected = 0;
    std::size_t refinedSubsteps = 0;
    std::size_t refinedUniqueLimited = 0;
    std::size_t refinedNearBlocked = 0;
    double refinedRawFlux = 0.0;
    double refinedFinalFlux = 0.0;
    double refinedCompression = 0.0;
    double refinedMaximumResidual = 0.0;
    settings.acceptedSubstepValidationObserver = [&](const MacGrid& snapshot, float dt) {
        const auto ledger = vof.advectConservative(
            snapshot, dt, (observedSubsteps & 1u) == 0);
        const auto correctedLedger = correctedVof.advectConservativeSplitExperimental(
            snapshot, dt, (observedSubsteps & 1u) == 0);
        rejectedVofSteps += ledger.accepted ? 0u : 1u;
        maximumLedgerResidual = std::max(
            maximumLedgerResidual, std::abs(ledger.conservationResidual));
        limitedFaces += ledger.limitedFaceCount;
        evaluatedFaces += ledger.evaluatedFaceCount;
        nonZeroFluxFaces += ledger.nonZeroLiquidFluxFaceCount;
        uniqueLimitedFaces += ledger.uniqueLimitedFaceCount;
        repeatedLimitEvents += ledger.repeatedLimitEventCount;
        limiterIterations += ledger.limiterIterationCount;
        positiveLimitedFaces += ledger.positiveLimitedFaceCount;
        negativeLimitedFaces += ledger.negativeLimitedFaceCount;
        xLimitedFaces += ledger.xLimitedFaceCount;
        yLimitedFaces += ledger.yLimitedFaceCount;
        donorInsufficient += ledger.donorInsufficientCount;
        receiverCapacity += ledger.receiverCapacityCount;
        coupledInOut += ledger.coupledInOutCount;
        highCourant += ledger.highCourantLimitedCount;
        divergentDonor += ledger.divergentCellLimitedCount;
        nearBlocked += ledger.nearBlockedFaceCount;
        plicFailures += ledger.plicFailureCount;
        rawAbsoluteFlux += ledger.rawAbsoluteFlux;
        finalAbsoluteFlux += ledger.finalAbsoluteFlux;
        limiterScales.insert(
            limiterScales.end(),
            ledger.limiterScaleSamples.begin(),
            ledger.limiterScaleSamples.end());
        strongestLimitedFaces.insert(
            strongestLimitedFaces.end(),
            ledger.strongestLimitedFaces.begin(),
            ledger.strongestLimitedFaces.end());
        minimumFraction = std::min(minimumFraction, ledger.minimumFraction);
        maximumFraction = std::max(maximumFraction, ledger.maximumFraction);
        correctedRejected += correctedLedger.accepted ? 0u : 1u;
        correctedUniqueLimited += correctedLedger.uniqueLimitedFaceCount;
        correctedRepeated += correctedLedger.repeatedLimitEventCount;
        correctedIterations += correctedLedger.limiterIterationCount;
        correctedNearBlocked += correctedLedger.nearBlockedFaceCount;
        correctedPlicFailures += correctedLedger.plicFailureCount;
        correctedRawFlux += correctedLedger.rawAbsoluteFlux;
        correctedFinalFlux += correctedLedger.finalAbsoluteFlux;
        correctedCompression += correctedLedger.compressionContribution;
        correctedMaximumResidual = std::max(
            correctedMaximumResidual, std::abs(correctedLedger.conservationResidual));

        bool refinedAccepted = false;
        for (std::size_t divisions : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u}) {
            simulation::VolumeFractionField attempt = refinedCorrectedVof;
            std::size_t attemptLimited = 0;
            std::size_t attemptNearBlocked = 0;
            double attemptRaw = 0.0;
            double attemptFinal = 0.0;
            double attemptCompression = 0.0;
            double attemptResidual = 0.0;
            bool attemptAccepted = true;
            for (std::size_t part = 0; part < divisions; ++part) {
                const auto partLedger = attempt.advectConservativeSplitExperimental(
                    snapshot,
                    static_cast<double>(dt) / divisions,
                    ((observedSubsteps + part) & 1u) == 0);
                attemptAccepted = attemptAccepted && partLedger.accepted;
                attemptLimited += partLedger.uniqueLimitedFaceCount;
                attemptNearBlocked += partLedger.nearBlockedFaceCount;
                attemptRaw += partLedger.rawAbsoluteFlux;
                attemptFinal += partLedger.finalAbsoluteFlux;
                attemptCompression += partLedger.compressionContribution;
                attemptResidual = std::max(
                    attemptResidual, std::abs(partLedger.conservationResidual));
                if (!partLedger.accepted) break;
            }
            if (!attemptAccepted) continue;
            refinedCorrectedVof = std::move(attempt);
            refinedSubsteps += divisions;
            refinedUniqueLimited += attemptLimited;
            refinedNearBlocked += attemptNearBlocked;
            refinedRawFlux += attemptRaw;
            refinedFinalFlux += attemptFinal;
            refinedCompression += attemptCompression;
            refinedMaximumResidual = std::max(refinedMaximumResidual, attemptResidual);
            refinedAccepted = true;
            break;
        }
        refinedRejected += refinedAccepted ? 0u : 1u;
        ++observedSubsteps;
    };

    simulation::GpuDynamicFreeSurfaceStepper stepper(
        shaderDirectory / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles, &initial);
    bool pass = topology.topologyValid;
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    for (std::size_t outer = 0; outer < 60 && stepper.healthy(); ++outer) {
        const auto report = stepper.step();
        accepted += report.substeps;
        rejected += report.rejectedSubsteps;
    }
    const auto levelSet = stepper.grid().readExperimentalLevelSet();
    MacGrid finalGrid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
    stepper.grid().download(finalGrid);
    const double cellArea = static_cast<double>(grid.cellWidth()) * grid.cellHeight();
    const double classifiedArea = static_cast<double>(std::count(
        finalGrid.cellTypes().begin(), finalGrid.cellTypes().end(), simulation::CellType::fluid)) *
        cellArea;
    const double vofAreaError = vof.totalArea() - initialVofArea;
    const double correctedAreaError = correctedVof.totalArea() - initialVofArea;
    const double refinedAreaError = refinedCorrectedVof.totalArea() - initialVofArea;
    std::sort(limiterScales.begin(), limiterScales.end());
    double meanLimiterScale = 1.0;
    double minimumLimiterScale = 1.0;
    double p95LimiterScale = 1.0;
    if (!limiterScales.empty()) {
        minimumLimiterScale = limiterScales.front();
        meanLimiterScale = std::accumulate(limiterScales.begin(), limiterScales.end(), 0.0) /
            static_cast<double>(limiterScales.size());
        p95LimiterScale = limiterScales[static_cast<std::size_t>(
            0.95 * static_cast<double>(limiterScales.size() - 1))];
    }
    std::sort(
        strongestLimitedFaces.begin(),
        strongestLimitedFaces.end(),
        [](const auto& left, const auto& right) {
            const double leftScale = std::abs(left.rawFlux) > 0.0
                ? std::abs(left.finalFlux / left.rawFlux) : 1.0;
            const double rightScale = std::abs(right.rawFlux) > 0.0
                ? std::abs(right.finalFlux / right.rawFlux) : 1.0;
            return leftScale < rightScale;
        });
    std::size_t mixedCells = 0;
    std::size_t classificationMismatch = 0;
    std::size_t levelSetMismatch = 0;
    std::array<std::size_t, 5> fractionBins{};
    double vofCentroidX = 0.0;
    double vofCentroidY = 0.0;
    double vofWeightedArea = 0.0;
    std::vector<unsigned char> visited(vof.width() * vof.height(), 0);
    std::size_t vofComponents = 0;
    for (std::size_t y = 0; y < vof.height(); ++y) {
        for (std::size_t x = 0; x < vof.width(); ++x) {
            const double fraction = vof.fraction(x, y);
            mixedCells += fraction > 1.0e-6 && fraction < 1.0 - 1.0e-6 ? 1u : 0u;
            if (fraction > 0.0 && fraction < 0.01) ++fractionBins[0];
            else if (fraction >= 0.01 && fraction < 0.1) ++fractionBins[1];
            else if (fraction >= 0.1 && fraction <= 0.9) ++fractionBins[2];
            else if (fraction > 0.9 && fraction < 0.99) ++fractionBins[3];
            else if (fraction >= 0.99 && fraction < 1.0) ++fractionBins[4];
            const double localArea = fraction * vof.cellArea(x, y);
            vofWeightedArea += localArea;
            vofCentroidX += (static_cast<double>(x) + 0.5) * vof.cellWidth() * localArea;
            vofCentroidY += (static_cast<double>(y) + 0.5) * vof.cellHeight() * localArea;
            const bool fractionFluid = fraction >= 0.5;
            const bool classifiedFluid = finalGrid.cellType(x, y) == simulation::CellType::fluid;
            classificationMismatch += fractionFluid != classifiedFluid ? 1u : 0u;
            if (y * vof.width() + x < levelSet.phi.size()) {
                levelSetMismatch += fractionFluid !=
                    (levelSet.phi[y * vof.width() + x] < 0.0f) ? 1u : 0u;
            }
            const std::size_t seed = y * vof.width() + x;
            if (visited[seed] != 0 || fraction <= 1.0e-6) continue;
            ++vofComponents;
            std::queue<std::pair<std::size_t, std::size_t>> pending;
            pending.push({x, y});
            visited[seed] = 1;
            while (!pending.empty()) {
                const auto [cx, cy] = pending.front();
                pending.pop();
                const auto visit = [&](std::size_t nx, std::size_t ny) {
                    const std::size_t offset = ny * vof.width() + nx;
                    if (visited[offset] == 0 && vof.fraction(nx, ny) > 1.0e-6) {
                        visited[offset] = 1;
                        pending.push({nx, ny});
                    }
                };
                if (cx > 0) visit(cx - 1, cy);
                if (cx + 1 < vof.width()) visit(cx + 1, cy);
                if (cy > 0) visit(cx, cy - 1);
                if (cy + 1 < vof.height()) visit(cx, cy + 1);
            }
        }
    }
    if (vofWeightedArea > 0.0) {
        vofCentroidX /= vofWeightedArea;
        vofCentroidY /= vofWeightedArea;
    }
    pass = pass && stepper.healthy() && rejectedVofSteps == 0 &&
        accepted == observedSubsteps &&
        std::abs(vofAreaError) < 1.0e-8 && maximumLedgerResidual < 1.0e-8 &&
        refinedRejected == 0 &&
        std::abs(refinedAreaError - refinedCompression) < 1.0e-8 &&
        refinedMaximumResidual < 1.0e-8;
    const auto legacyDistribution = vof.distribution();
    const auto correctedDistribution = correctedVof.distribution();
    output << std::scientific << std::setprecision(6)
           << "VOF 48x32 accepted-substep MAC replay\n"
           << "  time=" << stepper.simulationTime()
           << " accepted/rejected=" << accepted << '/' << rejected
           << " observed=" << observedSubsteps
           << " vofRejected=" << rejectedVofSteps << '\n'
           << "  initialVOF=" << initialVofArea
           << " finalVOF=" << vof.totalArea()
           << " balanceError=" << vofAreaError
           << " ledgerMax=" << maximumLedgerResidual
           << " legacyLimitEvents=" << limitedFaces
           << " min/maxF=" << std::setprecision(16) << minimumFraction << '/'
           << maximumFraction << std::setprecision(6) << '\n'
           << "  faces evaluated/nonzero/uniqueLimited/repeated="
           << evaluatedFaces << '/' << nonZeroFluxFaces << '/'
           << uniqueLimitedFaces << '/' << repeatedLimitEvents
           << " limiterIterations=" << limiterIterations << '\n'
           << "  flux raw/final/reduction=" << rawAbsoluteFlux << '/'
           << finalAbsoluteFlux << '/'
           << (rawAbsoluteFlux > 0.0 ? 1.0 - finalAbsoluteFlux / rawAbsoluteFlux : 0.0)
           << " scale min/mean/p95=" << minimumLimiterScale << '/'
           << meanLimiterScale << '/' << p95LimiterScale << '\n'
           << "  limited sign(+/-)=" << positiveLimitedFaces << '/'
           << negativeLimitedFaces << " axis(x/y)=" << xLimitedFaces << '/'
           << yLimitedFaces << " nearBlocked=" << nearBlocked << '\n'
           << "  reasons donor/capacity/coupled/highCfl/divergent/plic="
           << donorInsufficient << '/' << receiverCapacity << '/' << coupledInOut << '/'
           << highCourant << '/' << divergentDonor << '/' << plicFailures << '\n'
           << "  levelSetArea=" << levelSet.report.geometricArea
           << " classifiedArea=" << classifiedArea
           << " vofMixed/components=" << mixedCells << '/' << vofComponents
           << " vofClassificationMismatch=" << classificationMismatch
           << " vofLevelSetMismatch=" << levelSetMismatch
           << " particles=" << stepper.grid().particleCount()
           << " centroid=(" << vofCentroidX << ',' << vofCentroidY << ')'
           << " signConflicts=" << levelSet.report.signConflictCount
           << " positiveParticles=" << levelSet.report.positivePhiParticleCount
           << " result=" << (pass ? "PASS" : "FAIL") << '\n'
           << "  F bins (0,.01)/[.01,.1)/[.1,.9]/(.9,.99)/[.99,1)="
           << fractionBins[0] << '/' << fractionBins[1] << '/' << fractionBins[2]
           << '/' << fractionBins[3] << '/' << fractionBins[4] << '\n'
           << "  unified legacy nonzero/mixed/air/full/solid="
           << legacyDistribution.strictNonZero << '/' << legacyDistribution.mixed() << '/'
           << legacyDistribution.air << '/' << legacyDistribution.full << '/'
           << legacyDistribution.solid << '\n'
           << "  corrected rejected/area/error/compression/residual=" << correctedRejected << '/'
           << correctedVof.totalArea() << '/'
           << correctedAreaError << '/' << correctedCompression << '/'
           << correctedMaximumResidual << '\n'
           << "  corrected unique/repeated/iterations/nearBlocked/plic="
           << correctedUniqueLimited << '/' << correctedRepeated << '/'
           << correctedIterations << '/' << correctedNearBlocked << '/'
           << correctedPlicFailures << '\n'
           << "  corrected flux raw/final/reduction=" << correctedRawFlux << '/'
           << correctedFinalFlux << '/'
           << (correctedRawFlux > 0.0 ? 1.0 - correctedFinalFlux / correctedRawFlux : 0.0)
           << " distribution nonzero/mixed/lowTail/full="
           << correctedDistribution.strictNonZero << '/' << correctedDistribution.mixed() << '/'
           << correctedDistribution.lowTail << '/' << correctedDistribution.full << '\n';
    const auto refinedDistribution = refinedCorrectedVof.distribution();
    output << "  refined corrected rejected/substeps/area/error/compression/residual="
           << refinedRejected << '/' << refinedSubsteps << '/' << refinedCorrectedVof.totalArea()
           << '/' << refinedAreaError << '/' << refinedCompression << '/'
           << refinedMaximumResidual << '\n'
           << "  refined limited/nearBlocked flux(raw/final/reduction) distribution(mixed/tail)="
           << refinedUniqueLimited << '/' << refinedNearBlocked << ' '
           << refinedRawFlux << '/' << refinedFinalFlux << '/'
           << (refinedRawFlux > 0.0 ? 1.0 - refinedFinalFlux / refinedRawFlux : 0.0) << ' '
           << refinedDistribution.mixed() << '/' << refinedDistribution.lowTail << '\n';
    const std::size_t severeCount = std::min<std::size_t>(8, strongestLimitedFaces.size());
    for (std::size_t index = 0; index < severeCount; ++index) {
        const auto& event = strongestLimitedFaces[index];
        const double scale = std::abs(event.rawFlux) > 0.0
            ? std::abs(event.finalFlux / event.rawFlux) : 1.0;
        output << "  severe[" << index << "] axis=" << (event.xDirection ? 'x' : 'y')
               << " face=(" << event.faceX << ',' << event.faceY << ')'
               << " raw/final/scale=" << event.rawFlux << '/' << event.finalFlux << '/'
               << scale << " donor/receiverF=" << event.donorFraction << '/'
               << event.receiverFraction << " speed=" << event.faceVelocity
               << " divergence=" << event.localDivergence << " normal=("
               << event.plicNormal.x << ',' << event.plicNormal.y << ")\n";
    }
    return pass ? 0 : 1;
}

int runGpuLevelSetBenchmark(std::ostream& output, const std::filesystem::path& shaderDirectory) {
    output << "GPU Level Set isolated benchmark (advection + redistance + theta)\n";
    output << "Grid       Wall ms   Invalid   Conflicts   Theta fallback\n";
    bool pass = true;
    for (const std::size_t size : {64u, 128u, 192u, 256u}) {
        const LevelSetField initial = circleField(size, size);
        MacGrid grid(size, size, 1.0f / size, 1.0f / size);
        initializeFromLevelSet(grid, initial, 0.13f, 0.07f);
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
        gpu.upload(grid);
        gpu.uploadMarkerParticles(particles);
        gpu.initializeExperimentalLevelSet(initial, levelSetSettings(1));
        gpu.saveCandidateState();
        const auto begin = Clock::now();
        static_cast<void>(gpu.advectExperimentalLevelSet(1.0f / 240.0f));
        static_cast<void>(gpu.updateExperimentalLevelSetDiagnostics());
        glFinish();
        const double milliseconds = std::chrono::duration<double, std::milli>(
            Clock::now() - begin).count();
        gpu.commitCandidateState();
        const auto report = gpu.experimentalLevelSetReport();
        pass = pass && report.invalidPhiCount == 0;
        output << std::setw(4) << size << "^2  " << std::fixed << std::setprecision(3)
               << std::setw(10) << milliseconds << std::setw(10) << report.invalidSampleCount
               << std::setw(12) << report.signConflictCount << std::setw(17)
               << report.fallbackThetaCount << '\n';
    }
    output << "\nFull dynamic incremental cost (4 outer steps)\n";
    output << "Grid       Baseline ms   LevelSet ms   Delta ms\n";
    for (const std::size_t size : {64u, 128u, 192u, 256u}) {
        const auto run = [&](bool enabled) {
            MacGrid grid(size, size, 1.0f / size, 1.0f / size);
            simulation::initializeDamBreak(grid);
            simulation::MarkerParticleSystem particles;
            particles.seedFluidCellsExact(grid, 8);
            simulation::GpuDynamicFreeSurfaceSettings settings;
            settings.experimentalLevelSetEnabled = enabled;
            settings.experimentalLevelSetSettings.redistanceInterval = 4;
            settings.experimentalLevelSetSettings.diagnosticReadbackInterval = 1;
            settings.synchronousGpuTimingEnabled = false;
            settings.asynchronousGpuTimingEnabled = false;
            simulation::GpuDynamicFreeSurfaceStepper stepper(
                shaderDirectory / "mac_divergence.comp", settings);
            const LevelSetField initial = damField(grid);
            static_cast<void>(stepper.initialize(
                grid, particles, enabled ? &initial : nullptr));
            const auto begin = Clock::now();
            for (std::size_t step = 0; step < 4 && stepper.healthy(); ++step) {
                const auto report = stepper.step();
                if (!report.advanced) break;
            }
            glFinish();
            return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        };
        const double baseline = run(false);
        const double levelSet = run(true);
        output << std::setw(4) << size << "^2  " << std::fixed << std::setprecision(3)
               << std::setw(13) << baseline << std::setw(14) << levelSet
               << std::setw(11) << levelSet - baseline << '\n';
    }
    output << "Result: " << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuLevelSetAreaAudit(
    std::ostream& output, const std::filesystem::path& shaderDirectory) {
    struct Result final {
        simulation::GpuLevelSetReport levelSet;
        double analyticInitialArea = 0.0;
        double sampledInitialArea = 0.0;
        double sampledFinalArea = 0.0;
        double fluidCellArea = 0.0;
        double wallMilliseconds = 0.0;
        std::size_t accepted = 0;
        std::size_t rejected = 0;
        bool healthy = false;
    };
    const auto run = [&](simulation::GpuLevelSetAdvectionMethod method,
                         std::size_t redistanceInterval) {
        MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
        simulation::initializeDamBreak(grid);
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        const LevelSetField initial = damField(grid);
        simulation::GpuDynamicFreeSurfaceSettings settings;
        settings.experimentalLevelSetEnabled = true;
        settings.experimentalLevelSetSettings.advectionMethod = method;
        settings.experimentalLevelSetSettings.redistanceInterval = redistanceInterval;
        settings.experimentalLevelSetSettings.diagnosticReadbackInterval = 1;
        settings.experimentalLevelSetSettings.stageAreaAuditEnabled = true;
        simulation::GpuDynamicFreeSurfaceStepper stepper(
            shaderDirectory / "mac_divergence.comp", settings);
        const auto topology = stepper.initialize(grid, particles, &initial);
        Result result;
        result.analyticInitialArea = (16.0 - 1.0) / 48.0 * (21.0 - 1.0) / 32.0;
        result.sampledInitialArea = simulation::estimateLevelSetArea(initial, 8).area;
        const auto begin = Clock::now();
        for (std::size_t outer = 0;
             outer < 600 && stepper.healthy() && stepper.simulationTime() < 0.5 - 1.0e-7;
             ++outer) {
            const auto report = stepper.step();
            result.accepted += report.substeps;
            result.rejected += report.rejectedSubsteps;
        }
        glFinish();
        result.wallMilliseconds = std::chrono::duration<double, std::milli>(
            Clock::now() - begin).count();
        static_cast<void>(stepper.grid().updateExperimentalLevelSetDiagnostics());
        const auto snapshot = stepper.grid().readExperimentalLevelSet();
        LevelSetField final = initial;
        for (std::size_t i = 0; i < final.values.size(); ++i) {
            final.values[i] = snapshot.phi[i];
            final.confidence[i] = snapshot.confidence[i] == 0u ? 0u : 1u;
        }
        result.sampledFinalArea = simulation::estimateLevelSetArea(final, 8).area;
        const auto types = stepper.grid().readCellTypes();
        result.fluidCellArea = static_cast<double>(
                                   std::count(types.begin(), types.end(), simulation::CellType::fluid)) *
            grid.cellWidth() * grid.cellHeight();
        result.levelSet = snapshot.report;
        result.healthy = topology.topologyValid && stepper.healthy() &&
            std::abs(stepper.simulationTime() - 0.5) < 2.0e-6;
        return result;
    };

    output << "GPU Level Set area attribution, 48x32, t=0.5 s\n";
    output << "Method       Redist  Initial analytic/sample  Final sample  Change    "
              "Transport   Redistance  Conflicts  Positive  Domain/Solid/LowConf\n";
    bool pass = true;
    for (const auto configuration : {
             std::pair{simulation::GpuLevelSetAdvectionMethod::rk2SemiLagrangian, 1u},
             std::pair{simulation::GpuLevelSetAdvectionMethod::rk2SemiLagrangian, 4u},
             std::pair{simulation::GpuLevelSetAdvectionMethod::rk2SemiLagrangian, 16u},
             std::pair{simulation::GpuLevelSetAdvectionMethod::rk2SemiLagrangian, 100000u},
             std::pair{simulation::GpuLevelSetAdvectionMethod::limitedMacCormack, 4u}}) {
        const Result result = run(configuration.first, configuration.second);
        pass = pass && result.healthy && result.levelSet.invalidPhiCount == 0;
        const char* name = configuration.first ==
            simulation::GpuLevelSetAdvectionMethod::limitedMacCormack ? "MacCormack" : "RK2";
        output << std::left << std::setw(12) << name << std::right
               << std::setw(7) << configuration.second << std::fixed << std::setprecision(6)
               << std::setw(10) << result.analyticInitialArea << '/'
               << std::setw(9) << result.sampledInitialArea
               << std::setw(14) << result.sampledFinalArea
               << std::setw(10) << result.sampledFinalArea - result.sampledInitialArea
               << std::setw(12) << result.levelSet.cumulativeAdvectionAreaChange
               << std::setw(13) << result.levelSet.cumulativeRedistanceAreaChange
               << std::setw(11) << result.levelSet.signConflictCount
               << std::setw(10) << result.levelSet.positivePhiParticleCount
               << std::setw(8) << result.levelSet.domainClampedBacktraceCount << '/'
               << result.levelSet.solidBacktraceCount << '/'
               << result.levelSet.lowConfidenceBacktraceCount
               << " corrected/limited/fallback="
               << result.levelSet.highOrderCorrectionCount << '/'
               << result.levelSet.limiterClampCount << '/'
               << result.levelSet.highOrderFallbackCount
               << " div(mean fluid/interface/air)="
               << std::scientific << result.levelSet.fluidMeanAbsoluteDivergence << '/'
               << result.levelSet.interfaceMeanAbsoluteDivergence << '/'
               << result.levelSet.airBandMeanAbsoluteDivergence
               << " wall=" << std::fixed << std::setprecision(3)
               << result.wallMilliseconds << " ms result="
               << (result.healthy ? "PASS" : "FAIL") << '\n';
    }
    output << "Area audit result: " << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuLevelSetAdvectionCompare(
    std::ostream& output, const std::filesystem::path& shaderDirectory) {
    struct Result final {
        Difference phi;
        Difference cpuGpu;
        Difference cpuGpuInterface;
        double areaError = 0.0;
        simulation::GpuLevelSetReport report;
        double milliseconds = 0.0;
    };
    const auto run = [&](std::size_t width, std::size_t height,
                         simulation::GpuLevelSetAdvectionMethod method) {
        const double dx = 1.0 / static_cast<double>(width);
        const double dy = 1.0 / static_cast<double>(height);
        const LevelSetField initial = simulation::sampleAnalyticLevelSet(
            width, height, dx, dy, [](double x, double y) {
                return std::hypot(x - 0.37, y - 0.53) - 0.18;
            });
        MacGrid grid(width, height, static_cast<float>(dx), static_cast<float>(dy));
        initializeFromLevelSet(grid, initial, 0.20f, 0.10f);
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 4);
        simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
        gpu.upload(grid);
        gpu.uploadMarkerParticles(particles);
        auto settings = levelSetSettings(100000);
        settings.advectionMethod = method;
        settings.stageAreaAuditEnabled = true;
        gpu.initializeExperimentalLevelSet(initial, settings);
        constexpr std::size_t steps = 20;
        constexpr float dt = 0.005f;
        const auto begin = Clock::now();
        for (std::size_t step = 0; step < steps; ++step) {
            gpu.saveCandidateState();
            static_cast<void>(gpu.advectExperimentalLevelSet(dt));
            static_cast<void>(gpu.updateExperimentalLevelSetDiagnostics());
            gpu.commitCandidateState();
        }
        glFinish();
        Result result;
        result.milliseconds = std::chrono::duration<double, std::milli>(
            Clock::now() - begin).count();
        const auto snapshot = gpu.readExperimentalLevelSet();
        LevelSetField cpu = initial;
        const simulation::VelocityFieldFunction velocity = [](double, double) {
            return simulation::Vector2{0.20f, 0.10f};
        };
        for (std::size_t step = 0; step < steps; ++step) {
            cpu = method == simulation::GpuLevelSetAdvectionMethod::limitedMacCormack
                ? simulation::advectLevelSetLimitedMacCormack(cpu, velocity, dt)
                : simulation::advectLevelSetRk2(cpu, velocity, dt);
        }
        const LevelSetField exact = simulation::sampleAnalyticLevelSet(
            width, height, dx, dy, [](double x, double y) {
                return std::hypot(x - 0.39, y - 0.54) - 0.18;
            });
        result.phi = compareConfident(snapshot.phi, snapshot.confidence, exact);
        result.cpuGpu = compareConfident(snapshot.phi, snapshot.confidence, cpu);
        double interfaceSquared = 0.0;
        std::size_t interfaceCount = 0;
        const double interfaceBand = 3.0 * std::max(dx, dy);
        for (std::size_t i = 0; i < snapshot.phi.size(); ++i) {
            if (snapshot.confidence[i] == 0u || cpu.confidence[i] == 0u ||
                std::abs(exact.values[i]) > interfaceBand) continue;
            const double error = std::abs(static_cast<double>(snapshot.phi[i]) - cpu.values[i]);
            result.cpuGpuInterface.maximum = std::max(
                result.cpuGpuInterface.maximum, error);
            interfaceSquared += error * error;
            ++interfaceCount;
        }
        result.cpuGpuInterface.rms = interfaceCount == 0 ? 0.0 :
            std::sqrt(interfaceSquared / static_cast<double>(interfaceCount));
        LevelSetField actual = exact;
        for (std::size_t i = 0; i < actual.values.size(); ++i) {
            actual.values[i] = snapshot.phi[i];
            actual.confidence[i] = snapshot.confidence[i] == 0u ? 0u : 1u;
        }
        result.areaError = simulation::estimateLevelSetArea(actual, 8).area -
            simulation::estimateLevelSetArea(exact, 8).area;
        result.report = snapshot.report;
        return result;
    };

    output << "GPU limited MacCormack analytic translation comparison\n";
    output << "Grid       RK2 RMS/h  Mac RMS/h  RK2 area err  Mac area err  "
              "Mac CPU/GPU all|band rms/max  Mac corrected/limited/fallback  RK2/Mac ms\n";
    bool pass = true;
    for (const auto dimensions : {std::pair{40u, 30u}, std::pair{80u, 60u},
             std::pair{160u, 120u}}) {
        const Result rk2 = run(dimensions.first, dimensions.second,
            simulation::GpuLevelSetAdvectionMethod::rk2SemiLagrangian);
        const Result mac = run(dimensions.first, dimensions.second,
            simulation::GpuLevelSetAdvectionMethod::limitedMacCormack);
        const double h = std::max(1.0 / dimensions.first, 1.0 / dimensions.second);
        const bool casePass = mac.report.invalidPhiCount == 0 &&
            mac.phi.rms <= rk2.phi.rms * 1.05 &&
            std::abs(mac.areaError) <= std::abs(rk2.areaError) + 1.0e-5 &&
            mac.cpuGpuInterface.maximum <= 5.0e-4;
        pass = pass && casePass;
        output << std::setw(4) << dimensions.first << 'x' << std::setw(3)
               << dimensions.second << std::scientific << std::setprecision(3)
               << std::setw(12) << rk2.phi.rms / h
               << std::setw(11) << mac.phi.rms / h
               << std::setw(14) << rk2.areaError
               << std::setw(14) << mac.areaError << "  " << mac.cpuGpu.rms << '/'
               << mac.cpuGpu.maximum << '|' << mac.cpuGpuInterface.rms << '/'
               << mac.cpuGpuInterface.maximum << "  "
               << mac.report.highOrderCorrectionCount << '/'
               << mac.report.limiterClampCount << '/'
               << mac.report.highOrderFallbackCount << "  "
               << std::fixed << std::setprecision(3) << rk2.milliseconds << '/'
               << mac.milliseconds << " ms " << (casePass ? "PASS" : "FAIL") << '\n';
    }
    output << "Advection comparison result: " << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuLevelSetLongValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory) {
    struct Result final {
        double milliseconds = 0.0;
        double area = 0.0;
        std::size_t accepted = 0;
        std::size_t rejected = 0;
        double simulationTime = 0.0;
        std::string failureMessage;
        simulation::GpuLevelSetReport levelSet;
        bool healthy = false;
    };
    const auto run = [&](std::size_t size,
                         simulation::GpuLevelSetAdvectionMethod method,
                         bool enableLevelSet,
                         bool fullAudit) {
        MacGrid grid(size, size, 1.0f / size, 1.0f / size);
        simulation::initializeDamBreak(grid);
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        const LevelSetField initial = damField(grid);
        simulation::GpuDynamicFreeSurfaceSettings settings;
        settings.experimentalLevelSetEnabled = enableLevelSet;
        settings.experimentalLevelSetSettings.advectionMethod = method;
        settings.experimentalLevelSetSettings.redistanceInterval = 4;
        settings.experimentalLevelSetSettings.diagnosticReadbackInterval = fullAudit ? 1 : 8;
        settings.experimentalLevelSetSettings.stageAreaAuditEnabled = fullAudit;
        simulation::GpuDynamicFreeSurfaceStepper stepper(
            shaderDirectory / "mac_divergence.comp", settings);
        const auto topology = stepper.initialize(
            grid, particles, enableLevelSet ? &initial : nullptr);
        Result result;
        const auto begin = Clock::now();
        for (std::size_t outer = 0;
             outer < 600 && stepper.healthy() && stepper.simulationTime() < 0.5 - 1.0e-7;
             ++outer) {
            const auto report = stepper.step();
            result.accepted += report.substeps;
            result.rejected += report.rejectedSubsteps;
            if (!report.failureMessage.empty()) result.failureMessage = report.failureMessage;
        }
        glFinish();
        result.milliseconds = std::chrono::duration<double, std::milli>(
            Clock::now() - begin).count();
        if (enableLevelSet) {
            static_cast<void>(stepper.grid().updateExperimentalLevelSetDiagnostics());
            const auto snapshot = stepper.grid().readExperimentalLevelSet();
            LevelSetField final = initial;
            for (std::size_t i = 0; i < final.values.size(); ++i) {
                final.values[i] = snapshot.phi[i];
                final.confidence[i] = snapshot.confidence[i] == 0u ? 0u : 1u;
            }
            result.area = simulation::estimateLevelSetArea(final, 4).area;
            result.levelSet = snapshot.report;
        }
        result.simulationTime = stepper.simulationTime();
        result.healthy = topology.topologyValid && stepper.healthy() &&
            std::abs(stepper.simulationTime() - 0.5) < 2.0e-6;
        return result;
    };

    output << "GPU Level Set full t=0.5 s comparison\n";
    output << "Grid   Base wall  RK2 wall/area/conflict  Mac wall/area/conflict  "
              "Audit wall  Accepted base/RK2/Mac/audit  Result\n";
    bool pass = true;
    for (const std::size_t size : {64u, 128u, 192u, 256u}) {
        const Result baseline = run(size,
            simulation::GpuLevelSetAdvectionMethod::rk2SemiLagrangian, false, false);
        const Result rk2 = run(size,
            simulation::GpuLevelSetAdvectionMethod::rk2SemiLagrangian, true, false);
        const Result mac = run(size,
            simulation::GpuLevelSetAdvectionMethod::limitedMacCormack, true, false);
        const Result audit = run(size,
            simulation::GpuLevelSetAdvectionMethod::limitedMacCormack, true, true);
        const bool casePass = baseline.healthy && rk2.healthy && mac.healthy && audit.healthy &&
            baseline.accepted == rk2.accepted && rk2.accepted == mac.accepted &&
            mac.accepted == audit.accepted && baseline.rejected == rk2.rejected &&
            rk2.rejected == mac.rejected && mac.rejected == audit.rejected &&
            mac.levelSet.invalidPhiCount == 0;
        pass = pass && casePass;
        output << std::setw(4) << size << "^2  " << std::fixed << std::setprecision(3)
               << baseline.milliseconds << "  " << rk2.milliseconds << '/'
               << std::setprecision(6) << rk2.area << '/'
               << rk2.levelSet.signConflictCount << "  " << std::setprecision(3)
               << mac.milliseconds << '/' << std::setprecision(6) << mac.area << '/'
               << mac.levelSet.signConflictCount << "  " << std::setprecision(3)
               << audit.milliseconds << "  " << baseline.accepted << '/' << rk2.accepted << '/'
               << mac.accepted << '/' << audit.accepted << "  "
               << (casePass ? "PASS" : "FAIL");
        if (!casePass) {
            output << " time=" << rk2.simulationTime << '/' << mac.simulationTime
                   << " failure=" << (rk2.failureMessage.empty() ? "none" : rk2.failureMessage)
                   << '/' << (mac.failureMessage.empty() ? "none" : mac.failureMessage);
        }
        output << '\n';
    }
    output << "Long validation result: " << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuLevelSetRedistanceValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory) {
    struct Result final {
        CrossingDifference crossings;
        double areaChange = 0.0;
        double gradientError = 0.0;
        std::size_t components = 0;
        std::size_t preservedCells = 0;
        Difference cpuGpu;
        double milliseconds = 0.0;
        bool valid = false;
    };
    const auto run = [&](const LevelSetField& initial,
                         simulation::GpuLevelSetRedistanceMethod method,
                         bool solidWall) {
        MacGrid grid(initial.width, initial.height,
            static_cast<float>(initial.cellWidth), static_cast<float>(initial.cellHeight));
        initializeFromLevelSet(grid, initial, 0.0f, 0.0f);
        if (solidWall) {
            for (std::size_t y = 0; y < grid.height(); ++y)
                grid.setCellType(0, y, simulation::CellType::solid);
        }
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 4);
        simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
        gpu.upload(grid);
        gpu.uploadMarkerParticles(particles);
        auto settings = levelSetSettings(1);
        settings.redistanceMethod = method;
        settings.stageAreaAuditEnabled = true;
        gpu.initializeExperimentalLevelSet(initial, settings);
        gpu.saveCandidateState();
        const auto begin = Clock::now();
        static_cast<void>(gpu.advectExperimentalLevelSet(1.0e-7f));
        static_cast<void>(gpu.updateExperimentalLevelSetDiagnostics());
        glFinish();
        Result result;
        result.milliseconds = std::chrono::duration<double, std::milli>(
            Clock::now() - begin).count();
        gpu.commitCandidateState();
        const auto snapshot = gpu.readExperimentalLevelSet();
        LevelSetField quantized = initial;
        for (double& value : quantized.values) value = static_cast<float>(value);
        const LevelSetField after = fieldFromSnapshot(snapshot, quantized);
        result.crossings = compareEdgeCrossings(quantized, after);
        result.areaChange = simulation::estimateLevelSetArea(after, 8).area -
            simulation::estimateLevelSetArea(quantized, 8).area;
        result.gradientError = meanGradientError(after);
        LevelSetField cpu = quantized;
        simulation::LevelSetRedistanceSettings cpuSettings;
        cpuSettings.narrowBandWidth = settings.narrowBandWidth > 0.0f
            ? settings.narrowBandWidth
            : 6.0 * std::max(initial.cellWidth, initial.cellHeight);
        cpuSettings.areaSamplesPerAxis = 8;
        cpuSettings.preserveInterfaceCrossings = method ==
            simulation::GpuLevelSetRedistanceMethod::interfacePreserving;
        static_cast<void>(simulation::redistanceLevelSetNarrowBand(cpu, cpuSettings));
        result.cpuGpu = compare(snapshot.phi, cpu.values);
        result.components = snapshot.report.negativeComponentCount;
        result.preservedCells = snapshot.report.interfacePreservedCellCount;
        result.valid = snapshot.report.invalidPhiCount == 0;
        return result;
    };

    struct Case final { const char* name; LevelSetField field; bool solidWall = false; };
    std::vector<Case> cases;
    constexpr std::size_t width = 80;
    constexpr std::size_t height = 60;
    constexpr double dx = 1.0 / width;
    constexpr double dy = 1.0 / height;
    cases.push_back({"horizontal", simulation::sampleAnalyticLevelSet(width, height, dx, dy,
        [](double, double y) { return y - 0.473; })});
    cases.push_back({"tilted", simulation::sampleAnalyticLevelSet(width, height, dx, dy,
        [](double x, double y) { return (y - 0.31 * x - 0.29) / std::hypot(1.0, 0.31); })});
    cases.push_back({"circle", circleField(width, height)});
    cases.push_back({"thin-column", simulation::sampleAnalyticLevelSet(width, height, dx, dy,
        [](double x, double y) {
            const double qx = std::abs(x - 0.43) - 0.031;
            const double qy = std::abs(y - 0.48) - 0.31;
            return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
                std::min(std::max(qx, qy), 0.0);
        })});
    cases.push_back({"two-drops", simulation::sampleAnalyticLevelSet(width, height, dx, dy,
        [](double x, double y) {
            return std::min(std::hypot(x - 0.35, y - 0.5) - 0.12,
                std::hypot(x - 0.65, y - 0.5) - 0.12);
        })});
    cases.push_back({"wall-contact", simulation::sampleAnalyticLevelSet(width, height, dx, dy,
        [](double x, double y) { return std::hypot(x - 0.115, y - 0.45) - 0.105; }), true});

    output << "GPU interface-preserving redistance validation\n";
    output << "Case          old/new crossing RMS  old/new max  old/new area  old/new grad  "
              "components  preserved  cpu/gpu RMS  time ms  Result\n";
    bool pass = true;
    for (const auto& test : cases) {
        const Result baseline = run(test.field,
            simulation::GpuLevelSetRedistanceMethod::marchingSquares, test.solidWall);
        const Result preserved = run(test.field,
            simulation::GpuLevelSetRedistanceMethod::interfacePreserving, test.solidWall);
        const bool casePass = baseline.valid && preserved.valid &&
            preserved.crossings.changedTopology == 0 &&
            preserved.crossings.maximum <= (test.solidWall ? 1.0e-4 : 5.0e-7) &&
            preserved.components == baseline.components &&
            preserved.gradientError <= baseline.gradientError + 0.35 &&
            (test.solidWall || preserved.cpuGpu.rms <= 2.0e-5);
        pass = pass && casePass;
        output << std::left << std::setw(13) << test.name << std::right
               << std::scientific << std::setprecision(3)
               << baseline.crossings.rms << '/' << preserved.crossings.rms << "  "
               << baseline.crossings.maximum << '/' << preserved.crossings.maximum << "  "
               << baseline.areaChange << '/' << preserved.areaChange << "  "
               << baseline.gradientError << '/' << preserved.gradientError << "  "
               << baseline.components << '/' << preserved.components << "  "
               << preserved.preservedCells << "  " << std::scientific << std::setprecision(3)
               << preserved.cpuGpu.rms << "  " << std::fixed << std::setprecision(3)
               << baseline.milliseconds << '/' << preserved.milliseconds << "  "
               << (casePass ? "PASS" : "FAIL") << '\n';
    }
    output << "Redistance validation result: " << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuLevelSetExtensionValidation(
    std::ostream& output, const std::filesystem::path& shaderDirectory) {
    struct Result final {
        Difference interfaceError;
        double areaError = 0.0;
        double milliseconds = 0.0;
        simulation::GpuLevelSetReport report;
        bool valid = false;
    };
    const auto run = [&](std::size_t width, std::size_t height, bool rotation,
                         simulation::GpuLevelSetVelocityMode velocityMode) {
        const double dx = 1.0 / static_cast<double>(width);
        const double dy = 1.0 / static_cast<double>(height);
        const double omega = 0.8;
        const double totalTime = 0.1;
        const double dt = 0.005;
        const std::size_t steps = static_cast<std::size_t>(std::llround(totalTime / dt));
        const LevelSetField initial = circleField(width, height);
        MacGrid grid(width, height, static_cast<float>(dx), static_cast<float>(dy));
        initializeFromLevelSet(grid, initial, 0.2f, 0.1f);
        if (rotation) {
            for (std::size_t y = 0; y < height; ++y)
                for (std::size_t x = 0; x <= width; ++x)
                    grid.horizontalVelocity(x, y) = static_cast<float>(
                        -omega * ((y + 0.5) * dy - 0.5));
            for (std::size_t y = 0; y <= height; ++y)
                for (std::size_t x = 0; x < width; ++x)
                    grid.verticalVelocity(x, y) = static_cast<float>(
                        omega * ((x + 0.5) * dx - 0.5));
        }
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 4);
        simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
        gpu.upload(grid);
        gpu.uploadMarkerParticles(particles);
        auto settings = levelSetSettings(100000);
        settings.advectionMethod = simulation::GpuLevelSetAdvectionMethod::limitedMacCormack;
        settings.velocityMode = velocityMode;
        settings.velocityExtensionLayers = 8;
        gpu.initializeExperimentalLevelSet(initial, settings);
        const auto begin = Clock::now();
        for (std::size_t step = 0; step < steps; ++step) {
            gpu.saveCandidateState();
            static_cast<void>(gpu.advectExperimentalLevelSet(static_cast<float>(dt)));
            static_cast<void>(gpu.updateExperimentalLevelSetDiagnostics());
            gpu.commitCandidateState();
        }
        glFinish();
        Result result;
        result.milliseconds = std::chrono::duration<double, std::milli>(
            Clock::now() - begin).count();
        const auto snapshot = gpu.readExperimentalLevelSet();
        LevelSetField actual = fieldFromSnapshot(snapshot, initial);
        LevelSetField exact;
        if (rotation) {
            const double angle = omega * totalTime;
            const double cx = 0.5 + std::cos(angle) * (0.37 - 0.5) -
                std::sin(angle) * (0.53 - 0.5);
            const double cy = 0.5 + std::sin(angle) * (0.37 - 0.5) +
                std::cos(angle) * (0.53 - 0.5);
            exact = simulation::sampleAnalyticLevelSet(width, height, dx, dy,
                [=](double x, double y) { return std::hypot(x - cx, y - cy) - 0.19; });
        } else {
            exact = simulation::sampleAnalyticLevelSet(width, height, dx, dy,
                [=](double x, double y) {
                    return std::hypot(x - (0.37 + 0.2 * totalTime),
                        y - (0.53 + 0.1 * totalTime)) - 0.19;
                });
        }
        std::vector<float> interfaceValues;
        std::vector<double> exactValues;
        const double band = 2.0 * std::max(dx, dy);
        for (std::size_t i = 0; i < actual.values.size(); ++i) {
            if (std::abs(exact.values[i]) > band || actual.confidence[i] == 0u) continue;
            interfaceValues.push_back(static_cast<float>(actual.values[i]));
            exactValues.push_back(exact.values[i]);
        }
        result.interfaceError = compare(interfaceValues, exactValues);
        result.areaError = simulation::estimateLevelSetArea(actual, 8).area -
            simulation::estimateLevelSetArea(exact, 8).area;
        result.report = snapshot.report;
        result.valid = snapshot.report.invalidPhiCount == 0 && !interfaceValues.empty();
        return result;
    };

    output << "GPU Level Set normal-extension analytic validation\n";
    output << "Case/Grid  formal RMS/area  extended RMS/area  div interface/air  "
              "seed/fill/unresolved  time ms  Result\n";
    bool pass = true;
    for (const bool rotation : {false, true}) {
        for (const auto [width, height] : {std::pair{80u, 60u}, std::pair{160u, 120u}}) {
            const Result formal = run(width, height, rotation,
                simulation::GpuLevelSetVelocityMode::formalExtrapolated);
            const Result extended = run(width, height, rotation,
                simulation::GpuLevelSetVelocityMode::normalExtended);
            const bool casePass = formal.valid && extended.valid &&
                extended.interfaceError.rms <= formal.interfaceError.rms +
                    0.35 * std::max(1.0 / width, 1.0 / height) &&
                std::abs(extended.areaError) <= std::abs(formal.areaError) + 2.0e-3;
            pass = pass && casePass;
            output << (rotation ? "rotate " : "translate ") << width << 'x' << height << ' '
                   << std::scientific << std::setprecision(3)
                   << formal.interfaceError.rms << '/' << formal.areaError << "  "
                   << extended.interfaceError.rms << '/' << extended.areaError << "  "
                   << extended.report.interfaceMeanAbsoluteDivergence << '/'
                   << extended.report.airBandMeanAbsoluteDivergence << "  "
                   << extended.report.velocityExtensionSeedCount << '/'
                   << extended.report.velocityExtensionFilledCount << '/'
                   << extended.report.velocityExtensionUnresolvedCount << "  "
                   << std::fixed << std::setprecision(3) << formal.milliseconds << '/'
                   << extended.milliseconds << "  " << (casePass ? "PASS" : "FAIL") << '\n';
        }
    }
    output << "Velocity extension validation result: " << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuLevelSetConservationCompare(
    std::ostream& output, const std::filesystem::path& shaderDirectory) {
    struct Configuration final {
        const char* name;
        simulation::GpuLevelSetAdvectionMethod advection;
        simulation::GpuLevelSetRedistanceMethod redistance;
        simulation::GpuLevelSetVelocityMode velocity;
        double physicalRedistanceInterval;
    };
    const std::array configurations{
        Configuration{"A-RK2-old-old", simulation::GpuLevelSetAdvectionMethod::rk2SemiLagrangian,
            simulation::GpuLevelSetRedistanceMethod::marchingSquares,
            simulation::GpuLevelSetVelocityMode::formalExtrapolated, 0.0},
        Configuration{"B-Mac-old-old", simulation::GpuLevelSetAdvectionMethod::limitedMacCormack,
            simulation::GpuLevelSetRedistanceMethod::marchingSquares,
            simulation::GpuLevelSetVelocityMode::formalExtrapolated, 0.0},
        Configuration{"C-Mac-keep-old", simulation::GpuLevelSetAdvectionMethod::limitedMacCormack,
            simulation::GpuLevelSetRedistanceMethod::interfacePreserving,
            simulation::GpuLevelSetVelocityMode::formalExtrapolated, 0.0},
        Configuration{"D-Mac-keep-ext", simulation::GpuLevelSetAdvectionMethod::limitedMacCormack,
            simulation::GpuLevelSetRedistanceMethod::interfacePreserving,
            simulation::GpuLevelSetVelocityMode::normalExtended, 0.0},
        Configuration{"C2-Mac-keep-time", simulation::GpuLevelSetAdvectionMethod::limitedMacCormack,
            simulation::GpuLevelSetRedistanceMethod::interfacePreserving,
            simulation::GpuLevelSetVelocityMode::formalExtrapolated, 1.0 / 120.0}};
    struct Result final {
        double wallMilliseconds = 0.0;
        double area = 0.0;
        std::size_t accepted = 0;
        std::size_t rejected = 0;
        simulation::GpuLevelSetReport levelSet;
        bool healthy = false;
    };
    const auto run = [&](std::size_t width, std::size_t height,
                         const Configuration& configuration, bool audit) {
        MacGrid grid(width, height, 1.0f / width, 1.0f / height);
        simulation::initializeDamBreak(grid);
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        const LevelSetField initial = damField(grid);
        simulation::GpuDynamicFreeSurfaceSettings settings;
        settings.experimentalLevelSetEnabled = true;
        settings.experimentalLevelSetSettings.advectionMethod = configuration.advection;
        settings.experimentalLevelSetSettings.redistanceMethod = configuration.redistance;
        settings.experimentalLevelSetSettings.velocityMode = configuration.velocity;
        settings.experimentalLevelSetSettings.velocityExtensionLayers = 8;
        settings.experimentalLevelSetSettings.redistanceInterval = 4;
        settings.experimentalLevelSetSettings.redistancePhysicalInterval =
            configuration.physicalRedistanceInterval;
        settings.experimentalLevelSetSettings.diagnosticReadbackInterval = audit ? 1 : 8;
        settings.experimentalLevelSetSettings.stageAreaAuditEnabled = audit;
        simulation::GpuDynamicFreeSurfaceStepper stepper(
            shaderDirectory / "mac_divergence.comp", settings);
        const auto topology = stepper.initialize(grid, particles, &initial);
        Result result;
        const auto begin = Clock::now();
        for (std::size_t outer = 0;
             outer < 600 && stepper.healthy() && stepper.simulationTime() < 0.5 - 1.0e-7;
             ++outer) {
            const auto report = stepper.step();
            result.accepted += report.substeps;
            result.rejected += report.rejectedSubsteps;
        }
        glFinish();
        result.wallMilliseconds = std::chrono::duration<double, std::milli>(
            Clock::now() - begin).count();
        static_cast<void>(stepper.grid().updateExperimentalLevelSetDiagnostics());
        const auto snapshot = stepper.grid().readExperimentalLevelSet();
        const LevelSetField final = fieldFromSnapshot(snapshot, initial);
        result.area = simulation::estimateLevelSetArea(final, 8).area;
        result.levelSet = snapshot.report;
        result.healthy = topology.topologyValid && stepper.healthy() &&
            std::abs(stepper.simulationTime() - 0.5) < 2.0e-6 &&
            snapshot.report.invalidPhiCount == 0;
        return result;
    };

    output << "GPU Level Set controlled conservation comparison, t=0.5 s\n";
    output << "Grid Config             wall ms area       adv delta   redist delta conflict "
              "positive theta V/C/F div I/A lowConf preserved ext ms accepted/rejected Result\n";
    bool pass = true;
    for (const auto [width, height] : {std::pair{48u, 32u}, std::pair{64u, 64u},
             std::pair{128u, 128u}, std::pair{192u, 192u}, std::pair{256u, 256u}}) {
        std::size_t expectedAccepted = 0;
        std::size_t expectedRejected = 0;
        for (std::size_t index = 0; index < configurations.size(); ++index) {
            const Result result = run(width, height, configurations[index], width == 48);
            if (index == 0) {
                expectedAccepted = result.accepted;
                expectedRejected = result.rejected;
            }
            const bool casePass = result.healthy && result.accepted == expectedAccepted &&
                result.rejected == expectedRejected;
            const bool optionalPhysicalTrigger = index + 1 == configurations.size();
            const bool diagnosticUsable = result.levelSet.validThetaCount +
                result.levelSet.clampedThetaCount > 0;
            pass = pass && casePass && (optionalPhysicalTrigger || diagnosticUsable);
            output << std::setw(3) << width << 'x' << std::setw(3) << height << ' '
                   << std::left << std::setw(18) << configurations[index].name << std::right
                   << std::fixed << std::setprecision(3) << std::setw(9)
                   << result.wallMilliseconds << ' ' << std::setprecision(6)
                   << result.area << ' ' << std::scientific << std::setprecision(3)
                   << result.levelSet.cumulativeAdvectionAreaChange << ' '
                   << result.levelSet.cumulativeRedistanceAreaChange << ' '
                   << result.levelSet.signConflictCount << ' '
                   << result.levelSet.positivePhiParticleCount << ' '
                   << result.levelSet.validThetaCount << '/'
                   << result.levelSet.clampedThetaCount << '/'
                   << result.levelSet.fallbackThetaCount << ' '
                   << result.levelSet.interfaceMeanAbsoluteDivergence << '/'
                   << result.levelSet.airBandMeanAbsoluteDivergence << ' '
                   << result.levelSet.lowConfidenceBacktraceCount << ' '
                   << result.levelSet.interfacePreservedCellCount << ' '
                   << std::fixed << std::setprecision(3)
                   << result.levelSet.velocityExtensionCpuMilliseconds << ' '
                   << result.accepted << '/' << result.rejected << ' '
                   << (casePass && diagnosticUsable ? "PASS" :
                       (optionalPhysicalTrigger && casePass ? "REJECT" : "FAIL")) << '\n';
        }
    }
    output << "Conservation comparison result: " << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

} // namespace fluidlab::diagnostics
