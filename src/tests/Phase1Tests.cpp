#include "Phase1Tests.h"

#include "../simulation/mac/FluidStepper.h"
#include "../simulation/mac/DynamicFreeSurfaceStepper.h"
#include "../simulation/mac/FreeSurfacePressureProjection.h"
#include "../simulation/mac/MacGrid.h"
#include "../simulation/mac/PressureProjection.h"
#include "../simulation/mac/Scenarios.h"
#include "../simulation/mac/VelocityAdvection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <string_view>
#include <utility>

namespace fluidlab::tests {
namespace {

constexpr std::size_t kGridSize = 64;
constexpr float kCellSize = 1.0f / static_cast<float>(kGridSize);

class TestRecorder final {
public:
    explicit TestRecorder(std::ostream& output)
        : m_output(output) {
    }

    void check(bool condition, std::string_view description) {
        m_output << (condition ? "[PASS] " : "[FAIL] ") << description << '\n';
        if (!condition) {
            ++m_failures;
        }
    }

    [[nodiscard]] int failures() const noexcept { return m_failures; }

private:
    std::ostream& m_output;
    int m_failures = 0;
};

[[nodiscard]] float maximumAbsoluteVelocity(const simulation::MacGrid& grid) {
    float maximum = 0.0f;
    for (const float value : grid.horizontalVelocities()) {
        maximum = std::max(maximum, std::abs(value));
    }
    for (const float value : grid.verticalVelocities()) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

[[nodiscard]] float meanPressure(const simulation::MacGrid& grid) {
    double sum = 0.0;
    for (const float value : grid.pressures()) {
        sum += value;
    }
    return static_cast<float>(sum / static_cast<double>(grid.pressures().size()));
}

[[nodiscard]] float maximumAbsolutePressure(const simulation::MacGrid& grid) {
    float maximum = 0.0f;
    for (const float value : grid.pressures()) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

[[nodiscard]] float maximumVelocityDifference(
    const simulation::MacGrid& left,
    const simulation::MacGrid& right) {
    float maximum = 0.0f;
    for (std::size_t index = 0; index < left.horizontalVelocities().size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(left.horizontalVelocities()[index] - right.horizontalVelocities()[index]));
    }
    for (std::size_t index = 0; index < left.verticalVelocities().size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(left.verticalVelocities()[index] - right.verticalVelocities()[index]));
    }
    return maximum;
}

[[nodiscard]] float maximumPressureDifference(
    const simulation::MacGrid& left,
    const simulation::MacGrid& right) {
    float maximum = 0.0f;
    for (std::size_t index = 0; index < left.pressures().size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(left.pressures()[index] - right.pressures()[index]));
    }
    return maximum;
}

[[nodiscard]] simulation::ProjectionReport advanceLegacyCold(simulation::MacGrid& grid) {
    const simulation::VelocityAdvector advector;
    simulation::MacGrid advected(
        grid.width(), grid.height(), grid.cellWidth(), grid.cellHeight());
    advector.advect(grid, advected, 1.0f / 60.0f);
    advected.enforceClosedBoundary();
    grid = std::move(advected);
    const simulation::PressureProjector projector;
    return projector.project(grid);
}

void initializeLinearField(simulation::MacGrid& grid) {
    for (std::size_t y = 0; y < grid.height(); ++y) {
        const float positionY = (static_cast<float>(y) + 0.5f) * grid.cellHeight();
        for (std::size_t faceX = 0; faceX <= grid.width(); ++faceX) {
            const float positionX = static_cast<float>(faceX) * grid.cellWidth();
            grid.horizontalVelocity(faceX, y) = 1.0f + 2.0f * positionX - 3.0f * positionY;
        }
    }
    for (std::size_t faceY = 0; faceY <= grid.height(); ++faceY) {
        const float positionY = static_cast<float>(faceY) * grid.cellHeight();
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const float positionX = (static_cast<float>(x) + 0.5f) * grid.cellWidth();
            grid.verticalVelocity(x, faceY) = -2.0f - 0.5f * positionX + 1.25f * positionY;
        }
    }
    grid.computeDivergence();
}

void printProjectionReport(std::ostream& output, const simulation::ProjectionReport& report) {
    output << std::scientific << std::setprecision(6)
           << "  divergence max:  " << report.divergenceBefore.maximumAbsolute
           << " -> " << report.divergenceAfter.maximumAbsolute << '\n'
           << "  divergence mean: " << report.divergenceBefore.meanAbsolute
           << " -> " << report.divergenceAfter.meanAbsolute << '\n'
           << "  pressure residual RMS: " << report.pressureSolve.initialResidual.rootMeanSquare
           << " -> " << report.pressureSolve.finalResidual.rootMeanSquare << '\n'
           << "  pressure residual max: " << report.pressureSolve.initialResidual.maximumAbsolute
           << " -> " << report.pressureSolve.finalResidual.maximumAbsolute << '\n'
           << "  pressure residual RMS tolerance: " << report.pressureSolve.residualTolerance << '\n'
           << "  max absolute pressure: "
           << report.pressureSolve.initialMaximumAbsolutePressure << " -> "
           << report.pressureSolve.finalMaximumAbsolutePressure << '\n'
           << "  max velocity change: " << report.maximumVelocityChange << '\n'
           << "  iterations: " << report.pressureSolve.iterations
           << ", converged: " << (report.pressureSolve.converged ? "yes" : "no") << '\n';
}

} // namespace

int runPhase1Tests(std::ostream& output) {
    output << "Phase 1.1 and 1.2 CPU MAC tests\n";
    TestRecorder tests(output);
    const simulation::PressureProjector projector;

    {
        simulation::MacGrid zeroGrid(kGridSize, kGridSize, kCellSize, kCellSize);
        const simulation::ProjectionReport report = projector.project(zeroGrid);
        output << "\nZero velocity field:\n";
        printProjectionReport(output, report);

        tests.check(report.divergenceBefore.maximumAbsolute == 0.0f, "zero field starts divergence-free");
        tests.check(
            zeroGrid.horizontalVelocities().size() == (kGridSize + 1) * kGridSize &&
                zeroGrid.verticalVelocities().size() == kGridSize * (kGridSize + 1) &&
                zeroGrid.pressures().size() == kGridSize * kGridSize &&
                zeroGrid.divergences().size() == kGridSize * kGridSize,
            "MAC arrays use the required staggered dimensions");
        tests.check(report.divergenceAfter.maximumAbsolute <= 1.0e-7f, "zero field remains divergence-free");
        tests.check(maximumAbsoluteVelocity(zeroGrid) <= 1.0e-7f, "zero velocity remains unchanged");
        tests.check(report.pressureSolve.converged, "zero-field pressure solve reports convergence");
        tests.check(report.pressureSolve.iterations == 0, "zero field converges before the first pressure iteration");
        tests.check(
            report.pressureSolve.termination ==
                simulation::PressureSolveTermination::initialResidualWithinTolerance,
            "zero field reports initial residual as the early-convergence reason");
        tests.check(report.maximumVelocityChange == 0.0f, "zero-field projection does not change velocity");
        tests.check(zeroGrid.maximumBoundaryNormalSpeed() == 0.0f, "zero field satisfies closed-wall normals");
        tests.check(zeroGrid.allFinite(), "zero-field result contains only finite values");
    }

    {
        simulation::MacGrid grid(kGridSize, kGridSize, kCellSize, kCellSize);
        simulation::initializeDeterministicDivergentVortex(grid);
        const simulation::ProjectionReport report = projector.project(grid);
        output << "\nDivergent vortex field:\n";
        printProjectionReport(output, report);

        const float maxReduction = report.divergenceBefore.maximumAbsolute > 0.0f
            ? report.divergenceAfter.maximumAbsolute / report.divergenceBefore.maximumAbsolute
            : 1.0f;
        const float meanReduction = report.divergenceBefore.meanAbsolute > 0.0f
            ? report.divergenceAfter.meanAbsolute / report.divergenceBefore.meanAbsolute
            : 1.0f;

        tests.check(report.divergenceBefore.maximumAbsolute > 1.0f, "test input has significant divergence");
        tests.check(report.pressureSolve.iterations > 0, "nonzero-divergence input executes pressure iterations");
        tests.check(
            report.pressureSolve.termination ==
                simulation::PressureSolveTermination::iterativeResidualWithinTolerance,
            "nonzero-divergence input converges through pressure iterations");
        tests.check(report.maximumVelocityChange > 0.0f, "nonzero-divergence projection changes velocity");
        tests.check(
            report.pressureSolve.finalMaximumAbsolutePressure > 0.0f,
            "nonzero-divergence projection produces a nonzero pressure correction");
        tests.check(maxReduction < 0.01f, "maximum divergence is reduced by more than 99 percent");
        tests.check(meanReduction < 0.01f, "mean divergence is reduced by more than 99 percent");
        tests.check(report.pressureSolve.converged, "pressure residual reaches the configured tolerance");
        tests.check(std::abs(meanPressure(grid)) <= 1.0e-6f, "Neumann pressure gauge has approximately zero mean");
        tests.check(grid.maximumBoundaryNormalSpeed() <= 1.0e-7f, "projected field satisfies closed-wall normals");
        tests.check(grid.allFinite(), "projected result contains only finite values");
    }

    output << "\nPhase 1.2 velocity advection tests:\n";

    {
        simulation::MacGrid zeroGrid(kGridSize, kGridSize, kCellSize, kCellSize);
        simulation::CpuFluidStepper stepper;
        for (int step = 0; step < 8; ++step) {
            static_cast<void>(stepper.step(zeroGrid));
        }
        tests.check(maximumAbsoluteVelocity(zeroGrid) <= 1.0e-7f, "Test A: zero velocity remains zero across multiple steps");
        tests.check(zeroGrid.divergenceStatistics().maximumAbsolute <= 1.0e-7f, "Test A: zero field remains divergence-free");
    }

    {
        simulation::MacGrid source(24, 20, 1.0f / 24.0f, 1.0f / 20.0f);
        simulation::MacGrid destination(24, 20, 1.0f / 24.0f, 1.0f / 20.0f);
        simulation::initializeDeterministicDivergentVortex(source);
        const simulation::VelocityAdvector advector;
        advector.advect(source, destination, 0.0f);
        tests.check(maximumVelocityDifference(source, destination) == 0.0f, "Test B: zero timestep copies both staggered velocity fields exactly");
    }

    {
        simulation::MacGrid linearGrid(8, 6, 0.25f, 0.2f);
        initializeLinearField(linearGrid);
        const simulation::Vector2 position{0.83f, 0.71f};
        const float expectedU = 1.0f + 2.0f * position.x - 3.0f * position.y;
        const float expectedV = -2.0f - 0.5f * position.x + 1.25f * position.y;
        const float sampledU = simulation::MacVelocitySampler::sampleHorizontal(linearGrid, position);
        const float sampledV = simulation::MacVelocitySampler::sampleVertical(linearGrid, position);
        output << "  linear interpolation u/v: " << sampledU << ", " << sampledV << '\n';
        tests.check(std::abs(sampledU - expectedU) <= 1.0e-5f, "Test C: u interpolation respects its vertical-face offset");
        tests.check(std::abs(sampledV - expectedV) <= 1.0e-5f, "Test C: v interpolation respects its horizontal-face offset");

        simulation::MacGrid constantGrid(8, 8, 0.125f, 0.125f);
        for (std::size_t y = 0; y < constantGrid.height(); ++y) {
            for (std::size_t x = 0; x <= constantGrid.width(); ++x) {
                constantGrid.horizontalVelocity(x, y) = 0.2f;
            }
        }
        for (std::size_t y = 0; y <= constantGrid.height(); ++y) {
            for (std::size_t x = 0; x < constantGrid.width(); ++x) {
                constantGrid.verticalVelocity(x, y) = -0.1f;
            }
        }
        const simulation::Vector2 traced = simulation::MacVelocitySampler::traceBackwardRk2(
            constantGrid,
            {0.6f, 0.6f},
            0.25f);
        tests.check(
            std::abs(traced.x - 0.55f) <= 1.0e-6f &&
                std::abs(traced.y - 0.625f) <= 1.0e-6f,
            "Test C: RK2 tracing matches the exact constant-velocity backtrace");
    }

    {
        constexpr std::size_t evolutionSize = 32;
        constexpr float evolutionCellSize = 1.0f / static_cast<float>(evolutionSize);
        simulation::MacGrid vortex(evolutionSize, evolutionSize, evolutionCellSize, evolutionCellSize);
        simulation::initializeClosedVortex(vortex);
        const simulation::PressureProjector initialProjector;
        static_cast<void>(initialProjector.project(vortex));

        simulation::CpuFluidStepper stepper;
        const float initialEnergy = simulation::CpuFluidStepper::meanKineticEnergy(vortex);
        simulation::FluidStepReport lastReport;
        for (int step = 0; step < 20; ++step) {
            lastReport = stepper.step(vortex);
        }
        const float finalEnergy = simulation::CpuFluidStepper::meanKineticEnergy(vortex);
        output << std::scientific << std::setprecision(6)
               << "  vortex mean kinetic energy: " << initialEnergy << " -> " << finalEnergy << '\n'
               << "  final-step divergence max (advected -> projected): "
               << lastReport.divergenceAfterAdvection.maximumAbsolute << " -> "
               << lastReport.projection.divergenceAfter.maximumAbsolute << '\n'
               << "  final-step divergence mean (advected -> projected): "
               << lastReport.divergenceAfterAdvection.meanAbsolute << " -> "
               << lastReport.projection.divergenceAfter.meanAbsolute << '\n';

        const float divergenceRatio = lastReport.divergenceAfterAdvection.maximumAbsolute > 0.0f
            ? lastReport.projection.divergenceAfter.maximumAbsolute /
                  lastReport.divergenceAfterAdvection.maximumAbsolute
            : 0.0f;
        tests.check(finalEnergy < initialEnergy, "Test D: semi-Lagrangian vortex exhibits expected numerical dissipation");
        tests.check(finalEnergy > 0.25f * initialEnergy, "Test D: vortex remains meaningfully present after 20 steps");
        tests.check(divergenceRatio < 0.05f, "Test E: projection removes more than 95 percent of advected maximum divergence");
        tests.check(vortex.maximumBoundaryNormalSpeed() <= 1.0e-7f, "Test E: evolved vortex satisfies closed-wall normals");
        tests.check(vortex.allFinite(), "Test D/E: evolved vortex contains only finite values");
    }

    {
        simulation::MacGrid source(16, 12, 1.0f / 16.0f, 1.0f / 12.0f);
        simulation::MacGrid destination(16, 12, 1.0f / 16.0f, 1.0f / 12.0f);
        for (std::size_t y = 0; y < source.height(); ++y) {
            for (std::size_t x = 0; x <= source.width(); ++x) {
                source.horizontalVelocity(x, y) = 40.0f;
            }
        }
        for (std::size_t y = 0; y <= source.height(); ++y) {
            for (std::size_t x = 0; x < source.width(); ++x) {
                source.verticalVelocity(x, y) = -35.0f;
            }
        }

        const simulation::VelocityAdvector advector;
        advector.advect(source, destination, 2.0f);
        const simulation::Vector2 outsideSample =
            simulation::MacVelocitySampler::sampleVelocity(source, {-1000.0f, 1000.0f});
        destination.enforceClosedBoundary();
        tests.check(destination.allFinite(), "Test F: large boundary-crossing backtraces remain finite");
        tests.check(std::isfinite(outsideSample.x) && std::isfinite(outsideSample.y), "Test F: out-of-domain samples clamp safely");
        tests.check(destination.maximumBoundaryNormalSpeed() == 0.0f, "Test F: closed boundary enforcement restores zero normal speed");
    }

    output << "\nPhase 1.2.5 optimization regression tests:\n";

    {
        constexpr std::size_t comparisonSize = 32;
        constexpr float comparisonCellSize = 1.0f / static_cast<float>(comparisonSize);
        simulation::MacGrid legacy(comparisonSize, comparisonSize, comparisonCellSize, comparisonCellSize);
        simulation::MacGrid reused(comparisonSize, comparisonSize, comparisonCellSize, comparisonCellSize);
        simulation::initializeClosedVortex(legacy);
        reused = legacy;

        simulation::CpuFluidStepper reusedStepper(simulation::FluidStepSettings{
            .pressureWarmStartEnabled = false,
            .pressureMeanRemovalInterval = 1,
        });
        for (int step = 0; step < 12; ++step) {
            static_cast<void>(advanceLegacyCold(legacy));
            static_cast<void>(reusedStepper.step(reused));
        }

        const float velocityDifference = maximumVelocityDifference(legacy, reused);
        const float pressureDifference = maximumPressureDifference(legacy, reused);
        output << "  legacy/reused max velocity difference: " << velocityDifference << '\n'
               << "  legacy/reused max pressure difference: " << pressureDifference << '\n';
        tests.check(
            velocityDifference <= 1.0e-6f,
            "workspace reuse preserves the legacy cold-start velocity result");
        tests.check(
            pressureDifference <= 1.0e-5f,
            "workspace reuse preserves the legacy cold-start pressure result");
        tests.check(
            reusedStepper.workspaceAllocationCount() == 1,
            "fixed grid layout allocates the advection workspace only once");
        tests.check(
            reusedStepper.divergenceAfterAdvectionField().size() == comparisonSize * comparisonSize,
            "reused divergence diagnostics retain the complete cell field");
    }

    {
        constexpr std::size_t comparisonSize = 32;
        constexpr float comparisonCellSize = 1.0f / static_cast<float>(comparisonSize);
        simulation::MacGrid cold(comparisonSize, comparisonSize, comparisonCellSize, comparisonCellSize);
        simulation::MacGrid warm(comparisonSize, comparisonSize, comparisonCellSize, comparisonCellSize);
        simulation::initializeClosedVortex(cold);
        warm = cold;
        const simulation::PressureProjector initialProjector;
        static_cast<void>(initialProjector.project(cold));
        warm = cold;

        simulation::CpuFluidStepper coldStepper(simulation::FluidStepSettings{
            .pressureWarmStartEnabled = false,
            .pressureMeanRemovalInterval = 0,
        });
        simulation::CpuFluidStepper warmStepper(simulation::FluidStepSettings{
            .pressureWarmStartEnabled = true,
            .pressureMeanRemovalInterval = 0,
        });

        std::size_t coldIterations = 0;
        std::size_t warmIterations = 0;
        double coldMilliseconds = 0.0;
        double warmMilliseconds = 0.0;
        simulation::FluidStepReport coldReport;
        simulation::FluidStepReport warmReport;
        bool observedNonzeroRetainedPressure = false;
        bool pressureHistoryPreserved = true;
        for (int step = 0; step < 24; ++step) {
            coldReport = coldStepper.step(cold);
            const float retainedPressureMaximum = maximumAbsolutePressure(warm);
            warmReport = warmStepper.step(warm);
            if (retainedPressureMaximum > 1.0e-6f) {
                observedNonzeroRetainedPressure = true;
                pressureHistoryPreserved = pressureHistoryPreserved &&
                    std::abs(
                        warmReport.projection.pressureSolve.initialMaximumAbsolutePressure -
                        retainedPressureMaximum) <= 1.0e-5f;
            }
            coldIterations += coldReport.projection.pressureSolve.iterations;
            warmIterations += warmReport.projection.pressureSolve.iterations;
            coldMilliseconds += coldReport.timings.totalMilliseconds;
            warmMilliseconds += warmReport.timings.totalMilliseconds;
        }

        const float velocityDifference = maximumVelocityDifference(cold, warm);
        output << "  cold/warm pressure iterations: " << coldIterations << " / "
               << warmIterations << '\n'
               << "  cold/warm measured step ms: " << coldMilliseconds << " / "
               << warmMilliseconds << '\n'
               << "  cold/warm max velocity difference: " << velocityDifference << '\n';
        tests.check(
            coldReport.projection.pressureSolve.converged &&
                warmReport.projection.pressureSolve.converged,
            "cold and warm pressure solves both satisfy the unchanged tolerance");
        tests.check(
            coldReport.projection.divergenceAfter.maximumAbsolute <= 1.0e-3f &&
                warmReport.projection.divergenceAfter.maximumAbsolute <= 1.0e-3f,
            "cold and warm projection both keep maximum divergence below 1e-3");
        tests.check(
            velocityDifference <= 2.0e-4f,
            "warm start remains numerically consistent with cold start");
        tests.check(
            warmIterations <= coldIterations,
            "warm start does not increase total pressure iterations in the repeated vortex test");
        tests.check(
            warmReport.projection.pressureSolve.warmStartUsed,
            "warm-start diagnostics report the active initial-guess policy");
        tests.check(
            observedNonzeroRetainedPressure && pressureHistoryPreserved,
            "warm start retains the previous step pressure as its initial guess");
    }

    {
        simulation::MacGrid perIteration(kGridSize, kGridSize, kCellSize, kCellSize);
        simulation::MacGrid solveBoundary(kGridSize, kGridSize, kCellSize, kCellSize);
        simulation::initializeDeterministicDivergentVortex(perIteration);
        solveBoundary = perIteration;
        const simulation::PressureProjector originalGauge(simulation::PressureSolverSettings{
            .warmStartEnabled = false,
            .pressureMeanRemovalInterval = 1,
        });
        const simulation::PressureProjector boundaryGauge(simulation::PressureSolverSettings{
            .warmStartEnabled = false,
            .pressureMeanRemovalInterval = 0,
        });
        const simulation::ProjectionReport originalReport = originalGauge.project(perIteration);
        const simulation::ProjectionReport boundaryReport = boundaryGauge.project(solveBoundary);
        tests.check(
            originalReport.pressureSolve.converged && boundaryReport.pressureSolve.converged,
            "both pressure gauge normalization schedules converge");
        tests.check(
            maximumVelocityDifference(perIteration, solveBoundary) <= 1.0e-5f,
            "solve-boundary mean removal preserves the projected velocity");
        tests.check(
            std::abs(meanPressure(solveBoundary)) <= 1.0e-6f,
            "solve-boundary mean removal leaves a zero-mean pressure field");
        tests.check(
            boundaryReport.pressureSolve.pressureMeanRemovalCount <
                originalReport.pressureSolve.pressureMeanRemovalCount,
            "solve-boundary normalization removes repeated full-grid mean scans");
    }

    {
        constexpr std::size_t stabilitySize = 32;
        constexpr float stabilityCellSize = 1.0f / static_cast<float>(stabilitySize);
        simulation::MacGrid stable(stabilitySize, stabilitySize, stabilityCellSize, stabilityCellSize);
        simulation::initializeClosedVortex(stable);
        simulation::CpuFluidStepper stepper;
        bool everyStepConverged = true;
        for (int step = 0; step < 120; ++step) {
            const simulation::FluidStepReport report = stepper.step(stable);
            everyStepConverged = everyStepConverged && report.projection.pressureSolve.converged;
        }
        tests.check(stable.allFinite(), "120-step warm-start vortex remains finite");
        tests.check(everyStepConverged, "120-step warm-start vortex pressure solves all converge");
        tests.check(
            stable.divergenceStatistics().maximumAbsolute <= 1.0e-3f,
            "120-step warm-start vortex keeps bounded projected divergence");
    }

    {
        bool allResolutionsStable = true;
        bool allResolutionsConverged = true;
        for (const std::size_t gridSize : {16U, 24U, 40U}) {
            const float cellSize = 1.0f / static_cast<float>(gridSize);
            simulation::MacGrid localInput(gridSize, gridSize, cellSize, cellSize);
            localInput.horizontalVelocity(gridSize / 2, gridSize / 2) = 0.75f;
            localInput.verticalVelocity(gridSize / 2, gridSize / 2) = -0.45f;
            localInput.enforceClosedBoundary();
            simulation::CpuFluidStepper stepper;
            for (int step = 0; step < 40; ++step) {
                const simulation::FluidStepReport report = stepper.step(localInput);
                allResolutionsConverged =
                    allResolutionsConverged && report.projection.pressureSolve.converged;
                allResolutionsStable = allResolutionsStable && localInput.allFinite();
            }
        }
        tests.check(allResolutionsStable, "local velocity inputs remain finite across multiple grid sizes");
        tests.check(allResolutionsConverged, "local velocity pressure solves converge across multiple grid sizes");
    }

    {
        constexpr std::size_t limitedSize = 24;
        constexpr float limitedCellSize = 1.0f / static_cast<float>(limitedSize);
        simulation::MacGrid limited(limitedSize, limitedSize, limitedCellSize, limitedCellSize);
        simulation::initializeDeterministicDivergentVortex(limited);
        const simulation::PressureProjector limitedProjector(simulation::PressureSolverSettings{
            .maximumIterations = 20,
            .warmStartEnabled = false,
            .pressureMeanRemovalInterval = 0,
        });
        const simulation::ProjectionReport report = limitedProjector.project(limited);
        tests.check(report.pressureSolve.iterations == 20, "configured pressure iteration cap is enforced");
        tests.check(!report.pressureSolve.converged, "iteration-limited solve reports non-convergence explicitly");
        tests.check(limited.allFinite(), "iteration-limited pressure result remains finite");
    }

    output << "\nPhase 1.3 cell classification and free-surface pressure tests:\n";

    {
        simulation::MacGrid classified(10, 8, 0.1f, 0.125f);
        simulation::initializeLiquidRectangle(classified, 2, 1, 8, 4);
        classified.setCellType(1, 2, simulation::CellType::solid);
        std::size_t fluidCount = 0, airCount = 0, solidCount = 0;
        for (const simulation::CellType type : classified.cellTypes()) {
            fluidCount += type == simulation::CellType::fluid ? 1U : 0U;
            airCount += type == simulation::CellType::air ? 1U : 0U;
            solidCount += type == simulation::CellType::solid ? 1U : 0U;
        }
        tests.check(fluidCount == 18 && airCount == 61 && solidCount == 1,
            "cell classification stores fluid, air, and solid independently");
        tests.check(classified.pressures().size() == classified.cellTypes().size(),
            "cell classification matches the cell-centered grid dimensions");
    }

    {
        simulation::MacGrid boundaryGrid(8, 8, 0.125f, 0.125f);
        boundaryGrid.fillCellTypes(simulation::CellType::air);
        boundaryGrid.setCellType(3, 3, simulation::CellType::solid);
        boundaryGrid.horizontalVelocity(3, 3) = 2.0f;
        boundaryGrid.horizontalVelocity(4, 3) = -3.0f;
        boundaryGrid.verticalVelocity(3, 3) = 4.0f;
        boundaryGrid.verticalVelocity(3, 4) = -5.0f;
        boundaryGrid.enforceSolidBoundaries();
        tests.check(
            boundaryGrid.horizontalVelocity(3, 3) == 0.0f &&
            boundaryGrid.horizontalVelocity(4, 3) == 0.0f &&
            boundaryGrid.verticalVelocity(3, 3) == 0.0f &&
            boundaryGrid.verticalVelocity(3, 4) == 0.0f,
            "solid cells enforce zero normal velocity on every adjacent face");
    }

    {
        simulation::MacGrid surface(7, 7, 1.0f / 7.0f, 1.0f / 7.0f);
        surface.fillCellTypes(simulation::CellType::air);
        surface.setCellType(3, 3, simulation::CellType::fluid);
        surface.horizontalVelocity(4, 3) = 1.0f;
        const simulation::FreeSurfacePressureProjector surfaceProjector;
        const simulation::FreeSurfaceProjectionReport report = surfaceProjector.project(surface);
        output << "  single-cell free surface divergence: "
               << report.projection.divergenceBefore.maximumAbsolute << " -> "
               << report.projection.divergenceAfter.maximumAbsolute << '\n';
        tests.check(report.fluidCellCount == 1 && report.fluidAirFaceCount == 4,
            "single fluid cell identifies four atmospheric pressure faces");
        tests.check(report.pressureReferenceFixedByAir,
            "air contact supplies an absolute zero-gauge pressure reference");
        tests.check(report.projection.pressureSolve.pressureMeanRemovalCount == 0,
            "free-surface solve does not remove the pressure mean");
        tests.check(report.projection.pressureSolve.iterations > 0 &&
            report.projection.pressureSolve.converged,
            "nonzero free-surface divergence executes and converges pressure iterations");
        tests.check(report.projection.divergenceAfter.maximumAbsolute < 1.0e-3f,
            "free-surface projection removes single-cell liquid divergence");
        tests.check(surface.pressure(3, 3) != 0.0f,
            "free-surface pressure equation produces a nonzero liquid pressure");
        tests.check(surface.pressure(2, 3) == 0.0f && surface.pressure(3, 4) == 0.0f,
            "air cells retain zero gauge pressure");
        tests.check(surface.allFinite(), "free-surface projection remains finite");
    }

    {
        simulation::MacGrid closed(12, 10, 1.0f / 12.0f, 0.1f);
        simulation::initializeDeterministicDivergentVortex(closed);
        const simulation::FreeSurfacePressureProjector closedProjector;
        const simulation::FreeSurfaceProjectionReport closedReport = closedProjector.project(closed);

        simulation::MacGrid open(12, 10, 1.0f / 12.0f, 0.1f);
        simulation::initializeLiquidRectangle(open, 2, 1, 10, 5);
        open.setCellType(1, 2, simulation::CellType::solid);
        open.horizontalVelocity(6, 3) = 0.8f;
        const simulation::FreeSurfaceProjectionReport openReport = closedProjector.project(open);
        tests.check(!closedReport.pressureReferenceFixedByAir &&
            closedReport.projection.pressureSolve.pressureMeanRemovalCount > 0,
            "closed liquid uses a Neumann pressure gauge normalization");
        tests.check(closedReport.projection.pressureSolve.converged &&
            closedReport.projection.divergenceAfter.maximumAbsolute <
                0.01f * closedReport.projection.divergenceBefore.maximumAbsolute,
            "classified closed-liquid projection retains the full-domain convergence behavior");
        tests.check(openReport.pressureReferenceFixedByAir &&
            openReport.projection.pressureSolve.pressureMeanRemovalCount == 0,
            "liquid touching air uses the atmospheric Dirichlet pressure reference");
        tests.check(openReport.projection.divergenceAfter.maximumAbsolute <
            0.01f * openReport.projection.divergenceBefore.maximumAbsolute,
            "static rectangular liquid projection reduces fluid divergence by more than 99 percent");
        tests.check(open.maximumBoundaryNormalSpeed() == 0.0f && open.allFinite(),
            "free-surface corner and domain-edge handling remains bounded and impermeable");
    }

    {
        simulation::MacGrid corner(4, 4, 0.25f, 0.25f);
        corner.fillCellTypes(simulation::CellType::air);
        corner.setCellType(0, 0, simulation::CellType::fluid);
        corner.horizontalVelocity(1, 0) = 0.6f;
        const simulation::FreeSurfacePressureProjector cornerProjector;
        const simulation::FreeSurfaceProjectionReport report = cornerProjector.project(corner);
        tests.check(report.fluidAirFaceCount == 2 &&
            report.projection.pressureSolve.converged,
            "corner fluid cell distinguishes two air faces from two domain solid walls");
        tests.check(report.projection.divergenceAfter.maximumAbsolute < 1.0e-3f &&
            corner.maximumBoundaryNormalSpeed() == 0.0f && corner.allFinite(),
            "corner free-surface projection is finite, divergence-reducing, and impermeable");
    }

    {
        simulation::MacGrid warmSurface(10, 10, 0.1f, 0.1f);
        simulation::initializeLiquidRectangle(warmSurface, 2, 1, 8, 5);
        warmSurface.horizontalVelocity(5, 3) = 0.7f;
        const simulation::FreeSurfacePressureProjector warmProjector(simulation::PressureSolverSettings{
            .warmStartEnabled = true,
            .pressureMeanRemovalInterval = 0,
        });
        const simulation::FreeSurfaceProjectionReport first = warmProjector.project(warmSurface);
        const float retainedPressure = first.projection.pressureSolve.finalMaximumAbsolutePressure;
        const simulation::FreeSurfaceProjectionReport second = warmProjector.project(warmSurface);
        tests.check(retainedPressure > 0.0f &&
            std::abs(second.projection.pressureSolve.initialMaximumAbsolutePressure - retainedPressure) <= 1.0e-5f,
            "free-surface warm start retains pressure in cells that remain fluid");
        tests.check(second.projection.pressureSolve.converged && warmSurface.allFinite(),
            "free-surface warm start reconverges after the previous correction");
    }

    output << "\nPhase 1.4 dynamic free-surface tests:\n";

    {
        simulation::MacGrid still(20, 16, 0.05f, 0.05f);
        simulation::initializeDamBreak(still);
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCells(still);
        const auto initialParticles = particles.particles();
        simulation::DynamicFreeSurfaceStepper stepper(simulation::DynamicFreeSurfaceSettings{
            .gravity = {0.0f, 0.0f},
        });
        for (int step = 0; step < 30; ++step) static_cast<void>(stepper.step(still, particles));
        float maximumDisplacement = 0.0f;
        for (std::size_t i = 0; i < initialParticles.size(); ++i) {
            maximumDisplacement = std::max(maximumDisplacement, std::hypot(
                particles.particles()[i].position.x - initialParticles[i].position.x,
                particles.particles()[i].position.y - initialParticles[i].position.y));
        }
        tests.check(maximumDisplacement <= 1.0e-6f,
            "Test A: zero-gravity liquid at rest remains stationary");
        tests.check(particles.validParticleCount() == initialParticles.size(),
            "Test A/D: zero-gravity marker particle count is conserved");
    }

    {
        simulation::MacGrid column(20, 16, 0.05f, 0.05f);
        simulation::initializeDamBreak(column);
        for (std::size_t y = 1; y + 1 < column.height(); ++y) {
            for (std::size_t x = 1; x + 1 < column.width(); ++x) {
                column.setCellType(x, y, y < 6 ? simulation::CellType::fluid : simulation::CellType::air);
            }
        }
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCells(column);
        simulation::DynamicFreeSurfaceStepper stepper;
        simulation::DynamicFreeSurfaceReport report;
        for (int step = 0; step < 90; ++step) report = stepper.step(column, particles);
        output << "  hydrostatic maximum speed: " << report.maximumSpeed
               << ", divergence: " << report.projection.projection.divergenceAfter.maximumAbsolute << '\n';
        tests.check(report.maximumSpeed < 0.15f,
            "Test B: flat liquid column approaches a bounded hydrostatic state");
        tests.check(report.projection.projection.pressureSolve.converged,
            "Test B/F: hydrostatic pressure projection converges");
        tests.check(particles.allFiniteAndInside(column),
            "Test B/E: hydrostatic markers remain finite and inside solid walls");
    }

    {
        simulation::MacGrid dam(24, 18, 1.0f / 24.0f, 1.0f / 24.0f);
        simulation::initializeDamBreak(dam);
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCells(dam);
        const std::size_t initialParticleCount = particles.validParticleCount();
        const std::size_t initialFluidCells = simulation::DynamicFreeSurfaceStepper::fluidCellCount(dam);
        double initialMeanX = 0.0;
        for (const auto& particle : particles.particles()) initialMeanX += particle.position.x;
        initialMeanX /= static_cast<double>(initialParticleCount);

        simulation::DynamicFreeSurfaceStepper stepper;
        simulation::DynamicFreeSurfaceReport report;
        bool allConverged = true;
        float maximumRightwardPosition = 0.0f;
        for (int step = 0; step < 180; ++step) {
            report = stepper.step(dam, particles);
            allConverged = allConverged && report.projection.projection.pressureSolve.converged;
            for (const auto& particle : particles.particles()) {
                maximumRightwardPosition = std::max(maximumRightwardPosition, particle.position.x);
            }
        }
        double finalMeanX = 0.0;
        for (const auto& particle : particles.particles()) finalMeanX += particle.position.x;
        finalMeanX /= static_cast<double>(initialParticleCount);
        output << "  dam break mean x: " << initialMeanX << " -> " << finalMeanX
               << ", max x: " << maximumRightwardPosition
               << ", fluid cells: " << initialFluidCells << " -> " << report.fluidCellCount
               << ", particles: " << initialParticleCount << " -> " << report.particleCountAfter
               << ", max speed: " << report.maximumSpeed << '\n';
        tests.check(finalMeanX > initialMeanX + 0.08,
            "Test C: dam-break liquid center moves substantially to the right");
        tests.check(maximumRightwardPosition > 0.70f,
            "Test C: dam-break front traverses most of the container");
        tests.check(report.particleCountAfter == initialParticleCount,
            "Test D: closed-container marker particle count is conserved");
        tests.check(report.fluidCellCount != initialFluidCells,
            "Test D: particle-marked liquid area changes independently of particle count");
        tests.check(particles.allFiniteAndInside(dam) && particles.invalidPositionCount() == 0,
            "Test E/G: particles remain finite, valid, and outside solid cells");
        tests.check(allConverged &&
            report.projection.projection.divergenceAfter.maximumAbsolute < 2.0e-3f,
            "Test F: dynamic fluid pressure projections converge with bounded divergence");
        tests.check(dam.allFinite() && !report.cflLimitReached && report.maximumObservedCfl <= 0.85f,
            "Test G: long dam-break run remains finite and within the CFL substep policy");
    }

    output << "\nPhase 1.1/1.2/1.2.5/1.3/1.4 result: "
           << (tests.failures() == 0 ? "PASS" : "FAIL")
           << " (" << tests.failures() << " failed checks)\n";
    return tests.failures() == 0 ? 0 : 1;
}

} // namespace fluidlab::tests
