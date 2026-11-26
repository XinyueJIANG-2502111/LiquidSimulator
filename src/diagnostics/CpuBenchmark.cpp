#include "CpuBenchmark.h"

#include "../simulation/mac/FluidStepper.h"
#include "../simulation/mac/MacGrid.h"
#include "../simulation/mac/PressureProjection.h"
#include "../simulation/mac/Scenarios.h"
#include "../simulation/mac/VelocityAdvection.h"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <string_view>
#include <vector>

namespace fluidlab::diagnostics {
namespace {

using Clock = std::chrono::steady_clock;

struct BenchmarkResult final {
    std::size_t gridSize = 0;
    std::string_view mode;
    std::size_t measuredSteps = 0;
    double totalMilliseconds = 0.0;
    double advectionMilliseconds = 0.0;
    double pressureSolveMilliseconds = 0.0;
    double projectionOverheadMilliseconds = 0.0;
    double pressureIterations = 0.0;
    double meanRemovals = 0.0;
    float finalResidual = 0.0f;
    float finalDivergence = 0.0f;
    std::size_t convergedSteps = 0;
    std::size_t workspaceAllocations = 0;
    std::size_t workspaceBytes = 0;
};

struct Accumulator final {
    double totalMilliseconds = 0.0;
    double advectionMilliseconds = 0.0;
    double pressureSolveMilliseconds = 0.0;
    double projectionOverheadMilliseconds = 0.0;
    double pressureIterations = 0.0;
    double meanRemovals = 0.0;
    float finalResidual = 0.0f;
    float finalDivergence = 0.0f;
    std::size_t convergedSteps = 0;
};

[[nodiscard]] std::size_t measuredStepCount(std::size_t gridSize) noexcept {
    if (gridSize <= 64) {
        return 24;
    }
    if (gridSize <= 128) {
        return 10;
    }
    return 3;
}

void accumulate(Accumulator& target, const simulation::FluidStepReport& report) {
    target.totalMilliseconds += report.timings.totalMilliseconds;
    target.advectionMilliseconds += report.timings.velocityAdvectionMilliseconds;
    target.pressureSolveMilliseconds += report.projection.timings.pressureSolveMilliseconds;
    target.projectionOverheadMilliseconds +=
        report.projection.timings.totalMilliseconds - report.projection.timings.pressureSolveMilliseconds;
    target.pressureIterations += static_cast<double>(report.projection.pressureSolve.iterations);
    target.meanRemovals +=
        static_cast<double>(report.projection.pressureSolve.pressureMeanRemovalCount);
    target.finalResidual = report.projection.pressureSolve.finalResidual.rootMeanSquare;
    target.finalDivergence = report.projection.divergenceAfter.maximumAbsolute;
    target.convergedSteps += report.projection.pressureSolve.converged ? 1 : 0;
}

[[nodiscard]] BenchmarkResult finalize(
    std::size_t gridSize,
    std::string_view mode,
    std::size_t measuredSteps,
    const Accumulator& values,
    std::size_t allocations,
    std::size_t workspaceBytes) {
    const double inverseSteps = 1.0 / static_cast<double>(measuredSteps);
    return {
        gridSize,
        mode,
        measuredSteps,
        values.totalMilliseconds * inverseSteps,
        values.advectionMilliseconds * inverseSteps,
        values.pressureSolveMilliseconds * inverseSteps,
        values.projectionOverheadMilliseconds * inverseSteps,
        values.pressureIterations * inverseSteps,
        values.meanRemovals * inverseSteps,
        values.finalResidual,
        values.finalDivergence,
        values.convergedSteps,
        allocations,
        workspaceBytes,
    };
}

[[nodiscard]] simulation::MacGrid createProjectedVortex(std::size_t gridSize) {
    const float cellSize = 1.0f / static_cast<float>(gridSize);
    simulation::MacGrid grid(gridSize, gridSize, cellSize, cellSize);
    simulation::initializeClosedVortex(grid);
    const simulation::PressureProjector initialProjector;
    static_cast<void>(initialProjector.project(grid));
    return grid;
}

[[nodiscard]] BenchmarkResult runLegacyBenchmark(
    std::size_t gridSize,
    std::size_t warmupSteps,
    std::size_t measuredSteps) {
    simulation::MacGrid grid = createProjectedVortex(gridSize);
    const simulation::VelocityAdvector advector;
    const simulation::PressureProjector projector(simulation::PressureSolverSettings{
        .warmStartEnabled = false,
        .pressureMeanRemovalInterval = 1,
    });
    Accumulator measured;
    std::size_t allocationCount = 0;
    std::size_t diagnosticValueCount = 0;

    for (std::size_t step = 0; step < warmupSteps + measuredSteps; ++step) {
        const auto totalStart = Clock::now();
        static_cast<void>(simulation::CpuFluidStepper::meanKineticEnergy(grid));
        grid.enforceClosedBoundary();

        const auto advectionStart = Clock::now();
        simulation::MacGrid advected(
            grid.width(), grid.height(), grid.cellWidth(), grid.cellHeight());
        ++allocationCount;
        advector.advect(grid, advected, 1.0f / 60.0f);
        const auto advectionEnd = Clock::now();

        advected.enforceClosedBoundary();
        advected.computeDivergence();
        std::vector<float> divergenceSnapshot = advected.divergences();
        diagnosticValueCount += divergenceSnapshot.size();
        static_cast<void>(simulation::CpuFluidStepper::meanKineticEnergy(advected));
        grid = std::move(advected);
        const simulation::ProjectionReport projection = projector.project(grid);
        static_cast<void>(simulation::CpuFluidStepper::meanKineticEnergy(grid));
        const auto totalEnd = Clock::now();

        if (step >= warmupSteps) {
            measured.totalMilliseconds +=
                std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
            measured.advectionMilliseconds +=
                std::chrono::duration<double, std::milli>(advectionEnd - advectionStart).count();
            measured.pressureSolveMilliseconds += projection.timings.pressureSolveMilliseconds;
            measured.projectionOverheadMilliseconds +=
                projection.timings.totalMilliseconds - projection.timings.pressureSolveMilliseconds;
            measured.pressureIterations += static_cast<double>(projection.pressureSolve.iterations);
            measured.meanRemovals +=
                static_cast<double>(projection.pressureSolve.pressureMeanRemovalCount);
            measured.finalResidual = projection.pressureSolve.finalResidual.rootMeanSquare;
            measured.finalDivergence = projection.divergenceAfter.maximumAbsolute;
            measured.convergedSteps += projection.pressureSolve.converged ? 1 : 0;
        }
    }

    // Keep the diagnostic copy observable in optimized builds.
    if (diagnosticValueCount == 0) {
        return {};
    }
    return finalize(
        gridSize,
        "legacy-cold",
        measuredSteps,
        measured,
        allocationCount - warmupSteps,
        grid.storageBytes());
}

[[nodiscard]] BenchmarkResult runStepperBenchmark(
    std::size_t gridSize,
    std::size_t warmupSteps,
    std::size_t measuredSteps,
    bool warmStart,
    std::size_t meanRemovalInterval,
    std::string_view mode) {
    simulation::MacGrid grid = createProjectedVortex(gridSize);
    simulation::CpuFluidStepper stepper(simulation::FluidStepSettings{
        .pressureWarmStartEnabled = warmStart,
        .pressureMeanRemovalInterval = meanRemovalInterval,
    });

    for (std::size_t step = 0; step < warmupSteps; ++step) {
        static_cast<void>(stepper.step(grid));
    }
    const std::size_t allocationsBeforeMeasurement = stepper.workspaceAllocationCount();

    Accumulator measured;
    for (std::size_t step = 0; step < measuredSteps; ++step) {
        accumulate(measured, stepper.step(grid));
    }
    return finalize(
        gridSize,
        mode,
        measuredSteps,
        measured,
        stepper.workspaceAllocationCount() - allocationsBeforeMeasurement,
        stepper.workspaceBytes());
}

void printResult(std::ostream& output, const BenchmarkResult& result) {
    output << std::left << std::setw(6) << result.gridSize
           << std::setw(22) << result.mode
           << std::right << std::fixed << std::setprecision(3)
           << std::setw(10) << result.totalMilliseconds
           << std::setw(10) << result.advectionMilliseconds
           << std::setw(10) << result.pressureSolveMilliseconds
           << std::setw(10) << result.projectionOverheadMilliseconds
           << std::setw(10) << result.pressureIterations
           << std::setw(10) << result.meanRemovals
           << std::scientific << std::setprecision(3)
           << std::setw(13) << result.finalResidual
           << std::setw(13) << result.finalDivergence
           << std::fixed << std::setprecision(0)
           << std::setw(7) << result.convergedSteps << '/' << std::setw(3) << result.measuredSteps
           << std::fixed << std::setprecision(1)
           << std::setw(10) << result.workspaceAllocations
           << std::setw(12) << static_cast<double>(result.workspaceBytes) / 1024.0
           << '\n';
}

} // namespace

int runCpuBenchmark(std::ostream& output) {
    constexpr std::size_t warmupSteps = 3;
    constexpr std::size_t gridSizes[]{64, 128, 256};

    output << "LiquidSimulator Phase 1.2.5 CPU benchmark\n"
           << "Release x64 is required for meaningful performance comparisons.\n"
           << "dt=1/60, density=1, SOR=1.7, tolerance=5e-3, maxIterations=4000\n"
           << "Each case excludes three warmup steps. Allocations count measured steps only.\n\n"
           << std::left << std::setw(6) << "Grid"
           << std::setw(22) << "Mode"
           << std::right << std::setw(10) << "Step ms"
           << std::setw(10) << "Adv ms"
           << std::setw(10) << "Solve ms"
           << std::setw(10) << "Proj ms"
           << std::setw(10) << "Iters"
           << std::setw(10) << "MeanOps"
           << std::setw(13) << "Residual"
           << std::setw(13) << "Div max"
           << std::setw(11) << "Converged"
           << std::setw(10) << "Allocs"
           << std::setw(12) << "Work KiB"
           << '\n';

    for (const std::size_t gridSize : gridSizes) {
        const std::size_t measuredSteps = measuredStepCount(gridSize);
        printResult(output, runLegacyBenchmark(gridSize, warmupSteps, measuredSteps));
        printResult(output, runStepperBenchmark(
            gridSize, warmupSteps, measuredSteps, false, 1, "reuse-cold-mean1"));
        printResult(output, runStepperBenchmark(
            gridSize, warmupSteps, measuredSteps, false, 0, "reuse-cold-final"));
        printResult(output, runStepperBenchmark(
            gridSize, warmupSteps, measuredSteps, true, 0, "reuse-warm-final"));
    }
    return 0;
}

} // namespace fluidlab::diagnostics
