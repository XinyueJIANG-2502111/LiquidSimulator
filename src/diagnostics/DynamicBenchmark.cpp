#include "DynamicBenchmark.h"

#include "../simulation/mac/DynamicFreeSurfaceStepper.h"
#include "../simulation/mac/MarkerParticles.h"
#include "../simulation/mac/Scenarios.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

namespace fluidlab::diagnostics {
namespace {

using Clock = std::chrono::steady_clock;

struct Samples final {
    std::vector<double> step;
    double classification = 0.0;
    double extrapolation = 0.0;
    double advection = 0.0;
    double pressure = 0.0;
    double divergenceProjection = 0.0;
    double particles = 0.0;
    double iterations = 0.0;
    std::size_t maximumIterations = 0;
    std::size_t particlesCount = 0;
    std::size_t fluidCells = 0;
    std::size_t internalSubsteps = 0;
    std::size_t workspaceAllocations = 0;
    std::size_t failedSolves = 0;
    float maximumResidualRms = 0.0f;
    float maximumResidualAbsolute = 0.0f;
    float maximumDivergence = 0.0f;
    bool allConverged = true;
};

[[nodiscard]] double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

[[nodiscard]] double average(const std::vector<double>& values) {
    return values.empty() ? 0.0 :
        std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

[[nodiscard]] std::size_t measuredSteps(std::size_t size) noexcept {
    return size == 64 ? 30 : (size == 128 ? 12 : 4);
}

[[nodiscard]] Samples benchmarkDamBreak(
    std::size_t size,
    bool warmStart = true,
    std::size_t overrideSteps = 0) {
    const float cellSize = 1.0f / static_cast<float>(size);
    simulation::MacGrid grid(size, size, cellSize, cellSize);
    simulation::initializeDamBreak(grid);
    simulation::MarkerParticleSystem particles;
    particles.seedFluidCells(grid);
    simulation::DynamicFreeSurfaceSettings settings;
    settings.pressureWarmStartEnabled = warmStart;
    simulation::DynamicFreeSurfaceStepper stepper(settings);
    for (std::size_t i = 0; i < 2; ++i) static_cast<void>(stepper.step(grid, particles));

    Samples samples;
    const std::size_t count = overrideSteps != 0 ? overrideSteps : measuredSteps(size);
    samples.step.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const simulation::DynamicFreeSurfaceReport report = stepper.step(grid, particles);
        samples.step.push_back(report.timings.totalMilliseconds);
        samples.classification += report.timings.classificationMilliseconds;
        samples.extrapolation += report.timings.velocityExtrapolationMilliseconds;
        samples.advection += report.timings.velocityAdvectionMilliseconds;
        samples.pressure += report.timings.pressureSolveMilliseconds;
        samples.divergenceProjection += report.timings.divergenceMilliseconds +
            report.timings.pressureApplicationMilliseconds;
        samples.particles += report.timings.particleAdvectionAndCollisionMilliseconds;
        samples.iterations += static_cast<double>(report.pressureIterationsTotal);
        samples.maximumIterations = std::max(
            samples.maximumIterations, report.pressureIterationsMaximum);
        samples.internalSubsteps += report.substeps;
        samples.workspaceAllocations += report.timings.workspaceAllocations;
        samples.failedSolves += report.pressureSolveCount - report.pressureConvergedCount;
        samples.maximumResidualRms = std::max(samples.maximumResidualRms, report.maximumFinalResidualRms);
        samples.maximumResidualAbsolute = std::max(
            samples.maximumResidualAbsolute, report.maximumFinalResidualAbsolute);
        samples.maximumDivergence = std::max(
            samples.maximumDivergence, report.maximumPostProjectionDivergence);
        samples.allConverged = samples.allConverged &&
            report.pressureConvergedCount == report.pressureSolveCount;
        samples.particlesCount = report.particleCountAfter;
        samples.fluidCells = report.fluidCellCount;
    }
    return samples;
}

void printDamBreak(std::ostream& output, std::size_t size, const Samples& samples) {
    const double inverse = 1.0 / static_cast<double>(samples.step.size());
    output << std::setw(5) << size << std::setw(8) << samples.step.size()
           << std::setw(11) << samples.particlesCount << std::setw(9) << samples.fluidCells
           << std::fixed << std::setprecision(3)
           << std::setw(11) << average(samples.step)
           << std::setw(11) << percentile(samples.step, 0.95)
           << std::setw(11) << percentile(samples.step, 0.99)
           << std::setw(10) << samples.pressure * inverse
           << std::setw(10) << samples.advection * inverse
           << std::setw(10) << samples.extrapolation * inverse
           << std::setw(10) << samples.particles * inverse
           << std::setw(10) << samples.classification * inverse
           << std::setw(10) << samples.divergenceProjection * inverse
           << std::setw(10) << (samples.internalSubsteps != 0
               ? samples.iterations / static_cast<double>(samples.internalSubsteps) : 0.0)
           << std::setw(8) << samples.maximumIterations
           << std::setw(8) << static_cast<double>(samples.internalSubsteps) * inverse
           << std::setw(8) << samples.workspaceAllocations
           << std::setw(8) << (samples.allConverged ? "yes" : "no")
           << "  maxRms=" << std::scientific << samples.maximumResidualRms
           << " maxResidual=" << samples.maximumResidualAbsolute
           << " maxDiv=" << samples.maximumDivergence
           << " failedSolves=" << samples.failedSolves << '\n';
}

void runParticleScaling(std::ostream& output) {
    constexpr std::size_t size = 128;
    constexpr std::size_t repeats = 40;
    const float cellSize = 1.0f / static_cast<float>(size);
    output << "\nMarker-particle isolated diagnostic (128x128, fixed zero velocity)\n"
           << "PPC    Particles   Advect+collision ms   Classification ms\n";
    for (const std::size_t perAxis : {1U, 2U, 4U}) {
        simulation::MacGrid grid(size, size, cellSize, cellSize);
        simulation::initializeDamBreak(grid);
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCells(grid, perAxis);
        double advect = 0.0, classify = 0.0;
        for (std::size_t i = 0; i < repeats; ++i) {
            auto start = Clock::now();
            static_cast<void>(particles.advectRk2(grid, 1.0f / 120.0f));
            auto end = Clock::now();
            advect += std::chrono::duration<double, std::milli>(end - start).count();
            start = end;
            particles.rebuildFluidClassification(grid);
            end = Clock::now();
            classify += std::chrono::duration<double, std::milli>(end - start).count();
        }
        output << std::setw(3) << perAxis * perAxis << std::setw(13)
               << particles.validParticleCount() << std::fixed << std::setprecision(4)
               << std::setw(22) << advect / repeats << std::setw(20) << classify / repeats << '\n';
    }
}

void runWarmStartComparison(std::ostream& output) {
    output << "\nDynamic pressure warm-start diagnostic (64x64, six measured steps)\n"
           << "Mode    Step ms   Pressure ms   Avg iterations/solve   Max iterations   Converged\n";
    for (const bool warm : {false, true}) {
        const Samples samples = benchmarkDamBreak(64, warm, 6);
        const double inverse = 1.0 / static_cast<double>(samples.step.size());
        output << std::left << std::setw(7) << (warm ? "warm" : "cold") << std::right
               << std::fixed << std::setprecision(3)
               << std::setw(10) << average(samples.step)
               << std::setw(14) << samples.pressure * inverse
               << std::setw(23) << (samples.internalSubsteps != 0
                   ? samples.iterations / static_cast<double>(samples.internalSubsteps) : 0.0)
               << std::setw(17) << samples.maximumIterations
               << std::setw(12) << (samples.allConverged ? "yes" : "no") << '\n';
    }
}

} // namespace

int runDynamicFreeSurfaceBenchmark(std::ostream& output) {
    output << "LiquidSimulator Phase 1.4.5 dynamic CPU benchmark\n"
           << "Use Release x64. Two warmup steps are excluded. Times are milliseconds.\n"
           << "Grid Steps  Particles    Fluid   Mean step   P95 step   P99 step  Pressure Advection    Extrap Particles  Classify Div+Apply Avg iters Max iter Substep  Allocs Converge\n";
    for (const std::size_t size : {64U, 128U, 256U}) {
        printDamBreak(output, size, benchmarkDamBreak(size));
    }
    runParticleScaling(output);
    runWarmStartComparison(output);
    output << "\nCollision is intentionally grouped with particle RK2 integration because both share one stable loop.\n";
    return 0;
}

} // namespace fluidlab::diagnostics
