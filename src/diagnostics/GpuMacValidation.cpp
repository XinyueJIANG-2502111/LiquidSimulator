#include "GpuMacValidation.h"

#include "../simulation/gpu/GpuMacGrid.h"
#include "../simulation/gpu/GpuDynamicFreeSurfaceStepper.h"
#include "../simulation/mac/DynamicFreeSurfaceStepper.h"
#include "../simulation/mac/FreeSurfacePressureProjection.h"
#include "../simulation/mac/InterfaceGeometry.h"
#include "../simulation/mac/MarkerParticles.h"
#include "../simulation/mac/Scenarios.h"
#include "../simulation/mac/VelocityAdvection.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <utility>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fluidlab::diagnostics {
namespace {

using Clock = std::chrono::steady_clock;

struct Error final {
    float maximum = 0.0f;
    double rms = 0.0;
};

struct Comparison final {
    Error velocity;
    Error pressure;
};

[[nodiscard]] Error compareValues(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Comparison sizes differ.");
    Error error;
    double squared = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double difference = static_cast<double>(a[i]) - b[i];
        error.maximum = std::max(error.maximum, static_cast<float>(std::abs(difference)));
        squared += difference * difference;
    }
    error.rms = a.empty() ? 0.0 : std::sqrt(squared / static_cast<double>(a.size()));
    return error;
}

[[nodiscard]] std::vector<float> normalizedPressure(const simulation::MacGrid& grid) {
    std::vector<float> values = grid.pressures();
    std::vector<unsigned char> visited(values.size(), 0);
    const auto indexOf = [&](std::size_t x, std::size_t y) {
        return y * grid.width() + x;
    };
    for (std::size_t start = 0; start < values.size(); ++start) {
        if (visited[start] != 0 || grid.cellTypes()[start] != simulation::CellType::fluid) continue;
        std::deque<std::size_t> queue{start};
        std::vector<std::size_t> component;
        visited[start] = 1;
        bool touchesAir = false;
        while (!queue.empty()) {
            const std::size_t index = queue.front();
            queue.pop_front();
            component.push_back(index);
            const std::size_t x = index % grid.width();
            const std::size_t y = index / grid.width();
            const auto inspect = [&](std::size_t nx, std::size_t ny) {
                const simulation::CellType type = grid.cellType(nx, ny);
                if (type == simulation::CellType::air) touchesAir = true;
                if (type != simulation::CellType::fluid) return;
                const std::size_t neighbor = indexOf(nx, ny);
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    queue.push_back(neighbor);
                }
            };
            if (x > 0) inspect(x - 1, y);
            if (x + 1 < grid.width()) inspect(x + 1, y);
            if (y > 0) inspect(x, y - 1);
            if (y + 1 < grid.height()) inspect(x, y + 1);
        }
        if (touchesAir) continue;
        double sum = 0.0;
        for (const std::size_t index : component) sum += values[index];
        const float mean = static_cast<float>(sum / static_cast<double>(component.size()));
        for (const std::size_t index : component) values[index] -= mean;
    }
    return values;
}

[[nodiscard]] Comparison compareGrids(
    const simulation::MacGrid& cpu,
    const simulation::MacGrid& gpu) {
    std::vector<float> cpuVelocity = cpu.horizontalVelocities();
    cpuVelocity.insert(cpuVelocity.end(), cpu.verticalVelocities().begin(), cpu.verticalVelocities().end());
    std::vector<float> gpuVelocity = gpu.horizontalVelocities();
    gpuVelocity.insert(gpuVelocity.end(), gpu.verticalVelocities().begin(), gpu.verticalVelocities().end());
    return {compareValues(cpuVelocity, gpuVelocity),
        compareValues(normalizedPressure(cpu), normalizedPressure(gpu))};
}

void copyCellTypes(const simulation::MacGrid& source, simulation::MacGrid& destination) {
    for (std::size_t y = 0; y < source.height(); ++y)
        for (std::size_t x = 0; x < source.width(); ++x)
            destination.setCellType(x, y, source.cellType(x, y));
}

void fillSmoothField(simulation::MacGrid& grid) {
    const float domainWidth = static_cast<float>(grid.width()) * grid.cellWidth();
    const float domainHeight = static_cast<float>(grid.height()) * grid.cellHeight();
    for (std::size_t y = 0; y < grid.height(); ++y) for (std::size_t x = 0; x <= grid.width(); ++x) {
        const float px = static_cast<float>(x) * grid.cellWidth() / domainWidth;
        const float py = (static_cast<float>(y) + 0.5f) * grid.cellHeight() / domainHeight;
        grid.horizontalVelocity(x, y) = 0.08f * std::sin(6.2831853f * px) * std::cos(3.1415927f * py);
    }
    for (std::size_t y = 0; y <= grid.height(); ++y) for (std::size_t x = 0; x < grid.width(); ++x) {
        const float px = (static_cast<float>(x) + 0.5f) * grid.cellWidth() / domainWidth;
        const float py = static_cast<float>(y) * grid.cellHeight() / domainHeight;
        grid.verticalVelocity(x, y) = -0.05f * std::cos(3.1415927f * px) * std::sin(6.2831853f * py);
    }
}

void applyCpuAcceleration(simulation::MacGrid& grid, simulation::Vector2 gravity, float timeStep) {
    for (std::size_t y = 0; y < grid.height(); ++y) for (std::size_t x = 1; x < grid.width(); ++x) {
        const auto left = grid.cellType(x - 1, y), right = grid.cellType(x, y);
        if (left != simulation::CellType::solid && right != simulation::CellType::solid &&
            (left == simulation::CellType::fluid || right == simulation::CellType::fluid))
            grid.horizontalVelocity(x, y) += gravity.x * timeStep;
    }
    for (std::size_t y = 1; y < grid.height(); ++y) for (std::size_t x = 0; x < grid.width(); ++x) {
        const auto below = grid.cellType(x, y - 1), above = grid.cellType(x, y);
        if (below != simulation::CellType::solid && above != simulation::CellType::solid &&
            (below == simulation::CellType::fluid || above == simulation::CellType::fluid))
            grid.verticalVelocity(x, y) += gravity.y * timeStep;
    }
    grid.enforceSolidBoundaries();
}

[[nodiscard]] simulation::FreeSurfaceProjectionReport cpuProject(
    simulation::MacGrid& grid,
    float timeStep,
    float tolerance) {
    return simulation::FreeSurfacePressureProjector(simulation::PressureSolverSettings{
        .timeStep = timeStep,
        .residualTolerance = tolerance,
        .maximumIterations = 100000,
        .warmStartEnabled = false,
        .pressureMeanRemovalInterval = 0,
    }).project(grid);
}

[[nodiscard]] simulation::GpuPressureSettings gpuSettings(float timeStep, float tolerance) {
    return {.timeStep = timeStep, .residualTolerance = tolerance,
        .maximumIterations = 200000, .residualCheckInterval = 500,
        .warmStartEnabled = false, .collectVelocityChange = true};
}

[[nodiscard]] simulation::GpuPressureSettings pcgSettings(float timeStep, float tolerance) {
    return {.timeStep = timeStep, .residualTolerance = tolerance,
        .maximumIterations = 4000, .residualCheckInterval = 10,
        .solver = simulation::GpuPressureSolver::preconditionedConjugateGradient,
        .warmStartEnabled = false, .collectVelocityChange = true};
}

[[nodiscard]] simulation::PressureResidual legacyResidualStatistics(
    const simulation::MacGrid& grid,
    const std::vector<double>& values,
    std::size_t& validCount) {
    simulation::PressureResidual result;
    double squared = 0.0;
    validCount = 0;
    for (std::size_t index = 0; index < grid.cellTypes().size(); ++index) {
        if (grid.cellTypes()[index] != simulation::CellType::fluid) continue;
        result.maximumAbsolute = std::max(
            result.maximumAbsolute, static_cast<float>(values[index * 2]));
        squared += values[index * 2 + 1];
        ++validCount;
    }
    result.rootMeanSquare = validCount != 0
        ? static_cast<float>(std::sqrt(squared / static_cast<double>(validCount))) : 0.0f;
    return result;
}

[[nodiscard]] simulation::PressureResidual quantizedPressureResidual(
    const simulation::MacGrid& pressureGrid,
    const simulation::MacGrid& divergenceGrid,
    float density,
    float timeStep) {
    const double inverseDx2 = 1.0 /
        (static_cast<double>(pressureGrid.cellWidth()) * pressureGrid.cellWidth());
    const double inverseDy2 = 1.0 /
        (static_cast<double>(pressureGrid.cellHeight()) * pressureGrid.cellHeight());
    const double rhsScale = static_cast<double>(density) / timeStep;
    simulation::PressureResidual result;
    double squared = 0.0;
    std::size_t count = 0;
    for (std::size_t y = 0; y < pressureGrid.height(); ++y) {
        for (std::size_t x = 0; x < pressureGrid.width(); ++x) {
            if (pressureGrid.cellType(x, y) != simulation::CellType::fluid) continue;
            const double center = pressureGrid.pressure(x, y);
            double laplacian = 0.0;
            const auto add = [&](std::size_t nx, std::size_t ny, double coefficient) {
                const auto type = pressureGrid.cellType(nx, ny);
                if (type == simulation::CellType::fluid)
                    laplacian += coefficient * (pressureGrid.pressure(nx, ny) - center);
                else if (type == simulation::CellType::air) laplacian -= coefficient * center;
            };
            if (x > 0) add(x - 1, y, inverseDx2);
            if (x + 1 < pressureGrid.width()) add(x + 1, y, inverseDx2);
            if (y > 0) add(x, y - 1, inverseDy2);
            if (y + 1 < pressureGrid.height()) add(x, y + 1, inverseDy2);
            const double value = laplacian - rhsScale * divergenceGrid.divergence(x, y);
            result.maximumAbsolute = std::max(
                result.maximumAbsolute, static_cast<float>(std::abs(value)));
            squared += value * value;
            ++count;
        }
    }
    result.rootMeanSquare = count != 0
        ? static_cast<float>(std::sqrt(squared / static_cast<double>(count))) : 0.0f;
    return result;
}

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string("GPU MAC validation failed: ") + std::string(message));
}

void printProjection(
    std::ostream& output,
    std::string_view name,
    const simulation::GpuProjectionReport& report,
    const Comparison& comparison) {
    output << std::left << std::setw(22) << name << std::right
           << std::scientific << std::setprecision(3)
           << " residual=" << report.pressure.finalResidual.rootMeanSquare
           << " residualMax=" << report.pressure.finalResidual.maximumAbsolute
           << " div=" << report.divergenceBefore.maximumAbsolute << "->"
           << report.divergenceAfter.maximumAbsolute
           << " velMax=" << comparison.velocity.maximum
           << " velRms=" << comparison.velocity.rms
           << " pMax=" << comparison.pressure.maximum
           << " pRms=" << comparison.pressure.rms
           << " deltaV=" << report.maximumVelocityChange
           << std::fixed << std::setprecision(3)
           << " iter=" << report.pressure.iterations
           << " gpuMs=" << report.pressure.gpuMilliseconds + report.projectionGpuMilliseconds
           << '\n';
}

[[nodiscard]] simulation::MacGrid cpuFixedStep(
    const simulation::MacGrid& source,
    float timeStep,
    simulation::Vector2 gravity,
    float tolerance) {
    simulation::MacGrid result(source.width(), source.height(), source.cellWidth(), source.cellHeight());
    copyCellTypes(source, result);
    simulation::VelocityAdvector{}.advect(source, result, timeStep);
    applyCpuAcceleration(result, gravity, timeStep);
    static_cast<void>(cpuProject(result, timeStep, tolerance));
    return result;
}

} // namespace

int runGpuMacValidation(std::ostream& output, const std::filesystem::path& shaderDirectory) {
    constexpr float timeStep = 1.0f / 120.0f;
    constexpr float tolerance = 5.0e-3f;
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    output << "LiquidSimulator Phase 1.5 GPU MAC validation\n";

    simulation::MacGrid zero(16, 12, 1.0f / 16.0f, 1.0f / 12.0f);
    gpu.upload(zero);
    const auto zeroReport = gpu.project(gpuSettings(timeStep, tolerance));
    simulation::MacGrid zeroGpu = zero;
    gpu.download(zeroGpu);
    const Comparison zeroError = compareGrids(zero, zeroGpu);
    printProjection(output, "A zero projection", zeroReport, zeroError);
    require(zeroReport.pressure.converged && zeroReport.pressure.iterations == 0,
        "zero pressure projection should converge without iterations");
    require(zeroError.velocity.maximum <= 1.0e-7f, "zero velocity changed");

    simulation::MacGrid closed(32, 32, 1.0f / 32.0f, 1.0f / 32.0f);
    simulation::initializeDeterministicDivergentVortex(closed);
    simulation::MacGrid closedCpu = closed;
    const auto closedCpuReport = simulation::PressureProjector(simulation::PressureSolverSettings{
        .timeStep = timeStep, .residualTolerance = tolerance, .maximumIterations = 100000,
        .warmStartEnabled = false, .pressureMeanRemovalInterval = 0}).project(closedCpu);
    gpu.upload(closed);
    const auto closedGpuReport = gpu.project(gpuSettings(timeStep, tolerance));
    simulation::MacGrid closedGpu = closed;
    gpu.download(closedGpu);
    const Comparison closedError = compareGrids(closedCpu, closedGpu);
    printProjection(output, "B closed divergent", closedGpuReport, closedError);
    require(closedCpuReport.pressureSolve.converged && closedGpuReport.pressure.converged,
        "closed pressure solve did not converge");
    require(closedGpuReport.divergenceAfter.maximumAbsolute < 2.0e-3f,
        "closed projection left excessive divergence");
    require(closedError.velocity.maximum < 5.0e-4f, "closed projection velocity mismatch");

    simulation::MacGrid freeSurface(32, 24, 1.0f / 32.0f, 1.0f / 40.0f);
    simulation::initializeDamBreak(freeSurface);
    fillSmoothField(freeSurface);
    freeSurface.enforceSolidBoundaries();
    simulation::MacGrid freeCpu = freeSurface;
    const auto freeCpuReport = cpuProject(freeCpu, timeStep, tolerance);
    gpu.upload(freeSurface);
    const auto freeGpuReport = gpu.project(gpuSettings(timeStep, tolerance));
    simulation::MacGrid freeGpu = freeSurface;
    gpu.download(freeGpu);
    const Comparison freeError = compareGrids(freeCpu, freeGpu);
    printProjection(output, "C/D free surface", freeGpuReport, freeError);
    require(freeCpuReport.projection.pressureSolve.converged && freeGpuReport.pressure.converged,
        "free-surface pressure solve did not converge");
    require(freeGpu.maximumBoundaryNormalSpeed() == 0.0f, "solid wall is penetrable");
    require(freeGpuReport.divergenceAfter.maximumAbsolute < 2.0e-3f,
        "free-surface projection left excessive divergence");
    require(freeError.velocity.maximum < 7.5e-4f, "free-surface velocity mismatch");

    gpu.uploadVelocity(freeSurface);
    auto warmSettings = gpuSettings(timeStep, tolerance);
    warmSettings.warmStartEnabled = true;
    warmSettings.collectVelocityChange = false;
    const auto warmReport = gpu.project(warmSettings);
    output << "    warm-start repeat    initial=" << std::scientific
           << warmReport.pressure.initialResidual.rootMeanSquare
           << " final=" << warmReport.pressure.finalResidual.rootMeanSquare
           << std::fixed << " iter=" << warmReport.pressure.iterations << '\n';
    require(warmReport.pressure.converged &&
            warmReport.pressure.iterations <= freeGpuReport.pressure.iterations,
        "warm start did not preserve a useful pressure field");

    simulation::MacGrid advectionSource(31, 23, 0.031f, 0.047f);
    fillSmoothField(advectionSource);
    simulation::MacGrid advectionCpu(31, 23, 0.031f, 0.047f);
    simulation::VelocityAdvector{}.advect(advectionSource, advectionCpu, 0.013f);
    gpu.upload(advectionSource);
    const double advectionGpuMs = gpu.advectVelocity(0.013f);
    simulation::MacGrid advectionGpu = advectionSource;
    gpu.download(advectionGpu);
    const Comparison advectionError = compareGrids(advectionCpu, advectionGpu);
    output << "E/F non-square RK2     max=" << std::scientific << advectionError.velocity.maximum
           << " rms=" << advectionError.velocity.rms << std::fixed
           << " gpuMs=" << advectionGpuMs << '\n';
    require(advectionError.velocity.maximum < 2.0e-6f, "RK2 advection mismatch");

    simulation::MacGrid hydro(24, 18, 1.0f / 24.0f, 1.0f / 24.0f);
    simulation::initializeDamBreak(hydro);
    for (std::size_t y = 1; y + 1 < hydro.height(); ++y)
        for (std::size_t x = 1; x + 1 < hydro.width(); ++x)
            hydro.setCellType(x, y, simulation::CellType::air);
    for (std::size_t y = 1; y < 6; ++y) for (std::size_t x = 1; x + 1 < hydro.width(); ++x)
        hydro.setCellType(x, y, simulation::CellType::fluid);
    simulation::MacGrid hydroCpu = hydro;
    applyCpuAcceleration(hydroCpu, {0.0f, -9.81f}, timeStep);
    static_cast<void>(cpuProject(hydroCpu, timeStep, tolerance));
    gpu.upload(hydro);
    static_cast<void>(gpu.applyAccelerationAndBoundaries({0.0f, -9.81f}, timeStep));
    const auto hydroGpuReport = gpu.project(gpuSettings(timeStep, tolerance));
    simulation::MacGrid hydroGpu = hydro;
    gpu.download(hydroGpu);
    const Comparison hydroError = compareGrids(hydroCpu, hydroGpu);
    printProjection(output, "G hydrostatic", hydroGpuReport, hydroError);
    require(hydroGpuReport.pressure.converged && hydroError.velocity.maximum < 7.5e-4f,
        "hydrostatic projection mismatch");

    simulation::MacGrid snapshot(24, 18, 1.0f / 24.0f, 1.0f / 24.0f);
    simulation::initializeDamBreak(snapshot);
    simulation::MarkerParticleSystem markers;
    markers.seedFluidCells(snapshot);
    simulation::DynamicFreeSurfaceStepper dynamic;
    for (int i = 0; i < 12; ++i) static_cast<void>(dynamic.step(snapshot, markers));
    const simulation::MacGrid snapshotCpu = cpuFixedStep(snapshot, timeStep, {0.0f, -9.81f}, tolerance);
    gpu.upload(snapshot);
    const auto snapshotReport = gpu.stepFixedClassification(
        timeStep, {0.0f, -9.81f}, gpuSettings(timeStep, tolerance));
    simulation::MacGrid snapshotGpu = snapshot;
    gpu.download(snapshotGpu);
    const Comparison snapshotError = compareGrids(snapshotCpu, snapshotGpu);
    printProjection(output, "H dam snapshot", snapshotReport.projection, snapshotError);
    require(snapshotReport.projection.pressure.converged && snapshotError.velocity.maximum < 1.5e-3f,
        "dam-break snapshot mismatch");

    output << "\nI resolution performance (fixed classified snapshot)\n"
           << "Grid   CPUadv    CPUpress GPUadv GPUsolve GPUproj GPUdiv Upload ResidRead Endpoint Iter Residual Conv\n";
    bool allResolutionGpuSolvesConverged = true;
    for (const std::size_t size : {64U, 128U, 256U}) {
        simulation::MacGrid initial(size, size, 1.0f / static_cast<float>(size),
            1.0f / static_cast<float>(size));
        simulation::initializeDamBreak(initial);
        fillSmoothField(initial);
        initial.enforceSolidBoundaries();
        initial.computeDivergence();
        simulation::MacGrid cpu(size, size, initial.cellWidth(), initial.cellHeight());
        copyCellTypes(initial, cpu);
        auto cpuStart = Clock::now();
        simulation::VelocityAdvector{}.advect(initial, cpu, timeStep);
        const double cpuAdvectionMs =
            std::chrono::duration<double, std::milli>(Clock::now() - cpuStart).count();
        applyCpuAcceleration(cpu, {0.0f, 0.0f}, timeStep);
        const auto cpuReport = cpuProject(cpu, timeStep, tolerance);
        gpu.upload(initial);
        auto performanceSettings = gpuSettings(timeStep, tolerance);
        performanceSettings.collectVelocityChange = false;
        const auto gpuStep = gpu.stepFixedClassification(
            timeStep, {0.0f, 0.0f}, performanceSettings);
        output << std::setw(4) << size << std::fixed << std::setprecision(3)
               << std::setw(9) << cpuAdvectionMs
               << std::setw(12) << cpuReport.projection.timings.pressureSolveMilliseconds
               << std::setw(7) << gpuStep.advectionGpuMilliseconds
               << std::setw(9) << gpuStep.projection.pressure.gpuMilliseconds
               << std::setw(8) << gpuStep.projection.projectionGpuMilliseconds
               << std::setw(7) << gpuStep.projection.divergenceGpuMilliseconds
               << std::setw(7) << gpu.lastUploadMilliseconds()
               << std::setw(10) << gpuStep.projection.pressure.residualReadbackMilliseconds
               << std::setw(9) << gpuStep.totalCpuSubmissionMilliseconds
               << std::setw(6) << gpuStep.projection.pressure.iterations
               << std::scientific << std::setw(11)
               << gpuStep.projection.pressure.finalResidual.rootMeanSquare
               << std::setw(6) << (gpuStep.projection.pressure.converged ? "yes" : "no") << '\n';
        if (!cpuReport.projection.pressureSolve.converged) {
            output << "  note: CPU SOR reached its 100000-iteration validation budget.\n";
        }
        allResolutionGpuSolvesConverged = allResolutionGpuSolvesConverged &&
            gpuStep.projection.pressure.converged;
    }
    require(allResolutionGpuSolvesConverged,
        "at least one resolution did not meet the pressure tolerance");
    output << "Phase 1.5 GPU MAC result: PASS\n";
    return 0;
}

int runGpuPressureOptimizationValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    constexpr float timeStep = 1.0f / 120.0f;
    constexpr float tolerance = 5.0e-3f;
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    output << "LiquidSimulator Phase 1.5.5 GPU pressure optimization\n"
           << "Pressure equation: A=-L, b=-(rho/dt)div; tolerance RMS=5e-3.\n";

    const auto validatePcg = [&](std::string_view name, simulation::MacGrid source,
                                 float velocityTolerance) {
        source.enforceSolidBoundaries();
        source.computeDivergence();
        simulation::MacGrid cpu = source;
        const auto cpuReport = cpuProject(cpu, timeStep, tolerance);
        gpu.upload(source);
        const auto report = gpu.project(pcgSettings(timeStep, tolerance));
        simulation::MacGrid result = source;
        gpu.download(result);
        const Comparison error = compareGrids(cpu, result);
        std::size_t referenceCount = 0;
        const auto residual = legacyResidualStatistics(
            source, gpu.readResidualField(), referenceCount);
        output << std::left << std::setw(25) << name << std::right
               << std::scientific << std::setprecision(3)
               << " residual=" << report.pressure.finalResidual.rootMeanSquare
               << " cpuCheck=" << residual.rootMeanSquare
               << " div=" << report.divergenceBefore.maximumAbsolute << "->"
               << report.divergenceAfter.maximumAbsolute
               << " velErr=" << error.velocity.maximum
               << std::fixed << std::setprecision(3)
               << " iter=" << report.pressure.iterations
               << " kernelMs=" << report.pressure.gpuMilliseconds
               << " reduceMs=" << report.pressure.reductionGpuMilliseconds
               << " syncMs=" << report.pressure.residualReadbackMilliseconds
               << " bytes=" << report.pressure.residualBytesRead << '\n';
        require(cpuReport.projection.pressureSolve.converged && report.pressure.converged,
            "PCG or CPU reference did not converge");
        require(report.pressure.failureReason == simulation::GpuPressureFailureReason::none,
            "PCG reported a numerical failure");
        require(report.pressure.validEquationCount == referenceCount,
            "GPU reduction counted the wrong pressure equations");
        require(std::abs(report.pressure.finalResidual.rootMeanSquare - residual.rootMeanSquare) < 2.0e-4f,
            "GPU residual reduction disagrees with the CPU reference");
        require(error.velocity.maximum < velocityTolerance, "PCG velocity disagrees with CPU SOR");
        require(report.divergenceAfter.maximumAbsolute < 2.0e-3f,
            "PCG projection left excessive divergence");
        return report;
    };

    simulation::MacGrid zero(16, 12, 1.0f / 16.0f, 1.0f / 12.0f);
    const auto zeroReport = validatePcg("A zero velocity", zero, 1.0e-7f);
    require(zeroReport.pressure.iterations == 0, "zero field should converge before PCG iterations");

    simulation::MacGrid closed(32, 32, 1.0f / 32.0f, 1.0f / 32.0f);
    simulation::initializeDeterministicDivergentVortex(closed);
    static_cast<void>(validatePcg("B closed divergent", closed, 7.5e-4f));

    simulation::MacGrid freeSurface(32, 24, 1.0f / 32.0f, 1.0f / 40.0f);
    simulation::initializeDamBreak(freeSurface);
    fillSmoothField(freeSurface);
    static_cast<void>(validatePcg("C/F free non-square", freeSurface, 7.5e-4f));

    simulation::MacGrid hydro(24, 18, 1.0f / 24.0f, 1.0f / 24.0f);
    simulation::initializeDamBreak(hydro);
    for (std::size_t y = 1; y + 1 < hydro.height(); ++y)
        for (std::size_t x = 1; x + 1 < hydro.width(); ++x)
            hydro.setCellType(x, y, simulation::CellType::air);
    for (std::size_t y = 1; y < 6; ++y)
        for (std::size_t x = 1; x + 1 < hydro.width(); ++x)
            hydro.setCellType(x, y, simulation::CellType::fluid);
    applyCpuAcceleration(hydro, {0.0f, -9.81f}, timeStep);
    static_cast<void>(validatePcg("D hydrostatic", hydro, 7.5e-4f));

    simulation::MacGrid snapshot(24, 18, 1.0f / 24.0f, 1.0f / 24.0f);
    simulation::initializeDamBreak(snapshot);
    simulation::MarkerParticleSystem markers;
    markers.seedFluidCells(snapshot);
    simulation::DynamicFreeSurfaceStepper dynamic;
    for (int i = 0; i < 12; ++i) static_cast<void>(dynamic.step(snapshot, markers));
    static_cast<void>(validatePcg("E dam-break snapshot", snapshot, 1.5e-3f));

    simulation::MacGrid disconnected(32, 20, 1.0f / 32.0f, 1.0f / 25.0f);
    for (std::size_t y = 0; y < disconnected.height(); ++y)
        for (std::size_t x = 0; x < disconnected.width(); ++x)
            disconnected.setCellType(x, y, simulation::CellType::air);
    for (std::size_t x = 0; x < disconnected.width(); ++x) {
        disconnected.setCellType(x, 0, simulation::CellType::solid);
        disconnected.setCellType(x, disconnected.height() - 1, simulation::CellType::solid);
    }
    for (std::size_t y = 0; y < disconnected.height(); ++y) {
        disconnected.setCellType(0, y, simulation::CellType::solid);
        disconnected.setCellType(disconnected.width() - 1, y, simulation::CellType::solid);
    }
    for (std::size_t y = 3; y <= 8; ++y)
        for (std::size_t x = 3; x <= 9; ++x)
            disconnected.setCellType(x, y, simulation::CellType::fluid);
    for (std::size_t y = 5; y <= 12; ++y)
        for (std::size_t x = 19; x <= 27; ++x)
            disconnected.setCellType(x, y, simulation::CellType::solid);
    for (std::size_t y = 6; y <= 11; ++y)
        for (std::size_t x = 20; x <= 26; ++x)
            disconnected.setCellType(x, y, simulation::CellType::fluid);
    disconnected.horizontalVelocity(6, 5) = 0.12f;
    disconnected.horizontalVelocity(23, 8) = -0.09f;
    static_cast<void>(validatePcg("G mixed components", disconnected, 1.5e-3f));

    simulation::MacGrid unsupported(20, 12, 0.05f, 0.05f);
    for (std::size_t y = 0; y < unsupported.height(); ++y)
        for (std::size_t x = 0; x < unsupported.width(); ++x)
            unsupported.setCellType(x, y, simulation::CellType::solid);
    for (std::size_t y = 2; y <= 4; ++y) for (std::size_t x = 2; x <= 4; ++x)
        unsupported.setCellType(x, y, simulation::CellType::fluid);
    for (std::size_t y = 7; y <= 9; ++y) for (std::size_t x = 14; x <= 16; ++x)
        unsupported.setCellType(x, y, simulation::CellType::fluid);
    gpu.upload(unsupported);
    const auto unsupportedReport = gpu.project(pcgSettings(timeStep, tolerance));
    require(!unsupportedReport.pressure.converged &&
            unsupportedReport.pressure.failureReason ==
                simulation::GpuPressureFailureReason::unsupportedMultiplePureNeumannComponents,
        "multiple closed Neumann components were not rejected explicitly");
    output << "G unsupported topology: explicitly rejected multiple pure-Neumann components.\n";

    output << "\nH pressure performance (single deterministic run, Release x64)\n"
           << "Grid CPU-SORms CPUconv JacIter JacKernel JacReduce JacSync JacEnd PCGIter PCGKernel PCGReduce PCGSync PCGEnd StepEnd Residual FP32Quant\n";
    for (const std::size_t size : {64U, 128U, 256U}) {
        simulation::MacGrid initial(size, size, 1.0f / static_cast<float>(size),
            1.0f / static_cast<float>(size));
        simulation::initializeDamBreak(initial);
        fillSmoothField(initial);
        initial.enforceSolidBoundaries();
        initial.computeDivergence();
        simulation::MacGrid cpu = initial;
        const auto cpuReport = cpuProject(cpu, timeStep, tolerance);

        auto jacobi = gpuSettings(timeStep, tolerance);
        jacobi.collectVelocityChange = false;
        gpu.upload(initial);
        const auto jacobiStart = Clock::now();
        const auto jacobiReport = gpu.project(jacobi);
        const double jacobiEndToEnd =
            std::chrono::duration<double, std::milli>(Clock::now() - jacobiStart).count();

        auto pcg = pcgSettings(timeStep, tolerance);
        pcg.collectVelocityChange = false;
        gpu.upload(initial);
        const auto pcgStart = Clock::now();
        const auto pcgReport = gpu.project(pcg);
        const double pcgEndToEnd =
            std::chrono::duration<double, std::milli>(Clock::now() - pcgStart).count();
        simulation::MacGrid quantized = initial;
        gpu.download(quantized);
        const auto quantizedResidual = quantizedPressureResidual(
            quantized, initial, 1.0f, timeStep);
        gpu.upload(initial);
        const auto pcgStep = gpu.stepFixedClassification(
            timeStep, {0.0f, 0.0f}, pcg);
        output << std::setw(4) << size << std::fixed << std::setprecision(3)
               << std::setw(10) << cpuReport.projection.timings.pressureSolveMilliseconds
               << std::setw(8) << (cpuReport.projection.pressureSolve.converged ? "yes" : "no")
               << std::setw(8) << jacobiReport.pressure.iterations
               << std::setw(10) << jacobiReport.pressure.gpuMilliseconds
               << std::setw(10) << jacobiReport.pressure.reductionGpuMilliseconds
               << std::setw(9) << jacobiReport.pressure.residualReadbackMilliseconds
               << std::setw(10) << jacobiEndToEnd
               << std::setw(8) << pcgReport.pressure.iterations
               << std::setw(10) << pcgReport.pressure.gpuMilliseconds
               << std::setw(10) << pcgReport.pressure.reductionGpuMilliseconds
               << std::setw(8) << pcgReport.pressure.residualReadbackMilliseconds
               << std::setw(8) << pcgEndToEnd
               << std::setw(8) << pcgStep.totalCpuSubmissionMilliseconds
               << std::scientific << std::setprecision(3) << std::setw(11)
               << pcgReport.pressure.finalResidual.rootMeanSquare
               << std::setw(11) << quantizedResidual.rootMeanSquare << '\n';
        require(jacobiReport.pressure.converged && pcgReport.pressure.converged,
            "resolution pressure benchmark did not converge");
        require(pcgStep.projection.pressure.converged,
            "complete PCG grid step did not converge");
        std::vector<double> repeatedPcgMilliseconds;
        for (int sample = 0; sample < 5; ++sample) {
            gpu.upload(initial);
            const auto start = Clock::now();
            const auto repeated = gpu.project(pcg);
            repeatedPcgMilliseconds.push_back(
                std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            require(repeated.pressure.converged, "repeated PCG timing run did not converge");
        }
        std::sort(repeatedPcgMilliseconds.begin(), repeatedPcgMilliseconds.end());
        const double mean = std::accumulate(
            repeatedPcgMilliseconds.begin(), repeatedPcgMilliseconds.end(), 0.0) /
            repeatedPcgMilliseconds.size();
        output << "     PCG endpoint 5-run mean/P95/max ms: " << std::fixed
               << std::setprecision(3) << mean << " / "
               << repeatedPcgMilliseconds.back() << " / "
               << repeatedPcgMilliseconds.back() << '\n';
    }

    output << "Precision result: FP64 PCG validated. Existing FP32 Jacobi stalled near 8.4e-3 at 256^2;\n"
           << "FP32 PCG and iterative refinement are not implemented and are not claimed as tested.\n"
           << "Phase 1.5.5 GPU pressure optimization result: PASS\n";
    return 0;
}

int runGpuMultigridValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    constexpr float timeStep = 1.0f / 120.0f;
    constexpr float tolerance = 5.0e-3f;
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    const auto settingsFor = [&](simulation::GpuPressureSolver solver,
                                 std::size_t checkInterval = 10) {
        simulation::GpuPressureSettings settings = pcgSettings(timeStep, tolerance);
        settings.solver = solver;
        settings.residualCheckInterval = checkInterval;
        settings.multigridPreSmoothIterations = 2;
        settings.multigridPostSmoothIterations = 2;
        settings.multigridCoarseIterations = 40;
        settings.multigridRelaxationFactor = 0.8f;
        return settings;
    };
    const auto diagonalSettings = settingsFor(
        simulation::GpuPressureSolver::preconditionedConjugateGradient);
    const auto multigridSettings = settingsFor(
        simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient);

    output << "LiquidSimulator Phase 1.5.6 GPU MGPCG validation\n"
           << "Galerkin aggregation: R=P^T, piecewise-constant P, symmetric weighted-Jacobi smoothing.\n";

    const auto validate = [&](std::string_view name, simulation::MacGrid source,
                              float velocityTolerance) {
        source.enforceSolidBoundaries();
        source.computeDivergence();
        simulation::MacGrid cpu = source;
        const auto cpuReport = cpuProject(cpu, timeStep, tolerance);

        gpu.upload(source);
        gpu.dispatchDivergence();
        static_cast<void>(gpu.readDivergence());
        const auto cycle = gpu.testMultigridCycle(multigridSettings);

        gpu.upload(source);
        const auto diagonal = gpu.project(diagonalSettings);
        simulation::MacGrid diagonalGrid = source;
        gpu.download(diagonalGrid);

        gpu.upload(source);
        const auto multigrid = gpu.project(multigridSettings);
        simulation::MacGrid multigridGrid = source;
        gpu.download(multigridGrid);
        const Comparison cpuError = compareGrids(cpu, multigridGrid);
        const Comparison solverError = compareGrids(diagonalGrid, multigridGrid);

        output << std::left << std::setw(27) << name << std::right
               << std::scientific << std::setprecision(3)
               << " V=" << cycle.initialRootMeanSquare << "->"
               << cycle.correctedRootMeanSquare
               << " residual=" << multigrid.pressure.finalResidual.rootMeanSquare
               << " div=" << multigrid.divergenceBefore.maximumAbsolute << "->"
               << multigrid.divergenceAfter.maximumAbsolute
               << " velCpu=" << cpuError.velocity.maximum
               << "/" << cpuError.velocity.rms
               << " velDiag=" << solverError.velocity.maximum
               << "/" << solverError.velocity.rms
               << " pCpu=" << cpuError.pressure.maximum
               << "/" << cpuError.pressure.rms
               << std::fixed << std::setprecision(3)
               << " iter=" << diagonal.pressure.iterations << "->"
               << multigrid.pressure.iterations
               << " levels=" << multigrid.pressure.multigridLevelCount
               << " Vms=" << cycle.gpuMilliseconds << '\n';

        require(cpuReport.projection.pressureSolve.converged,
            "CPU reference did not converge in MGPCG validation");
        require(diagonal.pressure.converged && multigrid.pressure.converged,
            "diagonal PCG or MGPCG did not converge");
        require(multigrid.pressure.failureReason == simulation::GpuPressureFailureReason::none,
            "MGPCG reported a numerical failure");
        require(std::isfinite(multigrid.pressure.finalResidual.rootMeanSquare) &&
                std::isfinite(multigrid.pressure.finalResidual.maximumAbsolute) &&
                std::isfinite(multigrid.divergenceAfter.maximumAbsolute),
            "MGPCG produced a non-finite diagnostic");
        if (cycle.initialRootMeanSquare > tolerance) {
            require(cycle.correctedRootMeanSquare < cycle.initialRootMeanSquare,
                "one V-cycle did not reduce the pressure error residual");
        }
        require(cpuError.velocity.maximum < velocityTolerance,
            "MGPCG velocity disagrees with CPU SOR");
        require(multigrid.divergenceAfter.maximumAbsolute < 2.5e-3f,
            "MGPCG projection left excessive divergence");
        return multigrid;
    };

    simulation::MacGrid zero(16, 12, 1.0f / 16.0f, 1.0f / 12.0f);
    static_cast<void>(validate("1 zero velocity", zero, 1.0e-7f));

    simulation::MacGrid closed(32, 32, 1.0f / 32.0f, 1.0f / 32.0f);
    simulation::initializeDeterministicDivergentVortex(closed);
    static_cast<void>(validate("2/3 closed regular fluid", closed, 8.0e-4f));

    simulation::MacGrid freeSurface(32, 24, 1.0f / 32.0f, 1.0f / 40.0f);
    simulation::initializeDamBreak(freeSurface);
    fillSmoothField(freeSurface);
    static_cast<void>(validate("4/8 AIR non-square", freeSurface, 8.0e-4f));

    simulation::MacGrid irregular(36, 28, 1.0f / 36.0f, 1.0f / 32.0f);
    simulation::initializeDamBreak(irregular);
    for (std::size_t y = 4; y <= 12; ++y)
        for (std::size_t x = 10; x <= 12; ++x)
            irregular.setCellType(x, y, simulation::CellType::solid);
    fillSmoothField(irregular);
    static_cast<void>(validate("5 irregular SOLID", irregular, 1.2e-3f));

    simulation::MacGrid hydro(24, 18, 1.0f / 24.0f, 1.0f / 24.0f);
    simulation::initializeDamBreak(hydro);
    for (std::size_t y = 1; y + 1 < hydro.height(); ++y)
        for (std::size_t x = 1; x + 1 < hydro.width(); ++x)
            hydro.setCellType(x, y, simulation::CellType::air);
    for (std::size_t y = 1; y < 6; ++y)
        for (std::size_t x = 1; x + 1 < hydro.width(); ++x)
            hydro.setCellType(x, y, simulation::CellType::fluid);
    applyCpuAcceleration(hydro, {0.0f, -9.81f}, timeStep);
    static_cast<void>(validate("6 hydrostatic", hydro, 8.0e-4f));

    simulation::MacGrid snapshot(24, 18, 1.0f / 24.0f, 1.0f / 24.0f);
    simulation::initializeDamBreak(snapshot);
    simulation::MarkerParticleSystem markers;
    markers.seedFluidCells(snapshot);
    simulation::DynamicFreeSurfaceStepper dynamic;
    for (int i = 0; i < 12; ++i) static_cast<void>(dynamic.step(snapshot, markers));
    static_cast<void>(validate("7 dam-break snapshot", snapshot, 1.5e-3f));

    simulation::MacGrid mixed(40, 24, 1.0f / 40.0f, 1.0f / 30.0f);
    for (std::size_t y = 0; y < mixed.height(); ++y)
        for (std::size_t x = 0; x < mixed.width(); ++x)
            mixed.setCellType(x, y, simulation::CellType::air);
    for (std::size_t y = 4; y <= 12; ++y)
        for (std::size_t x = 3; x <= 11; ++x)
            mixed.setCellType(x, y, simulation::CellType::solid);
    for (std::size_t y = 5; y <= 11; ++y)
        for (std::size_t x = 4; x <= 10; ++x)
            mixed.setCellType(x, y, simulation::CellType::fluid);
    for (std::size_t y = 2; y <= 8; ++y)
        for (std::size_t x = 24; x <= 34; ++x)
            mixed.setCellType(x, y, simulation::CellType::fluid);
    fillSmoothField(mixed);
    static_cast<void>(validate("9/10 closed plus AIR", mixed, 1.8e-3f));

    simulation::MacGrid unsupported(20, 12, 0.05f, 0.05f);
    for (std::size_t y = 0; y < unsupported.height(); ++y)
        for (std::size_t x = 0; x < unsupported.width(); ++x)
            unsupported.setCellType(x, y, simulation::CellType::solid);
    for (std::size_t y = 2; y <= 4; ++y)
        for (std::size_t x = 2; x <= 4; ++x)
            unsupported.setCellType(x, y, simulation::CellType::fluid);
    for (std::size_t y = 7; y <= 9; ++y)
        for (std::size_t x = 14; x <= 16; ++x)
            unsupported.setCellType(x, y, simulation::CellType::fluid);
    gpu.upload(unsupported);
    const auto unsupportedReport = gpu.project(multigridSettings);
    require(!unsupportedReport.pressure.converged &&
            unsupportedReport.pressure.failureReason ==
                simulation::GpuPressureFailureReason::unsupportedMultiplePureNeumannComponents,
        "MGPCG did not preserve the multiple-Neumann-component topology guard");
    output << "Unsupported topology: multiple pure-Neumann components explicitly rejected.\n";

    output << "\nResidual-check interval comparison at 128x128\n"
           << "Interval Iter KernelMs ReduceMs SyncMs EndMs RMS\n";
    simulation::MacGrid intervalGrid(128, 128, 1.0f / 128.0f, 1.0f / 128.0f);
    simulation::initializeDamBreak(intervalGrid);
    fillSmoothField(intervalGrid);
    for (const std::size_t interval : {10U, 20U, 40U}) {
        auto settings = settingsFor(
            simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient,
            interval);
        gpu.upload(intervalGrid);
        const auto start = Clock::now();
        const auto report = gpu.project(settings);
        const double endMs = std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
        output << std::setw(8) << interval << std::setw(6) << report.pressure.iterations
               << std::fixed << std::setprecision(3)
               << std::setw(10) << report.pressure.gpuMilliseconds
               << std::setw(9) << report.pressure.reductionGpuMilliseconds
               << std::setw(8) << report.pressure.residualReadbackMilliseconds
               << std::setw(8) << endMs << std::scientific << std::setprecision(3)
               << ' ' << report.pressure.finalResidual.rootMeanSquare << '\n';
        require(report.pressure.converged, "MGPCG interval comparison did not converge");
    }

    output << "\nDiagonal PCG versus MGPCG (Release x64, deterministic fixed classification)\n"
           << "Grid DiagIter DiagKernel DiagReduce DiagSync DiagEnd DiagStep MGIter MGKernel MGPrec MGReduce MGSync MGEnd MGStep Levels MemoryMiB RMS MaxResidual PostDiv\n";
    for (const std::size_t size : {64U, 128U, 256U}) {
        simulation::MacGrid initial(size, size, 1.0f / static_cast<float>(size),
            1.0f / static_cast<float>(size));
        simulation::initializeDamBreak(initial);
        fillSmoothField(initial);
        initial.enforceSolidBoundaries();

        gpu.upload(initial);
        const auto diagonalStart = Clock::now();
        const auto diagonal = gpu.project(diagonalSettings);
        const double diagonalEnd = std::chrono::duration<double, std::milli>(
            Clock::now() - diagonalStart).count();
        gpu.upload(initial);
        const auto diagonalStep = gpu.stepFixedClassification(
            timeStep, {0.0f, 0.0f}, diagonalSettings);

        gpu.upload(initial);
        const auto multigridStart = Clock::now();
        const auto multigrid = gpu.project(multigridSettings);
        const double multigridEnd = std::chrono::duration<double, std::milli>(
            Clock::now() - multigridStart).count();

        gpu.upload(initial);
        const auto step = gpu.stepFixedClassification(
            timeStep, {0.0f, 0.0f}, multigridSettings);
        output << std::setw(4) << size << std::setw(9) << diagonal.pressure.iterations
               << std::fixed << std::setprecision(3)
               << std::setw(11) << diagonal.pressure.gpuMilliseconds
               << std::setw(11) << diagonal.pressure.reductionGpuMilliseconds
               << std::setw(9) << diagonal.pressure.residualReadbackMilliseconds
               << std::setw(8) << diagonalEnd
               << std::setw(9) << diagonalStep.totalCpuSubmissionMilliseconds
               << std::setw(7) << multigrid.pressure.iterations
               << std::setw(10) << multigrid.pressure.gpuMilliseconds
               << std::setw(8) << multigrid.pressure.preconditionerGpuMilliseconds
               << std::setw(9) << multigrid.pressure.reductionGpuMilliseconds
               << std::setw(7) << multigrid.pressure.residualReadbackMilliseconds
               << std::setw(8) << multigridEnd
               << std::setw(8) << step.totalCpuSubmissionMilliseconds
               << std::setw(7) << multigrid.pressure.multigridLevelCount
               << std::setw(10) << (static_cast<double>(multigrid.pressure.multigridMemoryBytes) /
                    (1024.0 * 1024.0))
               << std::scientific << std::setprecision(3)
               << ' ' << multigrid.pressure.finalResidual.rootMeanSquare
               << ' ' << multigrid.pressure.finalResidual.maximumAbsolute
               << ' ' << multigrid.divergenceAfter.maximumAbsolute << '\n';
        require(diagonal.pressure.converged && multigrid.pressure.converged &&
                diagonalStep.projection.pressure.converged &&
                step.projection.pressure.converged,
            "MGPCG resolution benchmark did not converge");
        if (size == 256U) {
            std::vector<double> repeated;
            for (int sample = 0; sample < 5; ++sample) {
                gpu.upload(initial);
                const auto start = Clock::now();
                const auto run = gpu.project(multigridSettings);
                repeated.push_back(std::chrono::duration<double, std::milli>(
                    Clock::now() - start).count());
                require(run.pressure.converged, "repeated 256 MGPCG solve did not converge");
            }
            std::sort(repeated.begin(), repeated.end());
            const double mean = std::accumulate(repeated.begin(), repeated.end(), 0.0) /
                static_cast<double>(repeated.size());
            output << "     256 MGPCG endpoint 5-run mean/P95/max ms: "
                   << std::fixed << std::setprecision(3) << mean << " / "
                   << repeated.back() << " / " << repeated.back() << '\n';
        }
    }

    output << "Phase 1.5.6 GPU MGPCG result: PASS\n";
    return 0;
}

int runGpuDynamicValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    simulation::MacGrid seeded(48, 32, 1.0f / 48.0f, 1.0f / 48.0f);
    simulation::initializeDamBreak(seeded);
    simulation::MarkerParticleSystem particles;
    particles.seedFluidCells(seeded);

    simulation::MacGrid initial = seeded;
    for (std::size_t y = 0; y < initial.height(); ++y) {
        for (std::size_t x = 0; x < initial.width(); ++x) {
            if (initial.cellType(x, y) != simulation::CellType::solid) {
                initial.setCellType(x, y, simulation::CellType::air);
            }
        }
    }
    simulation::MacGrid expected = initial;
    particles.rebuildFluidClassification(expected);

    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    gpu.upload(initial);
    gpu.uploadMarkerParticles(particles);
    const auto classification = gpu.rebuildClassificationFromParticles(false);
    const auto gpuTypes = gpu.readCellTypes();
    require(gpuTypes == expected.cellTypes(),
        "GPU particle classification disagrees with the CPU reference");
    require(classification.particleCount == particles.particles().size() &&
            classification.validParticleCount == particles.validParticleCount() &&
            classification.invalidParticleCount == 0,
        "GPU particle classification counters are incorrect");
    require(classification.fluidCellCount ==
            simulation::DynamicFreeSurfaceStepper::fluidCellCount(expected),
        "GPU classification counted the wrong number of fluid cells");

    auto pressureSettings = pcgSettings(1.0f / 120.0f, 5.0e-3f);
    pressureSettings.solver =
        simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
    const auto stale = gpu.solvePressure(pressureSettings);
    require(stale.failureReason ==
            simulation::GpuPressureFailureReason::dynamicPressureTopologyStale,
        "GPU pressure solve did not reject a stale dynamic operator");

    gpu.synchronizeDynamicPressureTopology();
    gpu.dispatchDivergence();
    static_cast<void>(gpu.readDivergence());
    const auto synchronized = gpu.solvePressure(pressureSettings);
    require(synchronized.converged && synchronized.iterations == 0,
        "synchronized zero-velocity dynamic pressure state did not converge immediately");

    const auto repeat = gpu.rebuildClassificationFromParticles(true);
    require(repeat.changedCellCount == 0,
        "repeated particle classification changed a stable cell map");
    const auto repeatedTypes = gpu.readCellTypes();
    require(repeatedTypes == expected.cellTypes(),
        "repeated GPU classification did not preserve solid and fluid cells");

    const auto compareParticles = [&](const simulation::MarkerParticleSystem& cpuParticles) {
        const auto gpuParticles = gpu.readMarkerParticles();
        require(gpuParticles.size() == cpuParticles.particles().size(),
            "GPU particle count changed unexpectedly");
        float maximumError = 0.0f;
        for (std::size_t i = 0; i < gpuParticles.size(); ++i) {
            maximumError = std::max(maximumError,
                std::hypot(gpuParticles[i].position.x - cpuParticles.particles()[i].position.x,
                    gpuParticles[i].position.y - cpuParticles.particles()[i].position.y));
        }
        return maximumError;
    };

    simulation::MarkerParticleSystem zeroParticles = particles;
    gpu.uploadMarkerParticles(particles);
    const auto zeroAdvection = gpu.advectMarkerParticles(1.0f / 120.0f);
    require(compareParticles(zeroParticles) < 1.0e-7f,
        "zero velocity moved GPU marker particles");

    simulation::MacGrid uniform = expected;
    for (std::size_t y = 0; y < uniform.height(); ++y)
        for (std::size_t x = 0; x <= uniform.width(); ++x)
            uniform.horizontalVelocity(x, y) = 0.12f;
    for (std::size_t y = 0; y <= uniform.height(); ++y)
        for (std::size_t x = 0; x < uniform.width(); ++x)
            uniform.verticalVelocity(x, y) = 0.04f;
    uniform.enforceSolidBoundaries();
    simulation::MarkerParticleSystem uniformCpu = particles;
    const auto uniformCpuReport = uniformCpu.advectRk2(uniform, 1.0f / 120.0f);
    gpu.uploadVelocity(uniform);
    gpu.uploadMarkerParticles(particles);
    const auto uniformGpuReport = gpu.advectMarkerParticles(1.0f / 120.0f);
    const float uniformError = compareParticles(uniformCpu);
    require(uniformError < 2.0e-6f &&
            uniformGpuReport.particles.collisionCount == uniformCpuReport.collisionCount,
        "uniform-field GPU particle RK2 disagrees with the CPU reference");

    simulation::MacGrid vortex = expected;
    fillSmoothField(vortex);
    vortex.enforceSolidBoundaries();
    simulation::MarkerParticleSystem vortexCpu = particles;
    static_cast<void>(vortexCpu.advectRk2(vortex, 1.0f / 120.0f));
    gpu.uploadVelocity(vortex);
    gpu.uploadMarkerParticles(particles);
    const auto vortexGpuReport = gpu.advectMarkerParticles(1.0f / 120.0f);
    const float vortexError = compareParticles(vortexCpu);
    require(vortexError < 2.0e-6f && vortexGpuReport.particles.invalidPositionCount == 0,
        "vortex-field GPU particle RK2 disagrees with the CPU reference");

    simulation::MacGrid extrapolationInput = expected;
    for (std::size_t y = 0; y < extrapolationInput.height(); ++y) {
        for (std::size_t x = 1; x < extrapolationInput.width(); ++x) {
            const auto left = extrapolationInput.cellType(x - 1, y);
            const auto right = extrapolationInput.cellType(x, y);
            if (left != simulation::CellType::solid && right != simulation::CellType::solid &&
                (left == simulation::CellType::fluid || right == simulation::CellType::fluid)) {
                extrapolationInput.horizontalVelocity(x, y) =
                    0.01f * static_cast<float>(x + 2 * y);
            }
        }
    }
    for (std::size_t y = 1; y < extrapolationInput.height(); ++y) {
        for (std::size_t x = 0; x < extrapolationInput.width(); ++x) {
            const auto below = extrapolationInput.cellType(x, y - 1);
            const auto above = extrapolationInput.cellType(x, y);
            if (below != simulation::CellType::solid && above != simulation::CellType::solid &&
                (below == simulation::CellType::fluid || above == simulation::CellType::fluid)) {
                extrapolationInput.verticalVelocity(x, y) =
                    -0.008f * static_cast<float>(2 * x + y);
            }
        }
    }
    simulation::MacGrid extrapolationCpu = extrapolationInput;
    simulation::VelocityExtrapolator extrapolator;
    extrapolator.extrapolate(extrapolationCpu, 3);
    gpu.uploadVelocity(extrapolationInput);
    const double extrapolationGpuMs = gpu.extrapolateVelocity(3);
    const Error uExtrapolationError = compareValues(
        extrapolationCpu.horizontalVelocities(), gpu.readHorizontalVelocity());
    const Error vExtrapolationError = compareValues(
        extrapolationCpu.verticalVelocities(), gpu.readVerticalVelocity());
    require(uExtrapolationError.maximum < 1.0e-6f &&
            vExtrapolationError.maximum < 1.0e-6f,
        "GPU layered velocity extrapolation disagrees with the CPU reference");

    output << "LiquidSimulator Phase 1.6A dynamic classification foundation\n"
           << "  particles total/valid/invalid: " << classification.particleCount << " / "
           << classification.validParticleCount << " / "
           << classification.invalidParticleCount << '\n'
           << "  fluid cells: " << classification.fluidCellCount
           << ", changed cells: " << classification.changedCellCount << '\n'
           << std::fixed << std::setprecision(3)
           << "  classification GPU/readback ms: " << classification.gpuMilliseconds
           << " / " << classification.readbackMilliseconds << '\n'
           << "  diagnostic bytes read: " << classification.bytesRead << '\n'
           << "  stale pressure operator: explicitly rejected\n"
           << "  synchronized zero-field pressure iterations: "
           << synchronized.iterations << '\n'
           << std::scientific << std::setprecision(3)
           << "  particle RK2 uniform/vortex max error: " << uniformError
           << " / " << vortexError << '\n'
           << std::fixed << std::setprecision(3)
           << "  particle GPU ms (zero/uniform/vortex): "
           << zeroAdvection.gpuMilliseconds << " / "
           << uniformGpuReport.gpuMilliseconds << " / "
           << vortexGpuReport.gpuMilliseconds << '\n'
           << std::scientific << std::setprecision(3)
           << "  extrapolation u/v max error: " << uExtrapolationError.maximum
           << " / " << vExtrapolationError.maximum << '\n'
           << std::fixed << std::setprecision(3)
           << "  extrapolation GPU ms: " << extrapolationGpuMs << '\n'
           << "Phase 1.6A foundation result: PASS\n";
    return 0;
}

int runGpuDynamicTopologyValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    output << "LiquidSimulator Phase 1.6A-2 GPU dynamic topology validation\n"
           << "Grid FineMax WeightMax ActionMax Levels FineGPU TopologyGPU CoarseGPU MGPCGGPU DynamicGPU SyncCPU Bytes PostDiv\n";

    for (const auto [width, height] : std::array<std::pair<std::size_t, std::size_t>, 5>{
             std::pair{48U, 32U}, std::pair{64U, 64U}, std::pair{128U, 128U},
             std::pair{256U, 256U}, std::pair{63U, 37U}}) {
        simulation::MacGrid grid(width, height, 1.0f / static_cast<float>(width),
            1.0f / static_cast<float>(width));
        simulation::initializeDamBreak(grid);
        fillSmoothField(grid);
        grid.enforceSolidBoundaries();
        gpu.upload(grid);

        std::vector<simulation::GpuPressureOperatorSnapshot> cpuBuilt;
        for (std::size_t level = 0;; ++level) {
            try {
                cpuBuilt.push_back(gpu.readPressureOperatorLevel(level));
            } catch (const std::out_of_range&) {
                break;
            }
        }
        const auto topology = gpu.rebuildDynamicPressureTopology();
        require(topology.fineOperatorValid && topology.topologyValid &&
                topology.multigridValid && !topology.usedFullClassificationReadback,
            "GPU dynamic topology build did not produce a valid hierarchy");
        require(topology.classificationVersion == topology.fineOperatorVersion &&
                topology.classificationVersion == topology.multigridVersion,
            "GPU dynamic pressure operator versions are inconsistent");

        double maximumDiagonalError = 0.0;
        double maximumWeightError = 0.0;
        double maximumActionError = 0.0;
        require(topology.activeMultigridLevelCount == cpuBuilt.size(),
            "GPU dynamic hierarchy selected a different safe level count");
        for (std::size_t level = 0; level < cpuBuilt.size(); ++level) {
            const auto dynamic = gpu.readPressureOperatorLevel(level);
            require(dynamic.width == cpuBuilt[level].width &&
                    dynamic.height == cpuBuilt[level].height &&
                    dynamic.active == cpuBuilt[level].active,
                "GPU dynamic operator active mask disagrees with CPU construction");
            for (std::size_t i = 0; i < dynamic.diagonal.size(); ++i) {
                maximumDiagonalError = std::max(maximumDiagonalError,
                    std::abs(dynamic.diagonal[i] - cpuBuilt[level].diagonal[i]));
                for (std::size_t direction = 0; direction < 4; ++direction) {
                    maximumWeightError = std::max(maximumWeightError,
                        std::abs(dynamic.weights[i][direction] -
                            cpuBuilt[level].weights[i][direction]));
                }
            }
        }
        require(maximumDiagonalError < 1.0e-9 && maximumWeightError < 1.0e-9,
            "GPU dynamic Galerkin coefficients disagree with CPU construction");
        const auto fineOperator = gpu.readPressureOperatorLevel(0);
        std::vector<double> testVector(width * height);
        for (std::size_t i = 0; i < testVector.size(); ++i) {
            testVector[i] = std::sin(0.013 * static_cast<double>(i));
        }
        const double inverseDx2 = 1.0 /
            (static_cast<double>(grid.cellWidth()) * grid.cellWidth());
        const double inverseDy2 = 1.0 /
            (static_cast<double>(grid.cellHeight()) * grid.cellHeight());
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t index = y * width + x;
                if (grid.cellType(x, y) != simulation::CellType::fluid) continue;
                double reference = 0.0;
                const auto addReference = [&](std::size_t nx, std::size_t ny, double coefficient) {
                    const auto type = grid.cellType(nx, ny);
                    if (type == simulation::CellType::fluid) {
                        reference += coefficient * (testVector[index] - testVector[ny * width + nx]);
                    } else if (type == simulation::CellType::air) {
                        reference += coefficient * testVector[index];
                    }
                };
                if (x > 0) addReference(x - 1, y, inverseDx2);
                if (x + 1 < width) addReference(x + 1, y, inverseDx2);
                if (y > 0) addReference(x, y - 1, inverseDy2);
                if (y + 1 < height) addReference(x, y + 1, inverseDy2);
                double action = fineOperator.diagonal[index] * testVector[index];
                if (x > 0) action -= fineOperator.weights[index][0] * testVector[index - 1];
                if (x + 1 < width) action -= fineOperator.weights[index][1] * testVector[index + 1];
                if (y > 0) action -= fineOperator.weights[index][2] * testVector[index - width];
                if (y + 1 < height) action -= fineOperator.weights[index][3] * testVector[index + width];
                maximumActionError = std::max(maximumActionError, std::abs(action - reference));
            }
        }
        require(maximumActionError < 1.0e-9,
            "GPU dynamic fine operator action disagrees with the reference equation");

        auto settings = pcgSettings(1.0f / 120.0f, 5.0e-3f);
        settings.solver = simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
        const auto projection = gpu.project(settings);
        require(projection.pressure.converged && std::isfinite(
                projection.pressure.finalResidual.rootMeanSquare),
            "dynamic MGPCG did not converge on a supported topology");

        output << std::setw(4) << width << 'x' << std::setw(3) << height << ' '
               << std::scientific << std::setprecision(2)
               << maximumDiagonalError << ' ' << maximumWeightError << ' '
               << maximumActionError << ' '
               << std::fixed << std::setprecision(3)
               << topology.activeMultigridLevelCount << ' '
               << topology.fineOperatorGpuMilliseconds << ' '
               << topology.topologyGpuMilliseconds << ' '
               << topology.coarseOperatorGpuMilliseconds << ' '
               << projection.pressure.gpuMilliseconds << ' '
               << topology.fineOperatorGpuMilliseconds + topology.topologyGpuMilliseconds +
                    topology.coarseOperatorGpuMilliseconds + projection.pressure.gpuMilliseconds << ' '
               << topology.statusReadbackMilliseconds << ' '
               << topology.statusBytesRead << ' ' << std::scientific << std::setprecision(2)
               << projection.divergenceAfter.maximumAbsolute << '\n';
    }

    simulation::MacGrid unsupported(24, 16, 1.0f / 24.0f, 1.0f / 24.0f);
    for (std::size_t y = 0; y < unsupported.height(); ++y)
        for (std::size_t x = 0; x < unsupported.width(); ++x)
            unsupported.setCellType(x, y, simulation::CellType::solid);
    for (std::size_t y = 2; y < 6; ++y)
        for (std::size_t x = 2; x < 6; ++x)
            unsupported.setCellType(x, y, simulation::CellType::fluid);
    for (std::size_t y = 8; y < 12; ++y)
        for (std::size_t x = 16; x < 20; ++x)
            unsupported.setCellType(x, y, simulation::CellType::fluid);
    gpu.upload(unsupported);
    const auto unsupportedTopology = gpu.rebuildDynamicPressureTopology();
    require(unsupportedTopology.pureNeumannComponentCount == 2,
        "GPU topology analysis did not identify two pure-Neumann components");
    gpu.dispatchDivergence();
    static_cast<void>(gpu.readDivergence());
    auto settings = pcgSettings(1.0f / 120.0f, 5.0e-3f);
    settings.solver = simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
    const auto rejected = gpu.solvePressure(settings);
    require(rejected.failureReason ==
            simulation::GpuPressureFailureReason::unsupportedMultiplePureNeumannComponents,
        "GPU dynamic topology did not safely reject multiple Neumann components");

    simulation::MacGrid dynamicSeed(48, 32, 1.0f / 48.0f, 1.0f / 48.0f);
    simulation::initializeDamBreak(dynamicSeed);
    simulation::MarkerParticleSystem particles;
    particles.seedFluidCells(dynamicSeed);
    for (std::size_t y = 0; y < dynamicSeed.height(); ++y)
        for (std::size_t x = 0; x < dynamicSeed.width(); ++x)
            if (dynamicSeed.cellType(x, y) != simulation::CellType::solid)
                dynamicSeed.setCellType(x, y, simulation::CellType::air);
    gpu.upload(dynamicSeed);
    gpu.uploadMarkerParticles(particles);
    static_cast<void>(gpu.rebuildClassificationFromParticles(true));
    const auto stale = gpu.solvePressure(settings);
    require(stale.failureReason == simulation::GpuPressureFailureReason::dynamicPressureTopologyStale,
        "classification change did not invalidate the pressure operator");
    const auto rebuilt = gpu.rebuildDynamicPressureTopology();
    require(rebuilt.topologyValid && !rebuilt.usedFullClassificationReadback,
        "dynamic rebuild required a full classification readback");

    simulation::MacGrid dynamicVelocity = dynamicSeed;
    particles.rebuildFluidClassification(dynamicVelocity);
    fillSmoothField(dynamicVelocity);
    dynamicVelocity.enforceSolidBoundaries();
    gpu.uploadVelocity(dynamicVelocity);
    auto coldSettings = settings;
    coldSettings.warmStartEnabled = false;
    coldSettings.collectVelocityChange = true;
    const auto cold = gpu.project(coldSettings);
    const auto coldU = gpu.readHorizontalVelocity();
    const auto coldV = gpu.readVerticalVelocity();
    require(cold.pressure.converged,
        "cold-start dynamic MGPCG did not converge");

    gpu.uploadVelocity(dynamicVelocity);
    const auto stableClassification = gpu.rebuildClassificationFromParticles(true);
    require(stableClassification.changedCellCount == 0,
        "stable dynamic classification unexpectedly changed before warm start");
    const auto warmTopology = gpu.reuseDynamicPressureTopology();
    require(warmTopology.multigridValid && warmTopology.reusedExistingHierarchy &&
            warmTopology.classificationVersion == warmTopology.fineOperatorVersion &&
            warmTopology.classificationVersion == warmTopology.multigridVersion,
        "unchanged classification did not safely reuse the dynamic hierarchy");
    auto warmSettings = coldSettings;
    warmSettings.warmStartEnabled = true;
    const auto warm = gpu.project(warmSettings);
    const Error warmUError = compareValues(coldU, gpu.readHorizontalVelocity());
    const Error warmVError = compareValues(coldV, gpu.readVerticalVelocity());
    require(warm.pressure.converged &&
            warm.pressure.finalResidual.rootMeanSquare <= warmSettings.residualTolerance &&
            warmUError.maximum < 2.0e-5f && warmVError.maximum < 2.0e-5f,
        "dynamic warm start changed the converged projected velocity");

    simulation::MacGrid equalCountGeometry(48, 32, 1.0f / 48.0f, 1.0f / 48.0f);
    for (std::size_t y = 0; y < equalCountGeometry.height(); ++y) {
        for (std::size_t x = 0; x < equalCountGeometry.width(); ++x) {
            equalCountGeometry.setCellType(x, y,
                x == 0 || y == 0 || x + 1 == equalCountGeometry.width() ||
                y + 1 == equalCountGeometry.height()
                    ? simulation::CellType::solid : simulation::CellType::air);
        }
    }
    const auto makeEqualCountState = [&](std::size_t x0) {
        simulation::MacGrid state = equalCountGeometry;
        for (std::size_t y = 2; y <= 8; ++y)
            for (std::size_t x = x0; x <= x0 + 6; ++x)
                state.setCellType(x, y, simulation::CellType::fluid);
        return state;
    };
    const simulation::MacGrid equalA = makeEqualCountState(2);
    const simulation::MacGrid equalB = makeEqualCountState(14);
    gpu.upload(equalCountGeometry);
    simulation::MarkerParticleSystem equalParticlesA;
    equalParticlesA.seedFluidCells(equalA, 1);
    gpu.uploadMarkerParticles(equalParticlesA);
    const auto equalClassificationA = gpu.rebuildClassificationFromParticles(true);
    static_cast<void>(gpu.rebuildDynamicPressureTopology());
    simulation::MarkerParticleSystem equalParticlesB;
    equalParticlesB.seedFluidCells(equalB, 1);
    gpu.uploadMarkerParticles(equalParticlesB);
    const auto equalClassificationB = gpu.rebuildClassificationFromParticles(true);
    const auto equalCountStale = gpu.solvePressure(settings);
    require(equalClassificationA.fluidCellCount == equalClassificationB.fluidCellCount &&
            equalClassificationB.changedCellCount != 0 &&
            equalCountStale.failureReason ==
                simulation::GpuPressureFailureReason::dynamicPressureTopologyStale,
        "equal fluid counts at different locations incorrectly reused pressure topology");

    simulation::MacGrid sequenceGeometry(48, 32, 1.0f / 48.0f, 1.0f / 48.0f);
    for (std::size_t y = 0; y < sequenceGeometry.height(); ++y) {
        for (std::size_t x = 0; x < sequenceGeometry.width(); ++x) {
            sequenceGeometry.setCellType(x, y,
                x == 0 || y == 0 || x + 1 == sequenceGeometry.width() ||
                y + 1 == sequenceGeometry.height()
                    ? simulation::CellType::solid : simulation::CellType::air);
        }
    }
    const auto addChamber = [&](std::size_t x0, std::size_t y0) {
        for (std::size_t y = y0; y <= y0 + 6; ++y)
            for (std::size_t x = x0; x <= x0 + 6; ++x)
                if (x == x0 || x == x0 + 6 || y == y0 || y == y0 + 6)
                    sequenceGeometry.setCellType(x, y, simulation::CellType::solid);
    };
    addChamber(27, 20);
    addChamber(38, 10);
    gpu.upload(sequenceGeometry);
    const auto makeState = [&]() { return sequenceGeometry; };
    const auto fillRect = [](simulation::MacGrid& state,
                             std::size_t x0, std::size_t y0,
                             std::size_t x1, std::size_t y1) {
        for (std::size_t y = y0; y <= y1; ++y)
            for (std::size_t x = x0; x <= x1; ++x)
                if (state.cellType(x, y) != simulation::CellType::solid)
                    state.setCellType(x, y, simulation::CellType::fluid);
    };
    std::vector<simulation::MacGrid> states;
    states.push_back(makeState()); fillRect(states.back(), 2, 2, 10, 12);
    states.push_back(makeState()); fillRect(states.back(), 2, 2, 16, 12);
    states.push_back(makeState()); fillRect(states.back(), 2, 2, 8, 9);
    states.push_back(makeState());
    fillRect(states.back(), 2, 2, 8, 8); fillRect(states.back(), 14, 2, 20, 8);
    fillRect(states.back(), 9, 5, 13, 5);
    states.push_back(makeState());
    fillRect(states.back(), 2, 2, 8, 8); fillRect(states.back(), 14, 2, 20, 8);
    states.push_back(makeState()); fillRect(states.back(), 2, 2, 20, 8);
    states.push_back(makeState()); fillRect(states.back(), 1, 1, 6, 6);
    states.push_back(makeState());
    fillRect(states.back(), 2, 2, 6, 6); fillRect(states.back(), 17, 11, 21, 15);
    states.push_back(makeState()); fillRect(states.back(), 39, 11, 43, 15);
    states.push_back(makeState());
    fillRect(states.back(), 39, 11, 43, 15); fillRect(states.back(), 28, 21, 32, 25);
    const std::array<std::size_t, 10> expectedComponents{1, 1, 1, 1, 2, 1, 1, 2, 1, 2};
    const std::array<std::size_t, 10> expectedNeumann{0, 0, 0, 0, 0, 0, 0, 0, 1, 2};
    for (std::size_t stateIndex = 0; stateIndex < states.size(); ++stateIndex) {
        simulation::MarkerParticleSystem stateParticles;
        stateParticles.seedFluidCells(states[stateIndex], 1);
        gpu.uploadMarkerParticles(stateParticles);
        static_cast<void>(gpu.rebuildClassificationFromParticles(true));
        const auto staleState = gpu.solvePressure(settings);
        require(staleState.failureReason ==
                simulation::GpuPressureFailureReason::dynamicPressureTopologyStale,
            "a sequence classification update did not invalidate the pressure operator");
        const auto stateTopology = gpu.rebuildDynamicPressureTopology();
        require(stateTopology.topologyValid &&
                stateTopology.connectedComponentCount == expectedComponents[stateIndex] &&
                stateTopology.pureNeumannComponentCount == expectedNeumann[stateIndex],
            "GPU dynamic sequence topology summary is incorrect");
    }

    output << "Multiple pure-Neumann components: safely rejected\n"
           << "Stale operator before rebuild: safely rejected\n"
           << "Exact unchanged classification topology reuse: passed\n"
           << "Equal-count displaced classification rebuild: passed\n"
           << "Full classification readback on GPU path: no\n"
           << std::scientific << std::setprecision(3)
           << "Dynamic cold/warm RMS: " << cold.pressure.finalResidual.rootMeanSquare
           << " / " << warm.pressure.finalResidual.rootMeanSquare << '\n'
           << std::fixed << std::setprecision(3)
           << "Dynamic cold/warm iterations: " << cold.pressure.iterations
           << " / " << warm.pressure.iterations << '\n'
           << "Dynamic cold/warm pressure kernel ms: " << cold.pressure.gpuMilliseconds
           << " / " << warm.pressure.gpuMilliseconds << '\n'
           << std::scientific << std::setprecision(3)
           << "Dynamic cold/warm post-divergence max: "
           << cold.divergenceAfter.maximumAbsolute << " / "
           << warm.divergenceAfter.maximumAbsolute << '\n'
           << "Dynamic cold/warm velocity max error u/v: "
           << warmUError.maximum << " / " << warmVError.maximum << '\n'
           << "Continuous topology sequence states: 10 / 10 passed\n"
           << "Phase 1.6A-2 GPU dynamic topology result: PASS\n";
    return 0;
}

int runGpuResidentPressureValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    constexpr float timeStep = 1.0f / 120.0f;
    constexpr float tolerance = 5.0e-3f;
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    gpu.setSynchronousTimingEnabled(false);

    auto settingsFor = [](simulation::GpuPressureControlMode mode) {
        simulation::GpuPressureSettings settings;
        settings.timeStep = timeStep;
        settings.residualTolerance = tolerance;
        settings.maximumIterations = 4000;
        settings.residualCheckInterval = 20;
        settings.solver =
            simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
        settings.warmStartEnabled = false;
        settings.collectVelocityChange = false;
        settings.controlMode = mode;
        settings.gpuResidentBatchCount = 1;
        return settings;
    };

    output << "LiquidSimulator Phase 1.8 GPU-resident pressure validation\n"
           << "Grid CpuIter GpuIter Submitted Skipped CpuReads GpuReads CpuWaitMs GpuWaitMs "
              "EndCpuMs EndGpuMs PressureMax VelocityMax RMS PostDiv\n";
    for (const std::size_t size : {64U, 128U, 192U, 256U}) {
        simulation::MacGrid source(size, size,
            1.0f / static_cast<float>(size), 1.0f / static_cast<float>(size));
        simulation::initializeDamBreak(source);
        fillSmoothField(source);
        source.enforceSolidBoundaries();
        source.computeDivergence();

        gpu.upload(source);
        const auto cpuStart = Clock::now();
        const auto cpuControl = gpu.project(
            settingsFor(simulation::GpuPressureControlMode::cpuSynchronized));
        const double cpuEnd = std::chrono::duration<double, std::milli>(
            Clock::now() - cpuStart).count();
        simulation::MacGrid cpuGrid = source;
        gpu.download(cpuGrid);

        gpu.upload(source);
        const auto gpuStart = Clock::now();
        const auto gpuControl = gpu.project(
            settingsFor(simulation::GpuPressureControlMode::gpuResidentBatches));
        const double gpuEnd = std::chrono::duration<double, std::milli>(
            Clock::now() - gpuStart).count();
        simulation::MacGrid gpuGrid = source;
        gpu.download(gpuGrid);
        const Comparison error = compareGrids(cpuGrid, gpuGrid);

        const std::size_t cpuReads = cpuControl.pressure.residualBytesRead / sizeof(double);
        output << std::setw(4) << size
               << std::setw(9) << cpuControl.pressure.iterations
               << std::setw(8) << gpuControl.pressure.iterations
               << std::setw(10) << gpuControl.pressure.submittedIterations
               << std::setw(8) << gpuControl.pressure.skippedIterations
               << std::setw(9) << cpuReads
               << std::setw(9) << gpuControl.pressure.stateReadbackCount
               << std::fixed << std::setprecision(3)
               << std::setw(11) << cpuControl.pressure.residualReadbackMilliseconds
               << std::setw(10) << gpuControl.pressure.residualReadbackMilliseconds
               << std::setw(9) << cpuEnd << std::setw(9) << gpuEnd
               << std::scientific << std::setprecision(3)
               << ' ' << error.pressure.maximum
               << ' ' << error.velocity.maximum
               << ' ' << gpuControl.pressure.finalResidual.rootMeanSquare
               << ' ' << gpuControl.divergenceAfter.maximumAbsolute << '\n';

        require(cpuControl.pressure.converged && gpuControl.pressure.converged,
            "CPU-controlled or GPU-resident MGPCG did not converge");
        require(gpuControl.pressure.solveState ==
                simulation::GpuPressureSolveState::converged,
            "GPU-resident state did not report convergence");
        require(error.velocity.maximum < 2.0e-5f && error.pressure.maximum < 2.0e-3f,
            "GPU-resident MGPCG disagrees with the synchronized control path");
        require(gpuControl.pressure.finalResidual.rootMeanSquare <= tolerance,
            "GPU-resident true residual exceeds the requested tolerance");
    }

    simulation::MacGrid zero(24, 16, 1.0f / 24.0f, 1.0f / 24.0f);
    gpu.upload(zero);
    const auto zeroReport = gpu.project(
        settingsFor(simulation::GpuPressureControlMode::gpuResidentBatches));
    require(zeroReport.pressure.converged && zeroReport.pressure.iterations == 0 &&
            zeroReport.pressure.solveState == simulation::GpuPressureSolveState::converged,
        "GPU-resident zero-iteration convergence is incorrect");

    simulation::MacGrid limited(48, 48, 1.0f / 48.0f, 1.0f / 48.0f);
    simulation::initializeDamBreak(limited);
    fillSmoothField(limited);
    auto limitedSettings = settingsFor(
        simulation::GpuPressureControlMode::gpuResidentBatches);
    limitedSettings.residualTolerance = 1.0e-12f;
    limitedSettings.maximumIterations = 1;
    limitedSettings.residualCheckInterval = 1;
    gpu.upload(limited);
    const auto limitedReport = gpu.project(limitedSettings);
    require(!limitedReport.pressure.converged &&
            limitedReport.pressure.solveState ==
                simulation::GpuPressureSolveState::maximumIterationsReached,
        "GPU-resident maximum-iteration state is incorrect");

    simulation::MacGrid invalid(24, 24, 1.0f / 24.0f, 1.0f / 24.0f);
    simulation::initializeDamBreak(invalid);
    invalid.horizontalVelocity(8, 8) = std::numeric_limits<float>::quiet_NaN();
    gpu.upload(invalid);
    const auto invalidReport = gpu.project(
        settingsFor(simulation::GpuPressureControlMode::gpuResidentBatches));
    require(!invalidReport.pressure.converged &&
            invalidReport.pressure.failureReason ==
                simulation::GpuPressureFailureReason::numericalBreakdown,
        "GPU-resident invalid residual did not close the pressure safety gate");

    simulation::MacGrid batched(128, 128, 1.0f / 128.0f, 1.0f / 128.0f);
    simulation::initializeDamBreak(batched);
    fillSmoothField(batched);
    auto batchedSettings = settingsFor(
        simulation::GpuPressureControlMode::gpuResidentBatches);
    batchedSettings.gpuResidentBatchCount = 2;
    gpu.upload(batched);
    const auto batchedReport = gpu.project(batchedSettings);
    require(batchedReport.pressure.converged &&
            batchedReport.pressure.stateReadbackCount == 1 &&
            batchedReport.pressure.submittedIterations == 40 &&
            batchedReport.pressure.skippedIterations == 20,
        "GPU-resident two-batch conditional skip is incorrect");

    output << "Zero-iteration convergence: passed\n"
           << "Maximum-iteration failure: passed\n"
           << "Invalid residual safety gate: passed\n"
           << "Two-batch conditional skip: passed\n"
           << "Phase 1.8 GPU-resident pressure result: PASS\n";
    return 0;
}

int runGpuMixedPressureValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    constexpr float timeStep = 1.0f / 120.0f;
    constexpr float tolerance = 5.0e-3f;
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    auto settingsFor = [](simulation::GpuPressurePrecisionMode precision) {
        simulation::GpuPressureSettings settings;
        settings.timeStep = timeStep;
        settings.residualTolerance = tolerance;
        settings.maximumIterations = 4000;
        settings.residualCheckInterval = 20;
        settings.solver =
            simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
        settings.warmStartEnabled = false;
        settings.collectVelocityChange = true;
        settings.controlMode = simulation::GpuPressureControlMode::cpuSynchronized;
        settings.precisionMode = precision;
        settings.fusedCoarseSmootherEnabled = false;
        return settings;
    };

    output << "LiquidSimulator Phase 1.8.2 mixed-precision pressure validation\n"
           << "FP64 outer PCG and true residual; FP32 V-cycle vectors; FP64 operator.\n"
           << "Grid F64Iter MixIter F64Ms MixMs F64Prec MixPrec Dispatch64 DispatchMix "
              "PressureMax VelocityMax RMS MaxResidual PostDiv\n";
    for (const std::size_t size : {64U, 128U, 192U, 256U}) {
        simulation::MacGrid source(size, size,
            1.0f / static_cast<float>(size), 1.0f / static_cast<float>(size));
        simulation::initializeDamBreak(source);
        fillSmoothField(source);
        source.enforceSolidBoundaries();

        gpu.upload(source);
        const auto fp64Start = Clock::now();
        const auto fp64 = gpu.project(settingsFor(simulation::GpuPressurePrecisionMode::fp64));
        const double fp64End = std::chrono::duration<double, std::milli>(
            Clock::now() - fp64Start).count();
        simulation::MacGrid fp64Grid = source;
        gpu.download(fp64Grid);

        gpu.upload(source);
        const auto mixedStart = Clock::now();
        const auto mixed = gpu.project(settingsFor(
            simulation::GpuPressurePrecisionMode::fp64OuterFp32Multigrid));
        const double mixedEnd = std::chrono::duration<double, std::milli>(
            Clock::now() - mixedStart).count();
        simulation::MacGrid mixedGrid = source;
        gpu.download(mixedGrid);
        const auto mixedResidualField = gpu.readStoredPressureResidualField();
        const auto maximumResidualIt = std::max_element(
            mixedResidualField.begin(), mixedResidualField.end());
        const std::size_t maximumResidualIndex = static_cast<std::size_t>(
            maximumResidualIt - mixedResidualField.begin());
        const Comparison error = compareGrids(fp64Grid, mixedGrid);

        auto fusedSettings = settingsFor(simulation::GpuPressurePrecisionMode::fp64);
        fusedSettings.fusedCoarseSmootherEnabled = true;
        gpu.upload(source);
        const auto fused = gpu.project(fusedSettings);
        simulation::MacGrid fusedGrid = source;
        gpu.download(fusedGrid);
        const Comparison fusedError = compareGrids(fp64Grid, fusedGrid);

        output << std::setw(4) << size
               << std::setw(8) << fp64.pressure.iterations
               << std::setw(8) << mixed.pressure.iterations
               << std::fixed << std::setprecision(3)
               << std::setw(8) << fp64End << std::setw(8) << mixedEnd
               << std::setw(9) << fp64.pressure.preconditionerGpuMilliseconds
               << std::setw(9) << mixed.pressure.preconditionerGpuMilliseconds
               << std::setw(11) << fp64.pressure.dispatchCount
               << std::setw(12) << mixed.pressure.dispatchCount
               << std::scientific << std::setprecision(3)
               << ' ' << error.pressure.maximum
               << ' ' << error.velocity.maximum
               << ' ' << mixed.pressure.finalResidual.rootMeanSquare
               << ' ' << mixed.pressure.finalResidual.maximumAbsolute
               << ' ' << mixed.divergenceAfter.maximumAbsolute << '\n';
        output << "     max-residual cell (" << maximumResidualIndex % size << ','
               << maximumResidualIndex / size << ") value " << *maximumResidualIt
               << " type " << static_cast<int>(mixedGrid.cellTypes()[maximumResidualIndex])
               << " boundary-distance " << std::min({maximumResidualIndex % size,
                      maximumResidualIndex / size, size - 1 - maximumResidualIndex % size,
                      size - 1 - maximumResidualIndex / size});
        const std::size_t residualX = maximumResidualIndex % size;
        const std::size_t residualY = maximumResidualIndex / size;
        const auto typeAt = [&](std::size_t x, std::size_t y) {
            return static_cast<int>(mixedGrid.cellType(x, y));
        };
        output << " neighbors L/R/D/U="
               << (residualX > 0 ? typeAt(residualX - 1, residualY) : -1) << '/'
               << (residualX + 1 < size ? typeAt(residualX + 1, residualY) : -1) << '/'
               << (residualY > 0 ? typeAt(residualX, residualY - 1) : -1) << '/'
               << (residualY + 1 < size ? typeAt(residualX, residualY + 1) : -1)
               << '\n';

        require(fp64.pressure.converged && mixed.pressure.converged,
            "FP64 or mixed-precision MGPCG did not converge");
        require(mixed.pressure.finalResidual.rootMeanSquare <= tolerance,
            "mixed-precision true residual exceeds the requested tolerance");
        require(mixed.pressure.failureReason == simulation::GpuPressureFailureReason::none,
            "mixed-precision MGPCG reported a numerical failure");
        require(error.velocity.maximum < 5.0e-4f && error.pressure.maximum < 5.0e-2f,
            "mixed-precision MGPCG differs excessively from FP64");
        require(fused.pressure.converged &&
                fused.pressure.finalResidual.rootMeanSquare <= tolerance,
            "fused coarse smoother did not converge");
        require(fusedError.velocity.maximum < 2.0e-5f &&
                fusedError.pressure.maximum < 2.0e-3f,
            "fused coarse smoother differs from the reference FP64 path");
        require(fused.pressure.dispatchCount < fp64.pressure.dispatchCount,
            "fused coarse smoother did not reduce pressure dispatches");
        output << "     fused-coarse dispatch " << fused.pressure.dispatchCount
               << " (saved " << fp64.pressure.dispatchCount - fused.pressure.dispatchCount
               << "), pressure/velocity max error " << std::scientific
               << fusedError.pressure.maximum << " / " << fusedError.velocity.maximum << '\n';
    }

    simulation::MacGrid neumann(48, 32, 1.0f / 48.0f, 1.0f / 48.0f);
    fillSmoothField(neumann);
    neumann.enforceSolidBoundaries();
    gpu.upload(neumann);
    const auto neumannReport = gpu.project(settingsFor(
        simulation::GpuPressurePrecisionMode::fp64OuterFp32Multigrid));
    require(neumannReport.pressure.converged,
        "mixed-precision pure-Neumann pressure solve did not converge");

    output << "Pure-Neumann mixed-precision solve: passed\n"
           << "Single-workgroup fused coarse smoother: passed\n"
           << "Multiple pure-Neumann rejection remains covered by topology validation.\n"
           << "Mixed precision remains experimental and FP64 remains the default.\n"
           << "Phase 1.8.2 mixed-precision pressure result: PASS\n";
    return 0;
}

int runGpuSolverBenchmark(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    constexpr std::size_t targetOuterSteps = 60;
    struct Variant final {
        simulation::GpuPressurePrecisionMode precision;
        bool fused;
        const char* name;
    };
    constexpr std::array variants{
        Variant{simulation::GpuPressurePrecisionMode::fp64, false, "FP64"},
        Variant{simulation::GpuPressurePrecisionMode::fp64, true, "FP64-FUSED"},
        Variant{simulation::GpuPressurePrecisionMode::fp64OuterFp32Multigrid, false, "MIXED"},
        Variant{simulation::GpuPressurePrecisionMode::fp64OuterFp32Multigrid, true, "MIXED-FUSED"}};
    output << "LiquidSimulator Phase 1.8.3 dynamic solver benchmark\n"
           << "Grid Mode Calls WallMs MeanMs P95Ms P99Ms PressureGpuMs Iter Dispatch Substeps Reject RMS MaxResidual PostDiv\n";
    for (const std::size_t size : {64U, 128U, 192U, 256U}) {
        for (const Variant& variant : variants) {
            simulation::MacGrid source(size, size,
                1.0f / static_cast<float>(size), 1.0f / static_cast<float>(size));
            simulation::initializeDamBreak(source);
            simulation::MarkerParticleSystem particles;
            particles.seedFluidCellsExact(source, 8);
            simulation::GpuDynamicFreeSurfaceSettings settings;
            settings.pressurePrecisionMode = variant.precision;
            settings.pressureFusedCoarseSmootherEnabled = variant.fused;
            settings.synchronousGpuTimingEnabled = true;
            settings.asynchronousGpuTimingEnabled = false;
            simulation::GpuDynamicFreeSurfaceStepper stepper(
                shaderDirectory / "mac_divergence.comp", settings);
            const auto topology = stepper.initialize(source, particles);
            require(topology.topologyValid, "dynamic solver benchmark topology is invalid");

            std::size_t calls = 0;
            std::size_t iterations = 0;
            std::size_t dispatches = 0;
            double pressureGpuMilliseconds = 0.0;
            std::size_t substeps = 0;
            std::size_t rejected = 0;
            std::vector<double> callMilliseconds;
            simulation::GpuDynamicFreeSurfaceReport report;
            const auto start = Clock::now();
            while (stepper.completedOuterSteps() < targetOuterSteps && stepper.healthy()) {
                const auto callStart = Clock::now();
                report = stepper.step();
                callMilliseconds.push_back(std::chrono::duration<double, std::milli>(
                    Clock::now() - callStart).count());
                ++calls;
                iterations += report.pressureIterationsTotal;
                dispatches += report.pressureDispatchCount;
                pressureGpuMilliseconds += report.timings.pressureGpuMilliseconds;
                substeps += report.substeps;
                rejected += report.rejectedSubsteps;
            }
            const double wallMilliseconds = std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
            require(stepper.healthy() && stepper.completedOuterSteps() == targetOuterSteps,
                "dynamic solver benchmark did not complete t=0.5 s");
            std::sort(callMilliseconds.begin(), callMilliseconds.end());
            const auto percentile = [&](double fraction) {
                const std::size_t index = static_cast<std::size_t>(std::ceil(
                    fraction * static_cast<double>(callMilliseconds.size()))) - 1;
                return callMilliseconds[std::min(index, callMilliseconds.size() - 1)];
            };
            output << std::setw(4) << size << ' ' << std::setw(12) << variant.name
                   << std::setw(6) << calls
                   << std::fixed << std::setprecision(3)
                   << std::setw(10) << wallMilliseconds
                   << std::setw(10) << wallMilliseconds / targetOuterSteps
                   << std::setw(10) << percentile(0.95)
                   << std::setw(10) << percentile(0.99)
                   << std::setw(14) << pressureGpuMilliseconds
                   << std::setw(7) << iterations
                   << std::setw(10) << dispatches
                   << std::setw(9) << substeps
                   << std::setw(7) << rejected
                   << std::scientific << std::setprecision(3)
                   << ' ' << report.projection.pressure.finalResidual.rootMeanSquare
                   << ' ' << report.projection.pressure.finalResidual.maximumAbsolute
                   << ' ' << report.projection.divergenceAfter.maximumAbsolute << '\n';
        }
    }
    return 0;
}

int runGpuMgpcgProfile(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    constexpr float timeStep = 1.0f / 120.0f;
    constexpr float tolerance = 5.0e-3f;
    constexpr std::array<const char*, 11> categoryNames{
        "matvec", "convert", "smoother", "mg-residual", "restrict", "prolong",
        "coarse", "vector-update", "dot-reduce", "true-residual", "v-cycle"};
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    output << "LiquidSimulator Phase 1.8.3 offline MGPCG profile\n"
           << "Timestamp sampling is limited to the first occurrence of each category/level.\n";
    for (const std::size_t size : {64U, 128U, 192U, 256U}) {
        for (const auto precision : {simulation::GpuPressurePrecisionMode::fp64,
                 simulation::GpuPressurePrecisionMode::fp64OuterFp32Multigrid}) {
            simulation::MacGrid source(size, size,
                1.0f / static_cast<float>(size), 1.0f / static_cast<float>(size));
            simulation::initializeDamBreak(source);
            fillSmoothField(source);
            source.enforceSolidBoundaries();
            simulation::GpuPressureSettings settings;
            settings.timeStep = timeStep;
            settings.residualTolerance = tolerance;
            settings.maximumIterations = 4000;
            settings.residualCheckInterval = 20;
            settings.solver =
                simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
            settings.precisionMode = precision;
            settings.controlMode = simulation::GpuPressureControlMode::cpuSynchronized;
            settings.fusedCoarseSmootherEnabled = false;

            gpu.upload(source);
            static_cast<void>(gpu.project(settings));
            std::vector<double> baselineSamples;
            std::vector<double> profileSamples;
            simulation::GpuProjectionReport baseline;
            simulation::GpuPressureProfileReport profile;
            for (std::size_t sample = 0; sample < 3; ++sample) {
                gpu.upload(source);
                const auto baselineStart = Clock::now();
                baseline = gpu.project(settings);
                baselineSamples.push_back(std::chrono::duration<double, std::milli>(
                    Clock::now() - baselineStart).count());
                gpu.upload(source);
                profile = gpu.profilePressureSolve(settings);
                profileSamples.push_back(profile.endToEndMilliseconds);
            }
            std::sort(baselineSamples.begin(), baselineSamples.end());
            std::sort(profileSamples.begin(), profileSamples.end());
            const double baselineMs = baselineSamples[baselineSamples.size() / 2];
            profile.endToEndMilliseconds = profileSamples[profileSamples.size() / 2];
            profile.instrumentationOverheadMilliseconds = profile.endToEndMilliseconds - baselineMs;

            auto fusedSettings = settings;
            fusedSettings.fusedCoarseSmootherEnabled = true;
            gpu.upload(source);
            const auto fusedStart = Clock::now();
            const auto fused = gpu.project(fusedSettings);
            const double fusedMs = std::chrono::duration<double, std::milli>(
                Clock::now() - fusedStart).count();

            require(baseline.pressure.converged && profile.pressure.converged &&
                    fused.pressure.converged,
                "profiled or fused pressure solve did not converge");
            require(fused.pressure.dispatchCount < baseline.pressure.dispatchCount,
                "offline profile found no dispatch reduction from coarse fusion");
            output << "Grid " << size << " mode "
                   << (precision == simulation::GpuPressurePrecisionMode::fp64
                           ? "FP64" : "Mixed")
                   << " baseline/profile/fused ms " << std::fixed << std::setprecision(3)
                   << baselineMs << '/' << profile.endToEndMilliseconds << '/' << fusedMs
                   << " instrumentation delta " << profile.instrumentationOverheadMilliseconds
                   << " dispatch " << baseline.pressure.dispatchCount << " -> "
                   << fused.pressure.dispatchCount << '\n';
            for (std::size_t categoryIndex = 0; categoryIndex < categoryNames.size();
                 ++categoryIndex) {
                const auto& category = profile.categories[categoryIndex];
                output << "  " << std::setw(13) << std::left << categoryNames[categoryIndex]
                       << std::right << " gpu " << std::setw(8) << category.gpuMilliseconds
                       << " cpu-submit " << std::setw(8) << category.cpuSubmissionMilliseconds
                       << " dispatch " << std::setw(4) << category.dispatchCount
                       << " samples " << category.sampleCount << '\n';
            }
            for (std::size_t level = 0; level < profile.levels.size(); ++level) {
                const auto& item = profile.levels[level];
                output << "    L" << level << ' ' << item.width << 'x' << item.height
                       << " active " << item.activeCount << " smooth-it "
                       << item.smootherIterations << " sampled-gpu "
                       << item.smootherGpuMilliseconds << " dispatch "
                       << item.smootherDispatchCount << '\n';
            }
        }
    }
    output << "Offline query pool reused: yes\n"
           << "Interactive path instrumentation: disabled\n"
           << "Phase 1.8.3 MGPCG profile result: PASS\n";
    return 0;
}

int runGpuMixedReliabilityValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    output << "LiquidSimulator Phase 1.8.3 mixed-preconditioner reliability validation\n"
           << "Grid Mode Symmetry Positive Linearity Repeatability\n";
    const auto dotFluid = [](const std::vector<double>& a, const std::vector<double>& b,
                              const simulation::MacGrid& grid) {
        double result = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (grid.cellTypes()[i] == simulation::CellType::fluid) result += a[i] * b[i];
        }
        return result;
    };
    for (const auto dimensions : {std::pair{48U, 32U}, std::pair{64U, 64U},
             std::pair{128U, 128U}, std::pair{192U, 192U}}) {
        simulation::MacGrid grid(dimensions.first, dimensions.second,
            1.0f / static_cast<float>(dimensions.first),
            1.0f / static_cast<float>(dimensions.second));
        simulation::initializeDamBreak(grid);
        gpu.upload(grid);
        const std::size_t count = grid.width() * grid.height();
        std::vector<double> u(count, 0.0);
        std::vector<double> v(count, 0.0);
        for (std::size_t i = 0; i < count; ++i) {
            if (grid.cellTypes()[i] != simulation::CellType::fluid) continue;
            u[i] = std::sin(0.173 * static_cast<double>(i + 1));
            v[i] = std::cos(0.097 * static_cast<double>(i + 3));
        }
        for (const auto precision : {simulation::GpuPressurePrecisionMode::fp64,
                 simulation::GpuPressurePrecisionMode::fp64OuterFp32Multigrid}) {
            simulation::GpuPressureSettings settings;
            settings.solver =
                simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
            settings.precisionMode = precision;
            settings.fusedCoarseSmootherEnabled = true;
            const auto bu = gpu.applyPressurePreconditionerForValidation(u, settings);
            const auto bv = gpu.applyPressurePreconditionerForValidation(v, settings);
            const auto repeat = gpu.applyPressurePreconditionerForValidation(u, settings);
            std::vector<double> combined(count);
            constexpr double a = 0.375;
            constexpr double b = -1.25;
            for (std::size_t i = 0; i < count; ++i) combined[i] = a * u[i] + b * v[i];
            const auto bCombined = gpu.applyPressurePreconditionerForValidation(combined, settings);
            const double uv = dotFluid(u, bv, grid);
            const double vu = dotFluid(bu, v, grid);
            const double symmetry = std::abs(uv - vu) /
                std::max({1.0, std::abs(uv), std::abs(vu)});
            const double positivity = dotFluid(u, bu, grid);
            double linearSquared = 0.0;
            double referenceSquared = 0.0;
            double repeatMaximum = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                if (grid.cellTypes()[i] != simulation::CellType::fluid) continue;
                const double reference = a * bu[i] + b * bv[i];
                const double difference = bCombined[i] - reference;
                linearSquared += difference * difference;
                referenceSquared += reference * reference;
                repeatMaximum = std::max(repeatMaximum, std::abs(repeat[i] - bu[i]));
            }
            const double linearity = std::sqrt(linearSquared /
                std::max(referenceSquared, std::numeric_limits<double>::min()));
            const double limit = precision == simulation::GpuPressurePrecisionMode::fp64
                ? 1.0e-11 : 2.0e-5;
            require(symmetry < limit, "multigrid preconditioner symmetry check failed");
            require(positivity > 0.0 && std::isfinite(positivity),
                "multigrid preconditioner positivity check failed");
            require(linearity < limit, "multigrid preconditioner linearity check failed");
            require(repeatMaximum == 0.0,
                "multigrid preconditioner repeatability check failed");
            output << dimensions.first << 'x' << dimensions.second << ' '
                   << (precision == simulation::GpuPressurePrecisionMode::fp64
                           ? "FP64 " : "Mixed")
                   << std::scientific << std::setprecision(3) << symmetry << ' '
                   << positivity << ' ' << linearity << ' ' << repeatMaximum << '\n';
        }
    }

    simulation::MacGrid neumann(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
    gpu.upload(neumann);
    std::vector<double> compatible(neumann.width() * neumann.height());
    double mean = 0.0;
    for (std::size_t i = 0; i < compatible.size(); ++i) {
        compatible[i] = std::sin(0.113 * static_cast<double>(i));
        mean += compatible[i];
    }
    mean /= static_cast<double>(compatible.size());
    for (double& value : compatible) value -= mean;
    simulation::GpuPressureSettings neumannSettings;
    neumannSettings.solver =
        simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
    neumannSettings.precisionMode =
        simulation::GpuPressurePrecisionMode::fp64OuterFp32Multigrid;
    neumannSettings.fusedCoarseSmootherEnabled = true;
    const auto neumannResult = gpu.applyPressurePreconditionerForValidation(
        compatible, neumannSettings);
    require(dotFluid(compatible, neumannResult, neumann) > 0.0,
        "mixed pure-Neumann preconditioner is not positive on the compatible subspace");
    output << "Pure-Neumann compatible-subspace positivity: passed\n"
           << "Dynamic hierarchy rebuild remains covered by dynamic topology validation.\n"
           << "Multiple pure-Neumann safe rejection remains covered by topology validation.\n"
           << "Phase 1.8.3 mixed-preconditioner reliability result: PASS\n";
    return 0;
}

int runGpuMixedLongValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    struct Case final { std::size_t size; std::size_t outerSteps; };
    constexpr std::array cases{
        Case{64, 240}, Case{128, 120}, Case{192, 72}, Case{256, 72}};
    output << "LiquidSimulator Phase 1.8.3 long mixed-precision validation\n"
           << "Grid Mode Time WallMs Calls Iter Dispatch Reject MaxCFL MaxSpeed MaxRMS MaxResidual MaxDiv Particles FluidRange Holes Flicker Invalid\n";
    for (const Case test : cases) {
        for (const auto precision : {simulation::GpuPressurePrecisionMode::fp64,
                 simulation::GpuPressurePrecisionMode::fp64OuterFp32Multigrid}) {
            simulation::MacGrid source(test.size, test.size,
                1.0f / static_cast<float>(test.size),
                1.0f / static_cast<float>(test.size));
            simulation::initializeDamBreak(source);
            simulation::MarkerParticleSystem particles;
            particles.seedFluidCellsExact(source, 8);
            simulation::GpuDynamicFreeSurfaceSettings settings;
            settings.pressurePrecisionMode = precision;
            settings.pressureFusedCoarseSmootherEnabled = true;
            settings.synchronousGpuTimingEnabled = true;
            settings.asynchronousGpuTimingEnabled = false;
            simulation::GpuDynamicFreeSurfaceStepper stepper(
                shaderDirectory / "mac_divergence.comp", settings);
            require(stepper.initialize(source, particles).topologyValid,
                "long mixed validation initial topology is invalid");
            std::size_t calls = 0;
            std::size_t iterations = 0;
            std::size_t dispatches = 0;
            std::size_t rejected = 0;
            std::size_t minimumFluid = std::numeric_limits<std::size_t>::max();
            std::size_t maximumFluid = 0;
            std::size_t maximumHoles = 0;
            std::size_t maximumFlicker = 0;
            std::size_t invalid = 0;
            float maximumCfl = 0.0f;
            float maximumSpeed = 0.0f;
            float maximumRms = 0.0f;
            float maximumResidual = 0.0f;
            float maximumDivergence = 0.0f;
            simulation::GpuDynamicFreeSurfaceReport report;
            const auto start = Clock::now();
            while (stepper.completedOuterSteps() < test.outerSteps && stepper.healthy()) {
                report = stepper.step();
                ++calls;
                iterations += report.pressureIterationsTotal;
                dispatches += report.pressureDispatchCount;
                rejected += report.rejectedSubsteps;
                minimumFluid = std::min(minimumFluid, report.fluidCellCount);
                maximumFluid = std::max(maximumFluid, report.fluidCellCount);
                maximumHoles = std::max(maximumHoles,
                    report.classification.enclosedAirCellCount);
                maximumFlicker = std::max(maximumFlicker,
                    report.classification.repeatedFlickerCellCount);
                invalid += report.classification.invalidParticleCount;
                maximumCfl = std::max(maximumCfl, report.maximumAcceptedCfl);
                maximumSpeed = std::max(maximumSpeed,
                    std::max(report.velocity.maximumAbsoluteU,
                        report.velocity.maximumAbsoluteV));
                maximumRms = std::max(maximumRms,
                    report.projection.pressure.finalResidual.rootMeanSquare);
                maximumResidual = std::max(maximumResidual,
                    report.projection.pressure.finalResidual.maximumAbsolute);
                maximumDivergence = std::max(maximumDivergence,
                    report.projection.divergenceAfter.maximumAbsolute);
            }
            const double wallMilliseconds = std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
            require(stepper.healthy() && stepper.completedOuterSteps() == test.outerSteps,
                "long mixed validation did not reach its target physical time");
            require(invalid == 0, "long mixed validation found invalid particles");
            require(maximumCfl <= 0.75f + 1.0e-5f,
                "long mixed validation accepted a CFL above the safety limit");
            output << test.size << ' ' <<
                (precision == simulation::GpuPressurePrecisionMode::fp64 ? "FP64 " : "MIXED")
                   << std::fixed << std::setprecision(3) << stepper.simulationTime() << ' '
                   << wallMilliseconds << ' ' << calls << ' ' << iterations << ' '
                   << dispatches << ' ' << rejected << ' ' << maximumCfl << ' '
                   << maximumSpeed << ' ' << std::scientific << maximumRms << ' '
                   << maximumResidual << ' ' << maximumDivergence << ' '
                   << report.particleCount << ' ' << minimumFluid << '-' << maximumFluid
                   << ' ' << maximumHoles << ' ' << maximumFlicker << ' ' << invalid << '\n';
        }
    }
    output << "Phase 1.8.3 long mixed-precision validation result: PASS\n";
    return 0;
}

int runGpuCoarseFusionRegression(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    using simulation::CellType;
    struct Case final {
        const char* name;
        std::size_t width;
        std::size_t height;
        std::function<void(simulation::MacGrid&)> classify;
        bool expectFallback = false;
        bool expectUnsupported = false;
    };
    const auto airWithSolidBorder = [](simulation::MacGrid& grid) {
        grid.fillCellTypes(CellType::air);
        for (std::size_t y = 0; y < grid.height(); ++y) {
            grid.setCellType(0, y, CellType::solid);
            grid.setCellType(grid.width() - 1, y, CellType::solid);
        }
        for (std::size_t x = 0; x < grid.width(); ++x) {
            grid.setCellType(x, 0, CellType::solid);
            grid.setCellType(x, grid.height() - 1, CellType::solid);
        }
    };
    std::vector<Case> cases;
    cases.push_back({"full-fluid-256", 16, 16, [](simulation::MacGrid& grid) {
        grid.fillCellTypes(CellType::fluid);
    }});
    cases.push_back({"mostly-air", 32, 24, [=](simulation::MacGrid& grid) {
        airWithSolidBorder(grid);
        for (std::size_t y = 4; y < 10; ++y)
            for (std::size_t x = 3; x < 9; ++x) grid.setCellType(x, y, CellType::fluid);
    }});
    cases.push_back({"single-fluid", 17, 13, [=](simulation::MacGrid& grid) {
        airWithSolidBorder(grid);
        grid.setCellType(8, 6, CellType::fluid);
    }});
    cases.push_back({"narrow-channel", 63, 37, [=](simulation::MacGrid& grid) {
        airWithSolidBorder(grid);
        for (std::size_t x = 2; x + 2 < grid.width(); ++x)
            grid.setCellType(x, 18, CellType::fluid);
    }});
    cases.push_back({"winding-channel", 63, 37, [=](simulation::MacGrid& grid) {
        airWithSolidBorder(grid);
        for (std::size_t x = 3; x < 55; ++x) grid.setCellType(x, 5, CellType::fluid);
        for (std::size_t y = 5; y < 27; ++y) grid.setCellType(54, y, CellType::fluid);
        for (std::size_t x = 15; x <= 54; ++x) grid.setCellType(x, 26, CellType::fluid);
    }});
    cases.push_back({"split-air-components", 48, 32, [=](simulation::MacGrid& grid) {
        airWithSolidBorder(grid);
        for (std::size_t y = 4; y < 13; ++y) {
            for (std::size_t x = 4; x < 14; ++x) grid.setCellType(x, y, CellType::fluid);
            for (std::size_t x = 30; x < 42; ++x) grid.setCellType(x, y, CellType::fluid);
        }
    }});
    cases.push_back({"merged-neck", 48, 32, [=](simulation::MacGrid& grid) {
        airWithSolidBorder(grid);
        for (std::size_t y = 4; y < 13; ++y) {
            for (std::size_t x = 4; x < 14; ++x) grid.setCellType(x, y, CellType::fluid);
            for (std::size_t x = 30; x < 42; ++x) grid.setCellType(x, y, CellType::fluid);
        }
        for (std::size_t x = 13; x <= 30; ++x) grid.setCellType(x, 8, CellType::fluid);
    }});
    cases.push_back({"early-stop-fallback", 20, 20, [=](simulation::MacGrid& grid) {
        airWithSolidBorder(grid);
        grid.setCellType(8, 8, CellType::fluid);
        grid.setCellType(9, 9, CellType::fluid);
    }, true});
    cases.push_back({"two-neumann-reject", 24, 20, [](simulation::MacGrid& grid) {
        grid.fillCellTypes(CellType::solid);
        for (std::size_t y = 3; y < 7; ++y)
            for (std::size_t x = 3; x < 7; ++x) grid.setCellType(x, y, CellType::fluid);
        for (std::size_t y = 12; y < 16; ++y)
            for (std::size_t x = 15; x < 20; ++x) grid.setCellType(x, y, CellType::fluid);
    }, false, true});

    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    output << "LiquidSimulator Phase 1.9A coarse-fusion regression\n"
           << "Case Grid Levels GenericDispatch FusedDispatch VectorMax PressureMax "
              "VelocityMax RMS MaxResidual PostDiv Path\n";
    for (const Case& test : cases) {
        simulation::MacGrid source(test.width, test.height,
            1.0f / static_cast<float>(test.width),
            1.0f / static_cast<float>(test.height));
        test.classify(source);
        if (std::string_view(test.name) != "full-fluid-256") fillSmoothField(source);
        source.enforceSolidBoundaries();
        simulation::GpuPressureSettings generic;
        generic.timeStep = 1.0f / 120.0f;
        generic.residualTolerance = 5.0e-3f;
        generic.maximumIterations = 12000;
        generic.residualCheckInterval = 20;
        generic.solver = simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
        generic.controlMode = simulation::GpuPressureControlMode::cpuSynchronized;
        generic.fusedCoarseSmootherEnabled = false;
        auto fused = generic;
        fused.fusedCoarseSmootherEnabled = true;

        gpu.upload(source);
        const auto unsupportedGeneric = gpu.project(generic);
        gpu.upload(source);
        const auto unsupportedFused = gpu.project(fused);
        if (test.expectUnsupported) {
            require(!unsupportedGeneric.pressure.converged &&
                    !unsupportedFused.pressure.converged &&
                    unsupportedGeneric.pressure.failureReason ==
                        simulation::GpuPressureFailureReason::unsupportedMultiplePureNeumannComponents &&
                    unsupportedFused.pressure.failureReason ==
                        simulation::GpuPressureFailureReason::unsupportedMultiplePureNeumannComponents,
                "multiple pure-Neumann components were not safely rejected");
            output << test.name << ' ' << test.width << 'x' << test.height
                   << " - - - - - - - - - safe-reject\n";
            continue;
        }

        const std::size_t count = test.width * test.height;
        std::vector<double> residual(count, 0.0);
        for (std::size_t i = 0; i < count; ++i) {
            if (source.cellTypes()[i] == CellType::fluid)
                residual[i] = std::sin(0.137 * static_cast<double>(i + 1));
        }
        if (std::string_view(test.name) == "full-fluid-256") {
            const double mean = std::accumulate(residual.begin(), residual.end(), 0.0) /
                static_cast<double>(residual.size());
            for (double& value : residual) value -= mean;
        }
        gpu.upload(source);
        const auto genericVector = gpu.applyPressurePreconditionerForValidation(residual, generic);
        gpu.upload(source);
        const auto fusedVector = gpu.applyPressurePreconditionerForValidation(residual, fused);
        double vectorMaximum = 0.0;
        for (std::size_t i = 0; i < count; ++i)
            vectorMaximum = std::max(vectorMaximum, std::abs(genericVector[i] - fusedVector[i]));

        gpu.upload(source);
        const auto genericReport = gpu.project(generic);
        simulation::MacGrid genericGrid = source;
        gpu.download(genericGrid);
        gpu.upload(source);
        const auto fusedReport = gpu.project(fused);
        simulation::MacGrid fusedGrid = source;
        gpu.download(fusedGrid);
        const Comparison comparison = compareGrids(genericGrid, fusedGrid);
        if (!genericReport.pressure.converged || !fusedReport.pressure.converged) {
            output << "  diagnostic " << test.name << " generic/fused state="
                   << static_cast<unsigned int>(genericReport.pressure.solveState) << '/'
                   << static_cast<unsigned int>(fusedReport.pressure.solveState)
                   << " failure="
                   << static_cast<unsigned int>(genericReport.pressure.failureReason) << '/'
                   << static_cast<unsigned int>(fusedReport.pressure.failureReason)
                   << " iter=" << genericReport.pressure.iterations << '/'
                   << fusedReport.pressure.iterations << " rms="
                   << genericReport.pressure.finalResidual.rootMeanSquare << '/'
                   << fusedReport.pressure.finalResidual.rootMeanSquare << '\n';
        }
        require(genericReport.pressure.converged && fusedReport.pressure.converged,
            "coarse-fusion pressure solve did not converge");
        require(vectorMaximum <= 1.0e-12 && comparison.pressure.maximum <= 1.0e-6f &&
                comparison.velocity.maximum <= 1.0e-6f,
            "coarse-fusion result differs from the generic smoother");
        require(genericReport.pressure.iterations == fusedReport.pressure.iterations,
            "coarse-fusion changed the PCG iteration count");
        const bool zeroIteration = genericReport.pressure.iterations == 0;
        const bool fellBack = !zeroIteration && genericReport.pressure.dispatchCount ==
            fusedReport.pressure.dispatchCount;
        if (!zeroIteration) {
            require(fellBack == test.expectFallback,
                "coarse-fusion dispatch path did not match the expected fallback");
        }
        output << test.name << ' ' << test.width << 'x' << test.height << ' '
               << genericReport.pressure.multigridLevelCount << ' '
               << genericReport.pressure.dispatchCount << ' '
               << fusedReport.pressure.dispatchCount << ' '
               << std::scientific << std::setprecision(3) << vectorMaximum << ' '
               << comparison.pressure.maximum << ' ' << comparison.velocity.maximum << ' '
               << fusedReport.pressure.finalResidual.rootMeanSquare << ' '
               << fusedReport.pressure.finalResidual.maximumAbsolute << ' '
               << fusedReport.divergenceAfter.maximumAbsolute << ' '
               << (zeroIteration ? "fused-vector-only" : fellBack ? "fallback" : "fused")
               << '\n';
    }
    output << "FP64 default coarse fusion: enabled\n"
           << "Generic coarse path retained: yes\n"
           << "Hardware coverage: NVIDIA RTX 3070 only\n"
           << "Phase 1.9A coarse-fusion regression result: PASS\n";
    return 0;
}

int runGpuInterfaceAccuracyValidation(
    std::ostream& output,
    const std::filesystem::path& shaderDirectory) {
    using simulation::CellType;
    using Phi = std::function<double(double, double)>;
    struct Shape final {
        const char* name;
        Phi phi;
        double particleSpacingScale = 0.5;
    };
    constexpr std::size_t width = 40;
    constexpr std::size_t height = 30;
    constexpr double dx = 1.0 / static_cast<double>(width);
    constexpr double dy = 1.0 / static_cast<double>(height);
    const double baseSpacing = 0.5 * std::min(dx, dy);
    const auto circle = [](double cx, double cy, double radius) {
        return [=](double x, double y) {
            return std::hypot(x - cx, y - cy) - radius;
        };
    };
    const auto rectangle = [](double cx, double cy, double hx, double hy) {
        return [=](double x, double y) {
            const double qx = std::abs(x - cx) - hx;
            const double qy = std::abs(y - cy) - hy;
            const double outside = std::hypot(std::max(qx, 0.0), std::max(qy, 0.0));
            return outside + std::min(std::max(qx, qy), 0.0);
        };
    };
    const std::vector<Shape> shapes{
        {"horizontal", [](double, double y) { return y - 0.58; }},
        {"inclined", [](double x, double y) {
            return (y - (0.25 + 0.45 * x)) / std::sqrt(1.0 + 0.45 * 0.45);
        }},
        {"circle", circle(0.52, 0.52, 0.24)},
        {"wall-contact", circle(0.08, 0.52, 0.25)},
        {"corner-contact", circle(0.06, 0.06, 0.22)},
        {"narrow-column", rectangle(0.48, 0.45, 0.05, 0.32)},
        {"subcell-shift", [=](double, double y) { return y - (0.58 + 0.37 * dy); }},
        {"sparse-horizontal", [](double, double y) { return y - 0.58; }, 1.0},
    };
    output << "LiquidSimulator Phase 1.9A interface-accuracy validation\n"
           << "Particle proxy: union of equal-radius support disks; diagnostic only.\n"
           << "Shape ReferenceArea BinaryArea ProxyArea BinaryRel ProxyRel InterfaceRms NormalDeg Particles\n";
    for (const Shape& shape : shapes) {
        const double spacing = baseSpacing * shape.particleSpacingScale / 0.5;
        std::vector<simulation::MarkerParticle> particles;
        for (double y = 0.5 * spacing; y < 1.0; y += spacing) {
            for (double x = 0.5 * spacing; x < 1.0; x += spacing) {
                if (shape.phi(x, y) <= -0.5 * spacing) {
                    particles.push_back({{static_cast<float>(x), static_cast<float>(y)}, true});
                }
            }
        }
        simulation::InterfaceGeometrySettings geometrySettings;
        geometrySettings.particleSupportRadius = static_cast<float>(0.78 * spacing);
        geometrySettings.areaSamplesPerAxis = 4;
        const auto geometry = simulation::reconstructParticleInterface(
            width, height, static_cast<float>(dx), static_cast<float>(dy),
            particles, geometrySettings);

        constexpr std::size_t referenceSamples = 512;
        std::size_t referenceInside = 0;
        for (std::size_t y = 0; y < referenceSamples; ++y)
            for (std::size_t x = 0; x < referenceSamples; ++x)
                if (shape.phi((x + 0.5) / referenceSamples,
                              (y + 0.5) / referenceSamples) <= 0.0) ++referenceInside;
        const double referenceArea = static_cast<double>(referenceInside) /
            static_cast<double>(referenceSamples * referenceSamples);
        std::size_t binaryCells = 0;
        double interfaceSquared = 0.0;
        double normalAngle = 0.0;
        std::size_t interfaceSamples = 0;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const double px = (x + 0.5) * dx;
                const double py = (y + 0.5) * dy;
                const double analytic = shape.phi(px, py);
                if (analytic <= 0.0) ++binaryCells;
                if (std::abs(analytic) > 1.5 * std::max(dx, dy)) continue;
                const std::size_t index = y * width + x;
                const double difference = geometry.signedDistanceProxy[index] - analytic;
                interfaceSquared += difference * difference;
                const double epsilon = 0.1 * std::min(dx, dy);
                double nx = shape.phi(px + epsilon, py) - shape.phi(px - epsilon, py);
                double ny = shape.phi(px, py + epsilon) - shape.phi(px, py - epsilon);
                const double length = std::hypot(nx, ny);
                if (length > 0.0) {
                    nx /= length;
                    ny /= length;
                    const auto reconstructed = geometry.normals[index];
                    const double dot = std::clamp(nx * reconstructed.x + ny * reconstructed.y,
                        -1.0, 1.0);
                    normalAngle += std::acos(dot);
                }
                ++interfaceSamples;
            }
        }
        const double binaryArea = static_cast<double>(binaryCells) * dx * dy;
        const double interfaceRms = std::sqrt(interfaceSquared /
            std::max<std::size_t>(interfaceSamples, 1));
        const double normalDegrees = normalAngle /
            std::max<std::size_t>(interfaceSamples, 1) * 180.0 / 3.141592653589793;
        require(std::isfinite(geometry.estimatedArea) &&
                std::isfinite(interfaceRms) && std::isfinite(normalDegrees),
            "particle interface reconstruction produced invalid geometry");
        output << shape.name << ' ' << std::fixed << std::setprecision(6)
               << referenceArea << ' ' << binaryArea << ' ' << geometry.estimatedArea << ' '
               << std::scientific << std::abs(binaryArea - referenceArea) / referenceArea << ' '
               << std::abs(geometry.estimatedArea - referenceArea) / referenceArea << ' '
               << interfaceRms << ' ' << std::fixed << std::setprecision(3)
               << normalDegrees << ' ' << particles.size() << '\n';
    }

    output << "Local pressure residual diagnostics\n"
           << "Grid Cell Type Neighbors Particles RMS MaxResidual RelativeResidual RHS Diagonal "
              "Pressure DivBefore DivAfter GradX GradY InterfaceProxy\n";
    simulation::GpuMacGrid gpu(shaderDirectory / "mac_divergence.comp");
    for (const std::size_t size : {128U, 192U, 256U}) {
        simulation::MacGrid source(size, size,
            1.0f / static_cast<float>(size), 1.0f / static_cast<float>(size));
        simulation::initializeDamBreak(source);
        fillSmoothField(source);
        source.enforceSolidBoundaries();
        source.computeDivergence();
        simulation::MarkerParticleSystem particles;
        particles.seedFluidCellsExact(source, 8);
        std::vector<GLuint> counts(size * size, 0u);
        for (const auto& particle : particles.particles()) {
            if (!particle.valid) continue;
            const auto x = std::min(size - 1,
                static_cast<std::size_t>(particle.position.x * size));
            const auto y = std::min(size - 1,
                static_cast<std::size_t>(particle.position.y * size));
            ++counts[y * size + x];
        }
        gpu.upload(source);
        const auto divergenceBefore = source.divergences();
        simulation::GpuPressureSettings settings;
        settings.timeStep = 1.0f / 120.0f;
        settings.residualTolerance = 5.0e-3f;
        settings.maximumIterations = 4000;
        settings.residualCheckInterval = 20;
        settings.solver = simulation::GpuPressureSolver::multigridPreconditionedConjugateGradient;
        settings.controlMode = simulation::GpuPressureControlMode::cpuSynchronized;
        settings.fusedCoarseSmootherEnabled = true;
        const auto projection = gpu.project(settings);
        simulation::MacGrid projected = source;
        gpu.download(projected);
        const auto divergenceAfter = projected.divergences();
        const auto pressure = gpu.readPressure();
        const auto residual = gpu.readStoredPressureResidualField();
        const auto op = gpu.readPressureOperatorLevel(0);
        const auto maximum = std::max_element(residual.begin(), residual.end(),
            [](double a, double b) { return std::abs(a) < std::abs(b); });
        const std::size_t index = static_cast<std::size_t>(maximum - residual.begin());
        const std::size_t x = index % size;
        const std::size_t y = index / size;
        const auto typeName = [](CellType type) {
            return type == CellType::fluid ? 'F' : type == CellType::air ? 'A' : 'S';
        };
        std::string neighbors;
        neighbors += x > 0 ? typeName(source.cellType(x - 1, y)) : 'S';
        neighbors += x + 1 < size ? typeName(source.cellType(x + 1, y)) : 'S';
        neighbors += y > 0 ? typeName(source.cellType(x, y - 1)) : 'S';
        neighbors += y + 1 < size ? typeName(source.cellType(x, y + 1)) : 'S';
        const double rhs = -static_cast<double>(divergenceBefore[index]) / settings.timeStep;
        double scale = std::abs(rhs) + std::abs(op.diagonal[index] * pressure[index]);
        if (x > 0) scale += std::abs(op.weights[index][0] * pressure[index - 1]);
        if (x + 1 < size) scale += std::abs(op.weights[index][1] * pressure[index + 1]);
        if (y > 0) scale += std::abs(op.weights[index][2] * pressure[index - size]);
        if (y + 1 < size) scale += std::abs(op.weights[index][3] * pressure[index + size]);
        const double relative = std::abs(*maximum) / std::max(scale, 1.0e-12);
        const double gradX = ((x + 1 < size ? pressure[index + 1] : pressure[index]) -
            (x > 0 ? pressure[index - 1] : pressure[index])) * 0.5 * size;
        const double gradY = ((y + 1 < size ? pressure[index + size] : pressure[index]) -
            (y > 0 ? pressure[index - size] : pressure[index])) * 0.5 * size;
        double nearest = std::numeric_limits<double>::infinity();
        const double centerX = (x + 0.5) / static_cast<double>(size);
        const double centerY = (y + 0.5) / static_cast<double>(size);
        for (const auto& particle : particles.particles()) {
            nearest = std::min(nearest, std::hypot(centerX - particle.position.x,
                centerY - particle.position.y));
        }
        const double spacing = (1.0 / size) / std::sqrt(8.0);
        const double interfaceProxy = nearest - 0.78 * spacing;
        output << size << ' ' << x << ',' << y << ' ' << typeName(source.cellType(x, y))
               << ' ' << neighbors << ' ' << counts[index] << ' '
               << std::scientific << std::setprecision(3)
               << projection.pressure.finalResidual.rootMeanSquare << ' '
               << std::abs(*maximum) << ' ' << relative << ' ' << rhs << ' '
               << op.diagonal[index] << ' ' << pressure[index] << ' '
               << divergenceBefore[index] << ' ' << divergenceAfter[index] << ' '
               << gradX << ' ' << gradY << ' ' << interfaceProxy << '\n';
    }

    simulation::MacGrid dynamicSource(32, 32, 1.0f / 32.0f, 1.0f / 32.0f);
    simulation::initializeDamBreak(dynamicSource);
    simulation::MarkerParticleSystem dynamicParticles;
    dynamicParticles.seedFluidCellsExact(dynamicSource, 8);
    simulation::InterfaceGeometrySettings dynamicGeometrySettings;
    dynamicGeometrySettings.particleSupportRadius = static_cast<float>(
        0.78 / (32.0 * std::sqrt(8.0)));
    dynamicGeometrySettings.areaSamplesPerAxis = 2;
    const double initialProxyArea = simulation::reconstructParticleInterface(
        32, 32, 1.0f / 32.0f, 1.0f / 32.0f,
        dynamicParticles.particles(), dynamicGeometrySettings).estimatedArea;
    simulation::GpuDynamicFreeSurfaceSettings dynamicSettings;
    dynamicSettings.pressureFusedCoarseSmootherEnabled = true;
    dynamicSettings.synchronousGpuTimingEnabled = false;
    dynamicSettings.asynchronousGpuTimingEnabled = false;
    simulation::GpuDynamicFreeSurfaceStepper stepper(
        shaderDirectory / "mac_divergence.comp", dynamicSettings);
    require(stepper.initialize(dynamicSource, dynamicParticles).topologyValid,
        "interface area diagnostic topology is invalid");
    const double initialCellArea = static_cast<double>(std::count(
        dynamicSource.cellTypes().begin(), dynamicSource.cellTypes().end(), CellType::fluid)) /
        (32.0 * 32.0);
    double maximumPositive = 0.0;
    double maximumNegative = 0.0;
    std::size_t maximumHoles = 0;
    std::size_t totalFlips = 0;
    double finalProxyArea = 0.0;
    double finalCellArea = initialCellArea;
    for (std::size_t step = 0; step < 30; ++step) {
        const auto report = stepper.step();
        require(stepper.healthy(), "interface area diagnostic dynamic step failed");
        const double area = static_cast<double>(report.fluidCellCount) / (32.0 * 32.0);
        finalCellArea = area;
        maximumPositive = std::max(maximumPositive, area - initialCellArea);
        maximumNegative = std::min(maximumNegative, area - initialCellArea);
        maximumHoles = std::max(maximumHoles, report.classification.enclosedAirCellCount);
        totalFlips += report.classification.phaseToggleCount;
        if (step == 29) {
            finalProxyArea = simulation::reconstructParticleInterface(
                32, 32, 1.0f / 32.0f, 1.0f / 32.0f,
                stepper.grid().readMarkerParticles(), dynamicGeometrySettings).estimatedArea;
        }
    }
    output << "Dynamic area proxy initial-cell=" << std::fixed << std::setprecision(6)
           << initialCellArea << " final-cell="
           << finalCellArea
           << " particle-union-initial=" << initialProxyArea
           << " particle-union-final=" << finalProxyArea
           << " max-positive=" << maximumPositive
           << " max-negative=" << maximumNegative
           << " max-enclosed-air=" << maximumHoles
           << " phase-toggles=" << totalFlips << '\n';
    constexpr double ghostDx = 1.0 / 40.0;
    constexpr double leftTheta = 0.35;
    constexpr double rightTheta = 0.70;
    const double internalCoefficient = 1.0 / (ghostDx * ghostDx);
    const double leftBoundaryCoefficient = internalCoefficient / leftTheta;
    const double rightBoundaryCoefficient = internalCoefficient / rightTheta;
    const double a00 = leftBoundaryCoefficient + internalCoefficient;
    const double a01 = -internalCoefficient;
    const double a10 = -internalCoefficient;
    const double a11 = internalCoefficient + rightBoundaryCoefficient;
    const double determinant = a00 * a11 - a01 * a10;
    const double trace = a00 + a11;
    const double minimumEigenvalue = 0.5 * (trace -
        std::sqrt((a00 - a11) * (a00 - a11) + 4.0 * a01 * a01));
    const double b0 = 1.25;
    const double b1 = -0.40;
    const double p0 = (b0 * a11 - a01 * b1) / determinant;
    const double p1 = (a00 * b1 - a10 * b0) / determinant;
    const double residual0 = a00 * p0 + a01 * p1 - b0;
    const double residual1 = a10 * p0 + a11 * p1 - b1;
    require(std::abs(a01 - a10) == 0.0 && determinant > 0.0 &&
            minimumEigenvalue > 0.0 && std::abs(residual0) < 1.0e-12 &&
            std::abs(residual1) < 1.0e-12,
        "ghost-fluid two-cell reference operator is not symmetric positive definite");
    output << "Ghost-fluid two-cell reference theta=" << leftTheta << ',' << rightTheta
           << " diag=" << a00 << ',' << a11 << " offdiag=" << a01
           << " min-eigenvalue=" << minimumEigenvalue
           << " residual-max=" << std::max(std::abs(residual0), std::abs(residual1)) << '\n';
    output << "Ghost-fluid design: fluid-air diagonal contribution is 1/(theta*h^2), "
              "and the matching face gradient is (p_air-p_fluid)/(theta*h). "
              "Theta must be clamped away from zero and used consistently in projection.\n"
           << "Experimental interface geometry is not connected to the production pressure solve.\n"
           << "Phase 1.9A interface-accuracy validation result: PASS\n";
    return 0;
}

} // namespace fluidlab::diagnostics
