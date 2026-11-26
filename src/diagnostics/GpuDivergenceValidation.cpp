#include "GpuDivergenceValidation.h"

#include "../simulation/gpu/GpuMacGrid.h"
#include "../simulation/mac/Scenarios.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string_view>

namespace fluidlab::diagnostics {
namespace {

struct ErrorMetrics final {
    float maximumAbsolute = 0.0f;
    double meanAbsolute = 0.0;
    double rootMeanSquare = 0.0;
    bool finite = true;
};

[[nodiscard]] ErrorMetrics compare(
    const simulation::MacGrid& grid,
    const std::vector<float>& gpuValues) {
    ErrorMetrics result;
    double sum = 0.0, squared = 0.0;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const std::size_t index = y * grid.width() + x;
            const float expected = grid.cellType(x, y) == simulation::CellType::fluid
                ? grid.divergence(x, y) : 0.0f;
            const float value = gpuValues[index];
            result.finite = result.finite && std::isfinite(value);
            const double error = std::abs(static_cast<double>(value) - expected);
            result.maximumAbsolute = std::max(result.maximumAbsolute, static_cast<float>(error));
            sum += error;
            squared += error * error;
        }
    }
    const double count = static_cast<double>(gpuValues.size());
    result.meanAbsolute = sum / count;
    result.rootMeanSquare = std::sqrt(squared / count);
    return result;
}

void fillConstant(simulation::MacGrid& grid) {
    for (std::size_t y = 0; y < grid.height(); ++y)
        for (std::size_t x = 0; x <= grid.width(); ++x) grid.horizontalVelocity(x, y) = 0.375f;
    for (std::size_t y = 0; y <= grid.height(); ++y)
        for (std::size_t x = 0; x < grid.width(); ++x) grid.verticalVelocity(x, y) = -0.25f;
}

void fillLinear(simulation::MacGrid& grid) {
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x <= grid.width(); ++x) {
            grid.horizontalVelocity(x, y) = 0.75f * static_cast<float>(x) * grid.cellWidth();
        }
    }
    for (std::size_t y = 0; y <= grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            grid.verticalVelocity(x, y) = -0.20f * static_cast<float>(y) * grid.cellHeight();
        }
    }
}

void fillDeterministic(simulation::MacGrid& grid) {
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x <= grid.width(); ++x) {
            grid.horizontalVelocity(x, y) =
                0.013f * static_cast<float>((x * 17 + y * 7) % 31) - 0.19f;
        }
    }
    for (std::size_t y = 0; y <= grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            grid.verticalVelocity(x, y) =
                0.011f * static_cast<float>((x * 5 + y * 19) % 29) - 0.15f;
        }
    }
}

[[nodiscard]] ErrorMetrics execute(
    simulation::GpuMacGrid& gpu,
    simulation::MacGrid& grid,
    double& uploadMilliseconds,
    double& gpuMilliseconds) {
    grid.computeDivergence();
    gpu.upload(grid);
    gpu.dispatchDivergence();
    const std::vector<float> values = gpu.readDivergence();
    uploadMilliseconds = gpu.lastUploadMilliseconds();
    gpuMilliseconds = gpu.lastGpuMilliseconds();
    return compare(grid, values);
}

void printCase(
    std::ostream& output,
    std::string_view name,
    const simulation::MacGrid& grid,
    const ErrorMetrics& error,
    double uploadMilliseconds,
    double gpuMilliseconds,
    float tolerance) {
    const bool passed = error.finite && error.maximumAbsolute <= tolerance;
    output << std::left << std::setw(22) << name
           << std::right << std::setw(5) << grid.width() << 'x' << std::setw(4) << grid.height()
           << std::scientific << std::setprecision(3)
           << std::setw(13) << error.maximumAbsolute
           << std::setw(13) << error.meanAbsolute
           << std::setw(13) << error.rootMeanSquare
           << std::fixed << std::setprecision(4)
           << std::setw(12) << uploadMilliseconds
           << std::setw(12) << gpuMilliseconds
           << std::setw(8) << (passed ? "PASS" : "FAIL") << '\n';
    if (!passed) throw std::runtime_error("GPU divergence validation exceeded its error tolerance.");
}

} // namespace

int runGpuDivergenceValidation(
    std::ostream& output,
    const std::filesystem::path& divergenceShaderPath) {
    constexpr float tolerance = 1.0e-4f;
    simulation::GpuMacGrid gpu(divergenceShaderPath);
    output << "LiquidSimulator Phase 1.4.5 GPU MAC divergence validation\n"
           << "Non-FLUID output rule: zero. CPU/GPU tolerance: 1e-4.\n"
           << std::left << std::setw(22) << "Case" << std::right << std::setw(10) << "Grid"
           << std::setw(13) << "Max abs" << std::setw(13) << "Mean abs"
           << std::setw(13) << "RMS" << std::setw(12) << "Upload ms"
           << std::setw(12) << "GPU ms" << std::setw(8) << "Result" << '\n';

    const auto run = [&](std::string_view name, simulation::MacGrid& grid) {
        double upload = 0.0, gpuTime = 0.0;
        const ErrorMetrics error = execute(gpu, grid, upload, gpuTime);
        printCase(output, name, grid, error, upload, gpuTime, tolerance);
    };

    simulation::MacGrid zero(17, 13, 1.0f / 17.0f, 1.0f / 13.0f);
    run("zero velocity", zero);
    simulation::MacGrid constant(17, 13, 1.0f / 17.0f, 1.0f / 13.0f);
    fillConstant(constant);
    run("constant velocity", constant);
    simulation::MacGrid linear(17, 13, 1.0f / 17.0f, 1.0f / 13.0f);
    fillLinear(linear);
    run("linear analytic", linear);
    const float analytic = 0.55f;
    float analyticError = 0.0f;
    linear.computeDivergence();
    for (const float value : linear.divergences()) analyticError = std::max(analyticError, std::abs(value - analytic));
    output << "  linear analytic divergence=0.55, CPU max analytic error="
           << std::scientific << analyticError << '\n';

    simulation::MacGrid classified(31, 23, 1.0f / 31.0f, 1.0f / 31.0f);
    simulation::initializeDamBreak(classified);
    fillDeterministic(classified);
    classified.enforceSolidBoundaries();
    run("fluid-air-solid", classified);

    for (const auto dimensions : {std::pair{64U, 64U}, std::pair{128U, 128U},
                                  std::pair{256U, 256U}, std::pair{96U, 128U}}) {
        simulation::MacGrid grid(dimensions.first, dimensions.second,
            1.0f / static_cast<float>(dimensions.first),
            1.0f / static_cast<float>(dimensions.first));
        simulation::initializeDamBreak(grid);
        fillDeterministic(grid);
        grid.enforceSolidBoundaries();
        run("resolution/classified", grid);
    }
    output << "GPU timer covers only the compute dispatch. Upload time is CPU submission time; readback synchronization is excluded.\n";
    return 0;
}

} // namespace fluidlab::diagnostics
