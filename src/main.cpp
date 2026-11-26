#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

#include "graphics/ShaderProgram.h"
#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
#include "diagnostics/CpuBenchmark.h"
#include "diagnostics/DynamicBenchmark.h"
#include "diagnostics/GpuDivergenceValidation.h"
#include "diagnostics/GpuMacValidation.h"
#include "diagnostics/InterfaceValidation.h"
#include "diagnostics/LevelSetDynamicValidation.h"
#include "diagnostics/GpuLevelSetValidation.h"
#include "diagnostics/GpuVofValidation.h"
#include "diagnostics/VofValidation.h"
#include "diagnostics/DesktopInteractionValidation.h"
#include "tests/Phase1Tests.h"
#endif
#include "desktop/ContainerMotionInput.h"
#include "desktop/DesktopPerformanceMetrics.h"
#include "desktop/DesktopSettledState.h"
#include "desktop/WindowsMoveLoopAdapter.h"
#include "simulation/mac/FluidStepper.h"
#include "simulation/mac/DynamicFreeSurfaceStepper.h"
#include "simulation/mac/MarkerParticles.h"
#include "simulation/mac/MacGrid.h"
#include "simulation/mac/PressureProjection.h"
#include "simulation/mac/Scenarios.h"
#include "simulation/gpu/GpuDynamicFreeSurfaceStepper.h"
#include "simulation/gpu/GpuAsyncDynamicFreeSurfaceStepper.h"
#include "visualization/FieldVisualization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace {

constexpr int kWindowWidth = 1024;
constexpr int kWindowHeight = 768;
#if defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
constexpr bool kDesktopRelease = true;
#else
constexpr bool kDesktopRelease = false;
constexpr GLsizei kPhase0TextureWidth = 256;
constexpr GLsizei kPhase0TextureHeight = 256;
constexpr GLuint kComputeLocalSizeX = 8;
constexpr GLuint kComputeLocalSizeY = 8;
constexpr std::size_t kPhase1GridSize = 64;
#endif

struct DesktopRenderStyle final {
    bool gridEnabled = true;
    float waterAlpha = 0.78f;
    float gridStrength = 1.0f;
    float gridCellSpacing = 8.0f;
};

struct DesktopFluidInjectionSettings final {
    bool enabled = true;
    double areaRate = 0.03;
    float radiusInCells = 2.5f;
};

void applyDesktopRenderStyle(
    const fluidlab::graphics::ShaderProgram& program,
    bool enabled,
    const DesktopRenderStyle& style) {
    glUniform1i(program.uniformLocation("u_desktopStyleEnabled"), enabled ? 1 : 0);
    glUniform1i(program.uniformLocation("u_gridEnabled"), style.gridEnabled ? 1 : 0);
    glUniform1f(program.uniformLocation("u_waterAlpha"), style.waterAlpha);
    glUniform1f(program.uniformLocation("u_gridStrength"), style.gridStrength);
    glUniform1f(program.uniformLocation("u_gridCellSpacing"), style.gridCellSpacing);
    glUniform3f(program.uniformLocation("u_backgroundColor"),
        16.0f / 255.0f, 23.0f / 255.0f, 34.0f / 255.0f);
    glUniform3f(program.uniformLocation("u_gridColor"),
        43.0f / 255.0f, 57.0f / 255.0f, 75.0f / 255.0f);
    glUniform3f(program.uniformLocation("u_waterColor"),
        66.0f / 255.0f, 155.0f / 255.0f, 206.0f / 255.0f);
}

struct GlfwTerminator final {
    void operator()(GLFWwindow* window) const noexcept {
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
    }
};

using WindowPtr = std::unique_ptr<GLFWwindow, GlfwTerminator>;

struct OpenGlObjects final {
    GLuint texture = 0;
    GLuint fullscreenVertexArray = 0;
    GLuint vectorVertexArray = 0;
    GLuint vectorBuffer = 0;

    ~OpenGlObjects() {
        if (vectorBuffer != 0) {
            glDeleteBuffers(1, &vectorBuffer);
        }
        if (vectorVertexArray != 0) {
            glDeleteVertexArrays(1, &vectorVertexArray);
        }
        if (fullscreenVertexArray != 0) {
            glDeleteVertexArrays(1, &fullscreenVertexArray);
        }
        if (texture != 0) {
            glDeleteTextures(1, &texture);
        }
    }
};

#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
enum class FieldView {
    velocity,
    divergenceBefore,
    divergenceAfter,
    pressure,
    cellTypes,
};

enum class ApplicationMode {
    phase1Interactive,
    phase1VisualValidation,
    phase1VisualBenchmark,
    phase0Validation,
    phase14DamBreak,
    phase14DamBreakValidation,
    phase145GpuDivergenceValidation,
    phase145FrameBenchmark,
    phase15GpuMacValidation,
    phase155GpuPressureOptimization,
    phase156GpuMultigridValidation,
    phase16GpuDynamicValidation,
    phase16GpuDynamicTopologyValidation,
    phase16GpuDamBreak,
    phase16GpuDamBreakValidation,
    phase17GpuInterfaceValidation,
    phase17AdaptiveValidation,
    phase18GpuResidentPressureValidation,
    phase181GpuAsyncValidation,
    phase181GpuAsyncBenchmark,
    phase182GpuMixedPressureValidation,
    phase182GpuSolverBenchmark,
    phase183GpuMgpcgProfile,
    phase183GpuMixedReliabilityValidation,
    phase183GpuMixedLongValidation,
    phase19GpuCoarseFusionRegression,
    phase19GpuInterfaceAccuracyValidation,
    phase19bInterfaceReconstructionValidation,
    phase19bGhostFluidCpuValidation,
    phase19b2LevelSetDynamicValidation,
    phase19b2LevelSetRedistanceBenchmark,
    phase19b2LevelSetParticleConsistencyValidation,
    phase19b2GhostFluidDynamicCpuValidation,
    phase19c1GpuLevelSetValidation,
    phase19c1GpuLevelSetDynamicValidation,
    phase19c1GpuLevelSetRollbackValidation,
    phase19c1GpuLevelSetBenchmark,
    phase19c1bGpuLevelSetAreaAudit,
    phase19c1bGpuLevelSetAdvectionCompare,
    phase19c1bGpuLevelSetLongValidation,
    phase19c1cGpuLevelSetRedistanceValidation,
    phase19c1cGpuLevelSetExtensionValidation,
    phase19c1cGpuLevelSetConservationCompare,
    phase19dVofMacReplayValidation,
    phase19eGpuVofFluxValidation,
    phase19eGpuVofPlicValidation,
    phase19eGpuVofSplitValidation,
    phase19eGpuVofRollbackValidation,
    phase19eGpuVofBenchmark,
    phase19eGpuVofMacDiagnostic,
    phase19eBGpuVofPrecisionAudit,
    phase19eBGpuVofMultistepValidation,
    phase19eBGpuVofBoundedFluxValidation,
    phase19eBGpuVofMacReplay,
    phase19eBGpuVofVelocityCompatibility,
    phase19eBGpuVofMultistepBenchmark,
    phase19eCGpuVofRoundoffAudit,
    phase19eCGpuVofCommitSafetyValidation,
    phase19eCGpuVofConservativeRepairValidation,
    phase19eCGpuVofBandVelocityValidation,
    phase19eCGpuVofLongValidation,
    phase19eCGpuVofDynamicCompare,
    phase19eCGpuVofSafetyBenchmark,
    phase20DesktopInteractive,
    phase20DesktopInteractiveDamBreak,
    phase20ExternalForceValidation,
    phase20ContainerMotionValidation,
    phase20ContainerInertiaValidation,
    phase20DesktopInputTimingValidation,
    phase20DesktopInteractionBenchmark,
    phase20DesktopFramePacingValidation,
    phase20DesktopModalLoopValidation,
    phase20DesktopSettleGenerate,
    phase20DesktopSettleLoadValidation,
    phase20DesktopInertiaFromSettledValidation,
    phase20DesktopInertiaPipelineAudit,
    phase20DesktopInertiaTimingValidation,
    phase20DesktopInertiaResponseValidation,
    phase20DesktopInertiaLatencyBenchmark,
    phase20DesktopGridRenderValidation,
    phase20DesktopWaterAlphaValidation,
    phase20DesktopRenderStyleBenchmark,
    phase20DesktopSchedulerValidation,
    phase20DesktopAsyncThroughputValidation,
    phase20DesktopBatchInputAudit,
    phase20DesktopThroughputAccountingValidation,
    phase20DesktopSustainedRealtimeBenchmark,
    phase20GpuFluidInjectionValidation,
};

struct Phase1State final {
    fluidlab::simulation::MacGrid grid;
    std::vector<float> divergenceBeforeProjection;
    fluidlab::simulation::ProjectionReport initialProjection;
    fluidlab::simulation::FluidStepReport lastStep;
    std::size_t completedSteps = 0;
};
#endif

struct TimingAccumulator final {
    double sum = 0.0;
    double maximum = 0.0;
    std::size_t samples = 0;
    std::vector<double> values;

    void add(double milliseconds) noexcept {
        sum += milliseconds;
        maximum = std::max(maximum, milliseconds);
        ++samples;
        values.push_back(milliseconds);
        if (values.size() > 240) values.erase(values.begin());
    }

    [[nodiscard]] double average() const noexcept {
        return samples != 0 ? sum / static_cast<double>(samples) : 0.0;
    }

    [[nodiscard]] double percentile(double fraction) const {
        if (values.empty()) return 0.0;
        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t index = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(sorted.size()))) - 1;
        return sorted[std::min(index, sorted.size() - 1)];
    }

    void clear() noexcept {
        sum = 0.0;
        maximum = 0.0;
        samples = 0;
        values.clear();
    }
};

#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
struct PerformanceWindow final {
    TimingAccumulator frame;
    TimingAccumulator physics;
    TimingAccumulator advection;
    TimingAccumulator pressureSolve;
    TimingAccumulator projectionOverhead;
    TimingAccumulator visualizationBuild;
    TimingAccumulator openGlUpload;
    std::size_t substeps = 0;
    std::size_t frames = 0;
    std::size_t lastPressureIterations = 0;
    float lastPressureResidual = 0.0f;
    double droppedWallTime = 0.0;

    void addStep(const fluidlab::simulation::FluidStepReport& report) noexcept {
        physics.add(report.timings.totalMilliseconds);
        advection.add(report.timings.velocityAdvectionMilliseconds);
        pressureSolve.add(report.projection.timings.pressureSolveMilliseconds);
        projectionOverhead.add(
            report.projection.timings.totalMilliseconds -
            report.projection.timings.pressureSolveMilliseconds);
        lastPressureIterations = report.projection.pressureSolve.iterations;
        lastPressureResidual = report.projection.pressureSolve.finalResidual.rootMeanSquare;
        ++substeps;
    }

    void clearSamples() noexcept {
        frame.clear();
        physics.clear();
        advection.clear();
        pressureSolve.clear();
        projectionOverhead.clear();
        visualizationBuild.clear();
        openGlUpload.clear();
        substeps = 0;
        frames = 0;
    }
};

struct VisualizationBuffers final {
    std::vector<float> pixels;
    std::vector<float> arrowVertices;
};
#endif

void glfwErrorCallback(int errorCode, const char* description) {
    std::cerr << "[GLFW " << errorCode << "] " << (description != nullptr ? description : "Unknown error") << '\n';
}

void APIENTRY openGlDebugCallback(
    GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei,
    const GLchar* message,
    const void*) {
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
        return;
    }

    std::cerr << "[OpenGL] source=0x" << std::hex << source
              << " type=0x" << type
              << " severity=0x" << severity
              << " id=" << std::dec << id
              << " message=" << (message != nullptr ? message : "<null>") << '\n';
}

void framebufferSizeCallback(GLFWwindow*, int width, int height) {
    glViewport(0, 0, std::max(width, 0), std::max(height, 0));
}

[[nodiscard]] std::filesystem::path executableDirectory() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(1024);
    while (true) {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            throw std::runtime_error("GetModuleFileNameW failed while locating shader resources.");
        }
        if (length < buffer.size() - 1) {
            return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
        }
        buffer.resize(buffer.size() * 2);
    }
#else
    return std::filesystem::current_path();
#endif
}

[[nodiscard]] std::filesystem::path findResource(const std::filesystem::path& relativePath) {
    const std::filesystem::path executablePath = executableDirectory();

#if defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
    // A distributed executable must use its own assets even from another working directory.
    const std::array candidates{executablePath / relativePath};
#else
    // Prefer packaged resources beside the executable, then support local and
    // Visual Studio development layouts without requiring a fixed work directory.
    const std::array candidates{
        executablePath / relativePath,
        std::filesystem::current_path() / relativePath,
        executablePath / ".." / ".." / ".." / relativePath,
    };
#endif

    for (const auto& candidate : candidates) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error)) {
            return std::filesystem::weakly_canonical(candidate);
        }
    }
    throw std::runtime_error("Unable to locate resource: " + relativePath.string());
}

[[nodiscard]] bool isAtLeastOpenGl46() {
    GLint major = 0;
    GLint minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    return major > 4 || (major == 4 && minor >= 6);
}

void printOpenGlInformation() {
    const auto text = [](GLenum name) {
        const GLubyte* value = glGetString(name);
        return value != nullptr ? reinterpret_cast<const char*>(value) : "<unavailable>";
    };

    std::cout << "OpenGL vendor:   " << text(GL_VENDOR) << '\n'
              << "OpenGL renderer: " << text(GL_RENDERER) << '\n'
              << "OpenGL version:  " << text(GL_VERSION) << '\n'
              << "GLSL version:    " << text(GL_SHADING_LANGUAGE_VERSION) << '\n';
}


#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
[[nodiscard]] std::string_view pressureTerminationDescription(
    fluidlab::simulation::PressureSolveTermination termination) noexcept {
    using fluidlab::simulation::PressureSolveTermination;
    switch (termination) {
    case PressureSolveTermination::initialResidualWithinTolerance:
        return "Initial RMS residual is within tolerance; no SOR iterations performed";
    case PressureSolveTermination::iterativeResidualWithinTolerance:
        return "RMS residual converged after SOR iterations";
    case PressureSolveTermination::maximumIterationsReached:
        return "Maximum iterations reached; RMS residual remains above tolerance";
    }
    return "Unknown";
}

void printProjectionReport(const fluidlab::simulation::ProjectionReport& report) {
    std::cout << std::scientific << std::setprecision(6)
        << "Initial Pressure Projection Diagnostics:\n"
        << "  Maximum divergence (before -> after): "
        << report.divergenceBefore.maximumAbsolute
        << " -> " << report.divergenceAfter.maximumAbsolute << '\n'
        << "  Mean absolute divergence (before -> after): "
        << report.divergenceBefore.meanAbsolute
        << " -> " << report.divergenceAfter.meanAbsolute << '\n'
        << "  Pressure residual RMS (initial -> final): "
        << report.pressureSolve.initialResidual.rootMeanSquare
        << " -> " << report.pressureSolve.finalResidual.rootMeanSquare << '\n'
        << "  Maximum pressure residual (initial -> final): "
        << report.pressureSolve.initialResidual.maximumAbsolute
        << " -> " << report.pressureSolve.finalResidual.maximumAbsolute << '\n'
        << "  Pressure convergence tolerance (RMS): "
        << report.pressureSolve.residualTolerance << '\n'
        << "  Maximum absolute pressure (initial -> final): "
        << report.pressureSolve.initialMaximumAbsolutePressure
        << " -> " << report.pressureSolve.finalMaximumAbsolutePressure << '\n'
        << "  Maximum velocity change after projection: "
        << report.maximumVelocityChange << '\n'
        << "  Iterations performed: " << report.pressureSolve.iterations << '\n'
        << "  Converged: " << (report.pressureSolve.converged ? "Yes" : "No") << '\n'
        << "  Termination reason: "
        << pressureTerminationDescription(report.pressureSolve.termination) << "\n\n";
}


[[nodiscard]] Phase1State createPhase1State(std::size_t gridSize = kPhase1GridSize) {
    const float cellSize = 1.0f / static_cast<float>(gridSize);
    fluidlab::simulation::MacGrid grid(
        gridSize,
        gridSize,
        cellSize,
        cellSize);
    fluidlab::simulation::initializeClosedVortex(grid);
    std::vector<float> divergenceBeforeProjection = grid.divergences();

    const fluidlab::simulation::PressureProjector projector;
    fluidlab::simulation::ProjectionReport initialProjection = projector.project(grid);
    fluidlab::simulation::initializeLiquidRectangle(grid, 1, 1, gridSize - 1, gridSize / 3);
    for (std::size_t y = 0; y < gridSize; ++y) {
        grid.setCellType(0, y, fluidlab::simulation::CellType::solid);
        grid.setCellType(gridSize - 1, y, fluidlab::simulation::CellType::solid);
    }
    for (std::size_t x = 0; x < gridSize; ++x) {
        grid.setCellType(x, 0, fluidlab::simulation::CellType::solid);
    }
    if (!grid.allFinite()) {
        throw std::runtime_error("Phase 1 projection produced NaN or infinity.");
    }
    printProjectionReport(initialProjection);
    return {
        std::move(grid),
        std::move(divergenceBeforeProjection),
        initialProjection,
        {},
        0,
    };
}

int runPhase0Validation(int framebufferWidth, int framebufferHeight) {
    const auto computeProgram = fluidlab::graphics::ShaderProgram::fromComputeFile(
        findResource("shaders/compute/phase0.comp"));
    const auto displayProgram = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/fullscreen.vert"),
        findResource("shaders/rendering/texture.frag"));
    const GLint timeLocation = computeProgram.uniformLocation("u_time");
    const GLint outputSizeLocation = computeProgram.uniformLocation("u_outputSize");

    OpenGlObjects objects;
    glCreateTextures(GL_TEXTURE_2D, 1, &objects.texture);
    glCreateVertexArrays(1, &objects.fullscreenVertexArray);
    if (objects.texture == 0 || objects.fullscreenVertexArray == 0) {
        throw std::runtime_error("Failed to create Phase 0 OpenGL resources.");
    }

    glTextureStorage2D(objects.texture, 1, GL_RGBA32F, kPhase0TextureWidth, kPhase0TextureHeight);
    glTextureParameteri(objects.texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(objects.texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(objects.texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(objects.texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindImageTexture(0, objects.texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    computeProgram.use();
    glUniform1f(timeLocation, 0.0f);
    glUniform2i(outputSizeLocation, kPhase0TextureWidth, kPhase0TextureHeight);
    glDispatchCompute(
        (static_cast<GLuint>(kPhase0TextureWidth) + kComputeLocalSizeX - 1) / kComputeLocalSizeX,
        (static_cast<GLuint>(kPhase0TextureHeight) + kComputeLocalSizeY - 1) / kComputeLocalSizeY,
        1);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    std::array<float, 4> texturePixel{};
    glGetTextureSubImage(
        objects.texture,
        0,
        kPhase0TextureWidth / 2,
        kPhase0TextureHeight / 2,
        0,
        1,
        1,
        1,
        GL_RGBA,
        GL_FLOAT,
        static_cast<GLsizei>(sizeof(texturePixel)),
        texturePixel.data());

    glClearColor(0.015f, 0.02f, 0.035f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    displayProgram.use();
    glBindTextureUnit(0, objects.texture);
    glBindVertexArray(objects.fullscreenVertexArray);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    std::array<float, 4> framebufferPixel{};
    glReadBuffer(GL_BACK);
    glReadPixels(
        framebufferWidth / 2,
        framebufferHeight / 2,
        1,
        1,
        GL_RGBA,
        GL_FLOAT,
        framebufferPixel.data());

    const bool finitePixels = std::all_of(texturePixel.begin(), texturePixel.end(), [](float value) {
        return std::isfinite(value);
    }) && std::all_of(framebufferPixel.begin(), framebufferPixel.end(), [](float value) {
        return std::isfinite(value);
    });
    const bool matchingColor = std::equal(
        texturePixel.begin(),
        texturePixel.end(),
        framebufferPixel.begin(),
        [](float expected, float actual) { return std::abs(expected - actual) < 0.006f; });
    if (!finitePixels || !matchingColor) {
        throw std::runtime_error("Phase 0 compute/render validation returned an invalid pixel.");
    }

    std::cout << "Compute validation pixel: ["
              << texturePixel[0] << ", " << texturePixel[1] << ", "
              << texturePixel[2] << ", " << texturePixel[3] << "]\n"
              << "Framebuffer validation pixel: ["
              << framebufferPixel[0] << ", " << framebufferPixel[1] << ", "
              << framebufferPixel[2] << ", " << framebufferPixel[3] << "]\n"
              << "Phase 0 GPU compute/render validation passed.\n";
    return 0;
}

[[nodiscard]] const char* fieldViewName(FieldView view) noexcept {
    switch (view) {
    case FieldView::velocity:
        return "Velocity";
    case FieldView::divergenceBefore:
        return "Divergence after advection";
    case FieldView::divergenceAfter:
        return "Divergence after projection";
    case FieldView::pressure:
        return "Pressure";
    case FieldView::cellTypes:
        return "Cell types";
    }
    return "Unknown";
}

void uploadFieldTexture(
    GLuint texture,
    const Phase1State& state,
    FieldView view,
    VisualizationBuffers& buffers,
    PerformanceWindow& performance) {
    using Clock = std::chrono::steady_clock;
    const auto buildStart = Clock::now();
    switch (view) {
    case FieldView::velocity:
        fluidlab::visualization::buildSpeedTexture(state.grid, buffers.pixels);
        break;
    case FieldView::divergenceBefore:
        fluidlab::visualization::buildSignedScalarTexture(
            state.divergenceBeforeProjection,
            buffers.pixels);
        break;
    case FieldView::divergenceAfter:
        fluidlab::visualization::buildSignedScalarTexture(state.grid.divergences(), buffers.pixels);
        break;
    case FieldView::pressure:
        fluidlab::visualization::buildSignedScalarTexture(state.grid.pressures(), buffers.pixels);
        break;
    case FieldView::cellTypes:
        fluidlab::visualization::buildCellTypeTexture(state.grid, buffers.pixels);
        break;
    }
    const auto buildEnd = Clock::now();

    glTextureSubImage2D(
        texture,
        0,
        0,
        0,
        static_cast<GLsizei>(state.grid.width()),
        static_cast<GLsizei>(state.grid.height()),
        GL_RGBA,
        GL_FLOAT,
        buffers.pixels.data());
    const auto uploadEnd = Clock::now();
    performance.visualizationBuild.add(
        std::chrono::duration<double, std::milli>(buildEnd - buildStart).count());
    performance.openGlUpload.add(
        std::chrono::duration<double, std::milli>(uploadEnd - buildEnd).count());
}

int runPhase1Visualization(
    GLFWwindow* window,
    ApplicationMode mode,
    std::size_t gridSize,
    bool warmStartEnabled) {
    const bool validationMode = mode == ApplicationMode::phase1VisualValidation;
    const bool benchmarkMode = mode == ApplicationMode::phase1VisualBenchmark;
    Phase1State state = createPhase1State(gridSize);
    fluidlab::simulation::CpuFluidStepper stepper(fluidlab::simulation::FluidStepSettings{
        .pressureWarmStartEnabled = warmStartEnabled,
    });
    PerformanceWindow performance;
    const auto advanceSimulation = [&]() {
        fluidlab::simulation::FluidStepReport report = stepper.step(state.grid);
        const auto& divergence = stepper.divergenceAfterAdvectionField();
        state.divergenceBeforeProjection.assign(divergence.begin(), divergence.end());
        performance.addStep(report);
        state.lastStep = std::move(report);
        ++state.completedSteps;
        if (!state.grid.allFinite()) {
            throw std::runtime_error("Phase 1.2 time step produced NaN or infinity.");
        }
    };
    if (validationMode) {
        advanceSimulation();
        std::cout << "Phase 1.2 validation step:\n"
                  << "  divergence max (advected -> projected): "
                  << state.lastStep.divergenceAfterAdvection.maximumAbsolute << " -> "
                  << state.lastStep.projection.divergenceAfter.maximumAbsolute << '\n'
                  << "  divergence mean (advected -> projected): "
                  << state.lastStep.divergenceAfterAdvection.meanAbsolute << " -> "
                  << state.lastStep.projection.divergenceAfter.meanAbsolute << '\n'
                  << "  kinetic energy (before -> advected -> projected): "
                  << state.lastStep.kineticEnergyBefore << " -> "
                  << state.lastStep.kineticEnergyAfterAdvection << " -> "
                  << state.lastStep.kineticEnergyAfterProjection << '\n'
                  << "  warm start: "
                  << (state.lastStep.projection.pressureSolve.warmStartUsed ? "enabled" : "disabled")
                  << ", iterations: " << state.lastStep.projection.pressureSolve.iterations << '\n'
                  << "  timings ms (step/advection/pressure): "
                  << state.lastStep.timings.totalMilliseconds << " / "
                  << state.lastStep.timings.velocityAdvectionMilliseconds << " / "
                  << state.lastStep.projection.timings.pressureSolveMilliseconds << "\n\n";
    }
    const auto displayProgram = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/fullscreen.vert"),
        findResource("shaders/rendering/texture.frag"));
    const auto vectorProgram = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/vectors.vert"),
        findResource("shaders/rendering/vectors.frag"));

    OpenGlObjects objects;
    glCreateTextures(GL_TEXTURE_2D, 1, &objects.texture);
    glCreateVertexArrays(1, &objects.fullscreenVertexArray);
    glCreateVertexArrays(1, &objects.vectorVertexArray);
    glCreateBuffers(1, &objects.vectorBuffer);
    if (objects.texture == 0 || objects.fullscreenVertexArray == 0 ||
        objects.vectorVertexArray == 0 || objects.vectorBuffer == 0) {
        throw std::runtime_error("Failed to create Phase 1 visualization resources.");
    }

    glTextureStorage2D(
        objects.texture,
        1,
        GL_RGBA32F,
        static_cast<GLsizei>(state.grid.width()),
        static_cast<GLsizei>(state.grid.height()));
    glTextureParameteri(objects.texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(objects.texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(objects.texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(objects.texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glVertexArrayVertexBuffer(objects.vectorVertexArray, 0, objects.vectorBuffer, 0, 2 * sizeof(float));
    glEnableVertexArrayAttrib(objects.vectorVertexArray, 0);
    glVertexArrayAttribFormat(objects.vectorVertexArray, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(objects.vectorVertexArray, 0, 0);

    VisualizationBuffers visualizationBuffers;
    visualizationBuffers.pixels.reserve(state.grid.width() * state.grid.height() * 4);
    constexpr std::size_t arrowSamplingStride = 4;
    const std::size_t maximumArrowSamples =
        ((state.grid.width() + arrowSamplingStride - 1) / arrowSamplingStride) *
        ((state.grid.height() + arrowSamplingStride - 1) / arrowSamplingStride);
    const GLsizeiptr arrowBufferCapacity = static_cast<GLsizeiptr>(
        std::max<std::size_t>(maximumArrowSamples * 12 * sizeof(float), 2 * sizeof(float)));
    visualizationBuffers.arrowVertices.reserve(maximumArrowSamples * 12);
    glNamedBufferData(objects.vectorBuffer, arrowBufferCapacity, nullptr, GL_DYNAMIC_DRAW);

    GLsizei arrowVertexCount = 0;
    const auto updateArrowBuffer = [&]() {
        using Clock = std::chrono::steady_clock;
        const auto buildStart = std::chrono::steady_clock::now();
        fluidlab::visualization::buildVelocityArrowVertices(
            state.grid,
            arrowSamplingStride,
            visualizationBuffers.arrowVertices);
        const auto buildEnd = std::chrono::steady_clock::now();
        const GLsizeiptr arrowByteCount = static_cast<GLsizeiptr>(
            visualizationBuffers.arrowVertices.size() * sizeof(float));
        if (arrowByteCount > 0) {
            glNamedBufferSubData(
                objects.vectorBuffer,
                0,
                arrowByteCount,
                visualizationBuffers.arrowVertices.data());
        }
        const auto uploadEnd = std::chrono::steady_clock::now();
        performance.visualizationBuild.add(
            std::chrono::duration<double, std::milli>(buildEnd - buildStart).count());
        performance.openGlUpload.add(
            std::chrono::duration<double, std::milli>(uploadEnd - buildEnd).count());
        arrowVertexCount = static_cast<GLsizei>(visualizationBuffers.arrowVertices.size() / 2);
    };
    updateArrowBuffer();

    FieldView currentView = FieldView::velocity;
    uploadFieldTexture(objects.texture, state, currentView, visualizationBuffers, performance);

    const auto renderFrame = [&](FieldView view) {
        glClearColor(0.01f, 0.015f, 0.025f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        displayProgram.use();
        glBindTextureUnit(0, objects.texture);
        glBindVertexArray(objects.fullscreenVertexArray);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        if (view == FieldView::velocity && arrowVertexCount > 0) {
            vectorProgram.use();
            glBindVertexArray(objects.vectorVertexArray);
            glDrawArrays(GL_LINES, 0, arrowVertexCount);
        }
    };

    if (validationMode) {
        const std::array views{
            FieldView::velocity,
            FieldView::divergenceBefore,
            FieldView::divergenceAfter,
            FieldView::pressure,
            FieldView::cellTypes,
        };
        for (const FieldView view : views) {
            uploadFieldTexture(
                objects.texture,
                state,
                view,
                visualizationBuffers,
                performance);
            renderFrame(view);

            int framebufferWidth = 0;
            int framebufferHeight = 0;
            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            std::array<float, 4> centerPixel{};
            glReadBuffer(GL_BACK);
            glReadPixels(
                framebufferWidth / 2,
                framebufferHeight / 2,
                1,
                1,
                GL_RGBA,
                GL_FLOAT,
                centerPixel.data());
            const bool finite = std::all_of(centerPixel.begin(), centerPixel.end(), [](float value) {
                return std::isfinite(value);
            });
            if (!finite || std::abs(centerPixel[3] - 1.0f) > 0.01f) {
                throw std::runtime_error("Phase 1 visualization produced an invalid framebuffer pixel.");
            }
        }
        std::cout << "Phase 1.3 visualization validation passed for all five field views.\n";
        return 0;
    }

    if (benchmarkMode) {
        constexpr std::size_t benchmarkWarmupSteps = 3;
        const std::size_t benchmarkFrames = gridSize <= 64 ? 60 : (gridSize <= 128 ? 20 : 3);
        for (std::size_t step = 0; step < benchmarkWarmupSteps; ++step) {
            advanceSimulation();
        }
        performance.clearSamples();
        for (std::size_t frame = 0; frame < benchmarkFrames; ++frame) {
            const auto frameStart = std::chrono::steady_clock::now();
            advanceSimulation();
            updateArrowBuffer();
            uploadFieldTexture(
                objects.texture,
                state,
                currentView,
                visualizationBuffers,
                performance);
            renderFrame(currentView);
            glFinish();
            const auto frameEnd = std::chrono::steady_clock::now();
            performance.frame.add(
                std::chrono::duration<double, std::milli>(frameEnd - frameStart).count());
            ++performance.frames;
        }

        const double averageFrame = performance.frame.average();
        std::cout << std::fixed << std::setprecision(3)
                  << "Phase 1.2.5 visual throughput benchmark (" << gridSize << " x "
                  << gridSize << ", warm start " << (warmStartEnabled ? "on" : "off") << ", "
                  << benchmarkFrames << " frames after " << benchmarkWarmupSteps
                  << " warmup steps):\n"
                  << "  frame ms average/max: " << averageFrame << " / "
                  << performance.frame.maximum << '\n'
                  << "  throughput FPS: " << (averageFrame > 0.0 ? 1000.0 / averageFrame : 0.0) << '\n'
                  << "  physics ms average/max: " << performance.physics.average() << " / "
                  << performance.physics.maximum << '\n'
                  << "  advection ms average/max: " << performance.advection.average() << " / "
                  << performance.advection.maximum << '\n'
                  << "  pressure solve ms average/max: " << performance.pressureSolve.average() << " / "
                  << performance.pressureSolve.maximum << '\n'
                  << "  divergence/projection ms average/max: "
                  << performance.projectionOverhead.average() << " / "
                  << performance.projectionOverhead.maximum << '\n'
                  << "  visualization build ms average/max: "
                  << performance.visualizationBuild.average() << " / "
                  << performance.visualizationBuild.maximum << '\n'
                  << "  OpenGL upload call ms average/max: "
                  << performance.openGlUpload.average() << " / "
                  << performance.openGlUpload.maximum << '\n'
                  << "  final pressure iterations/residual: "
                  << performance.lastPressureIterations << " / "
                  << std::scientific << performance.lastPressureResidual << '\n';
        return 0;
    }

    std::cout << "Visualization controls: 1 velocity, 2 divergence after advection, "
                 "3 divergence after, 4 pressure, 5 cell types, Space pause, N single step, "
                 "R reset, Escape exit.\n";

    bool paused = false;
    bool previousPauseKey = false;
    bool previousStepKey = false;
    bool previousResetKey = false;
    double previousWallTime = glfwGetTime();
    double statisticsStartTime = previousWallTime;
    double accumulatedTime = 0.0;
    constexpr std::size_t maximumSubstepsPerFrame = 4;

    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        const bool pauseKey = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        const bool stepKey = glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS;
        const bool resetKey = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
        if (pauseKey && !previousPauseKey) {
            paused = !paused;
        }

        bool simulationChanged = false;
        if (resetKey && !previousResetKey) {
            state = createPhase1State(gridSize);
            accumulatedTime = 0.0;
            performance.clearSamples();
            previousWallTime = glfwGetTime();
            statisticsStartTime = previousWallTime;
            simulationChanged = true;
        }
        if (paused && stepKey && !previousStepKey) {
            advanceSimulation();
            simulationChanged = true;
        }
        previousPauseKey = pauseKey;
        previousStepKey = stepKey;
        previousResetKey = resetKey;

        const double currentWallTime = glfwGetTime();
        const double frameDuration = std::clamp(currentWallTime - previousWallTime, 0.0, 0.25);
        previousWallTime = currentWallTime;
        performance.frame.add(frameDuration * 1000.0);
        ++performance.frames;
        if (!paused) {
            accumulatedTime += frameDuration;
            std::size_t substeps = 0;
            const double fixedTimeStep = static_cast<double>(stepper.settings().timeStep);
            while (accumulatedTime >= fixedTimeStep && substeps < maximumSubstepsPerFrame) {
                advanceSimulation();
                accumulatedTime -= fixedTimeStep;
                ++substeps;
                simulationChanged = true;
            }
            if (substeps == maximumSubstepsPerFrame) {
                // Drop excessive accumulated wall time instead of allowing an
                // unbounded catch-up loop after a debugger pause or slow frame.
                performance.droppedWallTime +=
                    std::max(0.0, accumulatedTime - fixedTimeStep);
                accumulatedTime = std::min(accumulatedTime, fixedTimeStep);
            }
        }

        if (simulationChanged) {
            if (currentView == FieldView::velocity) {
                updateArrowBuffer();
            }
            uploadFieldTexture(
                objects.texture,
                state,
                currentView,
                visualizationBuffers,
                performance);
            if (state.completedSteps % 60 == 0 && state.completedSteps != 0) {
                std::cout << "Step " << state.completedSteps
                          << ": energy=" << state.lastStep.kineticEnergyAfterProjection
                          << ", divergence max="
                          << state.lastStep.projection.divergenceAfter.maximumAbsolute << '\n';
            }
        }

        FieldView requestedView = currentView;
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
            requestedView = FieldView::velocity;
        } else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
            requestedView = FieldView::divergenceBefore;
        } else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
            requestedView = FieldView::divergenceAfter;
        } else if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
            requestedView = FieldView::pressure;
        } else if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
            requestedView = FieldView::cellTypes;
        }
        if (requestedView != currentView) {
            currentView = requestedView;
            if (currentView == FieldView::velocity) {
                updateArrowBuffer();
            }
            uploadFieldTexture(
                objects.texture,
                state,
                currentView,
                visualizationBuffers,
                performance);
        }

        const double statisticsDuration = currentWallTime - statisticsStartTime;
        if (statisticsDuration >= 0.5) {
            const double fps = performance.frames != 0
                ? static_cast<double>(performance.frames) / statisticsDuration
                : 0.0;
            const double substepsPerFrame = performance.frames != 0
                ? static_cast<double>(performance.substeps) /
                      static_cast<double>(performance.frames)
                : 0.0;
            std::ostringstream title;
            title << std::fixed << std::setprecision(2)
                  << "LiquidSimulator | " << fieldViewName(currentView)
                  << " | " << state.grid.width() << 'x' << state.grid.height()
                  << " dt=" << stepper.settings().timeStep
                  << " | FPS " << fps
                  << " | frame " << performance.frame.average() << '/'
                  << performance.frame.maximum << " ms"
                  << " | physics " << performance.physics.average() << '/'
                  << performance.physics.maximum << " ms"
                  << " | adv " << performance.advection.average() << '/'
                  << performance.advection.maximum << " ms"
                  << " | pressure " << performance.pressureSolve.average() << '/'
                  << performance.pressureSolve.maximum << " ms"
                  << " | proj " << performance.projectionOverhead.average() << '/'
                  << performance.projectionOverhead.maximum << " ms"
                  << " | vis " << performance.visualizationBuild.average() << '/'
                  << performance.visualizationBuild.maximum << " ms"
                  << " | upload " << performance.openGlUpload.average() << '/'
                  << performance.openGlUpload.maximum << " ms"
                  << " | steps/frame " << substepsPerFrame
                  << " | iter " << performance.lastPressureIterations
                  << " | residual " << std::scientific << std::setprecision(2)
                  << performance.lastPressureResidual;
            glfwSetWindowTitle(window, title.str().c_str());
            performance.clearSamples();
            statisticsStartTime = currentWallTime;
        }

        renderFrame(currentView);
        glfwSwapBuffers(window);
    }
    return 0;
}

fluidlab::simulation::Vector2 particleCentroid(
    const std::vector<fluidlab::simulation::MarkerParticle>& particles) {
    fluidlab::simulation::Vector2 centroid{};
    std::size_t count = 0;
    for (const auto& particle : particles) {
        if (!particle.valid) continue;
        centroid.x += particle.position.x;
        centroid.y += particle.position.y;
        ++count;
    }
    if (count != 0) {
        centroid.x /= static_cast<float>(count);
        centroid.y /= static_cast<float>(count);
    }
    return centroid;
}

struct InterfaceDiagnostics final {
    std::size_t internalAirCells = 0;
    std::size_t internalAirComponents = 0;
    std::size_t airCellsWithParticles = 0;
    std::size_t phaseToggles = 0;
    std::size_t maximumParticlesPerCell = 0;
};

InterfaceDiagnostics analyzeInterface(
    const std::vector<fluidlab::simulation::CellType>& cellTypes,
    const std::vector<GLuint>& particleCounts,
    const std::vector<fluidlab::simulation::CellType>* previous,
    std::size_t width,
    std::size_t height) {
    using fluidlab::simulation::CellType;
    InterfaceDiagnostics result;
    std::vector<unsigned char> visited(cellTypes.size(), 0);
    std::vector<std::size_t> airComponentSizes;
    for (std::size_t index = 0; index < cellTypes.size(); ++index) {
        result.maximumParticlesPerCell = std::max<std::size_t>(
            result.maximumParticlesPerCell, particleCounts[index]);
        if (cellTypes[index] == CellType::air && particleCounts[index] != 0) {
            ++result.airCellsWithParticles;
        }
        if (previous != nullptr &&
            ((cellTypes[index] == CellType::fluid && (*previous)[index] == CellType::air) ||
             (cellTypes[index] == CellType::air && (*previous)[index] == CellType::fluid))) {
            ++result.phaseToggles;
        }
        if (cellTypes[index] != CellType::air || visited[index] != 0) continue;
        std::queue<std::size_t> frontier;
        frontier.push(index);
        visited[index] = 1;
        std::size_t componentSize = 0;
        while (!frontier.empty()) {
            const std::size_t current = frontier.front();
            frontier.pop();
            ++componentSize;
            const std::size_t x = current % width;
            const std::size_t y = current / width;
            const auto append = [&](std::size_t neighbor) {
                if (visited[neighbor] == 0 && cellTypes[neighbor] == CellType::air) {
                    visited[neighbor] = 1;
                    frontier.push(neighbor);
                }
            };
            if (x > 0) append(current - 1);
            if (x + 1 < width) append(current + 1);
            if (y > 0) append(current - width);
            if (y + 1 < height) append(current + width);
        }
        airComponentSizes.push_back(componentSize);
    }
    if (!airComponentSizes.empty()) {
        const auto largest = std::max_element(
            airComponentSizes.begin(), airComponentSizes.end());
        for (auto it = airComponentSizes.begin(); it != airComponentSizes.end(); ++it) {
            if (it == largest) continue;
            result.internalAirCells += *it;
            ++result.internalAirComponents;
        }
    }
    return result;
}

void runGpuDamBreakValidationSuite(const std::filesystem::path& divergenceShader) {
    using namespace fluidlab::simulation;
    {
        constexpr std::size_t width = 48;
        constexpr std::size_t height = 32;
        constexpr float cellSize = 1.0f / 48.0f;
        MacGrid initial(width, height, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem seeded;
        seeded.seedFluidCells(initial);
        GpuDynamicFreeSurfaceSettings settings;
        settings.gravity = {0.0f, 0.0f};
        GpuDynamicFreeSurfaceStepper stationary(divergenceShader, settings);
        const auto initialTopology = stationary.initialize(initial, seeded);
        if (!initialTopology.topologyValid) {
            throw std::runtime_error("Zero-gravity GPU topology initialization failed.");
        }
        const auto before = stationary.grid().readMarkerParticles();
        GpuDynamicFreeSurfaceReport stationaryReport;
        for (std::size_t step = 0; step < 30; ++step) {
            stationaryReport = stationary.step();
            if (stationaryReport.failureReason != GpuDynamicFailureReason::none) {
                throw std::runtime_error("Zero-gravity GPU stability test failed.");
            }
        }
        const auto after = stationary.grid().readMarkerParticles();
        const auto pausedTypesBefore = stationary.grid().readCellTypes();
        GLuint diagnosticTexture = 0;
        glCreateTextures(GL_TEXTURE_2D, 1, &diagnosticTexture);
        glTextureStorage2D(diagnosticTexture, 1, GL_RGBA32F,
            static_cast<GLsizei>(width), static_cast<GLsizei>(height));
        std::vector<float> pausedPixelsA(width * height * 4);
        std::vector<float> pausedPixelsB(width * height * 4);
        stationary.grid().dispatchVisualization(diagnosticTexture, 5);
        glGetTextureImage(diagnosticTexture, 0, GL_RGBA, GL_FLOAT,
            static_cast<GLsizei>(pausedPixelsA.size() * sizeof(float)), pausedPixelsA.data());
        stationary.grid().dispatchVisualization(diagnosticTexture, 5);
        glGetTextureImage(diagnosticTexture, 0, GL_RGBA, GL_FLOAT,
            static_cast<GLsizei>(pausedPixelsB.size() * sizeof(float)), pausedPixelsB.data());
        glDeleteTextures(1, &diagnosticTexture);
        const auto pausedTypesAfter = stationary.grid().readCellTypes();
        if (pausedTypesBefore != pausedTypesAfter || pausedPixelsA != pausedPixelsB) {
            throw std::runtime_error("Paused GPU classification visualization was not deterministic.");
        }
        float maximumDisplacement = 0.0f;
        for (std::size_t i = 0; i < before.size(); ++i) {
            maximumDisplacement = std::max(maximumDisplacement,
                std::hypot(after[i].position.x - before[i].position.x,
                    after[i].position.y - before[i].position.y));
        }
        if (maximumDisplacement > 1.0e-5f) {
            throw std::runtime_error("Zero-gravity GPU liquid moved unexpectedly.");
        }
        std::cout << "GPU zero-gravity: PASS, steps=30, max displacement="
                  << maximumDisplacement << ", particles=" << after.size()
                  << ", paused diagnostic stable=yes.\n";
    }

    {
        constexpr std::size_t width = 48;
        constexpr std::size_t height = 32;
        constexpr float cellSize = 1.0f / 48.0f;
        MacGrid hydrostatic(width, height, cellSize, cellSize);
        hydrostatic.fillCellTypes(CellType::air);
        for (std::size_t y = 0; y < height; ++y) {
            hydrostatic.setCellType(0, y, CellType::solid);
            hydrostatic.setCellType(width - 1, y, CellType::solid);
        }
        for (std::size_t x = 0; x < width; ++x) {
            hydrostatic.setCellType(x, 0, CellType::solid);
            hydrostatic.setCellType(x, height - 1, CellType::solid);
        }
        for (std::size_t y = 1; y < height / 3; ++y) {
            for (std::size_t x = 1; x + 1 < width; ++x) {
                hydrostatic.setCellType(x, y, CellType::fluid);
            }
        }
        hydrostatic.enforceSolidBoundaries();
        MarkerParticleSystem particles;
        particles.seedFluidCells(hydrostatic);
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader);
        const auto topology = stepper.initialize(hydrostatic, particles);
        if (!topology.topologyValid) throw std::runtime_error("Hydrostatic GPU topology failed.");
        GpuDynamicFreeSurfaceReport report;
        for (std::size_t step = 0; step < 60; ++step) {
            report = stepper.step();
            if (report.failureReason != GpuDynamicFailureReason::none) {
                throw std::runtime_error("Hydrostatic GPU stability test failed.");
            }
        }
        const auto velocity = stepper.grid().velocityExtrema();
        const float maximumSpeed = std::max(
            velocity.maximumAbsoluteU, velocity.maximumAbsoluteV);
        if (maximumSpeed > 0.02f || !report.pressureConverged) {
            throw std::runtime_error("Hydrostatic GPU liquid did not remain bounded.");
        }
        std::cout << "GPU hydrostatic: PASS, steps=60, max face speed=" << maximumSpeed
                  << ", residual=" << report.projection.pressure.finalResidual.rootMeanSquare
                  << ", divergence=" << report.projection.divergenceAfter.maximumAbsolute << ".\n";
    }

    {
        constexpr std::size_t width = 48;
        constexpr std::size_t height = 32;
        constexpr float cellSize = 1.0f / 48.0f;
        MacGrid cpuGrid(width, height, cellSize, cellSize);
        initializeDamBreak(cpuGrid);
        MarkerParticleSystem cpuParticles;
        cpuParticles.seedFluidCells(cpuGrid);
        DynamicFreeSurfaceStepper cpuStepper;
        GpuDynamicFreeSurfaceStepper gpuStepper(divergenceShader);
        const auto topology = gpuStepper.initialize(cpuGrid, cpuParticles);
        if (!topology.topologyValid) throw std::runtime_error("CPU/GPU comparison topology failed.");
        GpuDynamicFreeSurfaceReport gpuReport;
        DynamicFreeSurfaceReport cpuReport;
        for (std::size_t step = 0; step < 60; ++step) {
            cpuReport = cpuStepper.step(cpuGrid, cpuParticles);
            gpuReport = gpuStepper.step();
            if (gpuReport.failureReason != GpuDynamicFailureReason::none) {
                throw std::runtime_error("CPU/GPU dam-break comparison GPU step failed.");
            }
        }
        const auto gpuParticles = gpuStepper.grid().readMarkerParticles();
        const Vector2 cpuCenter = particleCentroid(cpuParticles.particles());
        const Vector2 gpuCenter = particleCentroid(gpuParticles);
        const float centroidDifference = std::hypot(
            cpuCenter.x - gpuCenter.x, cpuCenter.y - gpuCenter.y);
        if (gpuParticles.size() != cpuParticles.particles().size() || centroidDifference > 0.08f) {
            throw std::runtime_error("CPU/GPU dam-break comparison exceeded its physical tolerance.");
        }
        std::cout << "CPU/GPU 48x32 comparison: PASS, steps=60, centroid error="
                  << centroidDifference << ", CPU/GPU fluid cells=" << cpuReport.fluidCellCount
                  << '/' << gpuReport.fluidCellCount << ".\n";
    }

    struct CoverageResult final {
        std::size_t initialDensity = 0;
        bool redistribution = false;
        std::size_t finalParticles = 0;
        std::size_t particlesAdded = 0;
        std::size_t maximumInternalAirCells = 0;
        std::size_t maximumInternalAirComponents = 0;
        std::size_t phaseToggles = 0;
        std::size_t repeatedFlickerEvents = 0;
        std::size_t airCellsWithParticles = 0;
        std::size_t minimumFluidCells = std::numeric_limits<std::size_t>::max();
        std::size_t maximumFluidCells = 0;
        double particleGpuMilliseconds = 0.0;
        double classificationGpuMilliseconds = 0.0;
        float finalResidual = 0.0f;
        float finalDivergence = 0.0f;
    };
    const auto runCoverageExperiment = [&](GpuDynamicFreeSurfaceStepper& stepper,
                                           std::size_t density,
                                           bool redistribute) {
        constexpr std::size_t size = 64;
        constexpr std::size_t steps = 30;
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid initial(size, size, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, density);
        const auto topology = stepper.initialize(initial, particles);
        if (!topology.topologyValid) {
            throw std::runtime_error("Particle coverage experiment topology failed.");
        }
        CoverageResult result;
        result.initialDensity = density;
        result.redistribution = redistribute;
        std::vector<CellType> previousTypes = stepper.grid().readCellTypes();
        for (std::size_t step = 0; step < steps; ++step) {
            const auto report = stepper.step();
            if (report.failureReason != GpuDynamicFailureReason::none) {
                throw std::runtime_error(std::string("Particle coverage experiment failed: ") +
                    report.failureMessage);
            }
            const auto types = stepper.grid().readCellTypes();
            const auto counts = stepper.grid().readParticleCellCounts();
            const auto interface = analyzeInterface(
                types, counts, &previousTypes, size, size);
            result.maximumInternalAirCells = std::max(
                result.maximumInternalAirCells, interface.internalAirCells);
            result.maximumInternalAirComponents = std::max(
                result.maximumInternalAirComponents, interface.internalAirComponents);
            result.phaseToggles += interface.phaseToggles;
            result.repeatedFlickerEvents += report.classification.repeatedFlickerCellCount;
            result.airCellsWithParticles += interface.airCellsWithParticles;
            result.minimumFluidCells = std::min(
                result.minimumFluidCells, report.fluidCellCount);
            result.maximumFluidCells = std::max(
                result.maximumFluidCells, report.fluidCellCount);
            result.particlesAdded += report.particlesAdded;
            result.particleGpuMilliseconds += report.timings.particleGpuMilliseconds;
            result.classificationGpuMilliseconds += report.timings.classificationGpuMilliseconds;
            result.finalResidual = report.projection.pressure.finalResidual.rootMeanSquare;
            result.finalDivergence = report.projection.divergenceAfter.maximumAbsolute;
            previousTypes = types;
            if ((step + 1) % 10 == 0) {
                std::cout << "  coverage " << density << (redistribute ? "R" : "")
                          << " step=" << step + 1
                          << " cpuMs=" << report.timings.totalCpuMilliseconds
                          << " substeps=" << report.substeps
                          << " cfl=" << report.maximumObservedCfl << '\n';
            }
        }
        result.finalParticles = stepper.grid().particleCount();
        result.particleGpuMilliseconds /= static_cast<double>(steps);
        result.classificationGpuMilliseconds /= static_cast<double>(steps);
        return result;
    };

    std::cout << "Particle coverage diagnostics (64x64, 30 steps, deterministic seed)\n"
                 "PPC Redistribute Particles Added HoleCells HoleComponents Toggles RepeatFlicker "
                 "AirWithParticle FluidRange ParticleGPU ClassifyGPU Residual Divergence\n";
    std::vector<CoverageResult> coverageResults;
    GpuDynamicFreeSurfaceSettings baselineCoverageSettings;
    GpuDynamicFreeSurfaceStepper baselineCoverageStepper(
        divergenceShader, baselineCoverageSettings);
    for (const std::size_t density : {4U, 8U, 16U}) {
        coverageResults.push_back(runCoverageExperiment(
            baselineCoverageStepper, density, false));
    }
    GpuDynamicFreeSurfaceSettings redistributionSettings;
    redistributionSettings.particleRedistributionEnabled = true;
    redistributionSettings.particleRedistributionInterval = 8;
    redistributionSettings.particleRedistributionTargetMinimum = 2;
    GpuDynamicFreeSurfaceStepper redistributionStepper(
        divergenceShader, redistributionSettings);
    coverageResults.push_back(runCoverageExperiment(
        redistributionStepper, 4, true));
    for (const auto& result : coverageResults) {
        std::cout << result.initialDensity << ' ' << (result.redistribution ? "on" : "off")
                  << ' ' << result.finalParticles << ' ' << result.particlesAdded << ' '
                  << result.maximumInternalAirCells << ' '
                  << result.maximumInternalAirComponents << ' ' << result.phaseToggles << ' '
                  << result.repeatedFlickerEvents << ' ' << result.airCellsWithParticles << ' '
                  << result.minimumFluidCells << '-' << result.maximumFluidCells << ' '
                  << std::fixed << std::setprecision(4)
                  << result.particleGpuMilliseconds << ' '
                  << result.classificationGpuMilliseconds << ' '
                  << std::scientific << result.finalResidual << ' '
                  << result.finalDivergence << '\n';
    }

    std::cout << "Timer synchronization comparison (64x64, 20 steps)\n"
                 "Mode StepCPUAvg StepP95 ExplicitGpuTiming\n";
    for (const bool synchronousTiming : {true, false}) {
        constexpr std::size_t size = 64;
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid initial(size, size, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.synchronousGpuTimingEnabled = synchronousTiming;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        const auto topology = stepper.initialize(initial, particles);
        if (!topology.topologyValid) {
            throw std::runtime_error("Timer synchronization comparison topology failed.");
        }
        TimingAccumulator cpuTime;
        double measuredGpuTime = 0.0;
        for (std::size_t step = 0; step < 20; ++step) {
            const auto report = stepper.step();
            if (report.failureReason != GpuDynamicFailureReason::none) {
                throw std::runtime_error("Timer synchronization comparison failed.");
            }
            cpuTime.add(report.timings.totalCpuMilliseconds);
            measuredGpuTime += report.timings.pressureGpuMilliseconds +
                report.timings.velocityAdvectionGpuMilliseconds +
                report.timings.particleGpuMilliseconds;
        }
        std::cout << (synchronousTiming ? "synchronous" : "realtime") << ' '
                  << std::fixed << std::setprecision(3) << cpuTime.average() << ' '
                  << cpuTime.percentile(0.95) << ' '
                  << measuredGpuTime / 20.0 << '\n';
    }

    std::cout << "GPU continuous performance (Release x64, synchronized diagnostics)\n"
                 "Grid Steps StepAvg P95 P99 Max Advect Extrap Pressure Project Particle Classify Topology Sync Iter Fluid\n";
    for (const std::size_t size : {64U, 128U, 256U}) {
        const std::size_t stepCount = size == 64 ? 60 : (size == 128 ? 20 : 5);
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid grid(size, size, cellSize, cellSize);
        initializeDamBreak(grid);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.detailedVelocityDiagnostics = size == 256;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        const auto topology = stepper.initialize(grid, particles);
        if (!topology.topologyValid) throw std::runtime_error("GPU performance topology failed.");
        TimingAccumulator total, advection, extrapolation, pressure, projection;
        TimingAccumulator particle, classification, topologyTime, synchronization;
        GpuDynamicFreeSurfaceReport report;
        std::string stoppedReason;
        for (std::size_t step = 0; step < stepCount; ++step) {
            report = stepper.step();
            if (report.failureReason != GpuDynamicFailureReason::none) {
                stoppedReason = report.failureMessage;
                break;
            }
            total.add(report.timings.totalCpuMilliseconds);
            advection.add(report.timings.velocityAdvectionGpuMilliseconds);
            extrapolation.add(report.timings.velocityExtrapolationGpuMilliseconds);
            pressure.add(report.timings.pressureGpuMilliseconds +
                report.timings.pressureReductionGpuMilliseconds);
            projection.add(report.timings.divergenceGpuMilliseconds +
                report.timings.projectionGpuMilliseconds);
            particle.add(report.timings.particleGpuMilliseconds);
            classification.add(report.timings.classificationGpuMilliseconds);
            topologyTime.add(report.timings.topologyFineGpuMilliseconds +
                report.timings.topologyAnalysisGpuMilliseconds +
                report.timings.topologyCoarseGpuMilliseconds);
            synchronization.add(report.timings.smallStatusReadbackMilliseconds +
                report.timings.pressureReadbackMilliseconds +
                report.timings.velocityReductionReadbackMilliseconds);
        }
        std::cout << size << 'x' << size << ' ' << stepCount << ' '
                  << std::fixed << std::setprecision(3)
                  << total.average() << ' ' << total.percentile(0.95) << ' '
                  << total.percentile(0.99) << ' ' << total.maximum << ' '
                  << advection.average() << ' ' << extrapolation.average() << ' '
                  << pressure.average() << ' ' << projection.average() << ' '
                  << particle.average() << ' ' << classification.average() << ' '
                  << topologyTime.average() << ' ' << synchronization.average() << ' '
                  << report.pressureIterationsTotal << ' ' << report.fluidCellCount;
        if (!stoppedReason.empty()) {
            const bool uDominant = report.velocity.maximumAbsoluteU >=
                report.velocity.maximumAbsoluteV;
            const std::size_t index = uDominant
                ? report.velocity.maximumUIndex : report.velocity.maximumVIndex;
            const std::size_t stride = uDominant ? size + 1 : size;
            std::cout << " STOPPED: " << stoppedReason
                      << ", peakStage=" << gpuVelocityStageName(report.peakVelocityStage)
                      << ", component=" << (uDominant ? 'u' : 'v')
                      << ", index=(" << index % stride << ',' << index / stride << ')'
                      << ", value=" << (uDominant ? report.velocity.maximumAbsoluteU
                          : report.velocity.maximumAbsoluteV)
                      << ", previousResidual="
                      << report.projection.pressure.finalResidual.rootMeanSquare
                      << ", previousDivergence="
                      << report.projection.divergenceAfter.maximumAbsolute;
        }
        std::cout << '\n';
    }

    {
        constexpr std::size_t size = 256;
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid initial(size, size, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.synchronousGpuTimingEnabled = false;
        settings.detailedVelocityDiagnostics = true;
        settings.adaptiveCflSchedulingEnabled = false;
        settings.projectionSnapshotFirstOuterStep = 37;
        settings.projectionSnapshotLastOuterStep = 37;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        const auto topology = stepper.initialize(initial, particles);
        if (!topology.topologyValid) {
            throw std::runtime_error("High-resolution CFL diagnostic topology failed.");
        }
        GpuDynamicFreeSurfaceReport report;
        std::optional<GpuProjectionDiagnosticSnapshot> diagnosticSnapshot;
        std::size_t completedSteps = 0;
        for (; completedSteps < 50; ++completedSteps) {
            report = stepper.step();
            if (report.projectionSnapshot) diagnosticSnapshot = report.projectionSnapshot;
            if (report.failureReason != GpuDynamicFailureReason::none) break;
        }
        const bool uDominant = report.velocity.maximumAbsoluteU >=
            report.velocity.maximumAbsoluteV;
        const std::size_t index = uDominant
            ? report.velocity.maximumUIndex : report.velocity.maximumVIndex;
        const std::size_t stride = uDominant ? size + 1 : size;
        const auto types = stepper.grid().readCellTypes();
        std::string adjacentTypes;
        if (uDominant) {
            const std::size_t x = index % stride;
            const std::size_t y = index / stride;
            if (x > 0) adjacentTypes += " left=" +
                std::to_string(static_cast<unsigned int>(types[y * size + x - 1]));
            if (x < size) adjacentTypes += " right=" +
                std::to_string(static_cast<unsigned int>(types[y * size + x]));
        } else {
            const std::size_t x = index % stride;
            const std::size_t y = index / stride;
            if (y > 0) adjacentTypes += " below=" +
                std::to_string(static_cast<unsigned int>(types[(y - 1) * size + x]));
            if (y < size) adjacentTypes += " above=" +
                std::to_string(static_cast<unsigned int>(types[y * size + x]));
        }
        std::cout << "256x256 CFL diagnostic: completed=" << completedSteps
                  << ", status=" << gpuDynamicFailureName(report.failureReason)
                  << ", peakStage=" << gpuVelocityStageName(report.peakVelocityStage)
                  << ", component=" << (uDominant ? 'u' : 'v')
                  << ", index=(" << index % stride << ',' << index / stride << ')'
                  << ", maxU/maxV=" << report.velocity.maximumAbsoluteU << '/'
                  << report.velocity.maximumAbsoluteV
                  << ", outerCfl=" << report.maximumObservedCfl
                  << adjacentTypes
                  << ", residual=" << std::scientific
                  << report.projection.pressure.finalResidual.rootMeanSquare
                  << ", divergence=" << report.projection.divergenceAfter.maximumAbsolute
                  << '\n';
        if (!diagnosticSnapshot) {
            throw std::runtime_error("The high-resolution projection snapshot was not captured.");
        }
        const auto& snapshot = *diagnosticSnapshot;
        const bool snapshotUsesU = snapshot.maximumComponent == 'u';
        const std::size_t faceStride = snapshotUsesU ? size + 1 : size;
        const std::size_t faceX = snapshot.maximumIndex % faceStride;
        const std::size_t faceY = snapshot.maximumIndex / faceStride;
        if ((snapshotUsesU && (faceX == 0 || faceX >= size || faceY >= size)) ||
            (!snapshotUsesU && (faceY == 0 || faceY >= size || faceX >= size))) {
            throw std::runtime_error("The reproduced high-resolution peak is not an interior MAC face.");
        }
        const std::size_t negativeCell = snapshotUsesU
            ? faceY * size + faceX - 1 : (faceY - 1) * size + faceX;
        const std::size_t positiveCell = snapshotUsesU
            ? faceY * size + faceX : faceY * size + faceX;
        const float faceBefore = snapshotUsesU
            ? snapshot.horizontalBefore[snapshot.maximumIndex]
            : snapshot.verticalBefore[snapshot.maximumIndex];
        const float expected = faceBefore - snapshot.timeStep *
            (snapshot.pressureAfter[positiveCell] - snapshot.pressureAfter[negativeCell]) /
            cellSize;
        const float actual = snapshotUsesU
            ? snapshot.horizontalAfter[snapshot.maximumIndex]
            : snapshot.verticalAfter[snapshot.maximumIndex];
        std::cout << "Projection snapshot outer/substep=" << snapshot.outerStep << '/'
                  << snapshot.substep << ", dt=" << snapshot.timeStep
                  << ", component=" << snapshot.maximumComponent
                  << ", faceBefore=" << faceBefore
                  << ", pNegative/pPositive=" << snapshot.pressureAfter[negativeCell] << '/'
                  << snapshot.pressureAfter[positiveCell]
                  << ", expected/actual=" << expected << '/' << actual
                  << ", formulaError=" << std::abs(expected - actual)
                  << ", maxDelta=" << snapshot.maximumVelocityChange
                  << ", energyBefore/After=" << snapshot.kineticEnergyBefore << '/'
                  << snapshot.kineticEnergyAfter << '\n';
        std::cout << "Local 5x5 cells around the peak: x y old/type/new count p residual diag divBefore divAfter\n";
        const auto divergenceAt = [&](const std::vector<float>& u,
                                      const std::vector<float>& v,
                                      std::size_t x, std::size_t y) {
            const std::size_t uIndex = y * (size + 1) + x;
            const std::size_t vIndex = y * size + x;
            return (u[uIndex + 1] - u[uIndex]) / cellSize +
                (v[vIndex + size] - v[vIndex]) / cellSize;
        };
        const std::size_t centerX = std::min(faceX, size - 1);
        const std::size_t centerY = std::min(faceY, size - 1);
        const std::size_t beginX = centerX > 2 ? centerX - 2 : 0;
        const std::size_t beginY = centerY > 2 ? centerY - 2 : 0;
        for (std::size_t y = beginY; y <= std::min(size - 1, centerY + 2); ++y) {
            for (std::size_t x = beginX; x <= std::min(size - 1, centerX + 2); ++x) {
                const std::size_t cell = y * size + x;
                std::cout << "  " << x << ' ' << y << ' '
                          << static_cast<unsigned int>(snapshot.previousCellTypes[cell]) << '/'
                          << static_cast<unsigned int>(snapshot.cellTypes[cell]) << '/'
                          << static_cast<unsigned int>(snapshot.cellTypesAfterClassification[cell]) << ' '
                          << snapshot.particleCounts[cell] << ' '
                          << snapshot.pressureAfter[cell] << ' '
                          << snapshot.residualAfter[cell * 2] << ' '
                          << snapshot.pressureOperator.diagonal[cell] << ' '
                          << divergenceAt(snapshot.horizontalBefore, snapshot.verticalBefore, x, y) << ' '
                          << divergenceAt(snapshot.horizontalAfter, snapshot.verticalAfter, x, y) << '\n';
            }
        }

        MacGrid cpuSnapshot(size, size, cellSize, cellSize);
        for (std::size_t y = 0; y < size; ++y) {
            for (std::size_t x = 0; x < size; ++x) {
                const std::size_t cell = y * size + x;
                cpuSnapshot.setCellType(x, y, snapshot.cellTypes[cell]);
                cpuSnapshot.pressure(x, y) = snapshot.pressureBefore[cell];
            }
            for (std::size_t x = 0; x <= size; ++x) {
                cpuSnapshot.horizontalVelocity(x, y) =
                    snapshot.horizontalBefore[y * (size + 1) + x];
            }
        }
        for (std::size_t y = 0; y <= size; ++y) {
            for (std::size_t x = 0; x < size; ++x) {
                cpuSnapshot.verticalVelocity(x, y) = snapshot.verticalBefore[y * size + x];
            }
        }
        PressureSolverSettings cpuPressure;
        cpuPressure.timeStep = snapshot.timeStep;
        cpuPressure.density = 1.0f;
        cpuPressure.residualTolerance = settings.pressureResidualTolerance;
        cpuPressure.maximumIterations = settings.pressureMaximumIterations;
        cpuPressure.residualCheckInterval = settings.pressureResidualCheckInterval;
        cpuPressure.warmStartEnabled = true;
        FreeSurfacePressureProjector cpuProjector(cpuPressure);
        const auto cpuProjection = cpuProjector.project(cpuSnapshot);
        const float cpuPeak = snapshotUsesU
            ? cpuSnapshot.horizontalVelocity(faceX, faceY)
            : cpuSnapshot.verticalVelocity(faceX, faceY);

        MacGrid diagonalInput = cpuSnapshot;
        for (std::size_t y = 0; y < size; ++y) {
            for (std::size_t x = 0; x < size; ++x) {
                diagonalInput.pressure(x, y) = snapshot.pressureBefore[y * size + x];
            }
            for (std::size_t x = 0; x <= size; ++x) {
                diagonalInput.horizontalVelocity(x, y) =
                    snapshot.horizontalBefore[y * (size + 1) + x];
            }
        }
        for (std::size_t y = 0; y <= size; ++y) {
            for (std::size_t x = 0; x < size; ++x) {
                diagonalInput.verticalVelocity(x, y) = snapshot.verticalBefore[y * size + x];
            }
        }
        GpuMacGrid diagonalGrid(divergenceShader);
        diagonalGrid.upload(diagonalInput);
        GpuPressureSettings diagonalSettings;
        diagonalSettings.timeStep = snapshot.timeStep;
        diagonalSettings.density = 1.0f;
        diagonalSettings.residualTolerance = settings.pressureResidualTolerance;
        diagonalSettings.maximumIterations = settings.pressureMaximumIterations;
        diagonalSettings.residualCheckInterval = settings.pressureResidualCheckInterval;
        diagonalSettings.warmStartEnabled = true;
        diagonalSettings.solver = GpuPressureSolver::preconditionedConjugateGradient;
        const auto diagonalProjection = diagonalGrid.project(diagonalSettings);
        const auto diagonalFace = snapshotUsesU
            ? diagonalGrid.readHorizontalVelocity() : diagonalGrid.readVerticalVelocity();
        std::cout << "Same-equation solver comparison at peak face: MGPCG/Diagonal/CPU="
                  << actual << '/' << diagonalFace[snapshot.maximumIndex] << '/' << cpuPeak
                  << ", RMS residual=" << snapshot.projection.pressure.finalResidual.rootMeanSquare
                  << '/' << diagonalProjection.pressure.finalResidual.rootMeanSquare
                  << '/' << cpuProjection.projection.pressureSolve.finalResidual.rootMeanSquare
                  << ", max residual=" << snapshot.projection.pressure.finalResidual.maximumAbsolute
                  << '/' << diagonalProjection.pressure.finalResidual.maximumAbsolute
                  << '/' << cpuProjection.projection.pressureSolve.finalResidual.maximumAbsolute
                  << '\n';
    }

    std::cout << "Phase 1.7B adaptive CFL and realtime pressure-path benchmark\n"
                 "Grid Requested Completed StepAvg P95 P99 Max AsyncGPU TopologyE2E "
                 "Rebuild Reuse Labels IterMax CFLMax Residual Divergence Status\n";
    for (const std::size_t size : {64U, 128U, 256U}) {
        const std::size_t requestedSteps = size == 64 ? 120 : (size == 128 ? 80 : 60);
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid initial(size, size, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.synchronousGpuTimingEnabled = false;
        settings.detailedVelocityDiagnostics = size == 256;
        settings.adaptiveCflSchedulingEnabled = true;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        const auto topology = stepper.initialize(initial, particles);
        if (!topology.topologyValid) {
            throw std::runtime_error("Phase 1.7B benchmark topology failed.");
        }
        TimingAccumulator stepTimes;
        TimingAccumulator asynchronousGpuTimes;
        double topologyEndToEnd = 0.0;
        std::size_t rebuilds = 0, reuses = 0, labels = 0, completed = 0;
        std::size_t maximumIterations = 0;
        float maximumCfl = 0.0f;
        GpuDynamicFreeSurfaceReport report;
        for (; completed < requestedSteps; ++completed) {
            report = stepper.step();
            if (report.failureReason != GpuDynamicFailureReason::none) break;
            stepTimes.add(report.timings.totalCpuMilliseconds);
            if (report.timings.asynchronousStepTimingAvailable) {
                asynchronousGpuTimes.add(report.timings.asynchronousStepGpuMilliseconds);
            }
            topologyEndToEnd += report.timings.topologyEndToEndMilliseconds;
            rebuilds += report.topologyRebuildCount;
            reuses += report.topologyReuseCount;
            labels += report.topologyLabelPropagationIterations;
            maximumIterations = std::max(maximumIterations, report.pressureIterationsMaximum);
            maximumCfl = std::max(maximumCfl, report.maximumObservedCfl);
        }
        std::cout << size << 'x' << size << ' ' << requestedSteps << ' ' << completed << ' '
                  << std::fixed << std::setprecision(3)
                  << stepTimes.average() << ' ' << stepTimes.percentile(0.95) << ' '
                  << stepTimes.percentile(0.99) << ' ' << stepTimes.maximum << ' '
                  << asynchronousGpuTimes.average() << ' '
                  << (completed != 0 ? topologyEndToEnd / completed : 0.0) << ' '
                  << rebuilds << ' ' << reuses << ' ' << labels << ' '
                  << maximumIterations << ' ' << maximumCfl << ' '
                  << std::scientific
                  << report.projection.pressure.finalResidual.rootMeanSquare << ' '
                  << report.projection.divergenceAfter.maximumAbsolute << ' '
                  << gpuDynamicFailureName(report.failureReason) << '\n';
    }

    std::cout << "MGPCG residual-check comparison (128x128, 30 outer steps)\n"
                 "Interval StepAvg PressureReadback IterTotal Residual Divergence Status\n";
    for (const std::size_t interval : {10U, 20U, 40U}) {
        constexpr std::size_t size = 128;
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid initial(size, size, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.synchronousGpuTimingEnabled = false;
        settings.pressureResidualCheckInterval = interval;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        const auto topology = stepper.initialize(initial, particles);
        if (!topology.topologyValid) {
            throw std::runtime_error("MGPCG residual-check comparison topology failed.");
        }
        TimingAccumulator steps;
        double readback = 0.0;
        std::size_t iterations = 0;
        GpuDynamicFreeSurfaceReport report;
        std::size_t completed = 0;
        for (; completed < 30; ++completed) {
            report = stepper.step();
            if (report.failureReason != GpuDynamicFailureReason::none) break;
            steps.add(report.timings.totalCpuMilliseconds);
            readback += report.timings.pressureReadbackMilliseconds;
            iterations += report.pressureIterationsTotal;
        }
        std::cout << interval << ' ' << std::fixed << std::setprecision(3)
                  << steps.average() << ' ' << (completed != 0 ? readback / completed : 0.0)
                  << ' ' << iterations << ' ' << std::scientific
                  << report.projection.pressure.finalResidual.rootMeanSquare << ' '
                  << report.projection.divergenceAfter.maximumAbsolute << ' '
                  << gpuDynamicFailureName(report.failureReason) << '\n';
    }
}

void runPhase17cAdaptiveValidation(const std::filesystem::path& divergenceShader) {
    using namespace fluidlab::simulation;
    const auto sameParticles = [](const std::vector<MarkerParticle>& left,
                                  const std::vector<MarkerParticle>& right) {
        if (left.size() != right.size()) return false;
        for (std::size_t i = 0; i < left.size(); ++i) {
            if (left[i].position.x != right[i].position.x ||
                left[i].position.y != right[i].position.y ||
                left[i].valid != right[i].valid) return false;
        }
        return true;
    };

    std::cout << "Phase 1.7C candidate rollback validation\n";
    {
        constexpr std::size_t width = 48;
        constexpr std::size_t height = 32;
        constexpr float cellSize = 1.0f / 48.0f;
        MacGrid initial(width, height, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.gravity = {0.0f, 0.0f};
        settings.maximumSubsteps = 1;
        settings.forcedCandidateRejectionsForValidation = 1;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        const auto topology = stepper.initialize(initial, particles);
        if (!topology.topologyValid) throw std::runtime_error("Rollback topology failed.");
        const auto uBefore = stepper.grid().readHorizontalVelocity();
        const auto vBefore = stepper.grid().readVerticalVelocity();
        const auto pBefore = stepper.grid().readPressure();
        const auto typesBefore = stepper.grid().readCellTypes();
        const auto particlesBefore = stepper.grid().readMarkerParticles();
        const auto rejected = stepper.step();
        const bool exactRollback = !rejected.advanced && rejected.computeBudgetExhausted &&
            rejected.rejectedSubsteps == 1 && rejected.rollbackCount == 1 &&
            rejected.remainingPhysicalTime == settings.timeStep &&
            stepper.grid().readHorizontalVelocity() == uBefore &&
            stepper.grid().readVerticalVelocity() == vBefore &&
            stepper.grid().readPressure() == pBefore &&
            stepper.grid().readCellTypes() == typesBefore &&
            sameParticles(stepper.grid().readMarkerParticles(), particlesBefore);
        if (!exactRollback) {
            throw std::runtime_error("Rejected GPU candidate did not restore the committed state.");
        }
        const auto resumed = stepper.step();
        if (!resumed.outerStepCompleted || resumed.failureReason != GpuDynamicFailureReason::none ||
            std::abs(resumed.simulationTime - settings.timeStep) > 1.0e-7) {
            throw std::runtime_error("A deferred GPU outer step did not resume correctly.");
        }
        std::cout << "  forced reject/rollback/resume: PASS, pending="
                  << rejected.remainingPhysicalTime << " -> " << resumed.remainingPhysicalTime
                  << ", committed time=" << resumed.simulationTime << '\n';
    }

    {
        constexpr std::size_t width = 48;
        constexpr std::size_t height = 32;
        constexpr float cellSize = 1.0f / 48.0f;
        MacGrid initial(width, height, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.gravity = {0.0f, 0.0f};
        settings.maximumSubsteps = 8;
        settings.maximumCandidateRetries = 2;
        settings.forcedCandidateRejectionsForValidation = 3;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        static_cast<void>(stepper.initialize(initial, particles));
        const auto uBefore = stepper.grid().readHorizontalVelocity();
        const auto particlesBefore = stepper.grid().readMarkerParticles();
        const auto failed = stepper.step();
        if (failed.failureReason != GpuDynamicFailureReason::cflSubstepLimit ||
            failed.advanced || failed.rollbackCount != 2 || stepper.simulationTime() != 0.0 ||
            stepper.grid().readHorizontalVelocity() != uBefore ||
            !sameParticles(stepper.grid().readMarkerParticles(), particlesBefore)) {
            throw std::runtime_error("GPU retry limit did not preserve the last committed state.");
        }
        std::cout << "  retry-limit safe stop: PASS, attempts=" << failed.candidateAttempts
                  << ", rollbacks=" << failed.rollbackCount << '\n';
    }

    struct GridResult final {
        std::size_t size = 0;
        std::size_t completed = 0;
        std::size_t calls = 0;
        std::size_t accepted = 0;
        std::size_t rejected = 0;
        std::size_t topologyLabels = 0;
        std::size_t topologyRebuilds = 0;
        std::size_t topologyReuses = 0;
        double elapsedMilliseconds = 0.0;
        double topologyMilliseconds = 0.0;
        double pressureReadbackMilliseconds = 0.0;
        float maximumCfl = 0.0f;
        float maximumAcceptedCfl = 0.0f;
        float maximumU = 0.0f;
        float maximumV = 0.0f;
        std::size_t maximumVIndex = 0;
        GpuDynamicFreeSurfaceReport report;
    };
    std::cout << "Phase 1.7C continuous grid study at physical t=0.5 s\n"
                 "Grid Completed Calls AvgOuterMs P95 P99 MaxOuter AvgAcceptedMs Reject Labels Rebuild Reuse "
                 "TopoMs PressureWaitMs CFLAccepted CFLCandidate MaxU MaxV MaxVx MaxVy Residual Divergence Status\n";
    for (const std::size_t size : {64U, 128U, 192U, 256U}) {
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid initial(size, size, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.synchronousGpuTimingEnabled = false;
        settings.detailedVelocityDiagnostics = size >= 192;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        const auto topology = stepper.initialize(initial, particles);
        if (!topology.topologyValid) {
            throw std::runtime_error("Phase 1.7C grid-study topology failed.");
        }
        GridResult result;
        result.size = size;
        const auto start = std::chrono::steady_clock::now();
        auto outerStart = start;
        TimingAccumulator outerTimes;
        while (result.completed < 60 && result.calls < 20000 && stepper.healthy()) {
            result.report = stepper.step();
            ++result.calls;
            result.accepted += result.report.substeps;
            result.rejected += result.report.rejectedSubsteps;
            result.topologyLabels += result.report.topologyLabelPropagationIterations;
            result.topologyRebuilds += result.report.topologyRebuildCount;
            result.topologyReuses += result.report.topologyReuseCount;
            result.topologyMilliseconds += result.report.timings.topologyEndToEndMilliseconds;
            result.pressureReadbackMilliseconds +=
                result.report.timings.pressureReadbackMilliseconds;
            result.maximumCfl = std::max(result.maximumCfl, result.report.maximumObservedCfl);
            result.maximumAcceptedCfl = std::max(
                result.maximumAcceptedCfl, result.report.maximumAcceptedCfl);
            result.maximumU = std::max(result.maximumU,
                result.report.velocity.maximumAbsoluteU);
            if (result.report.velocity.maximumAbsoluteV > result.maximumV) {
                result.maximumV = result.report.velocity.maximumAbsoluteV;
                result.maximumVIndex = result.report.velocity.maximumVIndex;
            }
            if (result.report.outerStepCompleted) {
                const auto now = std::chrono::steady_clock::now();
                outerTimes.add(std::chrono::duration<double, std::milli>(now - outerStart).count());
                outerStart = now;
                ++result.completed;
            }
            if (result.report.failureReason != GpuDynamicFailureReason::none) break;
        }
        result.elapsedMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        const std::size_t maximumVx = result.maximumVIndex % size;
        const std::size_t maximumVy = result.maximumVIndex / size;
        std::cout << size << 'x' << size << ' ' << result.completed << ' ' << result.calls << ' '
                  << std::fixed << std::setprecision(3)
                  << outerTimes.average() << ' ' << outerTimes.percentile(0.95) << ' '
                  << outerTimes.percentile(0.99) << ' ' << outerTimes.maximum << ' '
                  << (result.accepted != 0 ? result.elapsedMilliseconds / result.accepted : 0.0)
                  << ' ' << result.rejected << ' ' << result.topologyLabels << ' '
                  << result.topologyRebuilds << ' ' << result.topologyReuses << ' '
                  << result.topologyMilliseconds << ' ' << result.pressureReadbackMilliseconds
                  << ' ' << result.maximumAcceptedCfl << ' ' << result.maximumCfl
                  << ' ' << result.maximumU << ' ' << result.maximumV
                  << ' ' << maximumVx << ' ' << maximumVy << ' ' << std::scientific
                  << result.report.projection.pressure.finalResidual.rootMeanSquare << ' '
                  << result.report.projection.divergenceAfter.maximumAbsolute << ' '
                  << gpuDynamicFailureName(result.report.failureReason) << '\n';
        const auto finalVertical = stepper.grid().readVerticalVelocity();
        const auto finalPressure = stepper.grid().readPressure();
        const auto finalTypes = stepper.grid().readCellTypes();
        const auto finalParticleCounts = stepper.grid().readParticleCellCounts();
        float wallPeak = 0.0f;
        std::size_t wallPeakIndex = 0;
        for (std::size_t y = 0; y <= size; ++y) {
            for (std::size_t x = 3 * size / 4; x < size; ++x) {
                const std::size_t index = y * size + x;
                if (std::abs(finalVertical[index]) > wallPeak) {
                    wallPeak = std::abs(finalVertical[index]);
                    wallPeakIndex = index;
                }
            }
        }
        std::size_t highSpeedMinimumX = size - 1;
        for (std::size_t y = 0; y <= size; ++y) {
            for (std::size_t x = size / 2; x < size; ++x) {
                if (std::abs(finalVertical[y * size + x]) >= 0.8f * wallPeak) {
                    highSpeedMinimumX = std::min(highSpeedMinimumX, x);
                }
            }
        }
        std::size_t rightWallTopFluid = 0;
        for (std::size_t y = 0; y < size; ++y) {
            if (finalTypes[y * size + (size - 2)] == CellType::fluid) {
                rightWallTopFluid = y;
            }
        }
        const float maximumPressure = std::accumulate(finalPressure.begin(), finalPressure.end(),
            0.0f, [](float value, double pressure) {
                return std::max(value, static_cast<float>(std::abs(pressure)));
            });
        const std::size_t wallX = wallPeakIndex % size;
        const std::size_t wallY = wallPeakIndex / size;
        const std::size_t adjacentY = std::min(wallY, size - 1);
        const std::size_t adjacentIndex = adjacentY * size + wallX;
        std::cout << "  final-wall grid=" << size << " t=" << stepper.simulationTime()
                  << " peak=" << wallPeak << " at=("
                  << (static_cast<float>(wallX) + 0.5f) * cellSize << ','
                  << static_cast<float>(wallY) * cellSize << ") high-width="
                  << static_cast<float>(size - highSpeedMinimumX) * cellSize
                  << " surface-height=" << static_cast<float>(rightWallTopFluid + 1) * cellSize
                  << " adjacent-type=" << static_cast<unsigned int>(finalTypes[adjacentIndex])
                  << " adjacent-particles=" << finalParticleCounts[adjacentIndex]
                  << " max-pressure=" << maximumPressure
                  << " fluid=" << result.report.fluidCellCount
                  << " particles=" << stepper.grid().particleCount() << '\n';
    }

    std::cout << "Phase 1.7C extended low-resolution stability\n"
                 "Grid Completed Calls ElapsedMs Reject MaxAcceptedCFL Residual Divergence Status\n";
    for (const std::size_t size : {64U, 128U}) {
        const std::size_t targetSteps = size == 64 ? 240 : 120;
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid initial(size, size, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.synchronousGpuTimingEnabled = false;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        static_cast<void>(stepper.initialize(initial, particles));
        std::size_t completed = 0;
        std::size_t calls = 0;
        std::size_t rejected = 0;
        float maximumAcceptedCfl = 0.0f;
        GpuDynamicFreeSurfaceReport report;
        const auto start = std::chrono::steady_clock::now();
        while (completed < targetSteps && calls < 40000 && stepper.healthy()) {
            report = stepper.step();
            ++calls;
            rejected += report.rejectedSubsteps;
            maximumAcceptedCfl = std::max(maximumAcceptedCfl, report.maximumAcceptedCfl);
            if (report.outerStepCompleted) ++completed;
            if (report.failureReason != GpuDynamicFailureReason::none) break;
        }
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << size << 'x' << size << ' ' << completed << ' ' << calls << ' '
                  << std::fixed << std::setprecision(3) << elapsed << ' ' << rejected << ' '
                  << maximumAcceptedCfl << ' ' << std::scientific
                  << report.projection.pressure.finalResidual.rootMeanSquare << ' '
                  << report.projection.divergenceAfter.maximumAbsolute << ' '
                  << gpuDynamicFailureName(report.failureReason) << '\n';
    }
}

void runGpuRenderedPerformance(
    GLFWwindow* window,
    const std::filesystem::path& divergenceShader) {
    using namespace fluidlab::simulation;
    const auto displayProgram = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/fullscreen.vert"),
        findResource("shaders/rendering/texture.frag"));
    const auto particleProgram = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/gpu_particles.vert"),
        findResource("shaders/rendering/gpu_particles.frag"));
    GLuint vertexArray = 0;
    glCreateVertexArrays(1, &vertexArray);
    if (vertexArray == 0) throw std::runtime_error("Failed to create benchmark vertex array.");
    std::cout << "GPU rendered frame performance (two 1/120 s physics steps per frame)\n"
                 "Grid Frames Avg P95 P99 Max FPS SimLag\n";
    for (const std::size_t size : {64U, 128U, 256U}) {
        const std::size_t frameCount = size == 256 ? 20 : 60;
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid initial(size, size, cellSize, cellSize);
        initializeDamBreak(initial);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(initial, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.synchronousGpuTimingEnabled = false;
        settings.detailedVelocityDiagnostics = size == 256;
        GpuDynamicFreeSurfaceStepper stepper(divergenceShader, settings);
        const auto topology = stepper.initialize(initial, particles);
        if (!topology.topologyValid) throw std::runtime_error("Rendered benchmark topology failed.");
        GLuint texture = 0;
        glCreateTextures(GL_TEXTURE_2D, 1, &texture);
        glTextureStorage2D(texture, 1, GL_RGBA32F,
            static_cast<GLsizei>(size), static_cast<GLsizei>(size));
        glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        TimingAccumulator frames;
        const auto runStart = std::chrono::steady_clock::now();
        std::size_t completedFrames = 0;
        std::string stoppedReason;
        GpuDynamicFreeSurfaceReport lastReport;
        for (std::size_t frame = 0; frame < frameCount; ++frame) {
            const auto frameStart = std::chrono::steady_clock::now();
            for (std::size_t physicsStep = 0; physicsStep < 2; ++physicsStep) {
                lastReport = stepper.step();
                if (lastReport.failureReason != GpuDynamicFailureReason::none) {
                    stoppedReason = lastReport.failureMessage;
                    break;
                }
            }
            if (!stoppedReason.empty()) break;
            stepper.grid().dispatchVisualization(texture, 0);
            glClear(GL_COLOR_BUFFER_BIT);
            displayProgram.use();
            glBindTextureUnit(0, texture);
            glBindVertexArray(vertexArray);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            particleProgram.use();
            glUniform2f(particleProgram.uniformLocation("domainSize"), 1.0f, 1.0f);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, stepper.grid().particleBufferHandle());
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(stepper.grid().particleCount()));
            glfwSwapBuffers(window);
            glFinish();
            frames.add(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - frameStart).count());
            ++completedFrames;
        }
        const double realSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - runStart).count();
        const double simulationSeconds = static_cast<double>(completedFrames) / 60.0;
        std::cout << size << 'x' << size << ' ' << completedFrames << '/' << frameCount << ' '
                  << std::fixed << std::setprecision(3)
                  << frames.average() << ' ' << frames.percentile(0.95) << ' '
                  << frames.percentile(0.99) << ' ' << frames.maximum << ' '
                  << (frames.average() > 0.0 ? 1000.0 / frames.average() : 0.0) << ' '
                  << (realSeconds - simulationSeconds);
        if (!stoppedReason.empty()) {
            const bool uDominant = lastReport.velocity.maximumAbsoluteU >=
                lastReport.velocity.maximumAbsoluteV;
            const std::size_t index = uDominant
                ? lastReport.velocity.maximumUIndex : lastReport.velocity.maximumVIndex;
            const std::size_t stride = uDominant ? size + 1 : size;
            std::cout << " STOPPED: " << stoppedReason
                      << ", peakStage=" << gpuVelocityStageName(lastReport.peakVelocityStage)
                      << ", component=" << (uDominant ? 'u' : 'v')
                      << ", index=(" << index % stride << ',' << index / stride << ')'
                      << ", value=" << (uDominant ? lastReport.velocity.maximumAbsoluteU
                          : lastReport.velocity.maximumAbsoluteV)
                      << ", residual="
                      << lastReport.projection.pressure.finalResidual.rootMeanSquare
                      << ", divergence="
                      << lastReport.projection.divergenceAfter.maximumAbsolute;
        }
        std::cout << '\n';
        glDeleteTextures(1, &texture);
    }
    glDeleteVertexArrays(1, &vertexArray);
}

int runGpuAsyncCandidateValidation(GLFWwindow* window) {
    using namespace fluidlab::simulation;
    constexpr std::size_t size = 64;
    constexpr float cellSize = 1.0f / static_cast<float>(size);
    MacGrid grid(size, size, cellSize, cellSize);
    initializeDamBreak(grid);
    MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    const auto shader = findResource("shaders/compute/mac_divergence.comp");

    GpuDynamicFreeSurfaceSettings settings;
    settings.synchronousGpuTimingEnabled = false;
    settings.asynchronousGpuTimingEnabled = false;
    settings.forcedCandidateRejectionsForValidation = 1;
    GpuDynamicFreeSurfaceStepper synchronous(shader, settings);
    const auto synchronousTopology = synchronous.initialize(grid, particles);
    if (!synchronousTopology.topologyValid) {
        throw std::runtime_error("Synchronous topology is invalid in async validation.");
    }
    const auto synchronousReport = synchronous.step();
    if (synchronousReport.failureReason != GpuDynamicFailureReason::none ||
        synchronousReport.rollbackCount == 0) {
        throw std::runtime_error("Synchronous async-reference candidate did not recover.");
    }

    GpuAsyncDynamicFreeSurfaceStepper asynchronous(window, shader, settings);
    const auto asynchronousTopology = asynchronous.initialize(grid, particles);
    if (!asynchronousTopology.topologyValid || !asynchronous.submit()) {
        throw std::runtime_error("Failed to submit the asynchronous candidate.");
    }
    if (!asynchronous.stepper().grid().committedRenderStateEnabled()) {
        throw std::runtime_error("Committed render state was not enabled.");
    }
    const GLuint committedParticleBuffer =
        asynchronous.stepper().grid().particleBufferHandle();
    const std::size_t committedParticleCount =
        asynchronous.stepper().grid().particleCount();
    GLuint committedTexture = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &committedTexture);
    glTextureStorage2D(committedTexture, 1, GL_RGBA32F,
        static_cast<GLsizei>(size), static_cast<GLsizei>(size));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    std::size_t eventPolls = 0;
    while (!asynchronous.hasCompletedResult() &&
           std::chrono::steady_clock::now() < deadline) {
        glfwPollEvents();
        static_cast<void>(asynchronous.poll());
        if (asynchronous.inFlight()) {
            asynchronous.stepper().grid().dispatchVisualization(committedTexture, 0);
            if (asynchronous.stepper().grid().particleBufferHandle() != committedParticleBuffer ||
                asynchronous.stepper().grid().particleCount() != committedParticleCount) {
                throw std::runtime_error("In-flight candidate changed committed render bindings.");
            }
        }
        ++eventPolls;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!asynchronous.hasCompletedResult() ||
        asynchronous.state() != GpuAsyncCandidateState::accepted) {
        throw std::runtime_error("Asynchronous candidate did not reach Accepted.");
    }
    const auto& asynchronousReport = asynchronous.report();
    if (asynchronousReport.failureReason != GpuDynamicFailureReason::none ||
        asynchronousReport.rollbackCount == 0 ||
        asynchronousReport.simulationTime != synchronousReport.simulationTime) {
        throw std::runtime_error("Asynchronous candidate safety report differs from reference.");
    }

    const auto synchronousU = synchronous.grid().readHorizontalVelocity();
    const auto asynchronousU = asynchronous.stepper().grid().readHorizontalVelocity();
    const auto synchronousV = synchronous.grid().readVerticalVelocity();
    const auto asynchronousV = asynchronous.stepper().grid().readVerticalVelocity();
    const auto synchronousTypes = synchronous.grid().readCellTypes();
    const auto asynchronousTypes = asynchronous.stepper().grid().readCellTypes();
    float maximumVelocityError = 0.0f;
    for (std::size_t i = 0; i < synchronousU.size(); ++i) {
        maximumVelocityError = std::max(
            maximumVelocityError, std::abs(synchronousU[i] - asynchronousU[i]));
    }
    for (std::size_t i = 0; i < synchronousV.size(); ++i) {
        maximumVelocityError = std::max(
            maximumVelocityError, std::abs(synchronousV[i] - asynchronousV[i]));
    }
    if (maximumVelocityError > 1.0e-6f || synchronousTypes != asynchronousTypes) {
        throw std::runtime_error("Asynchronous candidate differs from synchronous physics.");
    }
    const std::size_t recoveredRollbackCount = asynchronousReport.rollbackCount;

    asynchronous.acknowledge();
    if (asynchronous.state() != GpuAsyncCandidateState::idle || !asynchronous.submit()) {
        throw std::runtime_error("Asynchronous candidate could not be resubmitted.");
    }
    asynchronous.requestCancel();
    while (!asynchronous.hasCompletedResult() &&
           std::chrono::steady_clock::now() < deadline) {
        glfwPollEvents();
        static_cast<void>(asynchronous.poll());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (asynchronous.state() != GpuAsyncCandidateState::canceled) {
        throw std::runtime_error("In-flight cancellation did not reach Canceled.");
    }
    glDeleteTextures(1, &committedTexture);

    const auto& statistics = asynchronous.statistics();
    if (statistics.workerThreadStartCount != 1 || statistics.completedTaskCount != 2) {
        throw std::runtime_error("Persistent GPU worker did not reuse one thread for both tasks.");
    }
    std::cout << "Phase 1.8.1 GPU async validation passed: state="
              << gpuAsyncCandidateStateName(asynchronous.state())
              << ", polls=" << statistics.fencePollCount
              << ", incomplete=" << statistics.incompleteFencePollCount
              << ", event polls=" << eventPolls
              << ", fence poll=" << statistics.fencePollingMilliseconds << " ms"
              << ", in-flight=" << statistics.lastInFlightMilliseconds << " ms"
              << ", worker starts=" << statistics.workerThreadStartCount
              << ", tasks=" << statistics.completedTaskCount
              << ", mailbox=" << statistics.lastMailboxLatencyMilliseconds << " ms"
              << ", max velocity error=" << maximumVelocityError
              << ", rollback=" << recoveredRollbackCount << ".\n";
    return 0;
}

int runGpuAsyncCandidateBenchmark(GLFWwindow* window) {
    using namespace fluidlab::simulation;
    const auto shader = findResource("shaders/compute/mac_divergence.comp");
    std::cout << "Phase 1.8.1 asynchronous candidate benchmark\n"
                 "Grid  outer avg(ms)  frame avg/p95/p99(ms)  poll cpu(ms)  "
                 "inflight avg(ms)  candidates  physical/wall  residual  divergence\n";
    for (const std::size_t size : {64U, 128U, 192U, 256U}) {
        const float cellSize = 1.0f / static_cast<float>(size);
        MacGrid grid(size, size, cellSize, cellSize);
        initializeDamBreak(grid);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        GpuDynamicFreeSurfaceSettings settings;
        settings.synchronousGpuTimingEnabled = false;
        settings.asynchronousGpuTimingEnabled = false;
        GpuAsyncDynamicFreeSurfaceStepper runner(window, shader, settings);
        const auto topology = runner.initialize(grid, particles);
        if (!topology.topologyValid) {
            throw std::runtime_error("Async benchmark topology is invalid.");
        }
        GLuint texture = 0;
        glCreateTextures(GL_TEXTURE_2D, 1, &texture);
        glTextureStorage2D(texture, 1, GL_RGBA32F,
            static_cast<GLsizei>(size), static_cast<GLsizei>(size));
        TimingAccumulator frameIntervals;
        TimingAccumulator outerSteps;
        TimingAccumulator inFlightTimes;
        double pollingCpuMilliseconds = 0.0;
        std::size_t previousOuterStep = 0;
        std::size_t acceptedOuterSteps = 0;
        auto outerStart = std::chrono::steady_clock::now();
        auto previousFrame = outerStart;
        const auto benchmarkStart = outerStart;
        GpuDynamicFreeSurfaceReport finalReport;
        while (acceptedOuterSteps < 60) {
            const auto frameStart = std::chrono::steady_clock::now();
            frameIntervals.add(std::chrono::duration<double, std::milli>(
                frameStart - previousFrame).count());
            previousFrame = frameStart;
            glfwPollEvents();
            const auto pollStart = std::chrono::steady_clock::now();
            const bool completed = runner.poll();
            pollingCpuMilliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - pollStart).count();
            if (completed) {
                finalReport = runner.report();
                if (runner.state() != GpuAsyncCandidateState::accepted ||
                    finalReport.failureReason != GpuDynamicFailureReason::none) {
                    glDeleteTextures(1, &texture);
                    throw std::runtime_error("Async benchmark candidate failed: " +
                        finalReport.failureMessage);
                }
                inFlightTimes.add(runner.statistics().lastInFlightMilliseconds);
                runner.acknowledge();
                const std::size_t completedOuter =
                    static_cast<std::size_t>(runner.stepper().completedOuterSteps());
                acceptedOuterSteps = completedOuter;
                if (completedOuter != previousOuterStep) {
                    const auto now = std::chrono::steady_clock::now();
                    outerSteps.add(std::chrono::duration<double, std::milli>(
                        now - outerStart).count());
                    outerStart = now;
                    previousOuterStep = completedOuter;
                }
            }
            if (acceptedOuterSteps < 60 && !runner.inFlight() &&
                runner.state() == GpuAsyncCandidateState::idle) {
                static_cast<void>(runner.submit());
            }
            runner.stepper().grid().dispatchVisualization(texture, 0);
            glFlush();
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
        const double wallSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - benchmarkStart).count();
        const auto statistics = runner.statistics();
        glDeleteTextures(1, &texture);
        std::cout << std::setw(4) << size << "  " << std::fixed << std::setprecision(3)
                  << std::setw(13) << outerSteps.average() << "  "
                  << frameIntervals.average() << '/' << frameIntervals.percentile(0.95)
                  << '/' << frameIntervals.percentile(0.99) << "  "
                  << pollingCpuMilliseconds << "  " << inFlightTimes.average() << "  "
                  << statistics.acceptedCount << "  "
                  << (wallSeconds > 0.0 ? 0.5 / wallSeconds : 0.0) << "  "
                  << std::scientific
                  << finalReport.projection.pressure.finalResidual.rootMeanSquare << "  "
                  << finalReport.projection.divergenceAfter.maximumAbsolute << '\n';
    }
    return 0;
}

#endif

void desktopWindowPositionCallback(GLFWwindow* window, int x, int y) {
    auto* motion = static_cast<fluidlab::desktop::ContainerMotionInput*>(
        glfwGetWindowUserPointer(window));
    if (motion != nullptr) {
        static_cast<void>(motion->addPositionSample(x, y, glfwGetTime()));
    }
}

int runDamBreakVisualization(
    GLFWwindow* window,
    bool validationMode,
    bool initialGpuMode = false,
    bool desktopInteractive = false,
    bool settledDesktopInitial = false,
    double desktopPixelsPerSimulationUnit = 600.0,
    DesktopRenderStyle desktopRenderStyle = {},
    DesktopFluidInjectionSettings injectionSettings = {}) {
    constexpr std::size_t width = 64;
    constexpr std::size_t height = 48;
    constexpr float cellSize = 1.0f / 64.0f;
    fluidlab::simulation::MacGrid grid(width, height, cellSize, cellSize);
    fluidlab::simulation::MarkerParticleSystem particles;
    auto stepper = std::make_unique<fluidlab::simulation::DynamicFreeSurfaceStepper>();
    fluidlab::simulation::DynamicFreeSurfaceReport report;
    std::unique_ptr<fluidlab::simulation::GpuDynamicFreeSurfaceStepper> gpuStepper;
    std::unique_ptr<fluidlab::simulation::GpuAsyncDynamicFreeSurfaceStepper> asyncGpuStepper;
    fluidlab::simulation::GpuDynamicFreeSurfaceReport gpuReport;
    bool gpuMode = initialGpuMode;
    bool asynchronousGpuMode = desktopInteractive;
    bool redistributionEnabled = false;
    bool simulationError = false;
    std::string simulationErrorMessage;
    fluidlab::desktop::ContainerMotionSettings containerMotionSettings;
    containerMotionSettings.pixelsPerSimulationUnit = desktopPixelsPerSimulationUnit;
    containerMotionSettings.horizontalOnly = false;
    fluidlab::desktop::ContainerMotionInput containerMotion(containerMotionSettings);
    fluidlab::desktop::DesktopPhysicsClock desktopPhysicsClock(1.0 / 120.0);
    std::uint64_t lastCommittedInputVersion = 0;
    double lastInputResponseDelay = 0.0;
    double totalInjectedArea = 0.0;
    std::size_t totalInjectedParticles = 0;
    bool inertiaInputEnabled = desktopInteractive;
    std::optional<fluidlab::desktop::DesktopSettledState> settledState;

    // The release starts from the packaged calm state. Development builds can
    // fall back to dam break when working without the asset.
    if (desktopInteractive && settledDesktopInitial) {
        try {
            const auto path = findResource("assets/desktop/settled_64x48.lsstate");
            auto loaded = fluidlab::desktop::loadDesktopSettledState(
                path, width, height, cellSize, cellSize);
            if (!loaded.valid) {
                throw std::runtime_error(loaded.message);
            }
            settledState = std::move(loaded.state);
            std::cout << "Desktop settled state loaded: source time="
                      << settledState->metadata.preEvolutionSeconds << " s, velocity RMS="
                      << settledState->metadata.velocityRms << ".\n";
        } catch (const std::exception& error) {
            if constexpr (kDesktopRelease) throw;
            std::cerr << "Desktop settled state unavailable; using explicit dam-break fallback: "
                      << error.what() << '\n';
        }
    }
    if (desktopInteractive) {
        int windowX = 0, windowY = 0;
        glfwGetWindowPos(window, &windowX, &windowY);
        static_cast<void>(containerMotion.addPositionSample(
            windowX, windowY, glfwGetTime()));
        glfwSetWindowUserPointer(window, &containerMotion);
        glfwSetWindowPosCallback(window, desktopWindowPositionCallback);
    }

    const auto reset = [&]() {
        // Recreate the complete simulation state so Reset and backend changes
        // cannot retain candidate data from the previous configuration.
        grid = fluidlab::simulation::MacGrid(width, height, cellSize, cellSize);
        particles = fluidlab::simulation::MarkerParticleSystem{};
        if (settledState.has_value()) {
            fluidlab::desktop::applyDesktopSettledState(*settledState, grid, particles);
        } else {
            fluidlab::simulation::initializeDamBreak(grid);
            particles.seedFluidCellsExact(grid, 8);
        }
        stepper = std::make_unique<fluidlab::simulation::DynamicFreeSurfaceStepper>();
        report = {};
        fluidlab::simulation::GpuDynamicFreeSurfaceSettings gpuSettings;
        gpuSettings.synchronousGpuTimingEnabled = validationMode;
        gpuSettings.particleRedistributionEnabled = redistributionEnabled;
        if (asynchronousGpuMode) {
            asyncGpuStepper =
                std::make_unique<fluidlab::simulation::GpuAsyncDynamicFreeSurfaceStepper>(
                    window, findResource("shaders/compute/mac_divergence.comp"), gpuSettings);
            gpuStepper.reset();
        } else {
            gpuStepper = std::make_unique<fluidlab::simulation::GpuDynamicFreeSurfaceStepper>(
                findResource("shaders/compute/mac_divergence.comp"), gpuSettings);
            asyncGpuStepper.reset();
        }
        const auto topology = asynchronousGpuMode
            ? asyncGpuStepper->initialize(grid, particles)
            : gpuStepper->initialize(grid, particles);
        gpuReport = {};
        gpuReport.topology = topology;
        gpuReport.fluidCellCount = static_cast<std::size_t>(std::count(
            grid.cellTypes().begin(), grid.cellTypes().end(),
            fluidlab::simulation::CellType::fluid));
        gpuReport.particleCount = particles.validParticleCount();
        gpuReport.markedArea = static_cast<float>(gpuReport.fluidCellCount) *
            cellSize * cellSize;
        gpuReport.topologyValid = topology.topologyValid && topology.multigridValid;
        simulationError = asynchronousGpuMode
            ? !asyncGpuStepper->stepper().healthy() : !gpuStepper->healthy();
        simulationErrorMessage = simulationError
            ? "Initial GPU pressure topology is unsupported." : std::string{};
        if (desktopInteractive) {
            int windowX = 0, windowY = 0;
            glfwGetWindowPos(window, &windowX, &windowY);
            containerMotion.reset();
            lastCommittedInputVersion = 0;
            lastInputResponseDelay = 0.0;
            totalInjectedArea = 0.0;
            totalInjectedParticles = 0;
            static_cast<void>(containerMotion.addPositionSample(
                windowX, windowY, glfwGetTime()));
            desktopPhysicsClock.reset(glfwGetTime(), 0.0);
        }
    };
    reset();
    if constexpr (kDesktopRelease) {
        if (simulationError) throw std::runtime_error(simulationErrorMessage);
    }
    const auto activeGpuStepper = [&]() -> fluidlab::simulation::GpuDynamicFreeSurfaceStepper& {
        return asynchronousGpuMode ? asyncGpuStepper->stepper() : *gpuStepper;
    };

    const auto displayProgram = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/fullscreen.vert"),
        findResource("shaders/rendering/texture.frag"));
#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
    const auto particleProgram = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/vectors.vert"),
        findResource("shaders/rendering/vectors.frag"));
    const auto gpuParticleProgram = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/gpu_particles.vert"),
        findResource("shaders/rendering/gpu_particles.frag"));
#endif
    OpenGlObjects objects;
    glCreateTextures(GL_TEXTURE_2D, 1, &objects.texture);
    glCreateVertexArrays(1, &objects.fullscreenVertexArray);
    glCreateVertexArrays(1, &objects.vectorVertexArray);
    glCreateBuffers(1, &objects.vectorBuffer);
    if (objects.texture == 0 || objects.fullscreenVertexArray == 0 ||
        objects.vectorVertexArray == 0 || objects.vectorBuffer == 0) {
        throw std::runtime_error("Failed to create dam-break visualization resources.");
    }
    glTextureStorage2D(objects.texture, 1, GL_RGBA32F,
        static_cast<GLsizei>(width), static_cast<GLsizei>(height));
    glTextureParameteri(objects.texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(objects.texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glVertexArrayVertexBuffer(objects.vectorVertexArray, 0, objects.vectorBuffer, 0, 2 * sizeof(float));
    glEnableVertexArrayAttrib(objects.vectorVertexArray, 0);
    glVertexArrayAttribFormat(objects.vectorVertexArray, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(objects.vectorVertexArray, 0, 0);
    glNamedBufferData(objects.vectorBuffer,
        static_cast<GLsizeiptr>(particles.particles().size() * 2 * sizeof(float)),
        nullptr, GL_DYNAMIC_DRAW);

    std::vector<float> pixels;
    std::vector<float> particleVertices;
    particleVertices.reserve(particles.particles().size() * 2);
    int view = 1;
    [[maybe_unused]] bool showParticles = false;
    bool paused = false;
    bool previousPause = false;
    bool previousStep = false;
    bool previousReset = false;
    bool previousParticles = false;
    bool previousBackend = false;
    bool previousRedistribution = false;
    bool previousAsync = false;
    bool previousInteraction = false;
    bool previousOverlay = false;
    bool previousVsync = false;
    bool performanceOverlayEnabled = desktopInteractive && !kDesktopRelease;
    bool vsyncEnabled = true;
    bool pendingBackendToggle = false;
    bool pendingAsyncToggle = false;
    bool pendingRedistributionToggle = false;
    bool singleStepPending = false;
    double previousTime = glfwGetTime();
    const double applicationStartTime = previousTime;
    double titleTime = previousTime;
    double accumulator = 0.0;
    TimingAccumulator frameIntervals, physicsSteps, classifications, extrapolations;
    TimingAccumulator advections, pressureSolves, divergenceAndProjection, particleUpdates;
    TimingAccumulator visualizationBuilds, openGlUploads;
    std::size_t windowFrames = 0, windowPhysicsSteps = 0;
    [[maybe_unused]] double droppedWallTime = 0.0;
    fluidlab::desktop::DesktopPerformanceMetrics desktopMetrics(600);
    fluidlab::desktop::WindowsMoveLoopAdapter moveLoopAdapter;
    std::uint64_t lastRenderedCommittedVersion = 0;
    double lastSnapshotPublishWallTime = 0.0;
    double lastSnapshotPresentationMilliseconds = 0.0;

    const auto upload = [&]() {
        const auto buildStart = std::chrono::steady_clock::now();
        glPointSize(2.0f);
        if (gpuMode) {
            activeGpuStepper().grid().dispatchVisualization(objects.texture,
                static_cast<std::size_t>(view - 1));
            visualizationBuilds.add(0.0);
            openGlUploads.add(0.0);
            return;
        }
        if (view == 1) fluidlab::visualization::buildCellTypeTexture(grid, pixels);
        else if (view == 2) fluidlab::visualization::buildSpeedTexture(grid, pixels);
        else if (view == 3) fluidlab::visualization::buildSignedScalarTexture(grid.pressures(), pixels);
        else fluidlab::visualization::buildSignedScalarTexture(grid.divergences(), pixels);
        particleVertices.clear();
        const float domainWidth = static_cast<float>(width) * cellSize;
        const float domainHeight = static_cast<float>(height) * cellSize;
        for (const auto& particle : particles.particles()) {
            if (!particle.valid) continue;
            particleVertices.push_back(particle.position.x / domainWidth);
            particleVertices.push_back(particle.position.y / domainHeight);
        }
        const auto buildEnd = std::chrono::steady_clock::now();
        glTextureSubImage2D(objects.texture, 0, 0, 0,
            static_cast<GLsizei>(width), static_cast<GLsizei>(height),
            GL_RGBA, GL_FLOAT, pixels.data());
        if (!particleVertices.empty()) glNamedBufferSubData(objects.vectorBuffer, 0,
            static_cast<GLsizeiptr>(particleVertices.size() * sizeof(float)), particleVertices.data());
        const auto uploadEnd = std::chrono::steady_clock::now();
        visualizationBuilds.add(
            std::chrono::duration<double, std::milli>(buildEnd - buildStart).count());
        openGlUploads.add(
            std::chrono::duration<double, std::milli>(uploadEnd - buildEnd).count());
    };
    upload();
#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
    if (validationMode) {
        if (gpuMode) {
            for (int step = 0; step < 240; ++step) {
                gpuReport = activeGpuStepper().step();
                if (gpuReport.failureReason !=
                    fluidlab::simulation::GpuDynamicFailureReason::none) {
                    throw std::runtime_error(std::string("GPU dam-break validation stopped: ") +
                        gpuReport.failureMessage);
                }
            }
        } else {
            for (int step = 0; step < 120; ++step) report = stepper->step(grid, particles);
        }
        upload();
        glClearColor(0.01f, 0.015f, 0.025f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        displayProgram.use();
        applyDesktopRenderStyle(displayProgram, desktopInteractive && view == 1,
            desktopRenderStyle);
        glBindTextureUnit(0, objects.texture);
        glBindVertexArray(objects.fullscreenVertexArray);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (gpuMode) {
            gpuParticleProgram.use();
            glUniform2f(gpuParticleProgram.uniformLocation("domainSize"),
                static_cast<float>(width) * cellSize,
                static_cast<float>(height) * cellSize);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0,
                activeGpuStepper().grid().particleBufferHandle());
            glBindVertexArray(objects.fullscreenVertexArray);
            glDrawArrays(GL_POINTS, 0,
                static_cast<GLsizei>(activeGpuStepper().grid().particleCount()));
        } else {
            particleProgram.use();
            glBindVertexArray(objects.vectorVertexArray);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particleVertices.size() / 2));
        }
        glFinish();
        if (gpuMode) {
            const auto gpuParticles = activeGpuStepper().grid().readMarkerParticles();
            const bool validParticles = std::all_of(gpuParticles.begin(), gpuParticles.end(),
                [&](const auto& particle) {
                    return particle.valid && std::isfinite(particle.position.x) &&
                        std::isfinite(particle.position.y) && particle.position.x >= 0.0f &&
                        particle.position.y >= 0.0f &&
                        particle.position.x <= static_cast<float>(width) * cellSize &&
                        particle.position.y <= static_cast<float>(height) * cellSize;
                });
            if (!validParticles || !gpuReport.pressureConverged ||
                activeGpuStepper().grid().particleCount() != particles.particles().size()) {
                throw std::runtime_error("GPU dam-break validation detected invalid state.");
            }
            TimingAccumulator validationFrames;
            for (std::size_t frame = 0; frame < 60; ++frame) {
                const auto frameStart = std::chrono::steady_clock::now();
                gpuReport = activeGpuStepper().step();
                if (gpuReport.failureReason !=
                    fluidlab::simulation::GpuDynamicFailureReason::none) {
                    throw std::runtime_error("GPU rendered continuity validation failed.");
                }
                upload();
                glClear(GL_COLOR_BUFFER_BIT);
                displayProgram.use();
                applyDesktopRenderStyle(displayProgram, desktopInteractive && view == 1,
                    desktopRenderStyle);
                glBindTextureUnit(0, objects.texture);
                glBindVertexArray(objects.fullscreenVertexArray);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                gpuParticleProgram.use();
                glUniform2f(gpuParticleProgram.uniformLocation("domainSize"),
                    static_cast<float>(width) * cellSize,
                    static_cast<float>(height) * cellSize);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0,
                    activeGpuStepper().grid().particleBufferHandle());
                glDrawArrays(GL_POINTS, 0,
                    static_cast<GLsizei>(activeGpuStepper().grid().particleCount()));
                glfwSwapBuffers(window);
                glFinish();
                validationFrames.add(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - frameStart).count());
            }
            std::cout << "Phase 1.6C GPU dam-break validation passed: steps=300, particles="
                      << activeGpuStepper().grid().particleCount() << ", fluid cells="
                      << gpuReport.fluidCellCount << ", residual="
                      << gpuReport.projection.pressure.finalResidual.rootMeanSquare
                      << ", divergence=" << gpuReport.projection.divergenceAfter.maximumAbsolute
                      << ", step=" << gpuReport.timings.totalCpuMilliseconds << " ms.\n";
            std::cout << "64x48 rendered validation frame avg/P95/P99/max="
                      << validationFrames.average() << '/'
                      << validationFrames.percentile(0.95) << '/'
                      << validationFrames.percentile(0.99) << '/'
                      << validationFrames.maximum << " ms, FPS="
                      << (validationFrames.average() > 0.0
                          ? 1000.0 / validationFrames.average() : 0.0)
                      << " (synchronized hidden-window measurement).\n";
            runGpuDamBreakValidationSuite(
                findResource("shaders/compute/mac_divergence.comp"));
            runGpuRenderedPerformance(
                window, findResource("shaders/compute/mac_divergence.comp"));
            return 0;
        }
        if (!grid.allFinite() || !particles.allFiniteAndInside(grid) ||
            report.particleCountBefore != report.particleCountAfter) {
            throw std::runtime_error("Dam-break visual validation detected invalid simulation state.");
        }
        std::cout << "Phase 1.4 dam-break visual validation passed: particles="
                  << report.particleCountAfter << ", fluid cells=" << report.fluidCellCount
                  << ", max speed=" << report.maximumSpeed << ".\n";
        return 0;
    }
    std::cout << "Dam-break controls: B CPU/GPU, A synchronous/asynchronous GPU, "
                 "1 cells, 2 speed, 3 pressure, 4 divergence, "
                 "5 particle density, 6 hole diagnostics, M redistribution, P particles, "
                 "I container inertia, O performance details, V VSync, "
                 "Space pause, N step, R reset, Escape exit.\n";
#endif
    [[maybe_unused]] bool firstFrame = true;

    std::function<void(bool)> frameTick;
    bool frameTickRunning = false;
    frameTick = [&](bool pollEvents) {
        const auto completeFrameStart = std::chrono::steady_clock::now();
        double eventMilliseconds = 0.0;
        if (pollEvents) {
            const auto eventStart = std::chrono::steady_clock::now();
            glfwPollEvents();
            eventMilliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - eventStart).count();
        }
        if (frameTickRunning || glfwWindowShouldClose(window) != GLFW_FALSE) return;
        frameTickRunning = true;
        struct FrameGuard final {
            bool& running;
            ~FrameGuard() { running = false; }
        } frameGuard{frameTickRunning};
        double fencePollMilliseconds = 0.0;
        double physicsSubmitMilliseconds = 0.0;
        bool asynchronousCompletion = false;
        bool asynchronousCancellationReconfigured = false;
        if (asynchronousGpuMode && asyncGpuStepper->inFlight()) {
            const auto pollStart = std::chrono::steady_clock::now();
            asynchronousCompletion = asyncGpuStepper->poll();
            fencePollMilliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - pollStart).count();
            if (asynchronousCompletion) {
                gpuReport = asyncGpuStepper->report();
                totalInjectedArea += gpuReport.injectionAreaAccepted;
                totalInjectedParticles += gpuReport.injectionParticlesAdded;
                lastSnapshotPublishWallTime = glfwGetTime();
            }
        }
        if (asynchronousCompletion &&
            asyncGpuStepper->state() ==
                fluidlab::simulation::GpuAsyncCandidateState::canceled) {
            if (pendingBackendToggle) gpuMode = !gpuMode;
            if (pendingAsyncToggle) asynchronousGpuMode = !asynchronousGpuMode;
            if (pendingRedistributionToggle) redistributionEnabled = !redistributionEnabled;
            reset();
            accumulator = 0.0;
            asynchronousCancellationReconfigured = true;
            pendingBackendToggle = pendingAsyncToggle = false;
            pendingRedistributionToggle = false;
        }
        if (asynchronousCompletion && desktopInteractive &&
            asyncGpuStepper->state() !=
                fluidlab::simulation::GpuAsyncCandidateState::canceled &&
            gpuReport.batchOuterStepsCompleted > 0) {
            // Commit only the completed prefix. The remaining motion impulse stays
            // queued and can be consumed by a later candidate.
            const double completedFraction = gpuReport.batchOuterStepsRequested > 0
                ? std::min(1.0, static_cast<double>(gpuReport.batchOuterStepsCompleted) /
                    static_cast<double>(gpuReport.batchOuterStepsRequested))
                : 0.0;
            if (containerMotion.commitPreparedFraction(
                gpuReport.inputVersion, completedFraction)) {
                lastCommittedInputVersion = gpuReport.inputVersion;
                lastInputResponseDelay = std::max(0.0, glfwGetTime() -
                    gpuReport.inputSourceTimeEnd);
            }
        } else if (asynchronousCompletion && desktopInteractive &&
            asyncGpuStepper->state() !=
                fluidlab::simulation::GpuAsyncCandidateState::canceled) {
            // A candidate that completed no outer step must not consume input.
            containerMotion.discardPreparedStep();
        }
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        const auto developmentKeyPressed = [window](int key) {
            if constexpr (kDesktopRelease) {
                return false;
            } else {
                return glfwGetKey(window, key) == GLFW_PRESS;
            }
        };
        const bool pauseKey = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        const bool stepKey = developmentKeyPressed(GLFW_KEY_N);
        const bool resetKey = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
        const bool particleKey = developmentKeyPressed(GLFW_KEY_P);
        const bool backendKey = developmentKeyPressed(GLFW_KEY_B);
        const bool asyncKey = developmentKeyPressed(GLFW_KEY_A);
        const bool redistributionKey = developmentKeyPressed(GLFW_KEY_M);
        const bool interactionKey = glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS;
        const bool overlayKey = developmentKeyPressed(GLFW_KEY_O);
        const bool vsyncKey = glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS;
        if (pauseKey && !previousPause) paused = !paused;
#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
        if (particleKey && !previousParticles) showParticles = !showParticles;
        if (backendKey && !previousBackend) {
            if (asynchronousGpuMode && asyncGpuStepper->inFlight()) {
                pendingBackendToggle = true;
                asyncGpuStepper->requestCancel();
            } else {
                gpuMode = !gpuMode;
                reset();
                accumulator = 0.0;
            }
            paused = false;
        }
        if (asyncKey && !previousAsync) {
            if (asynchronousGpuMode && asyncGpuStepper->inFlight()) {
                pendingAsyncToggle = true;
                asyncGpuStepper->requestCancel();
            } else {
                asynchronousGpuMode = !asynchronousGpuMode;
                reset();
                accumulator = 0.0;
            }
            paused = false;
        }
        if (redistributionKey && !previousRedistribution) {
            if (asynchronousGpuMode && asyncGpuStepper->inFlight()) {
                pendingRedistributionToggle = true;
                asyncGpuStepper->requestCancel();
            } else {
                redistributionEnabled = !redistributionEnabled;
                reset();
                accumulator = 0.0;
            }
            paused = false;
        }
#endif
        if (interactionKey && !previousInteraction && desktopInteractive) {
            inertiaInputEnabled = !inertiaInputEnabled;
            int windowX = 0, windowY = 0;
            glfwGetWindowPos(window, &windowX, &windowY);
            containerMotion.reset();
            static_cast<void>(containerMotion.addPositionSample(
                windowX, windowY, glfwGetTime()));
        }
        if constexpr (!kDesktopRelease) {
            if (overlayKey && !previousOverlay && desktopInteractive) {
                performanceOverlayEnabled = !performanceOverlayEnabled;
            }
        }
        if (vsyncKey && !previousVsync && desktopInteractive) {
            vsyncEnabled = !vsyncEnabled;
            glfwSwapInterval(vsyncEnabled ? 1 : 0);
        }
        if (resetKey && !previousReset) {
            if (asynchronousGpuMode && asyncGpuStepper->inFlight()) {
                asyncGpuStepper->requestCancel();
            } else {
                reset();
                accumulator = 0.0;
            }
        }
        const int oldView = view;
        if constexpr (!kDesktopRelease) {
            if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) view = 1;
            else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) view = 2;
            else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) view = 3;
            else if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) view = 4;
            else if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) view = 5;
            else if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) view = 6;
        }

        const double now = glfwGetTime();
        fluidlab::simulation::Vector2 emitterPosition{};
        bool emitterActive = false;
        if (desktopInteractive) {
            int windowX = 0, windowY = 0;
            glfwGetWindowPos(window, &windowX, &windowY);
            static_cast<void>(containerMotion.addPositionSample(windowX, windowY, now));

            double cursorX = 0.0, cursorY = 0.0;
            int windowWidth = 0, windowHeight = 0;
            glfwGetCursorPos(window, &cursorX, &cursorY);
            glfwGetWindowSize(window, &windowWidth, &windowHeight);
            emitterActive = injectionSettings.enabled && !paused && gpuMode &&
                glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS &&
                windowWidth > 0 && windowHeight > 0 &&
                cursorX >= 0.0 && cursorX < windowWidth &&
                cursorY >= 0.0 && cursorY < windowHeight;
            if (emitterActive) {
                emitterPosition = {
                    static_cast<float>(cursorX / windowWidth) * width * cellSize,
                    static_cast<float>(1.0 - cursorY / windowHeight) * height * cellSize};
            }
        }
        const double frameTime = std::clamp(now - previousTime, 0.0, 0.1);
        if (desktopInteractive) {
            const double committedTime = gpuMode
                ? gpuReport.simulationTime : report.simulationTime;
            desktopPhysicsClock.update(now, committedTime, paused);
        }
        frameIntervals.add(frameTime * 1000.0);
        ++windowFrames;
        previousTime = now;
        bool changed = asynchronousCompletion || (resetKey && !previousReset) ||
            (backendKey && !previousBackend) || (asyncKey && !previousAsync) ||
            (redistributionKey && !previousRedistribution) || view != oldView;
        const auto recordStep = [&]() {
            if (gpuMode) {
                physicsSteps.add(gpuReport.timings.totalCpuMilliseconds);
                classifications.add(gpuReport.timings.classificationGpuMilliseconds);
                extrapolations.add(gpuReport.timings.velocityExtrapolationGpuMilliseconds);
                advections.add(gpuReport.timings.velocityAdvectionGpuMilliseconds);
                pressureSolves.add(gpuReport.timings.pressureGpuMilliseconds +
                    gpuReport.timings.pressureReductionGpuMilliseconds);
                divergenceAndProjection.add(gpuReport.timings.divergenceGpuMilliseconds +
                    gpuReport.timings.projectionGpuMilliseconds);
                particleUpdates.add(gpuReport.timings.particleGpuMilliseconds);
                windowPhysicsSteps += std::max<std::size_t>(
                    1, gpuReport.batchOuterStepsCompleted);
                return;
            }
            physicsSteps.add(report.timings.totalMilliseconds);
            classifications.add(report.timings.classificationMilliseconds);
            extrapolations.add(report.timings.velocityExtrapolationMilliseconds);
            advections.add(report.timings.velocityAdvectionMilliseconds);
            pressureSolves.add(report.timings.pressureSolveMilliseconds);
            divergenceAndProjection.add(report.timings.divergenceMilliseconds +
                report.timings.pressureApplicationMilliseconds);
            particleUpdates.add(report.timings.particleAdvectionAndCollisionMilliseconds);
            ++windowPhysicsSteps;
        };
        const auto advance = [&](std::size_t requestedOuterSteps = 1) {
            const auto submitStart = std::chrono::steady_clock::now();
            requestedOuterSteps = std::max<std::size_t>(1, requestedOuterSteps);
            const double requestedDuration = stepper->settings().timeStep *
                static_cast<double>(requestedOuterSteps);

            // Freeze one immutable input transaction for the complete candidate.
            // Retries reuse this value, so an impulse is never applied twice.
            auto stepInput = desktopInteractive
                ? containerMotion.prepareStep(
                    now, requestedDuration, inertiaInputEnabled)
                : fluidlab::simulation::SimulationStepInput{};
            if (emitterActive) {
                stepInput.fluidInjection.active = true;
                stepInput.fluidInjection.position = emitterPosition;
                stepInput.fluidInjection.radius =
                    injectionSettings.radiusInCells * cellSize;
                stepInput.fluidInjection.requestedArea =
                    injectionSettings.areaRate * requestedDuration;
                stepInput.fluidInjection.version = stepInput.version;
            }
            if (gpuMode) {
                if (asynchronousGpuMode) {
                    if (!asyncGpuStepper->submitBatch(stepInput, requestedOuterSteps) &&
                        desktopInteractive) {
                        containerMotion.discardPreparedStep();
                    }
                    physicsSubmitMilliseconds += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - submitStart).count();
                    return;
                }
                gpuReport = activeGpuStepper().step(stepInput);
                totalInjectedArea += gpuReport.injectionAreaAccepted;
                totalInjectedParticles += gpuReport.injectionParticlesAdded;
                if (gpuReport.failureReason !=
                    fluidlab::simulation::GpuDynamicFailureReason::none) {
                    simulationError = true;
                    simulationErrorMessage = gpuReport.failureMessage;
                    paused = true;
                    std::cerr << "GPU simulation paused: " << simulationErrorMessage << '\n';
                    if (desktopInteractive) containerMotion.discardPreparedStep();
                } else if (desktopInteractive && gpuReport.outerStepCompleted &&
                    containerMotion.commitPreparedStep(gpuReport.inputVersion)) {
                    lastCommittedInputVersion = gpuReport.inputVersion;
                    lastInputResponseDelay = std::max(0.0,
                        glfwGetTime() - gpuReport.inputSourceTimeEnd);
                }
            } else {
                report = stepper->step(grid, particles, stepInput);
                if (desktopInteractive) {
                    if (!report.cflLimitReached &&
                        containerMotion.commitPreparedStep(stepInput.version)) {
                        lastCommittedInputVersion = stepInput.version;
                        lastInputResponseDelay = std::max(0.0,
                            glfwGetTime() - stepInput.sourceTimeEnd);
                    } else if (report.cflLimitReached) {
                        containerMotion.discardPreparedStep();
                    }
                }
            }
            physicsSubmitMilliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - submitStart).count();
            recordStep();
        };
        if (asynchronousCompletion && !asynchronousCancellationReconfigured &&
            asynchronousGpuMode &&
            asyncGpuStepper->state() !=
                fluidlab::simulation::GpuAsyncCandidateState::canceled) {
            recordStep();
            if (gpuReport.failureReason !=
                fluidlab::simulation::GpuDynamicFailureReason::none) {
                simulationError = true;
                simulationErrorMessage = gpuReport.failureMessage;
                paused = true;
                if (desktopInteractive) containerMotion.discardPreparedStep();
                std::cerr << "GPU simulation paused: " << simulationErrorMessage << '\n';
            } else {
                asyncGpuStepper->acknowledge();
            }
        }
        if (paused && stepKey && !previousStep) {
            if (!simulationError) {
                singleStepPending = asynchronousGpuMode && gpuMode;
                advance(1);
            }
            changed = true;
        }
        if (paused && singleStepPending && asynchronousGpuMode && gpuMode &&
            !asyncGpuStepper->inFlight() &&
            activeGpuStepper().pendingPhysicalTime() > 0.0 && !simulationError) {
            advance(1);
        }
        if (asynchronousCompletion && singleStepPending &&
            (!gpuMode || activeGpuStepper().pendingPhysicalTime() <= 0.0 || simulationError)) {
            singleStepPending = false;
        }
        if (!paused) {
            if (desktopInteractive && asynchronousGpuMode && gpuMode) {
                const bool continuingOuterStep =
                    activeGpuStepper().pendingPhysicalTime() > 0.0;
                std::size_t batchSize = desktopPhysicsClock.planBatch(
                    4, asyncGpuStepper->inFlight(), continuingOuterStep);
                if (emitterActive) batchSize = std::min<std::size_t>(batchSize, 1);
                if (batchSize > 1) {
                    // Keep a batch inside one motion segment. This preserves the
                    // time order of rapid start, stop, and reversal impulses.
                    const double continuousDuration =
                        containerMotion.maximumContinuousPreparedDuration();
                    if (std::isfinite(continuousDuration)) {
                        const auto segmentSteps = std::max<std::size_t>(1,
                            static_cast<std::size_t>(std::ceil(
                                continuousDuration / stepper->settings().timeStep - 1.0e-9)));
                        batchSize = std::min(batchSize, segmentSteps);
                    }
                }
                if (batchSize > 0 && !simulationError) {
                    advance(batchSize);
                    changed = true;
                }
            } else {
                accumulator += frameTime;
                std::size_t frameSteps = 0;
                while ((!asynchronousGpuMode || !gpuMode || !asyncGpuStepper->inFlight()) &&
                       (accumulator >= stepper->settings().timeStep ||
                        (gpuMode && activeGpuStepper().pendingPhysicalTime() > 0.0)) &&
                       frameSteps < 4) {
                    const bool continuingGpuOuterStep =
                        gpuMode && activeGpuStepper().pendingPhysicalTime() > 0.0;
                    advance(1);
                    if (!continuingGpuOuterStep) {
                        accumulator = std::max(0.0,
                            accumulator - stepper->settings().timeStep);
                    }
                    ++frameSteps;
                    changed = true;
                    if (asynchronousGpuMode && gpuMode) break;
                    if (simulationError) break;
                }
            }
        }
        previousPause = pauseKey;
        previousStep = stepKey;
        previousReset = resetKey;
        previousParticles = particleKey;
        previousBackend = backendKey;
        previousAsync = asyncKey;
        previousRedistribution = redistributionKey;
        previousInteraction = interactionKey;
        previousOverlay = overlayKey;
        previousVsync = vsyncKey;
        if (changed || particleKey) upload();

        if (now - titleTime >= 0.25) {
#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
            const double displayedSimulationTime = gpuMode
                ? gpuReport.simulationTime : report.simulationTime;
            const std::size_t displayedFluidCells = gpuMode
                ? gpuReport.fluidCellCount : report.fluidCellCount;
            const std::size_t displayedParticles = gpuMode
                ? activeGpuStepper().grid().particleCount() : particles.validParticleCount();
            const float displayedMaximumSpeed = gpuMode
                ? std::max(gpuReport.velocity.maximumAbsoluteU,
                    gpuReport.velocity.maximumAbsoluteV)
                : report.maximumSpeed;
            const std::size_t displayedIterations = gpuMode
                ? gpuReport.projection.pressure.iterations
                : report.projection.projection.pressureSolve.iterations;
            const float displayedDivergence = gpuMode
                ? gpuReport.projection.divergenceAfter.maximumAbsolute
                : report.projection.projection.divergenceAfter.maximumAbsolute;
            const float displayedResidual = gpuMode
                ? gpuReport.projection.pressure.finalResidual.rootMeanSquare
                : report.projection.projection.pressureSolve.finalResidual.rootMeanSquare;
            std::ostringstream title;
            const auto detailed = desktopMetrics.summary();
            const auto physicsClock = desktopPhysicsClock.snapshot();
            const auto motionState = containerMotion.state();
            const auto motionDiagnostics = containerMotion.diagnostics(now);
            title << std::fixed << std::setprecision(3)
                  << "LiquidSimulator Phase 2.0D | "
                  << (gpuMode ? (asynchronousGpuMode ? "GPU-ASYNC" : "GPU-SYNC") : "CPU")
                  << " | " << width << 'x' << height
                  << " | t=" << displayedSimulationTime
                  << " pending=" << (gpuMode ? gpuReport.remainingPhysicalTime : 0.0)
                  << " retry=" << (gpuMode ? gpuReport.rejectedSubsteps : 0)
                  << " | fluid=" << displayedFluidCells
                  << " | particles=" << displayedParticles
                  << " | holes=" << (gpuMode ? gpuReport.classification.enclosedAirCellCount : 0)
                  << " surface=" << (gpuMode ? gpuReport.classification.surfaceHoleCandidateCount : 0)
                  << " toggles=" << (gpuMode ? gpuReport.classification.phaseToggleCount : 0)
                  << " redistribute=" << (redistributionEnabled ? "on" : "off")
                  << " inertia=" << (desktopInteractive
                      ? (inertiaInputEnabled ? "on" : "off") : "n/a")
                  << " initial=" << (settledState.has_value() ? "settled" : "dam-break")
                  << " motionScale=" << desktopPixelsPerSimulationUnit
                  << " vsync=" << (vsyncEnabled ? "on" : "off")
                  << " ax=" << (gpuMode ? gpuReport.appliedExternalAcceleration.x : 0.0f)
                   << " ay=" << (gpuMode ? gpuReport.appliedExternalAcceleration.y : 0.0f)
                   << " inject=" << totalInjectedArea
                   << " added=" << totalInjectedParticles
                  << " | vmax=" << displayedMaximumSpeed
                  << " CFL=" << (gpuMode ? gpuReport.maximumObservedCfl : report.maximumObservedCfl)
                  << " | frame p50/p95/p99/max=" << detailed.frameP50Milliseconds << '/'
                  << detailed.frameP95Milliseconds << '/' << detailed.frameP99Milliseconds
                  << '/' << detailed.frameMaximumMilliseconds << "ms"
                  << " fps=" << (frameIntervals.average() > 0.0
                      ? 1000.0 / frameIntervals.average() : 0.0)
                  << " | physics=" << physicsSteps.average() << "ms"
                  << " state=" << (asynchronousGpuMode && gpuMode
                      ? fluidlab::simulation::gpuAsyncCandidateStateName(
                          asyncGpuStepper->state()) : "n/a")
                  << " gpu-step=" << (gpuMode &&
                      gpuReport.timings.asynchronousStepTimingAvailable
                          ? gpuReport.timings.asynchronousStepGpuMilliseconds : 0.0)
                  << "ms"
                  << " | pressure=" << pressureSolves.average() << "ms"
                  << " | particles=" << particleUpdates.average() << "ms"
                  << " | classify=" << classifications.average() << "ms"
                  << " | vis/upload=" << visualizationBuilds.average() << '/'
                  << openGlUploads.average() << "ms"
                  << " | steps/frame=" << (windowFrames != 0
                      ? static_cast<double>(windowPhysicsSteps) / windowFrames : 0.0)
                  << " | drift=" << (now - applicationStartTime - displayedSimulationTime)
                  << "s drop=" << droppedWallTime << "s"
                  << " | iter=" << displayedIterations
                  << " div=" << displayedDivergence
                  << " | residual=" << std::scientific
                  << displayedResidual;
            if (performanceOverlayEnabled) {
                title << " | event/submit/poll/render/swap="
                      << detailed.averageEventMilliseconds << '/'
                      << detailed.averagePhysicsSubmitMilliseconds << '/'
                      << detailed.averageFencePollMilliseconds << '/'
                      << detailed.averageRenderMilliseconds << '/'
                      << detailed.averageSwapMilliseconds << "ms"
                      << " lag=" << detailed.averageCommittedLagSeconds
                      << " rate=" << physicsClock.physicsRate
                      << " backlog=" << physicsClock.physicsBacklog
                      << " batch=" << (asynchronousGpuMode && gpuMode
                          ? asyncGpuStepper->statistics().lastBatchCompleted : 0)
                      << '/' << (asynchronousGpuMode && gpuMode
                          ? asyncGpuStepper->statistics().lastBatchSize : 0)
                      << " worker=" << (asynchronousGpuMode && gpuMode
                          ? asyncGpuStepper->statistics().lastWorkerBusyMilliseconds : 0.0)
                      << "ms publish=" << detailed.averageSnapshotPublishMilliseconds
                      << "ms present=" << detailed.averageSnapshotPresentationMilliseconds
                      << "ms"
                      << " repeat=" << detailed.repeatedSnapshotFrames
                      << " move=" << (moveLoopAdapter.inMoveSize() ? "yes" : "no")
                      << " winV=" << motionState.velocity.x
                      << " rawA=" << motionState.rawAcceleration.x
                      << " filtA=" << motionState.acceleration.x
                      << " queue=" << motionDiagnostics.pendingSegments
                      << '/' << motionDiagnostics.pendingTime
                      << " input=" << lastCommittedInputVersion
                      << " response=" << lastInputResponseDelay;
            }
            if (simulationError) title << " | PAUSED ERROR: " << simulationErrorMessage;
            glfwSetWindowTitle(window, title.str().c_str());
#else
            glfwSetWindowTitle(window, simulationError
                ? "LiquidSimulator Desktop - Simulation paused (R to reset)"
                : "LiquidSimulator Desktop");
#endif
            frameIntervals.clear();
            physicsSteps.clear();
            classifications.clear();
            extrapolations.clear();
            advections.clear();
            pressureSolves.clear();
            divergenceAndProjection.clear();
            particleUpdates.clear();
            visualizationBuilds.clear();
            openGlUploads.clear();
            windowFrames = windowPhysicsSteps = 0;
            titleTime = now;
        }

        const auto renderStart = std::chrono::steady_clock::now();
        glClearColor(0.01f, 0.015f, 0.025f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        displayProgram.use();
        applyDesktopRenderStyle(displayProgram, desktopInteractive && view == 1,
            desktopRenderStyle);
        glBindTextureUnit(0, objects.texture);
        glBindVertexArray(objects.fullscreenVertexArray);
        glDrawArrays(GL_TRIANGLES, 0, 3);

#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
        // Particles are an optional diagnostic layer; the fluid surface remains
        // the default product view and the solid/background pass stays unchanged.
        if (showParticles && gpuMode && activeGpuStepper().grid().particleCount() != 0) {
            gpuParticleProgram.use();
            glUniform2f(gpuParticleProgram.uniformLocation("domainSize"),
                static_cast<float>(width) * cellSize,
                static_cast<float>(height) * cellSize);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0,
                activeGpuStepper().grid().particleBufferHandle());
            glPointSize(2.0f);
            glBindVertexArray(objects.fullscreenVertexArray);
            glDrawArrays(GL_POINTS, 0,
                static_cast<GLsizei>(activeGpuStepper().grid().particleCount()));
        } else if (showParticles && !particleVertices.empty()) {
            particleProgram.use();
            glPointSize(2.0f);
            glBindVertexArray(objects.vectorVertexArray);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particleVertices.size() / 2));
        }
#endif
        const auto beforeSwap = std::chrono::steady_clock::now();
        glfwSwapBuffers(window);
#if defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
        if (firstFrame) {
            if (glGetError() != GL_NO_ERROR) {
                throw std::runtime_error("OpenGL failed to render the first desktop frame.");
            }
            std::cout << "LiquidSimulator Desktop rendered its first frame.\n" << std::flush;
            firstFrame = false;
        }
#endif
        const auto frameEnd = std::chrono::steady_clock::now();
        const double renderMilliseconds = std::chrono::duration<double, std::milli>(
            beforeSwap - renderStart).count();
        const double swapMilliseconds = std::chrono::duration<double, std::milli>(
            frameEnd - beforeSwap).count();
        const std::uint64_t committedVersion = gpuMode
            ? activeGpuStepper().completedOuterSteps()
            : static_cast<std::uint64_t>(report.simulationTime /
                stepper->settings().timeStep + 0.5);
        lastSnapshotPresentationMilliseconds = 0.0;
        if (committedVersion != lastRenderedCommittedVersion &&
            lastSnapshotPublishWallTime > 0.0) {
            lastSnapshotPresentationMilliseconds = std::max(
                0.0, (glfwGetTime() - lastSnapshotPublishWallTime) * 1000.0);
        }
        fluidlab::desktop::DesktopFrameTiming detailedTiming;
        detailedTiming.frameMilliseconds = std::chrono::duration<double, std::milli>(
            frameEnd - completeFrameStart).count();
        detailedTiming.eventMilliseconds = eventMilliseconds;
        detailedTiming.physicsSubmitMilliseconds = physicsSubmitMilliseconds;
        detailedTiming.fencePollMilliseconds = fencePollMilliseconds;
        detailedTiming.renderMilliseconds = renderMilliseconds;
        detailedTiming.swapMilliseconds = swapMilliseconds;
        const double displayedSimulationTime = gpuMode
            ? gpuReport.simulationTime : report.simulationTime;
        detailedTiming.committedLagSeconds = std::max(0.0,
            glfwGetTime() - applicationStartTime - displayedSimulationTime);
        if (desktopInteractive) {
            const auto clock = desktopPhysicsClock.snapshot();
            detailedTiming.physicsRate = clock.physicsRate;
            detailedTiming.physicsBacklogSeconds = clock.physicsBacklog;
        }
        if (asynchronousGpuMode && gpuMode) {
            detailedTiming.snapshotPublishMilliseconds =
                asyncGpuStepper->statistics().lastSnapshotPublishMilliseconds;
        }
        detailedTiming.snapshotPresentationMilliseconds =
            lastSnapshotPresentationMilliseconds;
        detailedTiming.committedOuterSteps = static_cast<std::size_t>(committedVersion);
        detailedTiming.moveSizeFrame = moveLoopAdapter.inMoveSize();
        detailedTiming.repeatedSnapshot = committedVersion == lastRenderedCommittedVersion;
        lastRenderedCommittedVersion = committedVersion;
        desktopMetrics.add(detailedTiming);
    };
    if (desktopInteractive) {
        if (!moveLoopAdapter.install(window, [&]() { frameTick(false); }, 16)) {
            std::cerr << "Windows move-loop adapter could not be installed; modal dragging may pause.\n";
        }
    }
    while (glfwWindowShouldClose(window) == GLFW_FALSE) frameTick(true);
    moveLoopAdapter.uninstall();
    if (desktopInteractive) {
        glfwSetWindowPosCallback(window, nullptr);
        glfwSetWindowUserPointer(window, nullptr);
    }
    return 0;
}

#if !defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
int runDynamicFrameBenchmark(GLFWwindow* window) {
    glfwSwapInterval(0);
    std::cout << "LiquidSimulator Phase 1.4.5 OpenGL frame diagnostic\n"
                 "Frame interval is measured between consecutive frame starts. OpenGL upload is CPU submission time.\n"
                 "Mode           Grid Frames Frame avg       P95       P99       Max   Physics  Vis build  GL upload Steps/frame Internal\n";
    const auto displayProgram = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/fullscreen.vert"),
        findResource("shaders/rendering/texture.frag"));

    for (const std::size_t size : {64U, 128U, 256U}) {
        const float cellSize = 1.0f / static_cast<float>(size);
        fluidlab::simulation::MacGrid grid(size, size, cellSize, cellSize);
        fluidlab::simulation::initializeDamBreak(grid);
        fluidlab::simulation::MarkerParticleSystem particles;
        particles.seedFluidCells(grid);
        fluidlab::simulation::DynamicFreeSurfaceStepper stepper;
        OpenGlObjects objects;
        glCreateTextures(GL_TEXTURE_2D, 1, &objects.texture);
        glCreateVertexArrays(1, &objects.fullscreenVertexArray);
        glTextureStorage2D(objects.texture, 1, GL_RGBA32F,
            static_cast<GLsizei>(size), static_cast<GLsizei>(size));
        glTextureParameteri(objects.texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(objects.texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        std::vector<float> pixels;
        fluidlab::visualization::buildCellTypeTexture(grid, pixels);
        glTextureSubImage2D(objects.texture, 0, 0, 0,
            static_cast<GLsizei>(size), static_cast<GLsizei>(size),
            GL_RGBA, GL_FLOAT, pixels.data());

        const auto draw = [&]() {
            glClear(GL_COLOR_BUFFER_BIT);
            displayProgram.use();
            glBindTextureUnit(0, objects.texture);
            glBindVertexArray(objects.fullscreenVertexArray);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glfwSwapBuffers(window);
        };
        const auto runMode = [&](std::string_view name, std::size_t frames, bool simulate, bool upload) {
            TimingAccumulator intervals, physics, build, transfers;
            std::size_t outerSteps = 0, internalSubsteps = 0;
            auto previousStart = std::chrono::steady_clock::now();
            for (std::size_t frame = 0; frame <= frames; ++frame) {
                const auto frameStart = std::chrono::steady_clock::now();
                if (frame != 0) intervals.add(
                    std::chrono::duration<double, std::milli>(frameStart - previousStart).count());
                previousStart = frameStart;
                if (frame == frames) break;
                if (simulate) {
                    const auto report = stepper.step(grid, particles);
                    physics.add(report.timings.totalMilliseconds);
                    ++outerSteps;
                    internalSubsteps += report.substeps;
                }
                if (upload) {
                    auto start = std::chrono::steady_clock::now();
                    fluidlab::visualization::buildCellTypeTexture(grid, pixels);
                    auto end = std::chrono::steady_clock::now();
                    build.add(std::chrono::duration<double, std::milli>(end - start).count());
                    start = end;
                    glTextureSubImage2D(objects.texture, 0, 0, 0,
                        static_cast<GLsizei>(size), static_cast<GLsizei>(size),
                        GL_RGBA, GL_FLOAT, pixels.data());
                    end = std::chrono::steady_clock::now();
                    transfers.add(std::chrono::duration<double, std::milli>(end - start).count());
                }
                draw();
            }
            std::cout << std::left << std::setw(15) << name << std::right
                      << std::setw(5) << size << std::setw(7) << frames
                      << std::fixed << std::setprecision(3)
                      << std::setw(10) << intervals.average()
                      << std::setw(10) << intervals.percentile(0.95)
                      << std::setw(10) << intervals.percentile(0.99)
                      << std::setw(10) << intervals.maximum
                      << std::setw(10) << physics.average()
                      << std::setw(11) << build.average()
                      << std::setw(11) << transfers.average()
                      << std::setw(12) << static_cast<double>(outerSteps) / frames
                      << std::setw(9) << (outerSteps != 0
                          ? static_cast<double>(internalSubsteps) / outerSteps : 0.0) << '\n';
        };
        runMode("render-only", 80, false, false);
        runMode("upload-only", 40, false, true);
        runMode("full-physics", size == 64 ? 24 : (size == 128 ? 8 : 3), true, true);
    }
    return 0;
}

int runDesktopRenderStyleValidation(bool benchmarkOnly) {
    constexpr int sourceWidth = 64;
    constexpr int sourceHeight = 48;
    const auto program = fluidlab::graphics::ShaderProgram::fromGraphicsFiles(
        findResource("shaders/rendering/fullscreen.vert"),
        findResource("shaders/rendering/texture.frag"));
    GLuint sourceTexture = 0;
    GLuint targetTexture = 0;
    GLuint framebuffer = 0;
    GLuint vertexArray = 0;
    GLuint timer = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &sourceTexture);
    glTextureStorage2D(sourceTexture, 1, GL_RGBA32F, sourceWidth, sourceHeight);
    glTextureParameteri(sourceTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(sourceTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    std::vector<float> source(sourceWidth * sourceHeight * 4, 1.0f);
    for (int y = 0; y < sourceHeight; ++y) {
        for (int x = 0; x < sourceWidth; ++x) {
            const std::size_t offset = static_cast<std::size_t>(
                (y * sourceWidth + x) * 4);
            const bool solid = x == 0 || y == 0 || x == sourceWidth - 1 ||
                y == sourceHeight - 1;
            const bool fluid = !solid && x >= 8 && x < 40 && y >= 8 && y < 32;
            const std::array<float, 3> color = solid
                ? std::array<float, 3>{0.34f, 0.36f, 0.40f}
                : (fluid ? std::array<float, 3>{0.04f, 0.35f, 0.92f}
                    : std::array<float, 3>{0.015f, 0.02f, 0.035f});
            std::copy(color.begin(), color.end(), source.begin() + offset);
        }
    }
    glTextureSubImage2D(sourceTexture, 0, 0, 0, sourceWidth, sourceHeight,
        GL_RGBA, GL_FLOAT, source.data());
    glCreateFramebuffers(1, &framebuffer);
    glCreateVertexArrays(1, &vertexArray);
    glCreateQueries(GL_TIME_ELAPSED, 1, &timer);

    const auto render = [&](int width, int height, const DesktopRenderStyle& style,
                            std::vector<float>* pixels, std::size_t repetitions) {
        if (targetTexture != 0) glDeleteTextures(1, &targetTexture);
        targetTexture = 0;
        glCreateTextures(GL_TEXTURE_2D, 1, &targetTexture);
        glTextureStorage2D(targetTexture, 1, GL_RGBA32F, width, height);
        glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, targetTexture, 0);
        if (glCheckNamedFramebufferStatus(framebuffer, GL_FRAMEBUFFER) !=
            GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Desktop style framebuffer is incomplete.");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glViewport(0, 0, width, height);
        program.use();
        applyDesktopRenderStyle(program, true, style);
        glBindTextureUnit(0, sourceTexture);
        glBindVertexArray(vertexArray);
        glBeginQuery(GL_TIME_ELAPSED, timer);
        const auto begin = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < repetitions; ++index) {
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        glEndQuery(GL_TIME_ELAPSED);
        glFinish();
        const double wallMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count() /
            static_cast<double>(repetitions);
        GLuint64 nanoseconds = 0;
        glGetQueryObjectui64v(timer, GL_QUERY_RESULT, &nanoseconds);
        if (pixels != nullptr) {
            pixels->resize(static_cast<std::size_t>(width * height * 4));
            glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, pixels->data());
        }
        return std::pair{static_cast<double>(nanoseconds) * 1.0e-6 /
            static_cast<double>(repetitions), wallMilliseconds};
    };

    int result = 0;
    if (benchmarkOnly) {
        std::cout << "Desktop render style benchmark (single full-screen draw, 9x50 samples)\n"
                     "Grid        Alpha  Size      GPU P50/P95/P99 ms       wall P50/P95/P99 ms  draws passes\n";
        const auto percentile = [](std::vector<double> samples, double fraction) {
            std::sort(samples.begin(), samples.end());
            const auto index = static_cast<std::size_t>(std::ceil(
                fraction * static_cast<double>(samples.size())) - 1.0);
            return samples[std::min(index, samples.size() - 1)];
        };
        for (const auto [width, height] : std::array{
                 std::pair{48, 32}, std::pair{64, 48},
                 std::pair{64, 64}, std::pair{128, 128}}) {
            for (const auto [grid, alpha] : std::array{
                     std::pair{false, 1.0f}, std::pair{true, 1.0f},
                     std::pair{false, 0.78f}, std::pair{true, 0.78f}}) {
                DesktopRenderStyle style;
                style.gridEnabled = grid;
                style.waterAlpha = alpha;
                std::vector<double> gpuSamples;
                std::vector<double> wallSamples;
                for (int sample = 0; sample < 9; ++sample) {
                    const auto [gpu, wall] = render(width * 8, height * 8,
                        style, nullptr, 50);
                    gpuSamples.push_back(gpu);
                    wallSamples.push_back(wall);
                }
                std::cout << std::setw(4) << (grid ? "on" : "off") << std::setw(13)
                          << alpha << std::setw(5) << width << 'x' << std::setw(3)
                          << height << std::fixed << std::setprecision(5)
                          << std::setw(10) << percentile(gpuSamples, 0.50) << '/'
                          << percentile(gpuSamples, 0.95) << '/'
                          << percentile(gpuSamples, 0.99) << std::setw(10)
                          << percentile(wallSamples, 0.50) << '/'
                          << percentile(wallSamples, 0.95) << '/'
                          << percentile(wallSamples, 0.99)
                          << "      1      1\n";
            }
        }
    } else {
        DesktopRenderStyle opaque;
        opaque.gridEnabled = false;
        opaque.waterAlpha = 1.0f;
        DesktopRenderStyle styled;
        std::vector<float> opaquePixels;
        std::vector<float> styledPixels;
        std::vector<float> transparentPixels;
        std::vector<float> resizedPixels;
        static_cast<void>(render(512, 384, opaque, &opaquePixels, 1));
        static_cast<void>(render(512, 384, styled, &styledPixels, 1));
        DesktopRenderStyle transparent = styled;
        transparent.waterAlpha = 0.0f;
        static_cast<void>(render(512, 384, transparent, &transparentPixels, 1));
        static_cast<void>(render(1024, 768, styled, &resizedPixels, 1));
        const auto pixel = [](const std::vector<float>& image, int x, int y) {
            const std::size_t offset = static_cast<std::size_t>((y * 512 + x) * 4);
            return std::array<float, 3>{image[offset], image[offset + 1], image[offset + 2]};
        };
        const auto distance = [](const auto& a, const auto& b) {
            return std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]),
                std::abs(a[2] - b[2])});
        };
        const auto fluidOpaque = pixel(opaquePixels, 160, 160);
        const auto fluidStyled = pixel(styledPixels, 160, 160);
        const auto fluidTransparent = pixel(transparentPixels, 160, 160);
        const auto airAway = pixel(styledPixels, 420, 160);
        const auto airGrid = pixel(styledPixels, 448, 160);
        const auto solidOpaque = pixel(opaquePixels, 4, 4);
        const auto solidStyled = pixel(styledPixels, 4, 4);
        const auto resizedPixel = [](const std::vector<float>& image, int x, int y) {
            const std::size_t offset = static_cast<std::size_t>((y * 1024 + x) * 4);
            return std::array<float, 3>{image[offset], image[offset + 1], image[offset + 2]};
        };
        const auto resizedAirAway = resizedPixel(resizedPixels, 840, 320);
        const auto resizedAirGrid = resizedPixel(resizedPixels, 896, 320);
        const bool alphaValid = distance(fluidOpaque, fluidStyled) > 0.02f &&
            distance(fluidTransparent, airAway) < 0.08f;
        const bool gridValid = distance(airAway, airGrid) > 0.02f &&
            distance(resizedAirAway, resizedAirGrid) > 0.02f;
        const bool solidValid = distance(solidOpaque, solidStyled) < 1.0e-5f;
        const GLenum error = glGetError();
        result = alphaValid && gridValid && solidValid && error == GL_NO_ERROR ? 0 : 1;
        std::cout << "Desktop render pixel validation: alpha="
                  << (alphaValid ? "PASS" : "FAIL") << ", grid="
                  << (gridValid ? "PASS" : "FAIL") << ", solid="
                  << (solidValid ? "PASS" : "FAIL") << ", GL=0x" << std::hex
                  << error << std::dec << ".\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteQueries(1, &timer);
    glDeleteVertexArrays(1, &vertexArray);
    glDeleteFramebuffers(1, &framebuffer);
    if (targetTexture != 0) glDeleteTextures(1, &targetTexture);
    glDeleteTextures(1, &sourceTexture);
    return result;
}

int runOpenGlApplication(
    ApplicationMode mode,
    std::size_t gridSize = kPhase1GridSize,
    bool warmStartEnabled = true,
    double desktopPixelsPerSimulationUnit = 600.0,
    DesktopRenderStyle desktopRenderStyle = {},
    DesktopFluidInjectionSettings injectionSettings = {}) {
    const char* runModeName = "Interactive Simulation";
    switch (mode) {
    case ApplicationMode::phase1Interactive: runModeName = "Interactive Simulation"; break;
    case ApplicationMode::phase14DamBreak: runModeName = "Interactive Dam Break"; break;
    case ApplicationMode::phase16GpuDynamicTopologyValidation:
        runModeName = "GPU Dynamic Topology Validation"; break;
    case ApplicationMode::phase16GpuDynamicValidation:
        runModeName = "GPU Dynamic Foundation Validation"; break;
    case ApplicationMode::phase16GpuDamBreak:
        runModeName = "Interactive GPU Dam Break"; break;
    case ApplicationMode::phase16GpuDamBreakValidation:
        runModeName = "GPU Dam Break Validation"; break;
    case ApplicationMode::phase17GpuInterfaceValidation:
        runModeName = "GPU Interface Diagnostics Validation"; break;
    case ApplicationMode::phase17AdaptiveValidation:
        runModeName = "GPU Adaptive Step Validation"; break;
    case ApplicationMode::phase181GpuAsyncValidation:
        runModeName = "GPU Asynchronous Candidate Validation"; break;
    case ApplicationMode::phase181GpuAsyncBenchmark:
        runModeName = "GPU Asynchronous Candidate Benchmark"; break;
    case ApplicationMode::phase182GpuMixedPressureValidation:
        runModeName = "GPU Mixed Precision Pressure Validation"; break;
    case ApplicationMode::phase182GpuSolverBenchmark:
        runModeName = "GPU Solver Benchmark"; break;
    case ApplicationMode::phase183GpuMgpcgProfile:
        runModeName = "GPU MGPCG Offline Profile"; break;
    case ApplicationMode::phase183GpuMixedReliabilityValidation:
        runModeName = "GPU Mixed Preconditioner Reliability Validation"; break;
    case ApplicationMode::phase183GpuMixedLongValidation:
        runModeName = "GPU Mixed Precision Long Validation"; break;
    case ApplicationMode::phase19GpuCoarseFusionRegression:
        runModeName = "GPU Coarse Fusion Regression"; break;
    case ApplicationMode::phase19GpuInterfaceAccuracyValidation:
        runModeName = "GPU Interface Accuracy Validation"; break;
    case ApplicationMode::phase19bInterfaceReconstructionValidation:
        runModeName = "Interface Reconstruction Validation"; break;
    case ApplicationMode::phase19bGhostFluidCpuValidation:
        runModeName = "CPU Ghost Fluid Validation"; break;
    case ApplicationMode::phase19b2LevelSetDynamicValidation:
        runModeName = "Dynamic Level Set Validation"; break;
    case ApplicationMode::phase19b2LevelSetRedistanceBenchmark:
        runModeName = "Level Set Redistance Benchmark"; break;
    case ApplicationMode::phase19b2LevelSetParticleConsistencyValidation:
        runModeName = "Level Set Particle Consistency Validation"; break;
    case ApplicationMode::phase19b2GhostFluidDynamicCpuValidation:
        runModeName = "Dynamic CPU Ghost Fluid Validation"; break;
    case ApplicationMode::phase19c1GpuLevelSetValidation:
        runModeName = "GPU Level Set Validation"; break;
    case ApplicationMode::phase19c1GpuLevelSetDynamicValidation:
        runModeName = "GPU Dynamic Level Set Validation"; break;
    case ApplicationMode::phase19c1GpuLevelSetRollbackValidation:
        runModeName = "GPU Level Set Rollback Validation"; break;
    case ApplicationMode::phase19c1GpuLevelSetBenchmark:
        runModeName = "GPU Level Set Benchmark"; break;
    case ApplicationMode::phase19c1bGpuLevelSetAreaAudit:
        runModeName = "GPU Level Set Area Audit"; break;
    case ApplicationMode::phase19c1bGpuLevelSetAdvectionCompare:
        runModeName = "GPU Level Set Advection Comparison"; break;
    case ApplicationMode::phase19c1bGpuLevelSetLongValidation:
        runModeName = "GPU Level Set Long Validation"; break;
    case ApplicationMode::phase19c1cGpuLevelSetRedistanceValidation:
        runModeName = "GPU Level Set Interface-Preserving Redistance Validation"; break;
    case ApplicationMode::phase19c1cGpuLevelSetExtensionValidation:
        runModeName = "GPU Level Set Velocity Extension Validation"; break;
    case ApplicationMode::phase19c1cGpuLevelSetConservationCompare:
        runModeName = "GPU Level Set Conservation Comparison"; break;
    case ApplicationMode::phase19dVofMacReplayValidation:
        runModeName = "VOF Accepted-Substep MAC Replay Validation"; break;
    case ApplicationMode::phase19eGpuVofFluxValidation: runModeName = "GPU VOF Flux Validation"; break;
    case ApplicationMode::phase19eGpuVofPlicValidation: runModeName = "GPU VOF PLIC Validation"; break;
    case ApplicationMode::phase19eGpuVofSplitValidation: runModeName = "GPU VOF Split Validation"; break;
    case ApplicationMode::phase19eGpuVofRollbackValidation: runModeName = "GPU VOF Rollback Validation"; break;
    case ApplicationMode::phase19eGpuVofBenchmark: runModeName = "GPU VOF Benchmark"; break;
    case ApplicationMode::phase19eGpuVofMacDiagnostic: runModeName = "GPU VOF MAC Diagnostic"; break;
    case ApplicationMode::phase19eBGpuVofPrecisionAudit: runModeName = "GPU VOF Precision Audit"; break;
    case ApplicationMode::phase19eBGpuVofMultistepValidation: runModeName = "GPU VOF Multistep Validation"; break;
    case ApplicationMode::phase19eBGpuVofBoundedFluxValidation: runModeName = "GPU VOF Bounded Flux Validation"; break;
    case ApplicationMode::phase19eBGpuVofMacReplay: runModeName = "GPU VOF MAC Replay"; break;
    case ApplicationMode::phase19eBGpuVofVelocityCompatibility: runModeName = "GPU VOF Velocity Compatibility"; break;
    case ApplicationMode::phase19eBGpuVofMultistepBenchmark: runModeName = "GPU VOF Multistep Benchmark"; break;
    case ApplicationMode::phase19eCGpuVofRoundoffAudit: runModeName = "GPU VOF Roundoff Audit"; break;
    case ApplicationMode::phase19eCGpuVofCommitSafetyValidation: runModeName = "GPU VOF Commit Safety Validation"; break;
    case ApplicationMode::phase19eCGpuVofConservativeRepairValidation: runModeName = "GPU VOF Conservative Repair Validation"; break;
    case ApplicationMode::phase19eCGpuVofBandVelocityValidation: runModeName = "GPU VOF Band Velocity Validation"; break;
    case ApplicationMode::phase19eCGpuVofLongValidation: runModeName = "GPU VOF Long Validation"; break;
    case ApplicationMode::phase19eCGpuVofDynamicCompare: runModeName = "GPU VOF Dynamic Compare"; break;
    case ApplicationMode::phase19eCGpuVofSafetyBenchmark: runModeName = "GPU VOF Safety Benchmark"; break;
    case ApplicationMode::phase20DesktopInteractive: runModeName = "Desktop Interactive Container"; break;
    case ApplicationMode::phase20DesktopInteractiveDamBreak: runModeName = "Desktop Interactive Dam Break"; break;
    case ApplicationMode::phase20ExternalForceValidation: runModeName = "External Force Validation"; break;
    case ApplicationMode::phase20ContainerMotionValidation: runModeName = "Container Motion Validation"; break;
    case ApplicationMode::phase20ContainerInertiaValidation: runModeName = "Container Inertia Validation"; break;
    case ApplicationMode::phase20DesktopInputTimingValidation: runModeName = "Desktop Input Timing Validation"; break;
    case ApplicationMode::phase20DesktopInteractionBenchmark: runModeName = "Desktop Interaction Benchmark"; break;
    case ApplicationMode::phase20DesktopFramePacingValidation: runModeName = "Desktop Frame Pacing Validation"; break;
    case ApplicationMode::phase20DesktopModalLoopValidation: runModeName = "Desktop Modal Loop Validation"; break;
    case ApplicationMode::phase20DesktopSettleGenerate: runModeName = "Desktop Settled State Generation"; break;
    case ApplicationMode::phase20DesktopSettleLoadValidation: runModeName = "Desktop Settled State Load Validation"; break;
    case ApplicationMode::phase20DesktopInertiaFromSettledValidation: runModeName = "Desktop Settled Inertia Validation"; break;
    case ApplicationMode::phase20DesktopInertiaPipelineAudit: runModeName = "Desktop Inertia Pipeline Audit"; break;
    case ApplicationMode::phase20DesktopInertiaTimingValidation: runModeName = "Desktop Inertia Timing Validation"; break;
    case ApplicationMode::phase20DesktopInertiaResponseValidation: runModeName = "Desktop Inertia Response Validation"; break;
    case ApplicationMode::phase20DesktopInertiaLatencyBenchmark: runModeName = "Desktop Inertia Latency Benchmark"; break;
    case ApplicationMode::phase20DesktopGridRenderValidation: runModeName = "Desktop Grid Render Validation"; break;
    case ApplicationMode::phase20DesktopWaterAlphaValidation: runModeName = "Desktop Water Alpha Validation"; break;
    case ApplicationMode::phase20DesktopRenderStyleBenchmark: runModeName = "Desktop Render Style Benchmark"; break;
    case ApplicationMode::phase20DesktopSchedulerValidation: runModeName = "Desktop Scheduler Validation"; break;
    case ApplicationMode::phase20DesktopAsyncThroughputValidation: runModeName = "Desktop Async Throughput Validation"; break;
    case ApplicationMode::phase20DesktopBatchInputAudit: runModeName = "Desktop Batch Input Audit"; break;
    case ApplicationMode::phase20DesktopThroughputAccountingValidation: runModeName = "Desktop Throughput Accounting Validation"; break;
    case ApplicationMode::phase20DesktopSustainedRealtimeBenchmark: runModeName = "Desktop Sustained Realtime Benchmark"; break;
    case ApplicationMode::phase20GpuFluidInjectionValidation: runModeName = "GPU Fluid Injection Validation"; break;
    default: runModeName = "OpenGL Validation or Benchmark"; break;
    }
    std::cout << "Run Mode: " << runModeName << '\n';
#else
int runOpenGlApplication() {
#endif
    glfwSetErrorCallback(glfwErrorCallback);
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("GLFW initialization failed.");
    }

    struct GlfwLifetime final {
        ~GlfwLifetime() { glfwTerminate(); }
    } glfwLifetime;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
#if defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
    constexpr bool hiddenWindow = false;
    const char* windowTitle = "LiquidSimulator Desktop";
#else
    const bool hiddenWindow = mode != ApplicationMode::phase1Interactive &&
        mode != ApplicationMode::phase14DamBreak &&
        mode != ApplicationMode::phase16GpuDamBreak &&
        mode != ApplicationMode::phase20DesktopInteractive &&
        mode != ApplicationMode::phase20DesktopInteractiveDamBreak;
    const char* windowTitle = "LiquidSimulator - Phase 1.9C-1C";
    if (mode == ApplicationMode::phase0Validation) {
        windowTitle = "LiquidSimulator - Phase 0 Validation";
    } else if (mode == ApplicationMode::phase20DesktopInteractive ||
        mode == ApplicationMode::phase20DesktopInteractiveDamBreak) {
        windowTitle = "LiquidSimulator Desktop - Interactive Container";
    } else if (mode == ApplicationMode::phase14DamBreak ||
        mode == ApplicationMode::phase16GpuDamBreak) {
        windowTitle = "LiquidSimulator - Phase 1.9C-1C Dam Break";
    }
#endif
    glfwWindowHint(GLFW_VISIBLE, hiddenWindow ? GLFW_FALSE : GLFW_TRUE);
#ifdef _DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

    WindowPtr window(glfwCreateWindow(
        kWindowWidth,
        kWindowHeight,
        windowTitle,
        nullptr,
        nullptr));
    if (!window) {
        throw std::runtime_error("Failed to create an OpenGL 4.6 GLFW window.");
    }

    if (!hiddenWindow) {
        // Keep the client area aligned with the 64x48 simulation domain so
        // interactive resizing cannot stretch the rendered fluid.
        glfwSetWindowAspectRatio(window.get(), kWindowWidth, kWindowHeight);
    }

    glfwMakeContextCurrent(window.get());
    glfwSwapInterval(1);

    const int loadedVersion = gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress));
    if (loadedVersion == 0) {
        throw std::runtime_error("GLAD failed to load OpenGL functions.");
    }
    if (!isAtLeastOpenGl46() || GLAD_GL_VERSION_4_6 == 0) {
        throw std::runtime_error("OpenGL 4.6 is required but was not provided by the active context.");
    }

    printOpenGlInformation();
    std::cout << "GLAD loaded OpenGL " << GLAD_VERSION_MAJOR(loadedVersion)
              << '.' << GLAD_VERSION_MINOR(loadedVersion) << '\n';

    GLint contextFlags = 0;
    glGetIntegerv(GL_CONTEXT_FLAGS, &contextFlags);
    if ((contextFlags & GL_CONTEXT_FLAG_DEBUG_BIT) != 0) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(openGlDebugCallback, nullptr);
        std::cout << "OpenGL debug output enabled.\n";
    }

    glfwSetFramebufferSizeCallback(window.get(), framebufferSizeCallback);
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window.get(), &framebufferWidth, &framebufferHeight);
    framebufferSizeCallback(window.get(), framebufferWidth, framebufferHeight);

#if defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
    return runDamBreakVisualization(window.get(), false, true, true, true);
#else
    if (mode == ApplicationMode::phase0Validation) {
        return runPhase0Validation(framebufferWidth, framebufferHeight);
    }
    if (mode == ApplicationMode::phase145GpuDivergenceValidation) {
        return fluidlab::diagnostics::runGpuDivergenceValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp"));
    }
    if (mode == ApplicationMode::phase145FrameBenchmark) {
        return runDynamicFrameBenchmark(window.get());
    }
    if (mode == ApplicationMode::phase15GpuMacValidation) {
        return fluidlab::diagnostics::runGpuMacValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase155GpuPressureOptimization) {
        return fluidlab::diagnostics::runGpuPressureOptimizationValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase156GpuMultigridValidation) {
        return fluidlab::diagnostics::runGpuMultigridValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase16GpuDynamicValidation) {
        return fluidlab::diagnostics::runGpuDynamicValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase16GpuDynamicTopologyValidation) {
        return fluidlab::diagnostics::runGpuDynamicTopologyValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase17GpuInterfaceValidation) {
        runGpuDamBreakValidationSuite(
            findResource("shaders/compute/mac_divergence.comp"));
        return 0;
    }
    if (mode == ApplicationMode::phase17AdaptiveValidation) {
        runPhase17cAdaptiveValidation(
            findResource("shaders/compute/mac_divergence.comp"));
        return 0;
    }
    if (mode == ApplicationMode::phase18GpuResidentPressureValidation) {
        return fluidlab::diagnostics::runGpuResidentPressureValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase181GpuAsyncValidation) {
        return runGpuAsyncCandidateValidation(window.get());
    }
    if (mode == ApplicationMode::phase181GpuAsyncBenchmark) {
        return runGpuAsyncCandidateBenchmark(window.get());
    }
    if (mode == ApplicationMode::phase182GpuMixedPressureValidation) {
        return fluidlab::diagnostics::runGpuMixedPressureValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase182GpuSolverBenchmark) {
        return fluidlab::diagnostics::runGpuSolverBenchmark(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase183GpuMgpcgProfile) {
        return fluidlab::diagnostics::runGpuMgpcgProfile(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase183GpuMixedReliabilityValidation) {
        return fluidlab::diagnostics::runGpuMixedReliabilityValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase183GpuMixedLongValidation) {
        return fluidlab::diagnostics::runGpuMixedLongValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19GpuCoarseFusionRegression) {
        return fluidlab::diagnostics::runGpuCoarseFusionRegression(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19GpuInterfaceAccuracyValidation) {
        return fluidlab::diagnostics::runGpuInterfaceAccuracyValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19bInterfaceReconstructionValidation) {
        return fluidlab::diagnostics::runInterfaceReconstructionValidation(std::cout);
    }
    if (mode == ApplicationMode::phase19bGhostFluidCpuValidation) {
        return fluidlab::diagnostics::runGhostFluidCpuValidation(std::cout);
    }
    if (mode == ApplicationMode::phase19b2LevelSetDynamicValidation) {
        return fluidlab::diagnostics::runLevelSetDynamicValidation(std::cout);
    }
    if (mode == ApplicationMode::phase19b2LevelSetRedistanceBenchmark) {
        return fluidlab::diagnostics::runLevelSetRedistanceBenchmark(std::cout);
    }
    if (mode == ApplicationMode::phase19b2LevelSetParticleConsistencyValidation) {
        return fluidlab::diagnostics::runLevelSetParticleConsistencyValidation(std::cout);
    }
    if (mode == ApplicationMode::phase19b2GhostFluidDynamicCpuValidation) {
        return fluidlab::diagnostics::runGhostFluidDynamicCpuValidation(std::cout);
    }
    if (mode == ApplicationMode::phase19c1GpuLevelSetValidation) {
        return fluidlab::diagnostics::runGpuLevelSetValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19c1GpuLevelSetDynamicValidation) {
        return fluidlab::diagnostics::runGpuLevelSetDynamicValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19c1GpuLevelSetRollbackValidation) {
        return fluidlab::diagnostics::runGpuLevelSetRollbackValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path(),
            window.get());
    }
    if (mode == ApplicationMode::phase19c1GpuLevelSetBenchmark) {
        return fluidlab::diagnostics::runGpuLevelSetBenchmark(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19c1bGpuLevelSetAreaAudit) {
        return fluidlab::diagnostics::runGpuLevelSetAreaAudit(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19c1bGpuLevelSetAdvectionCompare) {
        return fluidlab::diagnostics::runGpuLevelSetAdvectionCompare(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19c1bGpuLevelSetLongValidation) {
        return fluidlab::diagnostics::runGpuLevelSetLongValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19c1cGpuLevelSetRedistanceValidation) {
        return fluidlab::diagnostics::runGpuLevelSetRedistanceValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19c1cGpuLevelSetExtensionValidation) {
        return fluidlab::diagnostics::runGpuLevelSetExtensionValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19c1cGpuLevelSetConservationCompare) {
        return fluidlab::diagnostics::runGpuLevelSetConservationCompare(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    if (mode == ApplicationMode::phase19dVofMacReplayValidation) {
        return fluidlab::diagnostics::runVofMacReplayValidation(
            std::cout, findResource("shaders/compute/mac_divergence.comp").parent_path());
    }
    const auto shaderDirectory = findResource("shaders/compute/mac_divergence.comp").parent_path();
    if (mode == ApplicationMode::phase19eGpuVofFluxValidation)
        return fluidlab::diagnostics::runGpuVofFluxValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eGpuVofPlicValidation)
        return fluidlab::diagnostics::runGpuVofPlicValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eGpuVofSplitValidation)
        return fluidlab::diagnostics::runGpuVofSplitValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eGpuVofRollbackValidation)
        return fluidlab::diagnostics::runGpuVofRollbackValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eGpuVofBenchmark)
        return fluidlab::diagnostics::runGpuVofBenchmark(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eGpuVofMacDiagnostic)
        return fluidlab::diagnostics::runGpuVofMacDiagnostic(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eBGpuVofPrecisionAudit)
        return fluidlab::diagnostics::runGpuVofPrecisionAudit(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eBGpuVofMultistepValidation)
        return fluidlab::diagnostics::runGpuVofMultistepValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eBGpuVofBoundedFluxValidation)
        return fluidlab::diagnostics::runGpuVofBoundedFluxValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eBGpuVofMacReplay)
        return fluidlab::diagnostics::runGpuVofMacReplay(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eBGpuVofVelocityCompatibility)
        return fluidlab::diagnostics::runGpuVofVelocityCompatibility(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eBGpuVofMultistepBenchmark)
        return fluidlab::diagnostics::runGpuVofMultistepBenchmark(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eCGpuVofRoundoffAudit)
        return fluidlab::diagnostics::runGpuVofRoundoffAudit(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eCGpuVofCommitSafetyValidation)
        return fluidlab::diagnostics::runGpuVofCommitSafetyValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eCGpuVofConservativeRepairValidation)
        return fluidlab::diagnostics::runGpuVofConservativeRepairValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eCGpuVofBandVelocityValidation)
        return fluidlab::diagnostics::runGpuVofBandVelocityValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eCGpuVofLongValidation)
        return fluidlab::diagnostics::runGpuVofLongValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eCGpuVofDynamicCompare)
        return fluidlab::diagnostics::runGpuVofDynamicCompare(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase19eCGpuVofSafetyBenchmark)
        return fluidlab::diagnostics::runGpuVofSafetyBenchmark(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase20ExternalForceValidation)
        return fluidlab::diagnostics::runExternalForceValidation(
            std::cout, shaderDirectory, window.get());
    if (mode == ApplicationMode::phase20ContainerMotionValidation)
        return fluidlab::diagnostics::runContainerMotionValidation(std::cout);
    if (mode == ApplicationMode::phase20ContainerInertiaValidation)
        return fluidlab::diagnostics::runContainerInertiaValidation(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase20DesktopInputTimingValidation)
        return fluidlab::diagnostics::runDesktopInputTimingValidation(std::cout);
    if (mode == ApplicationMode::phase20DesktopInteractionBenchmark)
        return fluidlab::diagnostics::runDesktopInteractionBenchmark(std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase20DesktopFramePacingValidation)
        return fluidlab::diagnostics::runDesktopFramePacingValidation(std::cout);
    if (mode == ApplicationMode::phase20DesktopModalLoopValidation)
        return fluidlab::diagnostics::runDesktopModalLoopValidation(std::cout, window.get());
    const auto settledStatePath = findResource("README.md").parent_path() /
        "assets/desktop/settled_64x48.lsstate";
    if (mode == ApplicationMode::phase20DesktopSettleGenerate)
        return fluidlab::diagnostics::runDesktopSettleGenerate(
            std::cout, shaderDirectory, settledStatePath);
    if (mode == ApplicationMode::phase20DesktopSettleLoadValidation)
        return fluidlab::diagnostics::runDesktopSettleLoadValidation(
            std::cout, shaderDirectory, settledStatePath);
    if (mode == ApplicationMode::phase20DesktopInertiaFromSettledValidation)
        return fluidlab::diagnostics::runDesktopInertiaFromSettledValidation(
            std::cout, shaderDirectory, settledStatePath);
    if (mode == ApplicationMode::phase20DesktopInertiaPipelineAudit)
        return fluidlab::diagnostics::runDesktopInertiaPipelineAudit(std::cout);
    if (mode == ApplicationMode::phase20DesktopInertiaTimingValidation)
        return fluidlab::diagnostics::runDesktopInertiaTimingValidation(std::cout);
    if (mode == ApplicationMode::phase20DesktopInertiaResponseValidation)
        return fluidlab::diagnostics::runDesktopInertiaResponseValidation(
            std::cout, shaderDirectory, settledStatePath);
    if (mode == ApplicationMode::phase20DesktopInertiaLatencyBenchmark)
        return fluidlab::diagnostics::runDesktopInertiaLatencyBenchmark(
            std::cout, shaderDirectory, settledStatePath);
    if (mode == ApplicationMode::phase20DesktopGridRenderValidation ||
        mode == ApplicationMode::phase20DesktopWaterAlphaValidation)
        return runDesktopRenderStyleValidation(false);
    if (mode == ApplicationMode::phase20DesktopRenderStyleBenchmark)
        return runDesktopRenderStyleValidation(true);
    if (mode == ApplicationMode::phase20DesktopSchedulerValidation)
        return fluidlab::diagnostics::runDesktopSchedulerValidation(std::cout);
    if (mode == ApplicationMode::phase20DesktopAsyncThroughputValidation)
        return fluidlab::diagnostics::runDesktopAsyncThroughputValidation(
            std::cout, shaderDirectory, window.get());
    if (mode == ApplicationMode::phase20DesktopBatchInputAudit)
        return fluidlab::diagnostics::runDesktopBatchInputAudit(std::cout);
    if (mode == ApplicationMode::phase20DesktopThroughputAccountingValidation)
        return fluidlab::diagnostics::runDesktopThroughputAccountingValidation(
            std::cout, shaderDirectory,
            findResource("assets/desktop/settled_64x48.lsstate"), window.get());
    if (mode == ApplicationMode::phase20DesktopSustainedRealtimeBenchmark)
        return fluidlab::diagnostics::runDesktopSustainedRealtimeBenchmark(
            std::cout, shaderDirectory,
            findResource("assets/desktop/settled_64x48.lsstate"), window.get());
    if (mode == ApplicationMode::phase20GpuFluidInjectionValidation)
        return fluidlab::diagnostics::runGpuFluidInjectionValidation(
            std::cout, shaderDirectory);
    if (mode == ApplicationMode::phase14DamBreak ||
        mode == ApplicationMode::phase14DamBreakValidation ||
        mode == ApplicationMode::phase16GpuDamBreak ||
        mode == ApplicationMode::phase16GpuDamBreakValidation) {
        return runDamBreakVisualization(
            window.get(), mode == ApplicationMode::phase14DamBreakValidation ||
                mode == ApplicationMode::phase16GpuDamBreakValidation,
            mode == ApplicationMode::phase16GpuDamBreak ||
                mode == ApplicationMode::phase16GpuDamBreakValidation);
    }
    if (mode == ApplicationMode::phase20DesktopInteractive) {
        return runDamBreakVisualization(window.get(), false, true, true, true,
            desktopPixelsPerSimulationUnit, desktopRenderStyle, injectionSettings);
    }
    if (mode == ApplicationMode::phase20DesktopInteractiveDamBreak) {
        return runDamBreakVisualization(window.get(), false, true, true, false,
            desktopPixelsPerSimulationUnit, desktopRenderStyle, injectionSettings);
    }
    return runPhase1Visualization(window.get(), mode, gridSize, warmStartEnabled);
#endif
}

} // namespace

#if defined(LIQUIDSIMULATOR_DESKTOP_RELEASE)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR commandLine, int) {
    // The distributed executable has a single entry point and accepts no mode switches.
    if (*commandLine != '\0') return 2;
    try {
        return runOpenGlApplication();
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        MessageBoxA(nullptr, error.what(), "LiquidSimulator Desktop", MB_ICONERROR | MB_OK);
        return 1;
    }
}
#else
int main(int argumentCount, char* arguments[]) {
    try {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--test") {
            return fluidlab::tests::runPhase1Tests(std::cout);
        }
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--benchmark") {
            return fluidlab::diagnostics::runCpuBenchmark(std::cout);
        }
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--dynamic-benchmark") {
            return fluidlab::diagnostics::runDynamicFreeSurfaceBenchmark(std::cout);
        }
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--vof-geometry-validate") {
            return fluidlab::diagnostics::runVofGeometryValidation(std::cout);
        }
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--vof-conservation-validate") {
            return fluidlab::diagnostics::runVofConservationValidation(std::cout);
        }
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--vof-advection-validate") {
            return fluidlab::diagnostics::runVofAdvectionValidation(std::cout);
        }
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--clsvof-coupling-validate") {
            return fluidlab::diagnostics::runClsvofCouplingValidation(std::cout);
        }
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--clsvof-benchmark") {
            return fluidlab::diagnostics::runClsvofBenchmark(std::cout);
        }
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--vof-flux-audit") {
            return fluidlab::diagnostics::runVofFluxAudit(std::cout);
        }
        if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--vof-shape-convergence-validate") {
            return fluidlab::diagnostics::runVofShapeConvergenceValidation(std::cout);
        }
        if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--vof-height-function-validate") {
            return fluidlab::diagnostics::runVofHeightFunctionValidation(std::cout);
        }
        if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--clsvof-geometry-coupling-validate") {
            return fluidlab::diagnostics::runClsvofGeometryCouplingValidation(std::cout);
        }
        if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--plic-ghost-fluid-validate") {
            return fluidlab::diagnostics::runPlicGhostFluidValidation(std::cout);
        }
        if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--vof-split-conservation-validate") {
            return fluidlab::diagnostics::runVofSplitConservationValidation(std::cout);
        }
        if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--vof-split-boundedness-validate") {
            return fluidlab::diagnostics::runVofSplitBoundednessValidation(std::cout);
        }
        if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--vof-split-shape-validate") {
            return fluidlab::diagnostics::runVofSplitShapeValidation(std::cout);
        }
        if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--vof-split-benchmark") {
            return fluidlab::diagnostics::runVofSplitBenchmark(std::cout);
        }
        if (argumentCount == 4 &&
            std::string_view(arguments[1]) == "--visual-benchmark") {
            const std::size_t gridSize = std::stoull(arguments[2]);
            if (gridSize != 64 && gridSize != 128 && gridSize != 256) {
                throw std::invalid_argument("Visual benchmark grid must be 64, 128, or 256.");
            }
            const std::string_view warmStartMode(arguments[3]);
            if (warmStartMode != "cold" && warmStartMode != "warm") {
                throw std::invalid_argument("Visual benchmark mode must be cold or warm.");
            }
            return runOpenGlApplication(
                ApplicationMode::phase1VisualBenchmark,
                gridSize,
                warmStartMode == "warm");
        }

        ApplicationMode mode = ApplicationMode::phase20DesktopInteractive;
        double desktopPixelsPerSimulationUnit = 600.0;
        DesktopRenderStyle desktopRenderStyle;
        DesktopFluidInjectionSettings injectionSettings;
        if (argumentCount == 2 && std::string_view(arguments[1]) == "--validate") {
            mode = ApplicationMode::phase0Validation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--visual-validate") {
            mode = ApplicationMode::phase1VisualValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--visual-benchmark") {
            mode = ApplicationMode::phase1VisualBenchmark;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--dam-break") {
            mode = ApplicationMode::phase14DamBreak;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--dam-break-validate") {
            mode = ApplicationMode::phase14DamBreakValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-divergence-validate") {
            mode = ApplicationMode::phase145GpuDivergenceValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--dynamic-frame-benchmark") {
            mode = ApplicationMode::phase145FrameBenchmark;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-mac-validate") {
            mode = ApplicationMode::phase15GpuMacValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-pressure-optimize") {
            mode = ApplicationMode::phase155GpuPressureOptimization;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-mgpcg-validate") {
            mode = ApplicationMode::phase156GpuMultigridValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-dynamic-validate") {
            mode = ApplicationMode::phase16GpuDynamicValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-dynamic-topology-validate") {
            mode = ApplicationMode::phase16GpuDynamicTopologyValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-dam-break") {
            mode = ApplicationMode::phase16GpuDamBreak;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-dam-break-validate") {
            mode = ApplicationMode::phase16GpuDamBreakValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-interface-validate") {
            mode = ApplicationMode::phase17GpuInterfaceValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-adaptive-validate") {
            mode = ApplicationMode::phase17AdaptiveValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-pressure-resident-validate") {
            mode = ApplicationMode::phase18GpuResidentPressureValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-async-validate") {
            mode = ApplicationMode::phase181GpuAsyncValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-async-benchmark") {
            mode = ApplicationMode::phase181GpuAsyncBenchmark;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-mixed-pressure-validate") {
            mode = ApplicationMode::phase182GpuMixedPressureValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-solver-benchmark") {
            mode = ApplicationMode::phase182GpuSolverBenchmark;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-mgpcg-profile") {
            mode = ApplicationMode::phase183GpuMgpcgProfile;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-mixed-reliability-validate") {
            mode = ApplicationMode::phase183GpuMixedReliabilityValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-mixed-long-validate") {
            mode = ApplicationMode::phase183GpuMixedLongValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-coarse-fusion-regression") {
            mode = ApplicationMode::phase19GpuCoarseFusionRegression;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-interface-accuracy-validate") {
            mode = ApplicationMode::phase19GpuInterfaceAccuracyValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--interface-reconstruction-validate") {
            mode = ApplicationMode::phase19bInterfaceReconstructionValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--ghost-fluid-cpu-validate") {
            mode = ApplicationMode::phase19bGhostFluidCpuValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--levelset-dynamic-validate") {
            mode = ApplicationMode::phase19b2LevelSetDynamicValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--levelset-redistance-benchmark") {
            mode = ApplicationMode::phase19b2LevelSetRedistanceBenchmark;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--levelset-particle-consistency-validate") {
            mode = ApplicationMode::phase19b2LevelSetParticleConsistencyValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--ghost-fluid-dynamic-cpu-validate") {
            mode = ApplicationMode::phase19b2GhostFluidDynamicCpuValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-validate") {
            mode = ApplicationMode::phase19c1GpuLevelSetValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-dynamic-validate") {
            mode = ApplicationMode::phase19c1GpuLevelSetDynamicValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-rollback-validate") {
            mode = ApplicationMode::phase19c1GpuLevelSetRollbackValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-benchmark") {
            mode = ApplicationMode::phase19c1GpuLevelSetBenchmark;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-area-audit") {
            mode = ApplicationMode::phase19c1bGpuLevelSetAreaAudit;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-advection-compare") {
            mode = ApplicationMode::phase19c1bGpuLevelSetAdvectionCompare;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-long-validate") {
            mode = ApplicationMode::phase19c1bGpuLevelSetLongValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-redistance-validate") {
            mode = ApplicationMode::phase19c1cGpuLevelSetRedistanceValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-extension-validate") {
            mode = ApplicationMode::phase19c1cGpuLevelSetExtensionValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--gpu-levelset-conservation-compare") {
            mode = ApplicationMode::phase19c1cGpuLevelSetConservationCompare;
        } else if (argumentCount == 2 &&
            (std::string_view(arguments[1]) == "--vof-mac-replay-validate" ||
             std::string_view(arguments[1]) == "--vof-split-mac-replay")) {
            mode = ApplicationMode::phase19dVofMacReplayValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-flux-validate") {
            mode = ApplicationMode::phase19eGpuVofFluxValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-plic-validate") {
            mode = ApplicationMode::phase19eGpuVofPlicValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-split-validate") {
            mode = ApplicationMode::phase19eGpuVofSplitValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-rollback-validate") {
            mode = ApplicationMode::phase19eGpuVofRollbackValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-benchmark") {
            mode = ApplicationMode::phase19eGpuVofBenchmark;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-mac-diagnostic") {
            mode = ApplicationMode::phase19eGpuVofMacDiagnostic;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-precision-audit") {
            mode = ApplicationMode::phase19eBGpuVofPrecisionAudit;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-multistep-validate") {
            mode = ApplicationMode::phase19eBGpuVofMultistepValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-bounded-flux-validate") {
            mode = ApplicationMode::phase19eBGpuVofBoundedFluxValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-mac-replay") {
            mode = ApplicationMode::phase19eBGpuVofMacReplay;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-velocity-compatibility") {
            mode = ApplicationMode::phase19eBGpuVofVelocityCompatibility;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-multistep-benchmark") {
            mode = ApplicationMode::phase19eBGpuVofMultistepBenchmark;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-roundoff-audit") {
            mode = ApplicationMode::phase19eCGpuVofRoundoffAudit;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-commit-safety-validate") {
            mode = ApplicationMode::phase19eCGpuVofCommitSafetyValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-conservative-repair-validate") {
            mode = ApplicationMode::phase19eCGpuVofConservativeRepairValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-band-velocity-validate") {
            mode = ApplicationMode::phase19eCGpuVofBandVelocityValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-long-validate") {
            mode = ApplicationMode::phase19eCGpuVofLongValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-dynamic-compare") {
            mode = ApplicationMode::phase19eCGpuVofDynamicCompare;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--gpu-vof-safety-benchmark") {
            mode = ApplicationMode::phase19eCGpuVofSafetyBenchmark;
        } else if (argumentCount >= 2 &&
            (std::string_view(arguments[1]) == "--desktop-interactive" ||
             std::string_view(arguments[1]) == "--desktop-interactive-dam-break")) {
            mode = std::string_view(arguments[1]) == "--desktop-interactive"
                ? ApplicationMode::phase20DesktopInteractive
                : ApplicationMode::phase20DesktopInteractiveDamBreak;
            for (int index = 2; index < argumentCount; ++index) {
                const std::string_view option(arguments[index]);
                if (option.starts_with("--desktop-motion-scale=")) {
                    desktopPixelsPerSimulationUnit = std::stod(
                        std::string(option.substr(23)));
                    if (!std::isfinite(desktopPixelsPerSimulationUnit) ||
                        desktopPixelsPerSimulationUnit < 120.0 ||
                        desktopPixelsPerSimulationUnit > 2400.0) {
                        throw std::invalid_argument(
                            "Desktop motion scale must be between 120 and 2400 pixels per simulation unit.");
                    }
                } else if (option == "--desktop-grid=on") {
                    desktopRenderStyle.gridEnabled = true;
                } else if (option == "--desktop-grid=off") {
                    desktopRenderStyle.gridEnabled = false;
                } else if (option.starts_with("--desktop-water-alpha=")) {
                    desktopRenderStyle.waterAlpha = std::stof(
                        std::string(option.substr(22)));
                    if (!std::isfinite(desktopRenderStyle.waterAlpha) ||
                        desktopRenderStyle.waterAlpha < 0.0f ||
                        desktopRenderStyle.waterAlpha > 1.0f) {
                        throw std::invalid_argument(
                            "Desktop water alpha must be between 0 and 1.");
                    }
                } else if (option.starts_with("--desktop-grid-strength=")) {
                    desktopRenderStyle.gridStrength = std::stof(
                        std::string(option.substr(24)));
                    if (!std::isfinite(desktopRenderStyle.gridStrength) ||
                        desktopRenderStyle.gridStrength < 0.0f ||
                        desktopRenderStyle.gridStrength > 1.0f) {
                        throw std::invalid_argument(
                            "Desktop grid strength must be between 0 and 1.");
                    }
                } else if (option == "--desktop-injection=on") {
                    injectionSettings.enabled = true;
                } else if (option == "--desktop-injection=off") {
                    injectionSettings.enabled = false;
                } else if (option.starts_with("--desktop-injection-rate=")) {
                    injectionSettings.areaRate = std::stod(
                        std::string(option.substr(25)));
                    if (!std::isfinite(injectionSettings.areaRate) ||
                        injectionSettings.areaRate < 0.0 ||
                        injectionSettings.areaRate > 0.25) {
                        throw std::invalid_argument(
                            "Desktop injection rate must be between 0 and 0.25 simulation units squared per second.");
                    }
                } else if (option.starts_with("--desktop-injection-radius=")) {
                    injectionSettings.radiusInCells = std::stof(
                        std::string(option.substr(27)));
                    if (!std::isfinite(injectionSettings.radiusInCells) ||
                        injectionSettings.radiusInCells < 0.5f ||
                        injectionSettings.radiusInCells > 8.0f) {
                        throw std::invalid_argument(
                            "Desktop injection radius must be between 0.5 and 8 cells.");
                    }
                } else {
                    throw std::invalid_argument("Unknown Desktop option: " +
                        std::string(option));
                }
            }
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--external-force-validate") {
            mode = ApplicationMode::phase20ExternalForceValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--container-motion-validate") {
            mode = ApplicationMode::phase20ContainerMotionValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--container-inertia-validate") {
            mode = ApplicationMode::phase20ContainerInertiaValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-input-timing-validate") {
            mode = ApplicationMode::phase20DesktopInputTimingValidation;
        } else if (argumentCount == 2 &&
            (std::string_view(arguments[1]) == "--desktop-interaction-benchmark" ||
             std::string_view(arguments[1]) == "--desktop-frame-timing-benchmark")) {
            mode = ApplicationMode::phase20DesktopInteractionBenchmark;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-frame-pacing-validate") {
            mode = ApplicationMode::phase20DesktopFramePacingValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-modal-loop-validate") {
            mode = ApplicationMode::phase20DesktopModalLoopValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-settle-generate") {
            mode = ApplicationMode::phase20DesktopSettleGenerate;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-settle-load-validate") {
            mode = ApplicationMode::phase20DesktopSettleLoadValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-inertia-from-settled-validate") {
            mode = ApplicationMode::phase20DesktopInertiaFromSettledValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-inertia-pipeline-audit") {
            mode = ApplicationMode::phase20DesktopInertiaPipelineAudit;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-inertia-timing-validate") {
            mode = ApplicationMode::phase20DesktopInertiaTimingValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-inertia-response-validate") {
            mode = ApplicationMode::phase20DesktopInertiaResponseValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-inertia-latency-benchmark") {
            mode = ApplicationMode::phase20DesktopInertiaLatencyBenchmark;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-grid-render-validate") {
            mode = ApplicationMode::phase20DesktopGridRenderValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-water-alpha-validate") {
            mode = ApplicationMode::phase20DesktopWaterAlphaValidation;
        } else if (argumentCount == 2 && std::string_view(arguments[1]) == "--desktop-render-style-benchmark") {
            mode = ApplicationMode::phase20DesktopRenderStyleBenchmark;
        } else if (argumentCount == 2 &&
            (std::string_view(arguments[1]) == "--desktop-physics-clock-audit" ||
             std::string_view(arguments[1]) == "--desktop-scheduler-validate" ||
             std::string_view(arguments[1]) == "--desktop-scheduler-backlog-validate" ||
             std::string_view(arguments[1]) == "--desktop-modal-pacing-validate")) {
            mode = ApplicationMode::phase20DesktopSchedulerValidation;
        } else if (argumentCount == 2 &&
            (std::string_view(arguments[1]) == "--desktop-async-throughput-validate" ||
             std::string_view(arguments[1]) == "--desktop-snapshot-latency-validate" ||
             std::string_view(arguments[1]) == "--desktop-realtime-benchmark")) {
            mode = ApplicationMode::phase20DesktopAsyncThroughputValidation;
        } else if (argumentCount == 2 &&
            (std::string_view(arguments[1]) == "--desktop-batch-input-audit" ||
             std::string_view(arguments[1]) == "--desktop-batch-partial-validate")) {
            mode = ApplicationMode::phase20DesktopBatchInputAudit;
        } else if (argumentCount == 2 &&
            (std::string_view(arguments[1]) == "--desktop-async-turnaround-audit" ||
             std::string_view(arguments[1]) == "--desktop-throughput-accounting-validate" ||
             std::string_view(arguments[1]) == "--desktop-snapshot-cadence-validate")) {
            mode = ApplicationMode::phase20DesktopThroughputAccountingValidation;
        } else if (argumentCount == 2 &&
            std::string_view(arguments[1]) == "--desktop-realtime-sustained-benchmark") {
            mode = ApplicationMode::phase20DesktopSustainedRealtimeBenchmark;
        } else if (argumentCount == 2 &&
            (std::string_view(arguments[1]) == "--gpu-fluid-injection-validate" ||
             std::string_view(arguments[1]) == "--gpu-fluid-injection-rollback-validate" ||
             std::string_view(arguments[1]) == "--desktop-fluid-injection-volume-validate")) {
            mode = ApplicationMode::phase20GpuFluidInjectionValidation;
        } else if (argumentCount != 1) {
            throw std::runtime_error(
                "Usage: LiquidSimulator.exe "
                "[--test|--benchmark|--dynamic-benchmark|--vof-geometry-validate|--vof-conservation-validate|--vof-advection-validate|--vof-split-conservation-validate|--vof-split-boundedness-validate|--vof-split-shape-validate|--vof-split-mac-replay|--vof-split-benchmark|--clsvof-coupling-validate|--clsvof-benchmark|--vof-mac-replay-validate|--dynamic-frame-benchmark|--validate|--visual-validate|--dam-break|--dam-break-validate|--gpu-divergence-validate|--gpu-mac-validate|--gpu-pressure-optimize|--gpu-mgpcg-validate|--gpu-dynamic-validate|--gpu-dynamic-topology-validate|--gpu-dam-break|--gpu-dam-break-validate|--gpu-interface-validate|--gpu-adaptive-validate|--gpu-pressure-resident-validate|--gpu-async-validate|--gpu-async-benchmark|--gpu-mixed-pressure-validate|--gpu-solver-benchmark|--gpu-mgpcg-profile|--gpu-mixed-reliability-validate|--gpu-mixed-long-validate|--gpu-coarse-fusion-regression|--gpu-interface-accuracy-validate|--interface-reconstruction-validate|--ghost-fluid-cpu-validate|--levelset-dynamic-validate|--levelset-redistance-benchmark|--levelset-particle-consistency-validate|--ghost-fluid-dynamic-cpu-validate|--gpu-levelset-validate|--gpu-levelset-dynamic-validate|--gpu-levelset-rollback-validate|--gpu-levelset-benchmark|--gpu-levelset-area-audit|--gpu-levelset-advection-compare|--gpu-levelset-long-validate|--gpu-levelset-redistance-validate|--gpu-levelset-extension-validate|--gpu-levelset-conservation-compare|--visual-benchmark "
                "|--gpu-vof-flux-validate|--gpu-vof-plic-validate|--gpu-vof-split-validate|--gpu-vof-rollback-validate|--gpu-vof-benchmark|--gpu-vof-mac-diagnostic|--gpu-vof-precision-audit|--gpu-vof-multistep-validate|--gpu-vof-bounded-flux-validate|--gpu-vof-mac-replay|--gpu-vof-velocity-compatibility|--gpu-vof-multistep-benchmark|--gpu-vof-roundoff-audit|--gpu-vof-commit-safety-validate|--gpu-vof-conservative-repair-validate|--gpu-vof-band-velocity-validate|--gpu-vof-long-validate|--gpu-vof-dynamic-compare|--gpu-vof-safety-benchmark|--desktop-interactive|--desktop-interactive-dam-break|--external-force-validate|--container-motion-validate|--container-inertia-validate|--desktop-input-timing-validate|--desktop-interaction-benchmark|--desktop-frame-pacing-validate|--desktop-modal-loop-validate|--desktop-settle-generate|--desktop-settle-load-validate|--desktop-inertia-from-settled-validate|--desktop-inertia-pipeline-audit|--desktop-inertia-timing-validate|--desktop-inertia-response-validate|--desktop-inertia-latency-benchmark|--desktop-grid-render-validate|--desktop-water-alpha-validate|--desktop-render-style-benchmark|--desktop-physics-clock-audit|--desktop-scheduler-validate|--desktop-scheduler-backlog-validate|--desktop-async-throughput-validate|--desktop-snapshot-latency-validate|--desktop-realtime-benchmark|--desktop-modal-pacing-validate|--desktop-batch-input-audit|--desktop-batch-partial-validate|--desktop-async-turnaround-audit|--desktop-throughput-accounting-validate|--desktop-snapshot-cadence-validate|--desktop-realtime-sustained-benchmark|--gpu-fluid-injection-validate|--gpu-fluid-injection-rollback-validate|--desktop-fluid-injection-volume-validate] "
                "[64|128|256] [cold|warm] [--desktop-motion-scale=120..2400] [--desktop-grid=on|off] [--desktop-water-alpha=0..1] [--desktop-grid-strength=0..1] [--desktop-injection=on|off] [--desktop-injection-rate=0..0.25] [--desktop-injection-radius=0.5..8]]");
        }
        return runOpenGlApplication(mode, kPhase1GridSize, true,
            desktopPixelsPerSimulationUnit, desktopRenderStyle, injectionSettings);
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}
#endif
