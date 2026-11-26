#include "GpuVofValidation.h"

#include "../simulation/gpu/GpuVolumeFraction.h"
#include "../simulation/gpu/GpuDynamicFreeSurfaceStepper.h"
#include "../simulation/mac/LevelSetInterface.h"
#include "../simulation/mac/MarkerParticles.h"
#include "../simulation/mac/Scenarios.h"
#include "../simulation/mac/VolumeFractionField.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <numeric>
#include <ostream>
#include <queue>
#include <string_view>

namespace fluidlab::diagnostics {
namespace {

using simulation::GpuVolumeFraction;
using simulation::MacGrid;
using simulation::PlicReconstruction;
using simulation::VolumeFractionField;

VolumeFractionField makeCircle(std::size_t width, std::size_t height) {
    const double dx = 1.0 / static_cast<double>(width);
    const double dy = 0.75 / static_cast<double>(height);
    VolumeFractionField field(width, height, dx, dy);
    field.initializeFromSignedDistance([](double x, double y) {
        return std::hypot(x - 0.38, y - 0.36) - 0.19;
    }, 24);
    return field;
}

MacGrid makeTranslation(const VolumeFractionField& field, float xSpeed, float ySpeed) {
    MacGrid velocity(field.width(), field.height(),
        static_cast<float>(field.cellWidth()), static_cast<float>(field.cellHeight()));
    velocity.fillCellTypes(simulation::CellType::fluid);
    for (std::size_t y = 0; y < field.height(); ++y) {
        for (std::size_t x = 0; x <= field.width(); ++x) velocity.horizontalVelocity(x, y) = xSpeed;
    }
    for (std::size_t y = 0; y <= field.height(); ++y) {
        for (std::size_t x = 0; x < field.width(); ++x) velocity.verticalVelocity(x, y) = ySpeed;
    }
    velocity.enforceClosedBoundary();
    return velocity;
}

MacGrid makeDiscreteSolenoidal(const VolumeFractionField& field) {
    MacGrid velocity(field.width(), field.height(),
        static_cast<float>(field.cellWidth()), static_cast<float>(field.cellHeight()));
    velocity.fillCellTypes(simulation::CellType::fluid);
    const auto psi = [&](std::size_t x, std::size_t y) {
        const double px = static_cast<double>(x) / static_cast<double>(field.width());
        const double py = static_cast<double>(y) / static_cast<double>(field.height());
        return 0.01 * std::sin(2.0 * std::numbers::pi * px) *
            std::sin(2.0 * std::numbers::pi * py);
    };
    for (std::size_t y = 0; y < field.height(); ++y)
        for (std::size_t x = 0; x <= field.width(); ++x)
            velocity.horizontalVelocity(x, y) = static_cast<float>((psi(x, y + 1) - psi(x, y)) /
                field.cellHeight());
    for (std::size_t y = 0; y <= field.height(); ++y)
        for (std::size_t x = 0; x < field.width(); ++x)
            velocity.verticalVelocity(x, y) = static_cast<float>(-(psi(x + 1, y) - psi(x, y)) /
                field.cellWidth());
    velocity.enforceClosedBoundary();
    return velocity;
}

double cpuFlux(const VolumeFractionField& field, const MacGrid& velocity,
    bool xDirection, std::size_t faceX, std::size_t faceY, double dt) {
    if ((xDirection && (faceX == 0 || faceX == field.width())) ||
        (!xDirection && (faceY == 0 || faceY == field.height()))) return 0.0;
    const double speed = xDirection ? velocity.horizontalVelocity(faceX, faceY) :
        velocity.verticalVelocity(faceX, faceY);
    if (speed == 0.0) return 0.0;
    const std::size_t donorX = xDirection ? (speed > 0.0 ? faceX - 1 : faceX) : faceX;
    const std::size_t donorY = xDirection ? faceY : (speed > 0.0 ? faceY - 1 : faceY);
    if (!field.isValid(donorX, donorY)) return 0.0;
    const double f = field.fraction(donorX, donorY);
    const double extent = xDirection ? field.cellWidth() : field.cellHeight();
    const double swept = std::min(std::abs(speed) * dt, extent);
    if (f <= 1.0e-7 || swept == 0.0) return 0.0;
    double amount = 0.0;
    if (f >= 1.0 - 1.0e-7) {
        amount = swept * (xDirection ? field.cellHeight() : field.cellWidth());
    } else {
        const PlicReconstruction plic = field.reconstructCell(donorX, donorY);
        if (plic.valid) {
            const double minX = xDirection && speed > 0.0 ? field.cellWidth() - swept : 0.0;
            const double maxX = xDirection && speed < 0.0 ? swept : field.cellWidth();
            const double minY = !xDirection && speed > 0.0 ? field.cellHeight() - swept : 0.0;
            const double maxY = !xDirection && speed < 0.0 ? swept : field.cellHeight();
            amount = simulation::plicLiquidAreaInBox(plic, minX, minY, maxX, maxY);
        } else {
            amount = f * swept * (xDirection ? field.cellHeight() : field.cellWidth());
        }
    }
    return speed > 0.0 ? amount : -amount;
}

struct Error final { double rms = 0.0; double maximum = 0.0; };

struct FractionMetrics final {
    double area = 0.0;
    double centerX = 0.0;
    double centerY = 0.0;
    double secondMoment = 0.0;
    std::size_t lowTail = 0;
    std::size_t components = 0;
};

FractionMetrics metrics(
    const std::vector<float>& fractions, std::size_t width, std::size_t height,
    double dx, double dy) {
    FractionMetrics result;
    double weightedX = 0.0;
    double weightedY = 0.0;
    for (std::size_t y = 0; y < height; ++y) for (std::size_t x = 0; x < width; ++x) {
        const double value = fractions[y * width + x];
        const double area = value * dx * dy;
        result.area += area;
        weightedX += area * (static_cast<double>(x) + 0.5) * dx;
        weightedY += area * (static_cast<double>(y) + 0.5) * dy;
        if (value > 0.0 && value < 1.0e-4) ++result.lowTail;
    }
    if (result.area > 0.0) {
        result.centerX = weightedX / result.area;
        result.centerY = weightedY / result.area;
        for (std::size_t y = 0; y < height; ++y) for (std::size_t x = 0; x < width; ++x) {
            const double px = (static_cast<double>(x) + 0.5) * dx - result.centerX;
            const double py = (static_cast<double>(y) + 0.5) * dy - result.centerY;
            result.secondMoment += fractions[y * width + x] * dx * dy * (px * px + py * py);
        }
    }
    std::vector<unsigned char> visited(width * height, 0);
    for (std::size_t start = 0; start < fractions.size(); ++start) {
        if (visited[start] != 0 || fractions[start] <= 0.5f) continue;
        ++result.components;
        std::queue<std::size_t> pending;
        pending.push(start); visited[start] = 1;
        while (!pending.empty()) {
            const std::size_t index = pending.front(); pending.pop();
            const std::size_t x = index % width, y = index / width;
            const std::array<std::pair<int, int>, 4> neighbors{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
            for (const auto [ox, oy] : neighbors) {
                const int nx = static_cast<int>(x) + ox, ny = static_cast<int>(y) + oy;
                if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height)) continue;
                const std::size_t next = static_cast<std::size_t>(ny) * width + static_cast<std::size_t>(nx);
                if (visited[next] == 0 && fractions[next] > 0.5f) {
                    visited[next] = 1; pending.push(next);
                }
            }
        }
    }
    return result;
}

MacGrid makeRotation(const VolumeFractionField& field, float angularSpeed) {
    MacGrid velocity(field.width(), field.height(),
        static_cast<float>(field.cellWidth()), static_cast<float>(field.cellHeight()));
    velocity.fillCellTypes(simulation::CellType::fluid);
    const float centerX = static_cast<float>(0.5 * field.width() * field.cellWidth());
    const float centerY = static_cast<float>(0.5 * field.height() * field.cellHeight());
    for (std::size_t y = 0; y < field.height(); ++y) for (std::size_t x = 0; x <= field.width(); ++x)
        velocity.horizontalVelocity(x, y) = -angularSpeed *
            ((static_cast<float>(y) + 0.5f) * static_cast<float>(field.cellHeight()) - centerY);
    for (std::size_t y = 0; y <= field.height(); ++y) for (std::size_t x = 0; x < field.width(); ++x)
        velocity.verticalVelocity(x, y) = angularSpeed *
            ((static_cast<float>(x) + 0.5f) * static_cast<float>(field.cellWidth()) - centerX);
    velocity.enforceClosedBoundary();
    return velocity;
}

void scaleVelocity(MacGrid& velocity, float scale) {
    for (std::size_t y = 0; y < velocity.height(); ++y)
        for (std::size_t x = 0; x <= velocity.width(); ++x)
            velocity.horizontalVelocity(x, y) *= scale;
    for (std::size_t y = 0; y <= velocity.height(); ++y)
        for (std::size_t x = 0; x < velocity.width(); ++x)
            velocity.verticalVelocity(x, y) *= scale;
}

double absoluteFlux(const std::vector<float>& flux) {
    return std::accumulate(flux.begin(), flux.end(), 0.0,
        [](double sum, float value) { return sum + std::abs(static_cast<double>(value)); });
}

Error compare(const std::vector<float>& gpu, const std::vector<double>& cpu) {
    Error result;
    double squared = 0.0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const double error = std::abs(static_cast<double>(gpu[i]) - cpu[i]);
        squared += error * error;
        result.maximum = std::max(result.maximum, error);
    }
    result.rms = gpu.empty() ? 0.0 : std::sqrt(squared / static_cast<double>(gpu.size()));
    return result;
}

int runCoreValidation(std::ostream& output, const std::filesystem::path& shaderDirectory,
    bool includePlic, bool includeStep) {
    bool passed = true;
    for (const auto [width, height] : std::array<std::pair<std::size_t, std::size_t>, 3>{
             {{40, 30}, {80, 60}, {64, 64}}}) {
        auto field = makeCircle(width, height);
        auto velocity = makeTranslation(field, 0.12f, -0.07f);
        const double dt = 0.25 * std::min(field.cellWidth() / 0.12, field.cellHeight() / 0.07);
        GpuVolumeFraction gpu(shaderDirectory);
        gpu.upload(field);
        gpu.computeFluxDiagnostic(velocity, dt, true);
        auto snapshot = gpu.readDiagnosticState();
        std::vector<double> cpuX((width + 1) * height);
        for (std::size_t y = 0; y < height; ++y)
            for (std::size_t x = 0; x <= width; ++x)
                cpuX[y * (width + 1) + x] = cpuFlux(field, velocity, true, x, y, dt);
        const Error fluxError = compare(snapshot.horizontalFlux, cpuX);
        double alphaMaximum = 0.0;
        std::size_t mixed = 0;
        if (includePlic) {
            for (std::size_t y = 0; y < height; ++y) for (std::size_t x = 0; x < width; ++x) {
                const double f = field.fraction(x, y);
                if (f <= 1.0e-7 || f >= 1.0 - 1.0e-7) continue;
                const auto reference = field.reconstructCell(x, y);
                const std::size_t index = y * width + x;
                if (reference.valid && snapshot.plicValid[index]) {
                    alphaMaximum = std::max(alphaMaximum,
                        std::abs(reference.alpha - snapshot.alpha[index]));
                    ++mixed;
                }
            }
        }
        double fRms = 0.0;
        simulation::GpuVofStepReport report;
        if (includeStep) {
            auto cpu = field;
            const auto cpuLedger = cpu.advectConservativeSplitExperimental(velocity, dt, true);
            simulation::GpuVofCandidateSettings legacySettings;
            legacySettings.numericalBoundTolerance = 5.0e-6f;
            legacySettings.conservationTolerance = 5.0e-6;
            legacySettings.allowNumericalBoundSlack = true;
            report = gpu.computeCandidate(velocity, dt, true, legacySettings);
            const auto candidate = gpu.readCandidate();
            std::vector<double> cpuValues = cpu.fractions();
            fRms = compare(candidate.fractions, cpuValues).rms;
            passed = passed && cpuLedger.accepted == report.valid;
        }
        const bool casePass = fluxError.maximum < 2.0e-6 &&
            (!includePlic || (mixed != 0 && alphaMaximum < 2.0e-5)) &&
            (!includeStep || (report.valid && fRms < 2.0e-5 && report.outOfBoundsCount == 0));
        passed = passed && casePass;
        output << "GPU VOF grid=" << width << 'x' << height
               << " fluxRms=" << std::scientific << fluxError.rms
               << " fluxMax=" << fluxError.maximum
               << " alphaMax=" << alphaMaximum
               << " fRms=" << fRms
               << " result=" << (casePass ? "PASS" : "FAIL") << '\n';
    }
    return passed ? 0 : 1;
}

} // namespace

int runGpuVofFluxValidation(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF shared-face flux validation\n";
    return runCoreValidation(output, shaders, false, false);
}
int runGpuVofPlicValidation(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF PLIC validation\n";
    return runCoreValidation(output, shaders, true, false);
}
int runGpuVofSplitValidation(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF conservative split validation\n";
    bool passed = runCoreValidation(output, shaders, true, true) == 0;
    VolumeFractionField full(48, 32, 1.0 / 48.0, 0.75 / 32.0);
    for (std::size_t y = 0; y < full.height(); ++y)
        for (std::size_t x = 0; x < full.width(); ++x) full.setFraction(x, y, 1.0);
    const auto velocity = makeDiscreteSolenoidal(full);
    for (bool xFirst : {true, false}) {
        auto cpu = full;
        const auto cpuLedger = cpu.advectConservativeSplitExperimental(velocity, 0.005, xFirst);
        GpuVolumeFraction gpu(shaders); gpu.upload(full);
        const auto report = gpu.computeCandidate(velocity, 0.005, xFirst);
        const auto candidate = gpu.readCandidate();
        const Error error = compare(candidate.fractions, cpu.fractions());
        const bool casePass = cpuLedger.accepted && report.valid && error.maximum < 2.0e-5 &&
            report.minimumFraction >= -5.0e-6 && report.maximumFraction <= 1.0 + 5.0e-6;
        passed = passed && casePass;
        output << "GPU VOF full solenoidal order=" << (xFirst ? "xy" : "yx")
               << " fMax=" << std::scientific << error.maximum
               << " residual=" << report.conservationResidual
               << " result=" << (casePass ? "PASS" : "FAIL") << '\n';
    }
    return passed ? 0 : 1;
}
int runGpuVofRollbackValidation(std::ostream& output, const std::filesystem::path& shaders) {
    auto field = makeCircle(40, 30); auto velocity = makeTranslation(field, 0.1f, 0.0f);
    GpuVolumeFraction gpu(shaders); gpu.upload(field);
    const auto before = gpu.readCommitted(); const auto version = gpu.version(); const auto time = gpu.physicalTime();
    const auto report = gpu.computeCandidate(velocity, 0.01, true);
    gpu.rejectCandidate(); const auto after = gpu.readCommitted();
    const bool rejectedUnchanged = report.valid && before.fractions == after.fractions && gpu.version() == version &&
        gpu.physicalTime() == time && !gpu.hasCandidate();
    const auto acceptedReport = gpu.computeCandidate(velocity, 0.01, true);
    gpu.acceptCandidate();
    const bool accepted = acceptedReport.valid && gpu.version() == version + 1 &&
        std::abs(gpu.physicalTime() - time - 0.01) < 1.0e-12;
    gpu.reset(field);
    auto invalidVelocity = velocity;
    invalidVelocity.horizontalVelocity(20, 15) = 20.0f;
    const auto invalidBefore = gpu.readCommitted();
    const auto invalidReport = gpu.computeCandidate(invalidVelocity, 0.02, true);
    const auto invalidAfter = gpu.readCommitted();
    const bool failedUnchanged = !invalidReport.valid && !gpu.hasCandidate() &&
        invalidBefore.fractions == invalidAfter.fractions && gpu.version() == version && gpu.physicalTime() == time;
    gpu.reset(field);
    auto nonFiniteVelocity = velocity;
    nonFiniteVelocity.horizontalVelocity(20, 15) = std::numeric_limits<float>::quiet_NaN();
    const auto nonFiniteBefore = gpu.readCommitted();
    const auto nonFiniteReport = gpu.computeCandidate(nonFiniteVelocity, 0.01, true);
    const auto nonFiniteAfter = gpu.readCommitted();
    const bool nonFiniteRejected = !nonFiniteReport.valid && !gpu.hasCandidate() &&
        nonFiniteReport.invalidVelocityCount != 0 &&
        nonFiniteBefore.fractions == nonFiniteAfter.fractions &&
        gpu.version() == version && gpu.physicalTime() == time;
    simulation::GpuVofCandidateSettings localSettings;
    localSettings.boundedMode = simulation::GpuVofBoundedMode::localFixed;
    const auto localFailureReport = gpu.computeCandidate(invalidVelocity, 0.02, true, localSettings);
    const bool localFailureRejected = !localFailureReport.valid && !gpu.hasCandidate() &&
        gpu.readCommitted().fractions == nonFiniteBefore.fractions;
    const bool pass = rejectedUnchanged && accepted && failedUnchanged &&
        nonFiniteRejected && localFailureRejected;
    output << "GPU VOF rollback rejectUnchanged=" << rejectedUnchanged
           << " acceptVersion=" << accepted << " failedUnchanged=" << failedUnchanged
           << " nonFiniteRejected=" << nonFiniteRejected
           << " localFailureRejected=" << localFailureRejected
           << " failureBounds=" << invalidReport.outOfBoundsCount
           << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}
int runGpuVofBenchmark(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF benchmark (8 timed dispatches per candidate)\n";
    for (const std::size_t size : {64u, 128u, 192u, 256u}) {
        auto field = makeCircle(size, size); auto velocity = makeTranslation(field, 0.1f, 0.05f);
        GpuVolumeFraction gpu(shaders); gpu.upload(field);
        std::array<double, 8> phase{}; double wall = 0.0; constexpr int samples = 8;
        for (int sample = 0; sample < samples; ++sample) {
            const auto start = std::chrono::steady_clock::now();
            const auto report = gpu.computeCandidate(velocity, 0.002, (sample & 1) == 0);
            wall += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            for (std::size_t i = 0; i < phase.size(); ++i) phase[i] += report.gpuPhaseMilliseconds[i];
            gpu.rejectCandidate();
        }
        output << "grid=" << size << 'x' << size << " wallMs=" << std::fixed << std::setprecision(3)
               << wall / samples << " gpuMs=" << std::accumulate(phase.begin(), phase.end(), 0.0) / samples
               << " memoryKiB=" << gpu.memoryBytes() / 1024.0 << " dispatch=8\n";
    }
    return 0;
}
int runGpuVofMacDiagnostic(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF accepted-substep MAC diagnostic, 48x32 to t=0.5 s\n";
    MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
    simulation::initializeDamBreak(grid);
    simulation::MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    VolumeFractionField cpu(48, 32, 1.0 / 48.0, 1.0 / 32.0);
    const std::size_t liquidEndX = 1 + (grid.width() - 2) / 3;
    const std::size_t liquidEndY = 1 + 2 * (grid.height() - 2) / 3;
    const double minimumX = grid.cellWidth(), minimumY = grid.cellHeight();
    const double maximumX = liquidEndX * grid.cellWidth(), maximumY = liquidEndY * grid.cellHeight();
    const double centerX = 0.5 * (minimumX + maximumX), centerY = 0.5 * (minimumY + maximumY);
    const double halfX = 0.5 * (maximumX - minimumX), halfY = 0.5 * (maximumY - minimumY);
    cpu.initializeFromSignedDistance([=](double x, double y) {
        const double qx = std::abs(x - centerX) - halfX;
        const double qy = std::abs(y - centerY) - halfY;
        return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
            std::min(std::max(qx, qy), 0.0);
    }, 32);
    cpu.applyWholeCellSolidMask(grid);
    GpuVolumeFraction gpu(shaders); gpu.upload(cpu);
    std::size_t observed = 0, cpuRejected = 0, gpuRejected = 0, agreement = 0;
    double maximumFError = 0.0, maximumResidual = 0.0;
    double mismatchMinimum = 1.0, mismatchMaximum = 0.0;
    simulation::GpuDynamicFreeSurfaceSettings settings;
    settings.maximumSubsteps = 32;
    settings.acceptedSubstepValidationObserver = [&](const MacGrid& snapshot, float dt) {
        const VolumeFractionField before = cpu;
        const bool xFirst = (observed & 1u) == 0u;
        const auto cpuLedger = cpu.advectConservativeSplitExperimental(snapshot, dt, xFirst);
        simulation::GpuVofCandidateSettings legacySettings;
        legacySettings.numericalBoundTolerance = 5.0e-6f;
        legacySettings.conservationTolerance = 5.0e-6;
        legacySettings.allowNumericalBoundSlack = true;
        const auto gpuReport = gpu.computeCandidate(snapshot, dt, xFirst, legacySettings);
        cpuRejected += cpuLedger.accepted ? 0u : 1u;
        gpuRejected += gpuReport.valid ? 0u : 1u;
        maximumResidual = std::max(maximumResidual, std::abs(gpuReport.conservationResidual));
        if (cpuLedger.accepted == gpuReport.valid) ++agreement;
        else {
            mismatchMinimum = std::min(mismatchMinimum, static_cast<double>(gpuReport.minimumFraction));
            mismatchMaximum = std::max(mismatchMaximum, static_cast<double>(gpuReport.maximumFraction));
        }
        if (cpuLedger.accepted && gpuReport.valid) {
            const auto candidate = gpu.readCandidate();
            const Error error = compare(candidate.fractions, cpu.fractions());
            maximumFError = std::max(maximumFError, error.maximum);
            gpu.acceptCandidate();
        } else {
            cpu = before;
            gpu.rejectCandidate();
        }
        ++observed;
    };
    simulation::GpuDynamicFreeSurfaceStepper stepper(shaders / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles);
    std::size_t accepted = 0, formalRejected = 0;
    for (std::size_t outer = 0; outer < 60 && stepper.healthy(); ++outer) {
        const auto report = stepper.step(); accepted += report.substeps; formalRejected += report.rejectedSubsteps;
    }
    const bool fp32ToleranceOnly = mismatchMinimum >= -5.0e-6 && mismatchMaximum <= 1.0 + 5.0e-6;
    const bool pass = topology.topologyValid && stepper.healthy() && observed == accepted &&
        maximumFError < 5.0e-5 && fp32ToleranceOnly;
    output << "formalAccepted=" << accepted << " formalRejected=" << formalRejected
           << " observed=" << observed << " cpuRejected=" << cpuRejected
           << " gpuRejected=" << gpuRejected << " decisionsAgree=" << agreement
           << " maxF=" << std::scientific << maximumFError
           << " maxLedgerResidual=" << maximumResidual
           << " mismatchRange=" << mismatchMinimum << '/' << mismatchMaximum
           << " fp32ToleranceOnly=" << fp32ToleranceOnly
           << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuVofPrecisionAudit(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF FP32 candidate precision audit, same-source snapshots\n";
    MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
    simulation::initializeDamBreak(grid);
    simulation::MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    VolumeFractionField reference(48, 32, 1.0 / 48.0, 1.0 / 32.0);
    reference.initializeFromSignedDistance([](double x, double y) {
        const double qx = std::abs(x - 0.177083333333) - 0.15625;
        const double qy = std::abs(y - 0.34375) - 0.3125;
        return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
            std::min(std::max(qx, qy), 0.0);
    }, 32);
    reference.applyWholeCellSolidMask(grid);
    GpuVolumeFraction gpu(shaders);
    std::array<std::size_t, 4> gpuAcceptedByTolerance{};
    constexpr std::array<double, 4> tolerances{0.0, 1.0e-6, 2.0e-6, 5.0e-6};
    std::size_t observed = 0, cpuAccepted = 0, decisionMismatch = 0;
    std::size_t quantizedDecisionMismatch = 0;
    double maximumSameSourceError = 0.0, maximumBelow = 0.0, maximumAbove = 0.0;
    double cpuMaximumBelow = 0.0, cpuMaximumAbove = 0.0;
    std::size_t minimumStep = 0, minimumIndex = 0, maximumStep = 0, maximumIndex = 0;
    std::size_t firstMismatch = std::numeric_limits<std::size_t>::max();
    simulation::GpuDynamicFreeSurfaceSettings settings;
    settings.maximumSubsteps = 32;
    settings.acceptedSubstepValidationObserver = [&](const MacGrid& snapshot, float dt) {
        const VolumeFractionField start = reference;
        auto cpuCandidate = start;
        const bool xFirst = (observed & 1u) == 0u;
        const auto cpuLedger = cpuCandidate.advectConservativeSplitExperimental(snapshot, dt, xFirst);
        cpuMaximumBelow = std::max(cpuMaximumBelow, std::max(0.0, -cpuLedger.minimumFraction));
        cpuMaximumAbove = std::max(cpuMaximumAbove, std::max(0.0, cpuLedger.maximumFraction - 1.0));

        auto quantized = start;
        for (std::size_t y = 0; y < quantized.height(); ++y)
            for (std::size_t x = 0; x < quantized.width(); ++x)
                quantized.setFraction(x, y, static_cast<float>(quantized.fraction(x, y)));
        const auto quantizedLedger = quantized.advectConservativeSplitExperimental(snapshot, dt, xFirst);
        quantizedDecisionMismatch += quantizedLedger.accepted == cpuLedger.accepted ? 0u : 1u;

        gpu.reset(start);
        simulation::GpuVofCandidateSettings gpuSettings;
        gpuSettings.numericalBoundTolerance = 5.0e-6f;
        gpuSettings.conservationTolerance = 5.0e-6;
        gpuSettings.allowNumericalBoundSlack = true;
        const auto gpuReport = gpu.computeCandidate(snapshot, dt, xFirst, gpuSettings);
        const auto candidate = gpu.readCandidate();
        const auto minimum = std::min_element(candidate.fractions.begin(), candidate.fractions.end());
        const auto maximum = std::max_element(candidate.fractions.begin(), candidate.fractions.end());
        const double below = std::max(0.0, -static_cast<double>(*minimum));
        const double above = std::max(0.0, static_cast<double>(*maximum) - 1.0);
        if (below > maximumBelow) {
            maximumBelow = below; minimumStep = observed;
            minimumIndex = static_cast<std::size_t>(minimum - candidate.fractions.begin());
        }
        if (above > maximumAbove) {
            maximumAbove = above; maximumStep = observed;
            maximumIndex = static_cast<std::size_t>(maximum - candidate.fractions.begin());
        }
        for (std::size_t i = 0; i < tolerances.size(); ++i) {
            const bool bounds = gpuReport.minimumFraction >= -tolerances[i] &&
                gpuReport.maximumFraction <= 1.0 + tolerances[i];
            gpuAcceptedByTolerance[i] += bounds && gpuReport.nonFiniteCount == 0 ? 1u : 0u;
        }
        if (cpuLedger.accepted) {
            ++cpuAccepted;
            const Error error = compare(candidate.fractions, cpuCandidate.fractions());
            maximumSameSourceError = std::max(maximumSameSourceError, error.maximum);
        }
        if (cpuLedger.accepted != gpuReport.valid) {
            ++decisionMismatch;
            if (firstMismatch == std::numeric_limits<std::size_t>::max()) firstMismatch = observed;
        }
        if (cpuLedger.accepted) reference = std::move(cpuCandidate);
        ++observed;
    };
    simulation::GpuDynamicFreeSurfaceStepper stepper(shaders / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles);
    for (std::size_t outer = 0; outer < 60 && stepper.healthy(); ++outer)
        static_cast<void>(stepper.step());
    const bool pass = topology.topologyValid && stepper.healthy() && observed != 0 &&
        maximumSameSourceError < 1.0e-4 && maximumBelow < 5.0e-6 && maximumAbove < 5.0e-6;
    output << "snapshots=" << observed << " cpuAccepted=" << cpuAccepted
           << " decisionMismatch=" << decisionMismatch
           << " firstMismatch=" << (firstMismatch == std::numeric_limits<std::size_t>::max() ? -1LL :
               static_cast<long long>(firstMismatch))
           << " fp32InputDecisionMismatch=" << quantizedDecisionMismatch
           << " maxSameSourceF=" << std::scientific << maximumSameSourceError
           << " gpuMaxBelow=" << maximumBelow << " gpuMaxAbove=" << maximumAbove
           << " minAt=" << minimumStep << ':' << minimumIndex % 48 << ',' << minimumIndex / 48
           << " maxAt=" << maximumStep << ':' << maximumIndex % 48 << ',' << maximumIndex / 48
           << " cpuMaxBelow=" << cpuMaximumBelow << " cpuMaxAbove=" << cpuMaximumAbove << '\n';
    for (std::size_t i = 0; i < tolerances.size(); ++i)
        output << "boundTolerance=" << tolerances[i] << " accepted=" << gpuAcceptedByTolerance[i] << '\n';
    output << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuVofMultistepValidation(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF continuous multistep validation\n";
    bool passed = true;
    enum class Case { stationary, translation, rotation, reversible, thinColumn, twoDrops };
    constexpr std::array<Case, 6> cases{
        Case::stationary, Case::translation, Case::rotation,
        Case::reversible, Case::thinColumn, Case::twoDrops};
    for (const auto [width, height] : std::array<std::pair<std::size_t, std::size_t>, 3>{
             {{40, 30}, {80, 60}, {160, 120}}}) {
        for (const Case test : cases) {
            VolumeFractionField cpu(width, height, 1.0 / width, 0.75 / height);
            if (test == Case::thinColumn) {
                cpu.initializeFromSignedDistance([](double x, double y) {
                    const double qx = std::abs(x - 0.38) - 0.025;
                    const double qy = std::abs(y - 0.36) - 0.22;
                    return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
                        std::min(std::max(qx, qy), 0.0);
                }, 24);
            } else if (test == Case::twoDrops) {
                cpu.initializeFromSignedDistance([](double x, double y) {
                    return std::min(std::hypot(x - 0.32, y - 0.36) - 0.09,
                        std::hypot(x - 0.57, y - 0.36) - 0.07);
                }, 24);
            } else {
                cpu = makeCircle(width, height);
            }
            const std::vector<double> initial = cpu.fractions();
            GpuVolumeFraction gpu(shaders);
            gpu.upload(cpu);
            bool casePass = true;
            std::size_t accepted = 0, rejected = 0;
            constexpr std::size_t steps = 24;
            constexpr double dt = 0.001;
            for (std::size_t step = 0; step < steps; ++step) {
                MacGrid velocity = test == Case::stationary || test == Case::twoDrops
                    ? makeTranslation(cpu, 0.0f, 0.0f)
                    : test == Case::rotation ? makeRotation(cpu, 0.35f)
                    : test == Case::reversible ? makeDiscreteSolenoidal(cpu)
                    : makeTranslation(cpu, 0.05f, test == Case::thinColumn ? 0.015f : 0.02f);
                if (test == Case::reversible && step >= steps / 2) scaleVelocity(velocity, -1.0f);
                auto cpuCandidate = cpu;
                const bool xFirst = (step & 1u) == 0u;
                const auto cpuLedger = cpuCandidate.advectConservativeSplitExperimental(velocity, dt, xFirst);
                simulation::GpuVofCandidateSettings gpuSettings;
                gpuSettings.numericalBoundTolerance = 5.0e-6f;
                gpuSettings.allowNumericalBoundSlack = true;
                const auto report = gpu.computeCandidate(velocity, dt, xFirst, gpuSettings);
                if (cpuLedger.accepted && report.valid) {
                    cpu = std::move(cpuCandidate);
                    gpu.acceptCandidate();
                    ++accepted;
                } else {
                    gpu.rejectCandidate();
                    ++rejected;
                    casePass = false;
                    break;
                }
            }
            const auto snapshot = gpu.readCommitted();
            const Error cpuGpu = compare(snapshot.fractions, cpu.fractions());
            const auto gpuMetrics = metrics(snapshot.fractions, width, height, cpu.cellWidth(), cpu.cellHeight());
            VolumeFractionField analytic(width, height, cpu.cellWidth(), cpu.cellHeight());
            const double elapsed = accepted * dt;
            if (test == Case::translation) {
                analytic.initializeFromSignedDistance([=](double x, double y) {
                    return std::hypot(x - (0.38 + 0.05 * elapsed),
                        y - (0.36 + 0.02 * elapsed)) - 0.19;
                }, 24);
            } else if (test == Case::rotation) {
                const double angle = 0.35 * elapsed;
                const double offsetX = 0.38 - 0.5, offsetY = 0.36 - 0.375;
                const double centerX = 0.5 + std::cos(angle) * offsetX - std::sin(angle) * offsetY;
                const double centerY = 0.375 + std::sin(angle) * offsetX + std::cos(angle) * offsetY;
                analytic.initializeFromSignedDistance([=](double x, double y) {
                    return std::hypot(x - centerX, y - centerY) - 0.19;
                }, 24);
            } else if (test == Case::thinColumn) {
                analytic.initializeFromSignedDistance([=](double x, double y) {
                    const double qx = std::abs(x - (0.38 + 0.05 * elapsed)) - 0.025;
                    const double qy = std::abs(y - (0.36 + 0.015 * elapsed)) - 0.22;
                    return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
                        std::min(std::max(qx, qy), 0.0);
                }, 24);
            } else {
                for (std::size_t y = 0; y < height; ++y)
                    for (std::size_t x = 0; x < width; ++x)
                        analytic.setFraction(x, y, initial[y * width + x]);
            }
            double analyticL1 = 0.0;
            for (std::size_t i = 0; i < snapshot.fractions.size(); ++i)
                analyticL1 += std::abs(snapshot.fractions[i] - analytic.fractions()[i]) *
                    cpu.cellWidth() * cpu.cellHeight();
            double reverseL1 = 0.0;
            if (test == Case::reversible) {
                for (std::size_t i = 0; i < snapshot.fractions.size(); ++i)
                    reverseL1 += std::abs(snapshot.fractions[i] - initial[i]) * cpu.cellWidth() * cpu.cellHeight();
            }
            casePass = casePass && cpuGpu.maximum < 2.0e-4 &&
                std::abs(gpuMetrics.area - cpu.totalArea()) < 5.0e-6;
            passed = passed && casePass;
            const char* name = test == Case::stationary ? "static" : test == Case::translation ? "translate" :
                test == Case::rotation ? "rotate" : test == Case::reversible ? "reverse" :
                test == Case::thinColumn ? "thin" : "two-drops";
            output << "case=" << name << " grid=" << width << 'x' << height
                   << " accepted=" << accepted << " rejected=" << rejected
                   << " area=" << std::scientific << gpuMetrics.area
                   << " cpuGpuRms=" << cpuGpu.rms << " cpuGpuMax=" << cpuGpu.maximum
                   << " analyticL1=" << analyticL1 << " reverseL1=" << reverseL1
                   << " centroid=" << gpuMetrics.centerX << ',' << gpuMetrics.centerY
                   << " moment=" << gpuMetrics.secondMoment
                   << " components=" << gpuMetrics.components << " lowTail=" << gpuMetrics.lowTail
                   << " result=" << (casePass ? "PASS" : "FAIL") << '\n';
        }
    }
    return passed ? 0 : 1;
}

int runGpuVofBoundedFluxValidation(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF fixed-stage local bounded-flux validation\n";
    bool passed = true;
    VolumeFractionField full(48, 32, 1.0 / 48.0, 0.75 / 32.0);
    for (std::size_t y = 0; y < full.height(); ++y)
        for (std::size_t x = 0; x < full.width(); ++x) full.setFraction(x, y, 1.0);
    auto solenoidal = makeDiscreteSolenoidal(full);
    GpuVolumeFraction base(shaders), bounded(shaders);
    base.upload(full); bounded.upload(full);
    const auto baseReport = base.computeCandidate(solenoidal, 0.005, true);
    simulation::GpuVofCandidateSettings boundedSettings;
    boundedSettings.boundedMode = simulation::GpuVofBoundedMode::localFixed;
    const auto boundedReport = bounded.computeCandidate(solenoidal, 0.005, true, boundedSettings);
    const auto baseSnapshot = base.readCandidate();
    const auto boundedSnapshot = bounded.readCandidate();
    const Error noOpError = compare(boundedSnapshot.fractions,
        std::vector<double>(baseSnapshot.fractions.begin(), baseSnapshot.fractions.end()));
    const bool noOpPass = baseReport.valid && boundedReport.valid &&
        boundedReport.uniqueLimitedFaceCount == 0 && noOpError.maximum < 2.0e-7;
    passed = passed && noOpPass;
    output << "solenoidal limited=" << boundedReport.uniqueLimitedFaceCount
           << " fMax=" << std::scientific << noOpError.maximum
           << " result=" << (noOpPass ? "PASS" : "FAIL") << '\n';

    VolumeFractionField circle(40, 30, 1.0 / 40.0, 0.75 / 30.0);
    circle.setFraction(10, 10, 0.8);
    circle.setFraction(11, 10, 0.4);
    auto stressed = makeTranslation(circle, 0.0f, 0.0f);
    stressed.horizontalVelocity(11, 10) = 2.0f;
    GpuVolumeFraction stressedBase(shaders), stressedBounded(shaders);
    stressedBase.upload(circle); stressedBounded.upload(circle);
    const auto rawReport = stressedBase.computeCandidate(stressed, 0.01, true);
    const auto limitedReport = stressedBounded.computeCandidate(stressed, 0.01, true, boundedSettings);
    const auto limitedSnapshot = stressedBounded.readCandidate();
    const double rawAbs = absoluteFlux(limitedSnapshot.rawHorizontalFlux) +
        absoluteFlux(limitedSnapshot.rawVerticalFlux);
    const double finalAbs = absoluteFlux(limitedSnapshot.horizontalFlux) +
        absoluteFlux(limitedSnapshot.verticalFlux);
    const bool stressedPass = limitedReport.nonFiniteCount == 0 &&
        limitedReport.uniqueLimitedFaceCount != 0 && finalAbs <= rawAbs + 1.0e-12 &&
        (!limitedReport.valid || (limitedReport.minimumFraction >= -2.0e-6f &&
            limitedReport.maximumFraction <= 1.0f + 2.0e-6f));
    passed = passed && stressedPass;
    output << "stress rawValid=" << rawReport.valid << " limitedValid=" << limitedReport.valid
           << " unique=" << limitedReport.uniqueLimitedFaceCount
           << " repeated=" << limitedReport.repeatedLimitedFaceCount
           << " nearBlocked=" << limitedReport.nearlyBlockedFaceCount
           << " fluxWeakening=" << (rawAbs > 0.0 ? 1.0 - finalAbs / rawAbs : 0.0)
           << " range=" << limitedReport.minimumFraction << '/' << limitedReport.maximumFraction
           << " result=" << (stressedPass ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

int runGpuVofMacReplay(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF baseline/local-bounded continuous MAC replay, 48x32\n";
    MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
    simulation::initializeDamBreak(grid);
    simulation::MarkerParticleSystem particles; particles.seedFluidCellsExact(grid, 8);
    VolumeFractionField initial(48, 32, 1.0 / 48.0, 1.0 / 32.0);
    const std::size_t endX = 1 + (grid.width() - 2) / 3, endY = 1 + 2 * (grid.height() - 2) / 3;
    initial.initializeFromSignedDistance([=, dx = grid.cellWidth(), dy = grid.cellHeight()](double x, double y) {
        const double minX = dx, minY = dy, maxX = endX * dx, maxY = endY * dy;
        const double qx = std::abs(x - 0.5 * (minX + maxX)) - 0.5 * (maxX - minX);
        const double qy = std::abs(y - 0.5 * (minY + maxY)) - 0.5 * (maxY - minY);
        return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) + std::min(std::max(qx, qy), 0.0);
    }, 32);
    initial.applyWholeCellSolidMask(grid);
    GpuVolumeFraction baseline(shaders), bounded(shaders), tolerantBaseline(shaders), tolerantBounded(shaders);
    baseline.upload(initial); bounded.upload(initial);
    tolerantBaseline.upload(initial); tolerantBounded.upload(initial);
    simulation::GpuVofCandidateSettings strict;
    strict.numericalBoundTolerance = 2.0e-6f;
    auto local = strict; local.boundedMode = simulation::GpuVofBoundedMode::localFixed;
    auto tolerant = strict; tolerant.allowNumericalBoundSlack = true; tolerant.numericalBoundTolerance = 5.0e-6f;
    auto tolerantLocal = tolerant; tolerantLocal.boundedMode = simulation::GpuVofBoundedMode::localFixed;
    bool baselineStalled = false, boundedStalled = false;
    bool tolerantBaselineStalled = false, tolerantBoundedStalled = false;
    std::size_t observed = 0, baselineRejects = 0, boundedRejects = 0, limitedFaces = 0;
    std::size_t tolerantBaselineRejects = 0, tolerantBoundedRejects = 0;
    std::size_t tolerantLimitedFaces = 0;
    double baselineTime = 0.0, boundedTime = 0.0, tolerantBaselineTime = 0.0, tolerantBoundedTime = 0.0;
    double fluxWeakening = 0.0, rawFlux = 0.0;
    double tolerantFluxWeakening = 0.0, tolerantRawFlux = 0.0;
    float strictMinimum = 0.0f, strictMaximum = 1.0f;
    double strictResidual = 0.0;
    simulation::GpuDynamicFreeSurfaceSettings settings;
    settings.maximumSubsteps = 32;
    settings.acceptedSubstepValidationObserver = [&](const MacGrid& snapshot, float dt) {
        const bool xFirst = (observed & 1u) == 0u;
        if (!baselineStalled) {
            const auto report = baseline.computeCandidate(snapshot, dt, xFirst, strict);
            if (report.valid) { baseline.acceptCandidate(); baselineTime += dt; }
            else {
                strictMinimum = report.minimumFraction; strictMaximum = report.maximumFraction;
                strictResidual = report.conservationResidual;
                baseline.rejectCandidate(); baselineStalled = true; ++baselineRejects;
            }
        }
        if (!boundedStalled) {
            const auto report = bounded.computeCandidate(snapshot, dt, xFirst, local);
            const auto candidate = bounded.readCandidate();
            const double raw = absoluteFlux(candidate.rawHorizontalFlux) + absoluteFlux(candidate.rawVerticalFlux);
            const double final = absoluteFlux(candidate.horizontalFlux) + absoluteFlux(candidate.verticalFlux);
            rawFlux += raw; fluxWeakening += raw - final;
            limitedFaces += report.uniqueLimitedFaceCount;
            if (report.valid) { bounded.acceptCandidate(); boundedTime += dt; }
            else { bounded.rejectCandidate(); boundedStalled = true; ++boundedRejects; }
        }
        if (!tolerantBaselineStalled) {
            const auto report = tolerantBaseline.computeCandidate(snapshot, dt, xFirst, tolerant);
            if (report.valid) { tolerantBaseline.acceptCandidate(); tolerantBaselineTime += dt; }
            else { tolerantBaseline.rejectCandidate(); tolerantBaselineStalled = true; ++tolerantBaselineRejects; }
        }
        if (!tolerantBoundedStalled) {
            const auto report = tolerantBounded.computeCandidate(snapshot, dt, xFirst, tolerantLocal);
            const auto candidate = tolerantBounded.readCandidate();
            const double raw = absoluteFlux(candidate.rawHorizontalFlux) +
                absoluteFlux(candidate.rawVerticalFlux);
            const double final = absoluteFlux(candidate.horizontalFlux) +
                absoluteFlux(candidate.verticalFlux);
            tolerantRawFlux += raw;
            tolerantFluxWeakening += raw - final;
            tolerantLimitedFaces += report.uniqueLimitedFaceCount;
            if (report.valid) { tolerantBounded.acceptCandidate(); tolerantBoundedTime += dt; }
            else { tolerantBounded.rejectCandidate(); tolerantBoundedStalled = true; ++tolerantBoundedRejects; }
        }
        ++observed;
    };
    simulation::GpuDynamicFreeSurfaceStepper stepper(shaders / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles);
    std::size_t formalAccepted = 0;
    for (std::size_t outer = 0; outer < 60 && stepper.healthy(); ++outer) formalAccepted += stepper.step().substeps;
    const auto baseMetrics = metrics(baseline.readCommitted().fractions, 48, 32, 1.0 / 48.0, 1.0 / 32.0);
    const auto boundedMetrics = metrics(bounded.readCommitted().fractions, 48, 32, 1.0 / 48.0, 1.0 / 32.0);
    const auto tolerantMetrics = metrics(tolerantBaseline.readCommitted().fractions,
        48, 32, 1.0 / 48.0, 1.0 / 32.0);
    const auto tolerantBoundedMetrics = metrics(tolerantBounded.readCommitted().fractions,
        48, 32, 1.0 / 48.0, 1.0 / 32.0);
    const bool pass = topology.topologyValid && stepper.healthy() && observed == formalAccepted;
    output << "formalAccepted=" << formalAccepted << " observed=" << observed
           << " baselineTime=" << std::fixed << std::setprecision(6) << baselineTime
           << " baselineRejects=" << baselineRejects << " baselineArea=" << baseMetrics.area
           << " strictFailureRange=" << std::scientific << std::setprecision(9)
           << strictMinimum << '/' << strictMaximum
           << " strictResidual=" << strictResidual << std::fixed << std::setprecision(6)
           << " boundedTime=" << boundedTime << " boundedRejects=" << boundedRejects
           << " boundedArea=" << boundedMetrics.area << " limitedFaces=" << limitedFaces
           << " fluxWeakening=" << (rawFlux > 0.0 ? fluxWeakening / rawFlux : 0.0)
           << " tolerantBaselineTime=" << tolerantBaselineTime
           << " tolerantBaselineRejects=" << tolerantBaselineRejects
           << " tolerantBaselineArea=" << tolerantMetrics.area
           << " tolerantBoundedTime=" << tolerantBoundedTime
           << " tolerantBoundedRejects=" << tolerantBoundedRejects
           << " tolerantBoundedArea=" << tolerantBoundedMetrics.area
           << " tolerantLimitedFaces=" << tolerantLimitedFaces
           << " tolerantFluxWeakening=" << std::scientific << (tolerantRawFlux > 0.0
               ? tolerantFluxWeakening / tolerantRawFlux : 0.0) << std::fixed
           << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuVofVelocityCompatibility(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF interface-band velocity compatibility audit, 48x32\n";
    MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
    simulation::initializeDamBreak(grid);
    simulation::MarkerParticleSystem particles; particles.seedFluidCellsExact(grid, 8);
    VolumeFractionField field(48, 32, 1.0 / 48.0, 1.0 / 32.0);
    field.initializeFromSignedDistance([](double x, double y) {
        const double qx = std::abs(x - 0.177083333333) - 0.15625;
        const double qy = std::abs(y - 0.34375) - 0.3125;
        return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
            std::min(std::max(qx, qy), 0.0);
    }, 32);
    field.applyWholeCellSolidMask(grid);
    GpuVolumeFraction gpu(shaders); gpu.upload(field);
    double fluidSquared = 0.0, mixedSquared = 0.0, airSquared = 0.0;
    double fluidMax = 0.0, mixedMax = 0.0, airMax = 0.0;
    std::size_t fluidCount = 0, mixedCount = 0, airCount = 0, observed = 0;
    simulation::GpuDynamicFreeSurfaceSettings settings; settings.maximumSubsteps = 32;
    settings.acceptedSubstepValidationObserver = [&](const MacGrid& snapshot, float dt) {
        const auto fractions = gpu.readCommitted().fractions;
        for (std::size_t y = 0; y < snapshot.height(); ++y) for (std::size_t x = 0; x < snapshot.width(); ++x) {
            const double divergence =
                (snapshot.horizontalVelocity(x + 1, y) - snapshot.horizontalVelocity(x, y)) / snapshot.cellWidth() +
                (snapshot.verticalVelocity(x, y + 1) - snapshot.verticalVelocity(x, y)) / snapshot.cellHeight();
            const double absolute = std::abs(divergence);
            const float f = fractions[y * snapshot.width() + x];
            if (f > 1.0e-4f && f < 1.0f - 1.0e-4f) {
                mixedSquared += divergence * divergence; mixedMax = std::max(mixedMax, absolute); ++mixedCount;
            } else if (snapshot.cellType(x, y) == simulation::CellType::fluid) {
                fluidSquared += divergence * divergence; fluidMax = std::max(fluidMax, absolute); ++fluidCount;
            } else {
                airSquared += divergence * divergence; airMax = std::max(airMax, absolute); ++airCount;
            }
        }
        const auto report = gpu.computeCandidate(snapshot, dt, (observed & 1u) == 0u);
        if (report.valid) gpu.acceptCandidate(); else gpu.rejectCandidate();
        ++observed;
    };
    simulation::GpuDynamicFreeSurfaceStepper stepper(shaders / "mac_divergence.comp", settings);
    const auto topology = stepper.initialize(grid, particles);
    for (std::size_t outer = 0; outer < 12 && stepper.healthy(); ++outer)
        static_cast<void>(stepper.step());
    const auto rms = [](double squared, std::size_t count) {
        return count == 0 ? 0.0 : std::sqrt(squared / static_cast<double>(count));
    };
    const bool pass = topology.topologyValid && stepper.healthy() && observed != 0;
    output << "samples=" << observed
           << " fluidRmsMax=" << std::scientific << rms(fluidSquared, fluidCount) << '/' << fluidMax
           << " mixedRmsMax=" << rms(mixedSquared, mixedCount) << '/' << mixedMax
           << " airRmsMax=" << rms(airSquared, airCount) << '/' << airMax
           << " counts=" << fluidCount << '/' << mixedCount << '/' << airCount
           << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuVofMultistepBenchmark(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF baseline/local-bounded multistep benchmark\n";
    for (const std::size_t size : {64u, 128u, 192u, 256u}) {
        auto field = makeCircle(size, size);
        auto velocity = makeTranslation(field, 0.1f, 0.05f);
        for (const auto mode : {simulation::GpuVofBoundedMode::none,
                                simulation::GpuVofBoundedMode::localFixed}) {
            GpuVolumeFraction gpu(shaders); gpu.upload(field);
            simulation::GpuVofCandidateSettings settings; settings.boundedMode = mode;
            std::vector<double> wallSamples, gpuSamples;
            constexpr std::size_t samples = 15;
            for (std::size_t sample = 0; sample < samples; ++sample) {
                const auto start = std::chrono::steady_clock::now();
                const auto report = gpu.computeCandidate(velocity, 0.001, (sample & 1u) == 0u, settings);
                wallSamples.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
                gpuSamples.push_back(std::accumulate(
                    report.gpuPhaseMilliseconds.begin(), report.gpuPhaseMilliseconds.end(), 0.0));
                gpu.rejectCandidate();
            }
            std::sort(wallSamples.begin(), wallSamples.end());
            std::sort(gpuSamples.begin(), gpuSamples.end());
            output << "grid=" << size << 'x' << size
                   << " mode=" << (mode == simulation::GpuVofBoundedMode::none ? "baseline" : "local")
                   << " samples=" << samples << " wallMedianMs=" << std::fixed << std::setprecision(4)
                   << wallSamples[samples / 2] << " gpuMedianMs=" << gpuSamples[samples / 2]
                   << " dispatch=" << (mode == simulation::GpuVofBoundedMode::none ? 8 : 24)
                   << " memoryKiB=" << gpu.memoryBytes() / 1024.0 << '\n';
        }
    }
    return 0;
}

namespace {

VolumeFractionField makeDamVof(MacGrid& grid, bool exactCellInitialization) {
    VolumeFractionField field(grid.width(), grid.height(), grid.cellWidth(), grid.cellHeight());
    const std::size_t endX = 1 + (grid.width() - 2) / 3;
    const std::size_t endY = 1 + 2 * (grid.height() - 2) / 3;
    if (exactCellInitialization) {
        for (std::size_t y = 0; y < grid.height(); ++y) {
            for (std::size_t x = 0; x < grid.width(); ++x) {
                const bool liquid = x >= 1 && x < endX && y >= 1 && y < endY;
                field.setFraction(x, y, liquid ? 1.0 : 0.0);
            }
        }
    } else {
        const auto centerSampled = simulation::sampleAnalyticLevelSet(
            grid.width(), grid.height(), grid.cellWidth(), grid.cellHeight(),
            [=, dx = grid.cellWidth(), dy = grid.cellHeight()](double x, double y) {
            const double minX = dx, minY = dy, maxX = endX * dx, maxY = endY * dy;
            const double qx = std::abs(x - 0.5 * (minX + maxX)) - 0.5 * (maxX - minX);
            const double qy = std::abs(y - 0.5 * (minY + maxY)) - 0.5 * (maxY - minY);
            return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
                std::min(std::max(qx, qy), 0.0);
        });
        field.initializeFromSignedDistance(
            [&](double x, double y) { return centerSampled.sample(x, y); }, 32);
    }
    field.applyWholeCellSolidMask(grid);
    return field;
}

struct VelocityCorrectionReport final {
    std::size_t modifiedFaces = 0;
    std::size_t constrainedCells = 0;
    double rmsCorrection = 0.0;
    double maximumCorrection = 0.0;
    double divergenceBefore = 0.0;
    double divergenceAfter = 0.0;
};

double selectedDivergenceRms(
    const MacGrid& velocity, const std::vector<float>& fractions, std::size_t& count) {
    double squared = 0.0;
    count = 0;
    for (std::size_t y = 0; y < velocity.height(); ++y) {
        for (std::size_t x = 0; x < velocity.width(); ++x) {
            const float f = fractions[y * velocity.width() + x];
            if (f <= 1.0e-6f || velocity.cellType(x, y) == simulation::CellType::solid) continue;
            const double divergence =
                (velocity.horizontalVelocity(x + 1, y) - velocity.horizontalVelocity(x, y)) /
                    velocity.cellWidth() +
                (velocity.verticalVelocity(x, y + 1) - velocity.verticalVelocity(x, y)) /
                    velocity.cellHeight();
            squared += divergence * divergence;
            ++count;
        }
    }
    return count == 0 ? 0.0 : std::sqrt(squared / static_cast<double>(count));
}

MacGrid makeVofCompatibleVelocity(
    const MacGrid& source, const std::vector<float>& fractions, VelocityCorrectionReport& report) {
    MacGrid result = source;
    const std::size_t width = source.width(), height = source.height();
    std::size_t count = 0;
    report.divergenceBefore = selectedDivergenceRms(source, fractions, count);
    report.constrainedCells = count;
    std::vector<double> du((width + 1) * height), dv(width * (height + 1));
    std::vector<unsigned int> wu(du.size()), wv(dv.size());
    for (std::size_t iteration = 0; iteration < 12; ++iteration) {
        std::fill(du.begin(), du.end(), 0.0); std::fill(dv.begin(), dv.end(), 0.0);
        std::fill(wu.begin(), wu.end(), 0u); std::fill(wv.begin(), wv.end(), 0u);
        for (std::size_t y = 0; y < height; ++y) for (std::size_t x = 0; x < width; ++x) {
            if (fractions[y * width + x] <= 1.0e-6f ||
                source.cellType(x, y) == simulation::CellType::solid) continue;
            const double divergence =
                (result.horizontalVelocity(x + 1, y) - result.horizontalVelocity(x, y)) /
                    result.cellWidth() +
                (result.verticalVelocity(x, y + 1) - result.verticalVelocity(x, y)) /
                    result.cellHeight();
            struct Face { bool horizontal; std::size_t index; double coefficient; };
            std::array<Face, 4> faces{}; std::size_t faceCount = 0; double norm = 0.0;
            const auto addU = [&](std::size_t fx, double coefficient) {
                if (fx == 0 || fx == width) return;
                const auto left = source.cellType(fx - 1, y), right = source.cellType(fx, y);
                if (left == simulation::CellType::solid || right == simulation::CellType::solid ||
                    (left == simulation::CellType::fluid && right == simulation::CellType::fluid)) return;
                faces[faceCount++] = {true, y * (width + 1) + fx, coefficient};
                norm += coefficient * coefficient;
            };
            const auto addV = [&](std::size_t fy, double coefficient) {
                if (fy == 0 || fy == height) return;
                const auto below = source.cellType(x, fy - 1), above = source.cellType(x, fy);
                if (below == simulation::CellType::solid || above == simulation::CellType::solid ||
                    (below == simulation::CellType::fluid && above == simulation::CellType::fluid)) return;
                faces[faceCount++] = {false, fy * width + x, coefficient};
                norm += coefficient * coefficient;
            };
            addU(x, -1.0 / result.cellWidth()); addU(x + 1, 1.0 / result.cellWidth());
            addV(y, -1.0 / result.cellHeight()); addV(y + 1, 1.0 / result.cellHeight());
            if (faceCount == 0 || norm == 0.0) continue;
            for (std::size_t i = 0; i < faceCount; ++i) {
                const double correction = -divergence * faces[i].coefficient / norm;
                if (faces[i].horizontal) { du[faces[i].index] += correction; ++wu[faces[i].index]; }
                else { dv[faces[i].index] += correction; ++wv[faces[i].index]; }
            }
        }
        for (std::size_t y = 0; y < height; ++y) for (std::size_t x = 1; x < width; ++x) {
            const std::size_t index = y * (width + 1) + x;
            if (wu[index] != 0) result.horizontalVelocity(x, y) +=
                static_cast<float>(du[index] / wu[index]);
        }
        for (std::size_t y = 1; y < height; ++y) for (std::size_t x = 0; x < width; ++x) {
            const std::size_t index = y * width + x;
            if (wv[index] != 0) result.verticalVelocity(x, y) +=
                static_cast<float>(dv[index] / wv[index]);
        }
        result.enforceSolidBoundaries(); result.enforceClosedBoundary();
    }
    double correctionSquared = 0.0;
    for (std::size_t i = 0; i < du.size(); ++i) {
        const double difference = result.horizontalVelocities()[i] - source.horizontalVelocities()[i];
        if (difference != 0.0) { ++report.modifiedFaces; correctionSquared += difference * difference;
            report.maximumCorrection = std::max(report.maximumCorrection, std::abs(difference)); }
    }
    for (std::size_t i = 0; i < dv.size(); ++i) {
        const double difference = result.verticalVelocities()[i] - source.verticalVelocities()[i];
        if (difference != 0.0) { ++report.modifiedFaces; correctionSquared += difference * difference;
            report.maximumCorrection = std::max(report.maximumCorrection, std::abs(difference)); }
    }
    report.rmsCorrection = report.modifiedFaces == 0 ? 0.0 :
        std::sqrt(correctionSquared / static_cast<double>(report.modifiedFaces));
    report.divergenceAfter = selectedDivergenceRms(result, fractions, count);
    return result;
}

struct Track final {
    const char* name = "";
    GpuVolumeFraction gpu;
    simulation::GpuVofCandidateSettings settings;
    bool stalled = false;
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    std::size_t repairs = 0;
    std::size_t limitedFaces = 0;
    double time = 0.0;
    double repairVolume = 0.0;
    std::size_t correctedVelocityFaces = 0;
    double velocityCorrectionSquared = 0.0;
    explicit Track(const char* label, const std::filesystem::path& shaders)
        : name(label), gpu(shaders) {}
};

} // namespace

int runGpuVofRoundoffAudit(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF roundoff and initialization audit\n";
    MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f); simulation::initializeDamBreak(grid);
    auto sampled = makeDamVof(grid, false); auto exact = makeDamVof(grid, true);
    output << std::setprecision(10) << "init sampledArea=" << sampled.totalArea()
           << " exactCellArea=" << exact.totalArea() << " analyticArea=" << 0.1953125
           << " delta=" << exact.totalArea() - sampled.totalArea() << '\n';
    auto velocity = makeDiscreteSolenoidal(exact);
    GpuVolumeFraction fp32(shaders), fp64(shaders); fp32.upload(exact); fp64.upload(exact);
    simulation::GpuVofCandidateSettings high; high.updatePrecision = simulation::GpuVofUpdatePrecision::fp64Local;
    const auto a = fp32.computeCandidate(velocity, 0.005, true);
    const auto b = fp64.computeCandidate(velocity, 0.005, true, high);
    const auto fp32Candidate = fp32.readCandidate();
    const auto fp64Candidate = fp64.readCandidate();
    const std::vector<double> fp32Values(
        fp32Candidate.fractions.begin(), fp32Candidate.fractions.end());
    const Error difference = compare(fp64Candidate.fractions, fp32Values);
    const double ulpOne = std::nextafter(1.0f, std::numeric_limits<float>::infinity()) - 1.0f;
    const bool algorithmicOvershoot = a.maximumFraction - 1.0f > 100.0f * ulpOne;
    const bool pass = std::abs(exact.totalArea() - 0.1953125) < 1.0e-7 &&
        std::isfinite(a.conservationResidual) && std::isfinite(b.conservationResidual);
    output << std::scientific << "floatUlpAtOne=" << ulpOne
           << " fp32Range=" << a.minimumFraction << '/' << a.maximumFraction
           << " fp64LocalRange=" << b.minimumFraction << '/' << b.maximumFraction
           << " fp32Fp64RmsMax=" << difference.rms << '/' << difference.maximum
           << " algorithmicOvershoot=" << algorithmicOvershoot
           << " fp32Valid=" << a.valid << " fp64LocalValid=" << b.valid
           << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuVofConservativeRepairValidation(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF local pairwise conservative repair validation\n";
    bool passed = true;
    for (const bool xFirst : {true, false}) {
        auto field = makeCircle(40, 30); auto velocity = makeDiscreteSolenoidal(field);
        GpuVolumeFraction baseline(shaders), repaired(shaders); baseline.upload(field); repaired.upload(field);
        simulation::GpuVofCandidateSettings settings;
        settings.repairMode = simulation::GpuVofRepairMode::localPairwise;
        const auto raw = baseline.computeCandidate(velocity, 0.004, xFirst);
        const auto fixed = repaired.computeCandidate(velocity, 0.004, xFirst, settings);
        const bool casePass = fixed.nonFiniteCount == 0 &&
            std::abs(fixed.conservationResidual) < 5.0e-6 &&
            (!fixed.valid || (fixed.minimumFraction >= 0.0f && fixed.maximumFraction <= 1.0f));
        passed = passed && casePass;
        output << "order=" << (xFirst ? "xy" : "yx") << " rawValid=" << raw.valid
               << " repairedValid=" << fixed.valid << " events=" << fixed.repairEventCount
               << " transfer=" << std::scientific << fixed.repairAbsoluteVolume
               << " range=" << fixed.minimumFraction << '/' << fixed.maximumFraction
               << " residual=" << fixed.conservationResidual
               << " result=" << (casePass ? "PASS" : "FAIL") << '\n';
    }
    return passed ? 0 : 1;
}

int runGpuVofCommitSafetyValidation(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF strict committed-state safety validation\n";
    auto field = makeCircle(40, 30); auto velocity = makeTranslation(field, 0.08f, 0.02f);
    GpuVolumeFraction gpu(shaders); gpu.upload(field);
    const auto original = gpu.readCommitted(); const auto version = gpu.version();
    simulation::GpuVofCandidateSettings settings;
    settings.updatePrecision = simulation::GpuVofUpdatePrecision::fp64Local;
    settings.repairMode = simulation::GpuVofRepairMode::localPairwise;
    const auto valid = gpu.computeCandidate(velocity, 0.001, true, settings);
    if (valid.valid) gpu.acceptCandidate(); else gpu.rejectCandidate();
    const auto committed = gpu.readCommitted();
    const bool legal = std::all_of(committed.fractions.begin(), committed.fractions.end(),
        [](float f) { return std::isfinite(f) && f >= 0.0f && f <= 1.0f; });
    auto invalidVelocity = velocity; invalidVelocity.horizontalVelocity(20, 15) =
        std::numeric_limits<float>::quiet_NaN();
    const auto beforeFailure = gpu.readCommitted(); const auto failureVersion = gpu.version();
    const auto failed = gpu.computeCandidate(invalidVelocity, 0.001, false, settings); gpu.rejectCandidate();
    const bool rollback = !failed.valid && gpu.readCommitted().fractions == beforeFailure.fractions &&
        gpu.version() == failureVersion;
    const bool pass = legal && rollback && (valid.valid ? gpu.version() == version + 1 :
        committed.fractions == original.fractions);
    output << "candidateValid=" << valid.valid << " legalCommitted=" << legal
           << " repairEvents=" << valid.repairEventCount << " failedRejected=" << !failed.valid
           << " rollback=" << rollback << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuVofBandVelocityValidation(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF-only interface-band velocity compatibility validation\n";
    auto field = makeCircle(80, 60); auto velocity = makeDiscreteSolenoidal(field);
    for (std::size_t y = 0; y < field.height(); ++y) for (std::size_t x = 0; x < field.width(); ++x)
        velocity.setCellType(x, y, field.fraction(x, y) > 0.5 ? simulation::CellType::fluid : simulation::CellType::air);
    VelocityCorrectionReport compatibleReport;
    std::vector<float> fractions(field.fractions().size());
    std::transform(field.fractions().begin(), field.fractions().end(), fractions.begin(),
        [](double value) { return static_cast<float>(value); });
    const auto compatible = makeVofCompatibleVelocity(velocity, fractions, compatibleReport);
    auto divergent = velocity;
    for (std::size_t y = 1; y + 1 < divergent.height(); ++y)
        for (std::size_t x = 1; x < divergent.width(); ++x)
            if (divergent.cellType(x, y) == simulation::CellType::air)
                divergent.horizontalVelocity(x, y) += 0.015f *
                    static_cast<float>(static_cast<int>((x + 2 * y) % 3) - 1);
    VelocityCorrectionReport divergentReport;
    const auto corrected = makeVofCompatibleVelocity(divergent, fractions, divergentReport);
    GpuVolumeFraction originalGpu(shaders), correctedGpu(shaders);
    originalGpu.upload(field); correctedGpu.upload(field);
    simulation::GpuVofCandidateSettings settings; settings.allowNumericalBoundSlack = true;
    settings.numericalBoundTolerance = 1.0e-6f;
    const auto original = originalGpu.computeCandidate(divergent, 0.001, true, settings);
    const auto adjusted = correctedGpu.computeCandidate(corrected, 0.001, true, settings);
    const bool noRegression = compatibleReport.maximumCorrection < 2.0e-6;
    const bool reduced = divergentReport.divergenceAfter <= divergentReport.divergenceBefore;
    struct Region final { double squared = 0.0; double maximum = 0.0; std::size_t count = 0; };
    std::array<Region, 7> regions{};
    MacGrid realGrid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f); simulation::initializeDamBreak(realGrid);
    simulation::MarkerParticleSystem particles; particles.seedFluidCellsExact(realGrid, 8);
    auto realField = makeDamVof(realGrid, true); GpuVolumeFraction realVof(shaders); realVof.upload(realField);
    std::size_t realSnapshots = 0, correctedFaces = 0; double realBefore = 0.0, realAfter = 0.0;
    simulation::GpuDynamicFreeSurfaceSettings stepSettings; stepSettings.maximumSubsteps = 32;
    stepSettings.acceptedSubstepValidationObserver = [&](const MacGrid& snapshot, float dt) {
        const auto state = realVof.readCommitted();
        for (std::size_t y = 0; y < snapshot.height(); ++y) for (std::size_t x = 0; x < snapshot.width(); ++x) {
            const double divergence =
                (snapshot.horizontalVelocity(x + 1, y) - snapshot.horizontalVelocity(x, y)) /
                    snapshot.cellWidth() +
                (snapshot.verticalVelocity(x, y + 1) - snapshot.verticalVelocity(x, y)) /
                    snapshot.cellHeight();
            const float f = state.fractions[y * snapshot.width() + x];
            const auto type = snapshot.cellType(x, y);
            bool airNeighbor = false, solidNeighbor = false, liquidNeighbor = false;
            for (const auto [ox, oy] : std::array<std::pair<int, int>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}) {
                const int nx = static_cast<int>(x) + ox, ny = static_cast<int>(y) + oy;
                if (nx < 0 || ny < 0 || nx >= static_cast<int>(snapshot.width()) ||
                    ny >= static_cast<int>(snapshot.height())) { solidNeighbor = true; continue; }
                const auto neighbor = snapshot.cellType(nx, ny);
                airNeighbor = airNeighbor || neighbor == simulation::CellType::air;
                solidNeighbor = solidNeighbor || neighbor == simulation::CellType::solid;
                liquidNeighbor = liquidNeighbor || state.fractions[static_cast<std::size_t>(ny) *
                    snapshot.width() + static_cast<std::size_t>(nx)] > 1.0e-6f;
            }
            const auto add = [&](std::size_t index) {
                regions[index].squared += divergence * divergence;
                regions[index].maximum = std::max(regions[index].maximum, std::abs(divergence));
                ++regions[index].count;
            };
            if (type == simulation::CellType::fluid && !airNeighbor && !solidNeighbor) add(0);
            if (type == simulation::CellType::fluid && airNeighbor) add(1);
            if (type == simulation::CellType::fluid && solidNeighbor) add(2);
            if (f > 1.0e-4f && f < 1.0f - 1.0e-4f) add(3);
            if (f > 0.0f && f <= 1.0e-4f) add(4);
            if (type == simulation::CellType::air && liquidNeighbor) add(5);
            if (type == simulation::CellType::air && !liquidNeighbor) add(6);
        }
        VelocityCorrectionReport correction;
        static_cast<void>(makeVofCompatibleVelocity(snapshot, state.fractions, correction));
        correctedFaces += correction.modifiedFaces; realBefore += correction.divergenceBefore;
        realAfter += correction.divergenceAfter;
        simulation::GpuVofCandidateSettings tolerant; tolerant.allowNumericalBoundSlack = true;
        tolerant.numericalBoundTolerance = 1.0e-6f;
        const auto report = realVof.computeCandidate(snapshot, dt, (realSnapshots & 1u) == 0u, tolerant);
        if (report.valid) realVof.acceptCandidate(); else realVof.rejectCandidate();
        ++realSnapshots;
    };
    simulation::GpuDynamicFreeSurfaceStepper stepper(shaders / "mac_divergence.comp", stepSettings);
    const auto topology = stepper.initialize(realGrid, particles);
    for (std::size_t outer = 0; outer < 12 && stepper.healthy(); ++outer) static_cast<void>(stepper.step());
    const std::array<const char*, 7> names{
        "fluidInterior", "fluidAir", "fluidSolid", "vofMixed", "vofLowTail", "airBand", "airFar"};
    for (std::size_t i = 0; i < regions.size(); ++i) {
        const double rms = regions[i].count == 0 ? 0.0 :
            std::sqrt(regions[i].squared / static_cast<double>(regions[i].count));
        output << "region=" << names[i] << " samples=" << regions[i].count
               << " divergenceRmsMax=" << rms << '/' << regions[i].maximum << '\n';
    }
    const bool pass = noRegression && reduced && original.nonFiniteCount == 0 && adjusted.nonFiniteCount == 0 &&
        topology.topologyValid && stepper.healthy() && realSnapshots != 0;
    output << std::scientific << "validFieldModified=" << compatibleReport.modifiedFaces
           << " validFieldMaxCorrection=" << compatibleReport.maximumCorrection
           << " divergentRmsBeforeAfter=" << divergentReport.divergenceBefore << '/'
           << divergentReport.divergenceAfter << " modifiedFaces=" << divergentReport.modifiedFaces
           << " correctionRmsMax=" << divergentReport.rmsCorrection << '/' << divergentReport.maximumCorrection
           << " candidateValid=" << original.valid << '/' << adjusted.valid
           << " realSnapshots=" << realSnapshots << " realCorrectedFaces=" << correctedFaces
           << " realSelectedRmsBeforeAfter=" << (realBefore / realSnapshots) << '/'
           << (realAfter / realSnapshots)
           << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuVofLongValidation(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF strict long analytic validation\n";
    bool passed = true;
    for (const auto [width, height] : std::array<std::pair<std::size_t, std::size_t>, 3>{
             {{40, 30}, {80, 60}, {160, 120}}}) {
        for (const char* test : {"translate", "rotate", "reverse", "two-drops"}) {
            VolumeFractionField field(width, height, 1.0 / width, 0.75 / height);
            if (std::string_view(test) == "two-drops") field.initializeFromSignedDistance([](double x, double y) {
                return std::min(std::hypot(x - 0.32, y - 0.36) - 0.08,
                    std::hypot(x - 0.57, y - 0.36) - 0.065);
            }, 24); else field = makeCircle(width, height);
            const auto initial = field.fractions(); GpuVolumeFraction gpu(shaders); gpu.upload(field);
            simulation::GpuVofCandidateSettings settings;
            settings.updatePrecision = simulation::GpuVofUpdatePrecision::fp64Local;
            settings.repairMode = simulation::GpuVofRepairMode::localPairwise;
            std::size_t accepted = 0, repairs = 0; constexpr std::size_t steps = 120;
            for (std::size_t step = 0; step < steps; ++step) {
                MacGrid velocity = std::string_view(test) == "rotate" ? makeRotation(field, 0.35f) :
                    std::string_view(test) == "reverse" ? makeDiscreteSolenoidal(field) :
                    std::string_view(test) == "two-drops" ? makeTranslation(field, 0.0f, 0.0f) :
                    makeTranslation(field, 0.05f, 0.02f);
                if (std::string_view(test) == "reverse" && step >= steps / 2) scaleVelocity(velocity, -1.0f);
                const auto report = gpu.computeCandidate(velocity, 0.001, (step & 1u) == 0u, settings);
                repairs += report.repairEventCount;
                if (!report.valid) { gpu.rejectCandidate(); break; }
                gpu.acceptCandidate(); ++accepted;
            }
            const auto snapshot = gpu.readCommitted(); const auto finalMetrics = metrics(
                snapshot.fractions, width, height, field.cellWidth(), field.cellHeight());
            const double elapsed = accepted * 0.001;
            VolumeFractionField analytic(width, height, field.cellWidth(), field.cellHeight());
            if (std::string_view(test) == "translate") {
                analytic.initializeFromSignedDistance([=](double x, double y) {
                    return std::hypot(x - (0.38 + 0.05 * elapsed),
                        y - (0.36 + 0.02 * elapsed)) - 0.19;
                }, 24);
            } else if (std::string_view(test) == "rotate") {
                const double angle = 0.35 * elapsed;
                const double ox = 0.38 - 0.5, oy = 0.36 - 0.375;
                const double cx = 0.5 + std::cos(angle) * ox - std::sin(angle) * oy;
                const double cy = 0.375 + std::sin(angle) * ox + std::cos(angle) * oy;
                analytic.initializeFromSignedDistance([=](double x, double y) {
                    return std::hypot(x - cx, y - cy) - 0.19;
                }, 24);
            } else {
                for (std::size_t i = 0; i < initial.size(); ++i)
                    analytic.setFraction(i % width, i / width, initial[i]);
            }
            double analyticL1 = 0.0;
            for (std::size_t i = 0; i < initial.size(); ++i)
                analyticL1 += std::abs(snapshot.fractions[i] - analytic.fractions()[i]) *
                    field.cellArea(i % width, i / width);
            const bool casePass = accepted == steps && std::abs(finalMetrics.area - field.totalArea()) < 2.0e-5;
            passed = passed && casePass;
            output << "case=" << test << " grid=" << width << 'x' << height
                   << " accepted=" << accepted << '/' << steps << " time=" << accepted * 0.001
                   << " areaError=" << std::scientific << finalMetrics.area - field.totalArea()
                   << " analyticL1=" << analyticL1 << " repairs=" << repairs
                   << " centroid=" << finalMetrics.centerX << ',' << finalMetrics.centerY
                   << " moment=" << finalMetrics.secondMoment
                   << " components=" << finalMetrics.components << " lowTail=" << finalMetrics.lowTail
                   << " result=" << (casePass ? "PASS" : "FAIL") << '\n';
        }
    }
    return passed ? 0 : 1;
}

int runGpuVofDynamicCompare(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF strict/tolerant/high-precision/repair dynamic comparison, 48x32\n";
    MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f); simulation::initializeDamBreak(grid);
    simulation::MarkerParticleSystem particles; particles.seedFluidCellsExact(grid, 8);
    auto initial = makeDamVof(grid, true);
    Track strict("strict-fp32", shaders), tolerant("research-slack", shaders),
        high("strict-fp64-update", shaders), limited("strict-limit4", shaders),
        repair("strict-pair-repair", shaders), compatible("repair-compatible-velocity", shaders);
    std::array<Track*, 6> tracks{&strict, &tolerant, &high, &limited, &repair, &compatible};
    for (auto* track : tracks) track->gpu.upload(initial);
    tolerant.settings.allowNumericalBoundSlack = true; tolerant.settings.numericalBoundTolerance = 1.0e-6f;
    high.settings.updatePrecision = simulation::GpuVofUpdatePrecision::fp64Local;
    limited.settings.boundedMode = simulation::GpuVofBoundedMode::localFixed;
    repair.settings.updatePrecision = simulation::GpuVofUpdatePrecision::fp64Local;
    repair.settings.repairMode = simulation::GpuVofRepairMode::localPairwise;
    compatible.settings = repair.settings;
    std::size_t observed = 0;
    simulation::GpuDynamicFreeSurfaceSettings stepSettings; stepSettings.maximumSubsteps = 32;
    stepSettings.acceptedSubstepValidationObserver = [&](const MacGrid& snapshot, float dt) {
        for (auto* track : tracks) {
            if (track->stalled) continue;
            MacGrid adjusted = snapshot;
            const MacGrid* inputVelocity = &snapshot;
            if (track == &compatible) {
                VelocityCorrectionReport correction;
                adjusted = makeVofCompatibleVelocity(
                    snapshot, track->gpu.readCommitted().fractions, correction);
                inputVelocity = &adjusted;
                track->correctedVelocityFaces += correction.modifiedFaces;
                track->velocityCorrectionSquared += correction.rmsCorrection * correction.rmsCorrection;
            }
            const auto report = track->gpu.computeCandidate(
                *inputVelocity, dt, (observed & 1u) == 0u, track->settings);
            track->repairs += report.repairEventCount; track->repairVolume += report.repairAbsoluteVolume;
            track->limitedFaces += report.uniqueLimitedFaceCount;
            if (report.valid) { track->gpu.acceptCandidate(); ++track->accepted; track->time += dt; }
            else { track->gpu.rejectCandidate(); ++track->rejected; track->stalled = true; }
        }
        ++observed;
    };
    simulation::GpuDynamicFreeSurfaceStepper stepper(shaders / "mac_divergence.comp", stepSettings);
    const auto topology = stepper.initialize(grid, particles); std::size_t formalAccepted = 0;
    for (std::size_t outer = 0; outer < 60 && stepper.healthy(); ++outer) formalAccepted += stepper.step().substeps;
    for (auto* track : tracks) {
        const auto state = track->gpu.readCommitted(); const auto m = metrics(state.fractions, 48, 32, 1.0 / 48.0, 1.0 / 32.0);
        output << "mode=" << track->name << " accepted=" << track->accepted
               << " rejected=" << track->rejected << " physicalTime=" << std::fixed << std::setprecision(6)
               << track->time << " area=" << std::setprecision(9) << m.area
               << " limitedFaces=" << track->limitedFaces << " repairs=" << track->repairs
               << " repairVolume=" << std::scientific << track->repairVolume
               << " correctedVelocityFaces=" << track->correctedVelocityFaces
               << " correctionRmsAggregate=" << std::sqrt(track->velocityCorrectionSquared)
               << " components=" << m.components << " lowTail=" << m.lowTail << '\n';
    }
    const bool pass = topology.topologyValid && stepper.healthy() && observed == formalAccepted;
    output << "formalAccepted=" << formalAccepted << " observed=" << observed
           << " initArea=" << initial.totalArea() << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

int runGpuVofSafetyBenchmark(std::ostream& output, const std::filesystem::path& shaders) {
    output << "GPU VOF numerical-safety benchmark\n";
    for (const std::size_t size : {64u, 128u, 192u, 256u}) {
        auto field = makeCircle(size, size); auto velocity = makeTranslation(field, 0.1f, 0.05f);
        for (int mode = 0; mode < 4; ++mode) {
            simulation::GpuVofCandidateSettings settings;
            const char* name = "baseline";
            if (mode == 1) { settings.boundedMode = simulation::GpuVofBoundedMode::localFixed; name = "limit4"; }
            if (mode == 2) { settings.updatePrecision = simulation::GpuVofUpdatePrecision::fp64Local; name = "fp64-update"; }
            if (mode == 3) { settings.updatePrecision = simulation::GpuVofUpdatePrecision::fp64Local;
                settings.repairMode = simulation::GpuVofRepairMode::localPairwise; name = "pair-repair"; }
            GpuVolumeFraction gpu(shaders); gpu.upload(field); std::vector<double> wall, kernel;
            constexpr std::size_t samples = 9;
            for (std::size_t sample = 0; sample < samples; ++sample) {
                const auto start = std::chrono::steady_clock::now();
                const auto report = gpu.computeCandidate(velocity, 0.001, (sample & 1u) == 0u, settings);
                wall.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
                kernel.push_back(std::accumulate(report.gpuPhaseMilliseconds.begin(),
                    report.gpuPhaseMilliseconds.end(), 0.0)); gpu.rejectCandidate();
            }
            std::sort(wall.begin(), wall.end()); std::sort(kernel.begin(), kernel.end());
            const auto probe = gpu.computeCandidate(velocity, 0.001, true, settings); gpu.rejectCandidate();
            output << "grid=" << size << 'x' << size << " mode=" << name << " samples=" << samples
                   << " wallMedianMs=" << std::fixed << std::setprecision(4) << wall[samples / 2]
                   << " gpuMedianMs=" << kernel[samples / 2] << " dispatch=" << probe.dispatchCount
                   << " memoryKiB=" << probe.memoryBytes / 1024.0 << '\n';
        }
    }
    return 0;
}

} // namespace fluidlab::diagnostics
