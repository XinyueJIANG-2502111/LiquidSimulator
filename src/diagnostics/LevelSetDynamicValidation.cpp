#include "LevelSetDynamicValidation.h"

#include "../simulation/mac/DynamicLevelSet.h"
#include "../simulation/mac/DynamicFreeSurfaceStepper.h"
#include "../simulation/mac/GhostFluidPressureProjection.h"
#include "../simulation/mac/Scenarios.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fluidlab::diagnostics {
namespace {

using Clock = std::chrono::steady_clock;
using fluidlab::simulation::CellType;
using fluidlab::simulation::DynamicLevelSetTracker;
using fluidlab::simulation::GhostFluidLinearSystem;
using fluidlab::simulation::GhostFluidThetaData;
using fluidlab::simulation::LevelSetField;
using fluidlab::simulation::MacGrid;
using fluidlab::simulation::MarkerParticleSystem;
using fluidlab::simulation::ScalarFieldFunction;
using fluidlab::simulation::Vector2;
using Velocity = fluidlab::simulation::VelocityFieldFunction;

constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

[[nodiscard]] double elapsedMilliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void classifyFromField(MacGrid& grid, const LevelSetField& field, bool solidWalls) {
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (solidWalls && (x == 0 || x + 1 == grid.width() || y == 0)) {
                grid.setCellType(x, y, CellType::solid);
            } else {
                grid.setCellType(x, y,
                    field.value(x, y) <= 0.0 ? CellType::fluid : CellType::air);
            }
        }
    }
}

void setMacVelocity(MacGrid& grid, const Velocity& velocity) {
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x <= grid.width(); ++x) {
            grid.horizontalVelocity(x, y) = velocity(
                x * grid.cellWidth(), (y + 0.5) * grid.cellHeight()).x;
        }
    }
    for (std::size_t y = 0; y <= grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            grid.verticalVelocity(x, y) = velocity(
                (x + 0.5) * grid.cellWidth(), y * grid.cellHeight()).y;
        }
    }
}

struct GeometryError final {
    double rms = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
    double areaRelative = 0.0;
};

[[nodiscard]] GeometryError fieldError(
    const LevelSetField& field,
    const ScalarFieldFunction& expected,
    double referenceArea) {
    std::vector<double> errors;
    double squared = 0.0;
    for (std::size_t y = 0; y < field.height; ++y) {
        for (std::size_t x = 0; x < field.width; ++x) {
            const double px = (x + 0.5) * field.cellWidth;
            const double py = (y + 0.5) * field.cellHeight;
            const double expectedValue = expected(px, py);
            if (std::abs(expectedValue) > 2.0 * std::max(field.cellWidth, field.cellHeight)) continue;
            const double error = std::abs(field.value(x, y) - expectedValue);
            errors.push_back(error);
            squared += error * error;
        }
    }
    std::sort(errors.begin(), errors.end());
    GeometryError result;
    result.rms = std::sqrt(squared / std::max<std::size_t>(errors.size(), 1));
    if (!errors.empty()) {
        result.p95 = errors[static_cast<std::size_t>(0.95 * (errors.size() - 1))];
        result.maximum = errors.back();
    }
    const double area = fluidlab::simulation::estimateLevelSetArea(field, 4).area;
    result.areaRelative = std::abs(area - referenceArea) / std::max(referenceArea, 1.0e-12);
    return result;
}

[[nodiscard]] double circleArea(double radius) noexcept { return kPi * radius * radius; }

[[nodiscard]] double ellipseDistance(
    double x, double y, double centerX, double centerY, double radiusX, double radiusY) {
    const double px = std::abs(x - centerX);
    const double py = std::abs(y - centerY);
    if (px < 1.0e-14 && py < 1.0e-14) return -std::min(radiusX, radiusY);
    double angle = std::atan2(py * radiusX, px * radiusY);
    for (int iteration = 0; iteration < 16; ++iteration) {
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const double ex = radiusX * cosine;
        const double ey = radiusY * sine;
        const double derivative =
            (ex - px) * (-radiusX * sine) + (ey - py) * (radiusY * cosine);
        const double secondDerivative = radiusX * radiusX * sine * sine -
            radiusX * (ex - px) * cosine + radiusY * radiusY * cosine * cosine -
            radiusY * (ey - py) * sine;
        if (std::abs(secondDerivative) < 1.0e-14) break;
        angle = std::clamp(angle - derivative / secondDerivative, 0.0, 0.5 * kPi);
    }
    const double boundaryX = radiusX * std::cos(angle);
    const double boundaryY = radiusY * std::sin(angle);
    const double distance = std::hypot(px - boundaryX, py - boundaryY);
    const double normalized = px * px / (radiusX * radiusX) +
        py * py / (radiusY * radiusY);
    return normalized <= 1.0 ? -distance : distance;
}

[[nodiscard]] std::size_t signMismatchCount(
    const LevelSetField& first, const LevelSetField& second) {
    std::size_t mismatch = 0;
    for (std::size_t i = 0; i < first.values.size(); ++i) {
        mismatch += (first.values[i] <= 0.0) != (second.values[i] <= 0.0) ? 1U : 0U;
    }
    return mismatch;
}

void setGravityPrediction(MacGrid& grid, double gravity, double timeStep) {
    for (std::size_t y = 0; y <= grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const bool belowFluid = y > 0 && grid.cellType(x, y - 1) == CellType::fluid;
            const bool aboveFluid = y < grid.height() && grid.cellType(x, y) == CellType::fluid;
            const bool solid = (y > 0 && grid.cellType(x, y - 1) == CellType::solid) ||
                (y < grid.height() && grid.cellType(x, y) == CellType::solid);
            grid.verticalVelocity(x, y) = !solid && (belowFluid || aboveFluid)
                ? static_cast<float>(-gravity * timeStep) : 0.0f;
        }
    }
    grid.enforceSolidBoundaries();
    grid.computeDivergence();
}

struct HydrostaticResult final {
    double pressureError = 0.0;
    double residualRms = 0.0;
    double residualMaximum = 0.0;
    double postDivergence = 0.0;
};

[[nodiscard]] HydrostaticResult solveHydrostatic(
    MacGrid grid,
    GhostFluidThetaData theta,
    double surfaceHeight,
    double density,
    double gravity,
    double timeStep) {
    using namespace fluidlab::simulation;
    setGravityPrediction(grid, gravity, timeStep);
    const GhostFluidLinearSystem system = assembleGhostFluidSystem(
        grid, theta, density, timeStep);
    const GhostFluidSolveResult solve = solveGhostFluidSystemDirect(system);
    require(system.supported && solve.solved, "dynamic ghost-fluid reference solve failed");
    writeGhostFluidPressure(grid, system, solve.pressure);
    HydrostaticResult result;
    result.residualRms = solve.residual.rootMeanSquare;
    result.residualMaximum = solve.residual.maximumAbsolute;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (grid.cellType(x, y) != CellType::fluid) continue;
            const double py = (y + 0.5) * grid.cellHeight();
            const double expected = density * gravity * std::max(surfaceHeight - py, 0.0);
            result.pressureError = std::max(result.pressureError,
                std::abs(static_cast<double>(grid.pressure(x, y)) - expected));
        }
    }
    const float maximumVelocityChange = applyGhostFluidPressureGradient(
        grid, theta, density, timeStep);
    (void)maximumVelocityChange;
    grid.computeDivergence();
    result.postDivergence = grid.divergenceStatistics().maximumAbsolute;
    return result;
}

} // namespace

int runLevelSetRedistanceBenchmark(std::ostream& output) {
    using namespace fluidlab::simulation;
    output << "LiquidSimulator Phase 1.9B-2 level-set redistance benchmark\n"
           << "Grid BandCells OldMs NewMs Speedup Segments Processed InterfaceRmsH "
              "InterfaceMaxH AreaDelta GradMean GradMax Components\n";
    for (const std::size_t width : {64U, 128U, 192U, 256U}) {
        const std::size_t height = width;
        const double h = 1.0 / width;
        const auto circle = [](double x, double y) {
            return std::hypot(x - 0.51, y - 0.49) - 0.235;
        };
        const LevelSetField initial = sampleAnalyticLevelSet(width, height, h, h, circle);
        for (const double bandCells : {3.0, 6.0, 9.0}) {
            LevelSetField oldField = initial;
            auto start = Clock::now();
            const auto oldReport = reinitializeLevelSet(oldField, bandCells * h);
            const double oldMilliseconds = elapsedMilliseconds(start);
            LevelSetField newField = initial;
            start = Clock::now();
            const auto report = redistanceLevelSetNarrowBand(newField,
                {.narrowBandWidth = bandCells * h, .areaSamplesPerAxis = 2});
            const double newMilliseconds = elapsedMilliseconds(start);
            output << width << 'x' << height << ' ' << bandCells << ' '
                   << std::fixed << std::setprecision(3) << oldMilliseconds << ' '
                   << newMilliseconds << ' ' << oldMilliseconds / std::max(newMilliseconds, 1.0e-9)
                   << ' ' << report.segmentCount << ' ' << report.processedCellCount << ' '
                   << std::scientific << report.rmsInterfaceDisplacement / h << ' '
                   << report.maximumInterfaceDisplacement / h << ' '
                   << std::abs(report.areaAfter - report.areaBefore) << ' '
                   << report.meanGradientError << ' ' << report.maximumGradientError << ' '
                   << report.negativeComponentCountBefore << '/'
                   << report.negativeComponentCountAfter << '\n';
            require(report.negativeComponentCountBefore == report.negativeComponentCountAfter,
                "narrow-band redistance changed interface topology");
            require(report.maximumInterfaceDisplacement < 0.35 * h,
                "narrow-band redistance moved the zero contour too far");
            if (width == 256 && bandCells == 6.0) {
                require(newMilliseconds < oldMilliseconds,
                    "new redistance did not improve the 256-square baseline");
                require(oldReport.crossingCount > 0, "old redistance baseline was empty");
            }
        }
    }
    output << "Dynamic geometry pipeline (correction disabled)\n"
           << "Grid Particles AdvectionMs RedistanceMs ConsistencyMs CorrectionMs ThetaMs "
              "AreaMs ValidTheta FallbackTheta ApproxStateKiB\n";
    for (const std::size_t width : {64U, 128U, 192U, 256U}) {
        const std::size_t height = width;
        const double h = 1.0 / width;
        const auto circle = [](double x, double y) {
            return std::hypot(x - 0.48, y - 0.51) - 0.235;
        };
        LevelSetField initial = sampleAnalyticLevelSet(width, height, h, h, circle);
        MacGrid grid(width, height, static_cast<float>(h), static_cast<float>(h));
        classifyFromField(grid, initial, true);
        setMacVelocity(grid, [](double x, double y) {
            return Vector2{static_cast<float>(0.08 * std::sin(kPi * x) * std::cos(kPi * y)),
                static_cast<float>(-0.08 * std::cos(kPi * x) * std::sin(kPi * y))};
        });
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        DynamicLevelSetTracker tracker;
        tracker.initialize(std::move(initial));
        const DynamicLevelSetAdvanceReport report = tracker.prepareCandidate(
            grid, particles, 1.0 / 120.0,
            {.narrowBandWidth = 6.0 * h, .areaSamplesPerAxis = 2},
            {.enabled = false, .interfaceTolerance = 1.25 * h,
             .maximumCorrectionDistance = 1.5 * h, .correctionDepth = 0.25 * h});
        const auto areaStart = Clock::now();
        const LevelSetAreaEstimate area = estimateLevelSetArea(
            tracker.candidate().field, 4);
        const double areaMilliseconds = elapsedMilliseconds(areaStart);
        const std::size_t cells = width * height;
        const std::size_t approximateStateBytes = 2 * cells *
            (sizeof(double) + sizeof(unsigned char)) +
            particles.particles().size() * sizeof(MarkerParticle);
        output << width << 'x' << height << ' ' << particles.validParticleCount() << ' '
               << std::fixed << std::setprecision(3) << report.advectionMilliseconds << ' '
               << report.redistanceMilliseconds << ' ' << report.consistencyMilliseconds << ' '
               << report.correctionMilliseconds << ' ' << report.thetaMilliseconds << ' '
               << areaMilliseconds << ' ' << report.theta.validFaceCount << ' '
               << report.theta.fallbackFaceCount << ' '
               << static_cast<double>(approximateStateBytes) / 1024.0 << '\n';
        require(std::isfinite(area.area) && report.correction.appliedCount == 0,
            "diagnostic geometry pipeline produced invalid data or enabled correction");
    }
    output << "Phase 1.9B-2 redistance benchmark result: PASS\n";
    return 0;
}

int runLevelSetDynamicValidation(std::ostream& output) {
    using namespace fluidlab::simulation;
    output << "LiquidSimulator Phase 1.9B-2 dynamic level-set validation\n"
           << "Grid Case Dt Steps RmsH P95H MaxH AreaRel Components Confidence Versions\n";
    for (const std::size_t width : {40U, 80U, 160U}) {
        const std::size_t height = width * 3 / 4;
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        const double h = std::min(dx, dy);
        for (const double dt : {0.01, 0.005}) {
            const auto initialCircle = [](double x, double y) {
                return std::hypot(x - 0.42, y - 0.48) - 0.17;
            };
            LevelSetField field = sampleAnalyticLevelSet(width, height, dx, dy, initialCircle);
            constexpr double vx = 0.12;
            constexpr double vy = -0.03;
            const std::size_t steps = static_cast<std::size_t>(0.2 / dt);
            for (std::size_t step = 0; step < steps; ++step) {
                field = advectLevelSetRk2(field, [](double, double) {
                    return Vector2{static_cast<float>(vx), static_cast<float>(vy)};
                }, dt);
                const std::size_t redistanceInterval = std::max<std::size_t>(
                    1, static_cast<std::size_t>(std::lround(0.04 / dt)));
                if ((step + 1) % redistanceInterval == 0) {
                    const auto redistance = redistanceLevelSetNarrowBand(field,
                        {.narrowBandWidth = 6.0 * std::max(dx, dy), .areaSamplesPerAxis = 2});
                    (void)redistance;
                }
            }
            const auto expected = [](double x, double y) {
                return std::hypot(x - (0.42 + vx * 0.2), y - (0.48 + vy * 0.2)) - 0.17;
            };
            const GeometryError error = fieldError(field, expected, circleArea(0.17));
            const std::size_t confidence = static_cast<std::size_t>(std::count(
                field.confidence.begin(), field.confidence.end(), static_cast<unsigned char>(1)));
            output << width << 'x' << height << " translation " << dt << ' ' << steps << ' '
                   << std::scientific << error.rms / h << ' ' << error.p95 / h << ' '
                   << error.maximum / h << ' ' << error.areaRelative << " 1 "
                   << confidence << ' ' << field.sourceVersion << '\n';
            require(error.rms / h < 0.35, "translated dynamic level set exceeded its error target");
        }

        const auto offCenter = [](double x, double y) {
            return std::hypot(x - 0.64, y - 0.48) - 0.13;
        };
        LevelSetField rotated = sampleAnalyticLevelSet(width, height, dx, dy, offCenter);
        constexpr double omega = 0.8;
        constexpr double rotationDt = 0.004;
        constexpr std::size_t rotationSteps = 100;
        for (std::size_t step = 0; step < rotationSteps; ++step) {
            rotated = advectLevelSetRk2(rotated, [](double x, double y) {
                return Vector2{static_cast<float>(-omega * (y - 0.5)),
                    static_cast<float>(omega * (x - 0.5))};
            }, rotationDt);
            if ((step + 1) % 5 == 0) {
                const auto redistance = redistanceLevelSetNarrowBand(rotated,
                    {.narrowBandWidth = 6.0 * std::max(dx, dy), .areaSamplesPerAxis = 2});
                (void)redistance;
            }
        }
        const double angle = omega * rotationDt * rotationSteps;
        const double cx = 0.5 + std::cos(angle) * 0.14 - std::sin(angle) * -0.02;
        const double cy = 0.5 + std::sin(angle) * 0.14 + std::cos(angle) * -0.02;
        const auto rotatedExpected = [=](double x, double y) {
            return std::hypot(x - cx, y - cy) - 0.13;
        };
        const GeometryError rotationError = fieldError(
            rotated, rotatedExpected, circleArea(0.13));
        output << width << 'x' << height << " rotation " << rotationDt << ' '
               << rotationSteps << ' ' << std::scientific << rotationError.rms / h << ' '
               << rotationError.p95 / h << ' ' << rotationError.maximum / h << ' '
               << rotationError.areaRelative << " 1 "
               << std::count(rotated.confidence.begin(), rotated.confidence.end(), 1u)
               << ' ' << rotated.sourceVersion << '\n';

        const auto twoDrops = [](double x, double y) {
            return std::min(std::hypot(x - 0.34, y - 0.5) - 0.12,
                std::hypot(x - 0.68, y - 0.5) - 0.12);
        };
        LevelSetField separated = sampleAnalyticLevelSet(width, height, dx, dy, twoDrops);
        const auto topology = redistanceLevelSetNarrowBand(separated,
            {.narrowBandWidth = 5.0 * std::max(dx, dy), .areaSamplesPerAxis = 2});
        require(topology.negativeComponentCountBefore == 2 &&
            topology.negativeComponentCountAfter == 2,
            "redistance merged separated droplets");
    }

    // Forward/reverse deformation checks reversibility without using the method as truth.
    {
        constexpr std::size_t width = 80;
        constexpr std::size_t height = 60;
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        const auto circle = [](double x, double y) {
            return std::hypot(x - 0.5, y - 0.5) - 0.19;
        };
        const LevelSetField initial = sampleAnalyticLevelSet(width, height, dx, dy, circle);
        LevelSetField deformed = initial;
        const auto flow = [](double x, double y) {
            return Vector2{static_cast<float>(std::sin(kPi * x) * std::cos(kPi * y)),
                static_cast<float>(-std::cos(kPi * x) * std::sin(kPi * y))};
        };
        for (std::size_t step = 0; step < 40; ++step) {
            deformed = advectLevelSetRk2(deformed, flow, 0.0025);
        }
        for (std::size_t step = 0; step < 40; ++step) {
            deformed = advectLevelSetRk2(deformed, [&](double x, double y) {
                const Vector2 value = flow(x, y);
                return Vector2{-value.x, -value.y};
            }, 0.0025);
        }
        const GeometryError reverse = fieldError(deformed, circle, circleArea(0.19));
        output << "80x60 reverse 0.0025 80 " << std::scientific
               << reverse.rms / std::min(dx, dy) << ' ' << reverse.p95 / std::min(dx, dy)
               << ' ' << reverse.maximum / std::min(dx, dy) << ' '
               << reverse.areaRelative << " 1 0 " << deformed.sourceVersion << '\n';
        require(reverse.rms / std::min(dx, dy) < 0.5,
            "forward/reverse level-set transport error is excessive");
    }

    // Additional analytic and topology cases use the same persistent transport path.
    {
        constexpr std::size_t width = 80;
        constexpr std::size_t height = 60;
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        const double h = std::min(dx, dy);
        struct AdditionalCase final {
            const char* name;
            ScalarFieldFunction initial;
            ScalarFieldFunction expected;
            Velocity velocity;
            double expectedArea;
            std::size_t expectedComponents;
        };
        const std::vector<AdditionalCase> cases{
            {"static-plane",
             [](double, double y) { return y - 0.55; },
             [](double, double y) { return y - 0.55; },
             [](double, double) { return Vector2{}; }, 0.55, 1},
            {"moving-plane",
             [](double, double y) { return y - 0.52; },
             [](double, double y) { return y - 0.54; },
             [](double, double) { return Vector2{0.0f, 0.1f}; }, 0.54, 1},
            {"moving-ellipse",
             [](double x, double y) {
                 return ellipseDistance(x, y, 0.45, 0.48, 0.18, 0.11);
             },
             [](double x, double y) {
                 return ellipseDistance(x, y, 0.47, 0.47, 0.18, 0.11);
             },
             [](double, double) { return Vector2{0.1f, -0.05f}; },
             kPi * 0.18 * 0.11, 1},
            {"wall-circle",
             [](double x, double y) { return std::hypot(x - 0.19, y - 0.48) - 0.13; },
             [](double x, double y) { return std::hypot(x - 0.17, y - 0.48) - 0.13; },
             [](double, double) { return Vector2{-0.1f, 0.0f}; }, circleArea(0.13), 1},
            {"two-drops",
             [](double x, double y) {
                 return std::min(std::hypot(x - 0.32, y - 0.5) - 0.11,
                     std::hypot(x - 0.70, y - 0.5) - 0.11);
             },
             [](double x, double y) {
                 return std::min(std::hypot(x - 0.33, y - 0.5) - 0.11,
                     std::hypot(x - 0.71, y - 0.5) - 0.11);
             },
             [](double, double) { return Vector2{0.05f, 0.0f}; },
             2.0 * circleArea(0.11), 2},
        };
        output << "Additional dynamic geometry cases\n"
               << "Case RmsH P95H MaxH AreaRel ComponentsBeforeAfter\n";
        for (const AdditionalCase& test : cases) {
            LevelSetField field = sampleAnalyticLevelSet(width, height, dx, dy, test.initial);
            std::size_t beforeComponents = 0;
            std::size_t afterComponents = 0;
            for (std::size_t step = 0; step < 20; ++step) {
                field = advectLevelSetRk2(field, test.velocity, 0.01);
                if ((step + 1) % 4 == 0) {
                    const auto report = redistanceLevelSetNarrowBand(field,
                        {.narrowBandWidth = 6.0 * std::max(dx, dy), .areaSamplesPerAxis = 2});
                    beforeComponents = report.negativeComponentCountBefore;
                    afterComponents = report.negativeComponentCountAfter;
                }
            }
            const GeometryError error = fieldError(field, test.expected, test.expectedArea);
            output << test.name << ' ' << std::scientific << error.rms / h << ' '
                   << error.p95 / h << ' ' << error.maximum / h << ' '
                   << error.areaRelative << ' ' << beforeComponents << '/' << afterComponents << '\n';
            require(beforeComponents == test.expectedComponents &&
                afterComponents == test.expectedComponents,
                "dynamic transport changed an expected interface component count");
        }

        const auto narrow = [](double x, double y) {
            const double qx = std::abs(x - 0.5) - 0.04;
            const double qy = std::abs(y - 0.5) - 0.27;
            return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
                std::min(std::max(qx, qy), 0.0);
        };
        LevelSetField stretched = sampleAnalyticLevelSet(width, height, dx, dy, narrow);
        const double areaBefore = estimateLevelSetArea(stretched, 4).area;
        std::size_t components = 0;
        for (std::size_t step = 0; step < 40; ++step) {
            stretched = advectLevelSetRk2(stretched, [](double x, double y) {
                return Vector2{static_cast<float>(0.35 * (x - 0.5)),
                    static_cast<float>(-0.35 * (y - 0.5))};
            }, 0.005);
            if ((step + 1) % 8 == 0) {
                const auto report = redistanceLevelSetNarrowBand(stretched,
                    {.narrowBandWidth = 6.0 * std::max(dx, dy), .areaSamplesPerAxis = 2});
                components = report.negativeComponentCountAfter;
            }
        }
        const double areaAfter = estimateLevelSetArea(stretched, 4).area;
        output << "narrow-stretch area-before/after=" << std::scientific << areaBefore << '/'
               << areaAfter << " relative=" << std::abs(areaAfter - areaBefore) / areaBefore
               << " components=" << components << '\n';
        require(components == 1, "narrow-column transport destroyed its component");
    }

    // Candidate state is persistent and reject does not alter committed geometry or time.
    {
        MacGrid grid(48, 32, 1.0f / 48.0f, 1.0f / 32.0f);
        const auto dam = [](double x, double y) {
            const double qx = std::max(x - 0.34, 0.0);
            const double qy = std::max(y - 0.68, 0.0);
            return std::hypot(qx, qy) + std::min(std::max(x - 0.34, y - 0.68), 0.0);
        };
        LevelSetField initial = sampleAnalyticLevelSet(48, 32, 1.0 / 48.0, 1.0 / 32.0, dam);
        classifyFromField(grid, initial, true);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        setMacVelocity(grid, [](double, double) { return Vector2{0.04f, 0.0f}; });
        DynamicLevelSetTracker tracker;
        tracker.initialize(initial, 0.0);
        const std::vector<double> committedValues = tracker.committed().field.values;
        const auto rejectedReport = tracker.prepareCandidate(grid, particles, 0.01,
            {.narrowBandWidth = 5.0 / 32.0, .areaSamplesPerAxis = 2},
            {.enabled = false, .interfaceTolerance = 1.0 / 32.0,
             .maximumCorrectionDistance = 1.5 / 32.0, .correctionDepth = 0.25 / 32.0});
        (void)rejectedReport;
        require(tracker.hasCandidate() && tracker.candidate().simulationTime == 0.01,
            "dynamic level-set candidate was not prepared");
        tracker.rejectCandidate();
        require(!tracker.hasCandidate() && tracker.committed().simulationTime == 0.0 &&
            tracker.committed().field.values == committedValues,
            "rejected level-set candidate polluted committed state");
        const auto acceptedReport = tracker.prepareCandidate(grid, particles, 0.01,
            {.narrowBandWidth = 5.0 / 32.0, .areaSamplesPerAxis = 2},
            {.enabled = false, .interfaceTolerance = 1.0 / 32.0,
             .maximumCorrectionDistance = 1.5 / 32.0, .correctionDepth = 0.25 / 32.0});
        (void)acceptedReport;
        tracker.acceptCandidate();
        require(tracker.committed().simulationTime == 0.01 &&
            tracker.committed().versions.geometry > 1 &&
            tracker.committed().versions.fineOperator == 0,
            "accepted level-set candidate version state is invalid");
    }

    // Replay the verified CPU MAC dam-break velocity at matching physical times.
    {
        constexpr std::size_t width = 48;
        constexpr std::size_t height = 32;
        MacGrid grid(width, height, 1.0f / width, 1.0f / height);
        initializeDamBreak(grid);
        MarkerParticleSystem particles;
        particles.seedFluidCellsExact(grid, 8);
        const double liquidRight = static_cast<double>(1 + (width - 2) / 3) / width;
        const double liquidTop = static_cast<double>(1 + 2 * (height - 2) / 3) / height;
        const auto dam = [=](double x, double y) {
            const double qx = std::max(x - liquidRight, 0.0);
            const double qy = std::max(y - liquidTop, 0.0);
            return std::hypot(qx, qy) +
                std::min(std::max(x - liquidRight, y - liquidTop), 0.0);
        };
        DynamicLevelSetTracker tracker;
        tracker.initialize(sampleAnalyticLevelSet(width, height, 1.0 / width,
            1.0 / height, dam), 0.0);
        DynamicFreeSurfaceSettings settings;
        settings.maximumSubsteps = 16;
        DynamicFreeSurfaceStepper stepper(settings);
        output << "CPU MAC replay\nStep Time Substeps ParticleCount FluidCells SignMismatch "
                  "PositiveParticles NegativeNoParticle LevelArea CellArea ThetaValid Fallback\n";
        for (std::size_t step = 1; step <= 60; ++step) {
            const auto physical = stepper.step(grid, particles);
            const auto interface = tracker.prepareCandidate(grid, particles, settings.timeStep,
                {.narrowBandWidth = 5.0 / height, .areaSamplesPerAxis = 2},
                {.enabled = false, .interfaceTolerance = 1.25 / height,
                 .maximumCorrectionDistance = 1.5 / height,
                 .correctionDepth = 0.2 / height});
            tracker.acceptCandidate();
            if (step == 1 || step == 15 || step == 30 || step == 45 || step == 60) {
                const double levelArea = estimateLevelSetArea(
                    tracker.committed().field, 2).area;
                output << step << ' ' << std::fixed << std::setprecision(6)
                       << physical.simulationTime << ' ' << physical.substeps << ' '
                       << particles.validParticleCount() << ' ' << physical.fluidCellCount << ' '
                       << interface.consistency.classificationSignMismatchCount << ' '
                       << interface.consistency.particleInPositivePhiCount << ' '
                       << interface.consistency.negativeCellWithoutParticleCount << ' '
                       << levelArea << ' ' << physical.markedArea << ' '
                       << interface.theta.validFaceCount << ' '
                       << interface.theta.fallbackFaceCount << '\n';
            }
        }
        require(std::abs(tracker.committed().simulationTime - stepper.simulationTime()) < 1.0e-6,
            "MAC replay level-set time does not match physical time");
        output << "Replay note: each offline outer step transports phi with the final projected "
                  "MAC velocity for that outer interval; it does not alter CPU classification or pressure.\n";
    }
    output << "Phase 1.9B-2 dynamic level-set validation result: PASS\n";
    return 0;
}

int runLevelSetParticleConsistencyValidation(std::ostream& output) {
    using namespace fluidlab::simulation;
    constexpr std::size_t width = 80;
    constexpr std::size_t height = 60;
    const double dx = 1.0 / width;
    const double dy = 1.0 / height;
    const auto circle = [](double x, double y) {
        return std::hypot(x - 0.5, y - 0.5) - 0.22;
    };
    MacGrid grid(width, height, static_cast<float>(dx), static_cast<float>(dy));
    LevelSetField field = sampleAnalyticLevelSet(width, height, dx, dy, circle);
    classifyFromField(grid, field, true);
    MarkerParticleSystem particles;
    particles.seedFluidCellsExact(grid, 8);
    const double tolerance = 1.25 * std::max(dx, dy);
    const auto baseline = diagnoseParticleLevelSetConsistency(grid, particles, field, tolerance);

    // A subcell offset creates a controlled one-sided conflict without changing particles.
    LevelSetField lagged = field;
    for (double& value : lagged.values) value += 0.45 * std::min(dx, dy);
    ++lagged.sourceVersion;
    const auto conflict = diagnoseParticleLevelSetConsistency(grid, particles, lagged, tolerance);
    const ParticleCorrectionSettings disabled{
        .enabled = false,
        .interfaceTolerance = 0.1 * std::min(dx, dy),
        .maximumCorrectionDistance = 0.8 * std::min(dx, dy),
        .correctionDepth = 0.2 * std::min(dx, dy)};
    const auto proposal = applyConstrainedParticleCorrection(grid, particles, lagged, disabled);
    LevelSetField corrected = lagged;
    ParticleCorrectionSettings enabled = disabled;
    enabled.enabled = true;
    const auto applied = applyConstrainedParticleCorrection(grid, particles, corrected, enabled);
    output << "LiquidSimulator Phase 1.9B-2 particle/level-set consistency validation\n"
           << "Case Particles Positive FarPositive NegativeNoParticle SignMismatch LowConfidence "
              "SolidConflict MaxPositive MeanBoundaryDistance Proposed Applied RejectedFar "
              "RejectedSolid AreaBefore AreaAfter\n";
    const auto print = [&](const char* name, const ParticleLevelSetConsistency& consistency,
                           const ParticleCorrectionReport& correction) {
        output << name << ' ' << consistency.validParticleCount << ' '
               << consistency.particleInPositivePhiCount << ' '
               << consistency.farPositiveParticleCount << ' '
               << consistency.negativeCellWithoutParticleCount << ' '
               << consistency.classificationSignMismatchCount << ' '
               << consistency.lowConfidenceInterfaceCellCount << ' '
               << consistency.solidAdjacentConflictCount << ' ' << std::scientific
               << consistency.maximumParticlePositiveDistance << ' '
               << consistency.meanClassificationInterfaceDistance << ' '
               << correction.proposedCount << ' ' << correction.appliedCount << ' '
               << correction.rejectedFarCount << ' ' << correction.rejectedSolidCount << ' '
               << correction.areaBefore << ' ' << correction.areaAfter << '\n';
    };
    print("baseline", baseline, {});
    print("disabled", conflict, proposal);
    print("enabled", diagnoseParticleLevelSetConsistency(grid, particles, corrected, tolerance), applied);
    require(baseline.farPositiveParticleCount == 0,
        "trusted initial particles conflict with the analytic interface");
    require(proposal.appliedCount == 0 && proposal.proposedCount > 0,
        "disabled particle correction changed geometry or found no conflict");
    require(applied.appliedCount > 0 && applied.areaAfter >= applied.areaBefore,
        "controlled particle correction did not repair the synthetic lag");
    output << "Policy: correction remains disabled by default because interior particles provide "
              "no symmetric air-side constraint. Far and solid-adjacent proposals are rejected.\n"
           << "Phase 1.9B-2 particle consistency validation result: PASS\n";
    return 0;
}

int runGhostFluidDynamicCpuValidation(std::ostream& output) {
    using namespace fluidlab::simulation;
    constexpr std::size_t width = 24;
    constexpr std::size_t height = 18;
    const double dx = 1.0 / width;
    const double dy = 1.0 / height;
    constexpr double density = 1.0;
    constexpr double gravity = 9.81;
    constexpr double dt = 1.0 / 120.0;
    double surface = 0.557;
    LevelSetField field = sampleAnalyticLevelSet(width, height, dx, dy,
        [=](double, double y) { return y - surface; }, 1);
    output << "LiquidSimulator Phase 1.9B-2 dynamic CPU ghost-fluid validation\n"
           << "Step Surface GeometryVersion ThetaVersion Valid Clamped Fallback SolidAdjacent "
              "ThetaDelta ThetaX ThetaY GhostPressureError DynamicPressureError "
              "CellPressureError RMS MaxResidual PostDiv\n";
    std::vector<double> previousTheta;
    bool observedSubcellChangeWithoutClassificationChange = false;
    std::vector<CellType> previousClassification;
    for (std::size_t step = 0; step < 6; ++step) {
        MacGrid grid(width, height, static_cast<float>(dx), static_cast<float>(dy));
        classifyFromField(grid, field, true);
        const std::vector<CellType> classification = grid.cellTypes();
        GhostFluidThetaData theta = buildGhostFluidTheta(grid, field, 0.1, true);
        GhostFluidThetaData cellTheta = theta;
        std::fill(cellTheta.horizontalTheta.begin(), cellTheta.horizontalTheta.end(), 1.0);
        std::fill(cellTheta.verticalTheta.begin(), cellTheta.verticalTheta.end(), 1.0);
        const HydrostaticResult dynamicResult = solveHydrostatic(
            grid, theta, surface, density, gravity, dt);
        const HydrostaticResult cellResult = solveHydrostatic(
            grid, cellTheta, surface, density, gravity, dt);
        const LevelSetField analyticField = sampleAnalyticLevelSet(width, height, dx, dy,
            [=](double, double y) { return y - surface; }, 100 + step);
        const GhostFluidThetaData analyticTheta = buildGhostFluidTheta(
            grid, analyticField, 0.1, true);
        const HydrostaticResult analyticResult = solveHydrostatic(
            grid, analyticTheta, surface, density, gravity, dt);
        double thetaDelta = 0.0;
        std::size_t thetaDeltaIndex = 0;
        if (!previousTheta.empty()) {
            for (std::size_t i = 0; i < theta.verticalTheta.size(); ++i) {
                const double delta = std::abs(theta.verticalTheta[i] - previousTheta[i]);
                if (delta > thetaDelta) {
                    thetaDelta = delta;
                    thetaDeltaIndex = i;
                }
            }
            if (classification == previousClassification && thetaDelta > 1.0e-8) {
                observedSubcellChangeWithoutClassificationChange = true;
            }
        }
        output << step << ' ' << std::fixed << std::setprecision(6) << surface << ' '
               << field.sourceVersion << ' ' << theta.interfaceVersion << ' '
               << theta.statistics.validFaceCount << ' '
               << theta.statistics.clampedFaceCount << ' '
               << theta.statistics.fallbackFaceCount << ' '
               << theta.statistics.solidAdjacentFaceCount << ' ' << std::scientific << thetaDelta
               << ' ' << (thetaDeltaIndex % width + 0.5) * dx << ' '
               << (thetaDeltaIndex / width) * dy << ' '
               << analyticResult.pressureError << ' ' << dynamicResult.pressureError << ' '
               << cellResult.pressureError << ' ' << dynamicResult.residualRms << ' '
               << dynamicResult.residualMaximum << ' ' << dynamicResult.postDivergence << '\n';
        require(dynamicResult.residualRms < 1.0e-9 && dynamicResult.postDivergence < 1.0e-5,
            "dynamic CPU ghost-fluid solve or projection is inconsistent");
        require(dynamicResult.pressureError < cellResult.pressureError,
            "dynamic ghost-fluid boundary did not improve hydrostatic pressure");
        previousTheta = theta.verticalTheta;
        previousClassification = classification;

        constexpr double upwardSpeed = 0.035;
        field = advectLevelSetRk2(field, [](double, double) {
            return Vector2{0.0f, static_cast<float>(upwardSpeed)};
        }, dt);
        const auto redistance = redistanceLevelSetNarrowBand(field,
            {.narrowBandWidth = 5.0 * dy, .areaSamplesPerAxis = 2});
        (void)redistance;
        surface += upwardSpeed * dt;
    }
    require(observedSubcellChangeWithoutClassificationChange,
        "theta did not change while classification remained unchanged");
    output << "Version rule: a geometry/theta change invalidates the fine operator and every "
              "Galerkin level even if cellType is unchanged. Warm start requires a new true residual.\n"
           << "Phase 1.9B-2 dynamic CPU ghost-fluid validation result: PASS\n";
    return 0;
}

} // namespace fluidlab::diagnostics
