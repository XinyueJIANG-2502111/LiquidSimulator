#include "VofValidation.h"

#include "../simulation/mac/CoupledLevelSetVOF.h"
#include "../simulation/mac/Scenarios.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace fluidlab::diagnostics {
namespace {

using Clock = std::chrono::steady_clock;
using fluidlab::simulation::CellType;
using fluidlab::simulation::ClsvofCouplingReport;
using fluidlab::simulation::ClsvofCouplingSettings;
using fluidlab::simulation::CoupledLevelSetVOF;
using fluidlab::simulation::LevelSetField;
using fluidlab::simulation::MacGrid;
using fluidlab::simulation::PlicReconstruction;
using fluidlab::simulation::PlicNormalMethod;
using fluidlab::simulation::GhostFluidThetaData;
using fluidlab::simulation::GhostThetaState;
using fluidlab::simulation::Vector2;
using fluidlab::simulation::Vector2d;
using fluidlab::simulation::VolumeFractionField;
using fluidlab::simulation::VolumeFractionLedger;
using fluidlab::simulation::VolumeFractionDistribution;

constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double elapsedMilliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void setVelocity(
    MacGrid& grid,
    const std::function<Vector2(double, double)>& velocity) {
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x <= grid.width(); ++x) {
            grid.horizontalVelocity(x, y) = velocity(
                x * grid.cellWidth(), (static_cast<double>(y) + 0.5) * grid.cellHeight()).x;
        }
    }
    for (std::size_t y = 0; y <= grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            grid.verticalVelocity(x, y) = velocity(
                (static_cast<double>(x) + 0.5) * grid.cellWidth(), y * grid.cellHeight()).y;
        }
    }
    grid.enforceClosedBoundary();
}

void setSplitTestVelocity(MacGrid& grid, double amplitude, double sign = 1.0) {
    const auto psi = [&](std::size_t x, std::size_t y) {
        const double px = static_cast<double>(x) / grid.width();
        const double py = static_cast<double>(y) / grid.height();
        return sign * amplitude * std::sin(2.0 * kPi * px) * std::sin(2.0 * kPi * py);
    };
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x <= grid.width(); ++x) {
            grid.horizontalVelocity(x, y) = static_cast<float>(
                (psi(x, y + 1) - psi(x, y)) / grid.cellHeight());
        }
    }
    for (std::size_t y = 0; y <= grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            grid.verticalVelocity(x, y) = static_cast<float>(
                -(psi(x + 1, y) - psi(x, y)) / grid.cellWidth());
        }
    }
    grid.enforceClosedBoundary();
}

VolumeFractionField makeField(
    std::size_t width,
    std::size_t height,
    const std::function<double(double, double)>& signedDistance,
    std::size_t samples = 24) {
    VolumeFractionField field(width, height, 1.0 / width, 1.0 / height);
    field.initializeFromSignedDistance(signedDistance, samples);
    return field;
}

struct FractionError final {
    double l1Area = 0.0;
    double centroidX = 0.0;
    double centroidY = 0.0;
};

FractionError compareFields(
    const VolumeFractionField& actual,
    const VolumeFractionField& expected) {
    FractionError result;
    double weightedX = 0.0;
    double weightedY = 0.0;
    double area = 0.0;
    for (std::size_t y = 0; y < actual.height(); ++y) {
        for (std::size_t x = 0; x < actual.width(); ++x) {
            const double cellArea = actual.cellArea(x, y);
            result.l1Area += std::abs(actual.fraction(x, y) - expected.fraction(x, y)) * cellArea;
            const double volume = actual.fraction(x, y) * cellArea;
            weightedX += (static_cast<double>(x) + 0.5) * actual.cellWidth() * volume;
            weightedY += (static_cast<double>(y) + 0.5) * actual.cellHeight() * volume;
            area += volume;
        }
    }
    if (area > 0.0) {
        result.centroidX = weightedX / area;
        result.centroidY = weightedY / area;
    }
    return result;
}

std::size_t occupiedComponents(const VolumeFractionField& field, double threshold = 1.0e-6) {
    std::vector<unsigned char> visited(field.width() * field.height(), 0);
    std::size_t components = 0;
    for (std::size_t seedY = 0; seedY < field.height(); ++seedY) {
        for (std::size_t seedX = 0; seedX < field.width(); ++seedX) {
            const std::size_t seed = seedY * field.width() + seedX;
            if (visited[seed] != 0 || field.fraction(seedX, seedY) <= threshold) {
                continue;
            }
            ++components;
            std::queue<std::pair<std::size_t, std::size_t>> pending;
            pending.push({seedX, seedY});
            visited[seed] = 1;
            while (!pending.empty()) {
                const auto [x, y] = pending.front();
                pending.pop();
                const auto visit = [&](std::size_t nx, std::size_t ny) {
                    const std::size_t offset = ny * field.width() + nx;
                    if (visited[offset] == 0 && field.fraction(nx, ny) > threshold) {
                        visited[offset] = 1;
                        pending.push({nx, ny});
                    }
                };
                if (x > 0) visit(x - 1, y);
                if (x + 1 < field.width()) visit(x + 1, y);
                if (y > 0) visit(x, y - 1);
                if (y + 1 < field.height()) visit(x, y + 1);
            }
        }
    }
    return components;
}

Vector2d analyticNormal(
    const std::function<double(double, double)>& signedDistance,
    double x,
    double y,
    double spacing) {
    const double gx = (signedDistance(x + spacing, y) - signedDistance(x - spacing, y)) /
        (2.0 * spacing);
    const double gy = (signedDistance(x, y + spacing) - signedDistance(x, y - spacing)) /
        (2.0 * spacing);
    const double length = std::hypot(gx, gy);
    return length > 1.0e-12 ? Vector2d{gx / length, gy / length} : Vector2d{};
}

struct GeometryMetrics final {
    double maximumAreaError = 0.0;
    double meanNormalErrorDegrees = 0.0;
    double p95NormalErrorDegrees = 0.0;
    double maximumPositionErrorCells = 0.0;
    std::size_t interfaceCells = 0;
    std::size_t failures = 0;
};

GeometryMetrics geometryMetrics(
    const VolumeFractionField& field,
    const std::function<double(double, double)>& signedDistance) {
    GeometryMetrics result;
    std::vector<double> angles;
    for (std::size_t y = 0; y < field.height(); ++y) {
        for (std::size_t x = 0; x < field.width(); ++x) {
            const double fraction = field.fraction(x, y);
            if (!(fraction > 1.0e-8 && fraction < 1.0 - 1.0e-8)) {
                continue;
            }
            ++result.interfaceCells;
            const double px = (static_cast<double>(x) + 0.5) * field.cellWidth();
            const double py = (static_cast<double>(y) + 0.5) * field.cellHeight();
            const Vector2d expectedNormal = analyticNormal(
                signedDistance, px, py, 0.05 * std::min(field.cellWidth(), field.cellHeight()));
            const Vector2d estimatedNormal = field.estimateNormal(x, y);
            const PlicReconstruction plic = field.reconstructCell(x, y, &expectedNormal);
            if (!plic.valid || std::hypot(estimatedNormal.x, estimatedNormal.y) < 0.5) {
                ++result.failures;
                continue;
            }
            const double targetArea = fraction * field.cellWidth() * field.cellHeight();
            result.maximumAreaError = std::max(
                result.maximumAreaError, std::abs(plic.reconstructedArea - targetArea));
            const double dot = std::clamp(
                expectedNormal.x * estimatedNormal.x + expectedNormal.y * estimatedNormal.y,
                -1.0,
                1.0);
            angles.push_back(std::acos(dot) * 180.0 / kPi);
            const double centerDistance = plic.normal.x * 0.5 * field.cellWidth() +
                plic.normal.y * 0.5 * field.cellHeight() - plic.alpha;
            result.maximumPositionErrorCells = std::max(
                result.maximumPositionErrorCells,
                std::abs(centerDistance - signedDistance(px, py)) /
                    std::max(field.cellWidth(), field.cellHeight()));
        }
    }
    std::sort(angles.begin(), angles.end());
    if (!angles.empty()) {
        for (double angle : angles) {
            result.meanNormalErrorDegrees += angle;
        }
        result.meanNormalErrorDegrees /= angles.size();
        result.p95NormalErrorDegrees = angles[static_cast<std::size_t>(0.95 * (angles.size() - 1))];
    }
    return result;
}

std::function<double(double, double)> circle(double cx, double cy, double radius) {
    return [=](double x, double y) { return std::hypot(x - cx, y - cy) - radius; };
}

std::function<double(double, double)> ellipse(
    double cx,
    double cy,
    double radiusX,
    double radiusY) {
    return [=](double x, double y) {
        const double nx = (x - cx) / radiusX;
        const double ny = (y - cy) / radiusY;
        return (std::hypot(nx, ny) - 1.0) * std::min(radiusX, radiusY);
    };
}

bool runAdvectionCase(
    std::ostream& output,
    std::size_t width,
    std::size_t height,
    std::size_t steps,
    double timeStep) {
    const auto initialShape = circle(0.30, 0.50, 0.12);
    VolumeFractionField field = makeField(width, height, initialShape, 24);
    const double initialArea = field.totalArea();
    MacGrid velocity(width, height, static_cast<float>(1.0 / width), static_cast<float>(1.0 / height));
    velocity.fillCellTypes(CellType::fluid);
    setVelocity(velocity, [](double, double) { return Vector2{0.20f, 0.0f}; });

    double maximumResidual = 0.0;
    std::size_t limitedFaces = 0;
    for (std::size_t step = 0; step < steps; ++step) {
        const VolumeFractionLedger ledger = field.advectConservative(
            velocity, timeStep, (step & 1u) == 0);
        require(ledger.accepted, "VOF translation step was rejected.");
        maximumResidual = std::max(maximumResidual, std::abs(ledger.conservationResidual));
        limitedFaces += ledger.limitedFaceCount;
    }
    const double displacement = 0.20 * steps * timeStep;
    const VolumeFractionField expected = makeField(
        width, height, circle(0.30 + displacement, 0.50, 0.12), 24);
    const FractionError error = compareFields(field, expected);
    const double areaError = std::abs(field.totalArea() - initialArea);
    output << "VOF translation " << width << 'x' << height
           << " steps=" << steps
           << " areaError=" << areaError
           << " ledgerMax=" << maximumResidual
           << " l1Area=" << error.l1Area
           << " centroid=(" << error.centroidX << ',' << error.centroidY << ')'
           << " limitedFaces=" << limitedFaces << '\n';
    return areaError < 2.0e-10 && maximumResidual < 2.0e-10 &&
        error.l1Area < 0.025;
}

bool runFeatureTranslationCase(
    std::ostream& output,
    const char* name,
    const std::function<double(double, double)>& initialShape,
    const std::function<double(double, double)>& expectedShape,
    Vector2 speed,
    std::size_t expectedComponents) {
    constexpr std::size_t width = 80;
    constexpr std::size_t height = 60;
    constexpr std::size_t steps = 30;
    constexpr double timeStep = 0.0025;
    VolumeFractionField field = makeField(width, height, initialShape, 32);
    const double initialArea = field.totalArea();
    MacGrid velocity(width, height, 1.0f / width, 1.0f / height);
    velocity.fillCellTypes(CellType::fluid);
    setVelocity(velocity, [=](double, double) { return speed; });
    for (std::size_t step = 0; step < steps; ++step) {
        require(field.advectConservative(velocity, timeStep, (step & 1u) == 0).accepted,
            "VOF feature translation step was rejected.");
    }
    const VolumeFractionField expected = makeField(width, height, expectedShape, 32);
    const FractionError error = compareFields(field, expected);
    const std::size_t components = occupiedComponents(field);
    const double areaError = std::abs(field.totalArea() - initialArea);
    output << "VOF feature " << name
           << " areaError=" << areaError
           << " l1Area=" << error.l1Area
           << " components=" << components << '/' << expectedComponents << '\n';
    return areaError < 2.0e-10 && error.l1Area < 0.012 && components == expectedComponents;
}

struct FieldMoments final {
    double area = 0.0;
    double centroidX = 0.0;
    double centroidY = 0.0;
    double secondX = 0.0;
    double secondY = 0.0;
    std::size_t mixedCells = 0;
    std::size_t tinyTailCells = 0;
};

FieldMoments measureMoments(const VolumeFractionField& field) {
    FieldMoments result;
    for (std::size_t y = 0; y < field.height(); ++y) {
        for (std::size_t x = 0; x < field.width(); ++x) {
            const double fraction = field.fraction(x, y);
            const double volume = fraction * field.cellArea(x, y);
            const double px = (static_cast<double>(x) + 0.5) * field.cellWidth();
            const double py = (static_cast<double>(y) + 0.5) * field.cellHeight();
            result.area += volume;
            result.centroidX += px * volume;
            result.centroidY += py * volume;
            result.secondX += px * px * volume;
            result.secondY += py * py * volume;
            result.mixedCells += fraction > 1.0e-8 && fraction < 1.0 - 1.0e-8 ? 1u : 0u;
            result.tinyTailCells += fraction > 1.0e-8 && fraction < 1.0e-2 ? 1u : 0u;
        }
    }
    if (result.area > 0.0) {
        result.centroidX /= result.area;
        result.centroidY /= result.area;
        result.secondX /= result.area;
        result.secondY /= result.area;
    }
    return result;
}

void setDiscreteDivergenceFreeVelocity(MacGrid& grid, double amplitude) {
    const auto streamFunction = [=](double x, double y) {
        const double sx = std::sin(kPi * x);
        const double sy = std::sin(kPi * y);
        return amplitude * sx * sx * sy * sy;
    };
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x <= grid.width(); ++x) {
            const double px = x * grid.cellWidth();
            const double y0 = y * grid.cellHeight();
            const double y1 = (y + 1) * grid.cellHeight();
            grid.horizontalVelocity(x, y) = static_cast<float>(
                (streamFunction(px, y1) - streamFunction(px, y0)) / grid.cellHeight());
        }
    }
    for (std::size_t y = 0; y <= grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const double py = y * grid.cellHeight();
            const double x0 = x * grid.cellWidth();
            const double x1 = (x + 1) * grid.cellWidth();
            grid.verticalVelocity(x, y) = static_cast<float>(
                -(streamFunction(x1, py) - streamFunction(x0, py)) / grid.cellWidth());
        }
    }
    grid.enforceClosedBoundary();
}

double maximumDiscreteDivergence(const MacGrid& grid) {
    double maximum = 0.0;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const double divergence =
                (grid.horizontalVelocity(x + 1, y) - grid.horizontalVelocity(x, y)) /
                    grid.cellWidth() +
                (grid.verticalVelocity(x, y + 1) - grid.verticalVelocity(x, y)) /
                    grid.cellHeight();
            maximum = std::max(maximum, std::abs(divergence));
        }
    }
    return maximum;
}

struct NormalStatistics final {
    std::size_t interfaceCount = 0;
    std::size_t validCount = 0;
    std::size_t invalidCount = 0;
    double mean = 0.0;
    double rms = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
};

NormalStatistics normalStatistics(
    const VolumeFractionField& field,
    const std::function<double(double, double)>& signedDistance,
    PlicNormalMethod method) {
    NormalStatistics result;
    std::vector<double> errors;
    for (std::size_t y = 0; y < field.height(); ++y) {
        for (std::size_t x = 0; x < field.width(); ++x) {
            const double fraction = field.fraction(x, y);
            if (!(fraction > 1.0e-8 && fraction < 1.0 - 1.0e-8)) continue;
            ++result.interfaceCount;
            bool valid = true;
            const Vector2d estimated = method == PlicNormalMethod::heightFunction
                ? field.estimateHeightFunctionNormal(x, y, &valid)
                : field.estimateNormal(x, y);
            const double px = (static_cast<double>(x) + 0.5) * field.cellWidth();
            const double py = (static_cast<double>(y) + 0.5) * field.cellHeight();
            const Vector2d exact = analyticNormal(
                signedDistance, px, py, 0.02 * std::min(field.cellWidth(), field.cellHeight()));
            if (!valid || std::hypot(estimated.x, estimated.y) < 0.5 ||
                std::hypot(exact.x, exact.y) < 0.5) {
                ++result.invalidCount;
                continue;
            }
            ++result.validCount;
            const double dot = std::clamp(
                estimated.x * exact.x + estimated.y * exact.y, -1.0, 1.0);
            errors.push_back(std::acos(dot) * 180.0 / kPi);
        }
    }
    std::sort(errors.begin(), errors.end());
    if (!errors.empty()) {
        double squareSum = 0.0;
        for (double value : errors) {
            result.mean += value;
            squareSum += value * value;
        }
        result.mean /= errors.size();
        result.rms = std::sqrt(squareSum / errors.size());
        result.p95 = errors[static_cast<std::size_t>(0.95 * (errors.size() - 1))];
        result.maximum = errors.back();
    }
    return result;
}

void classifyFromSignedDistance(
    MacGrid& grid,
    const std::function<double(double, double)>& signedDistance) {
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (x == 0 || y == 0 || x + 1 == grid.width() || y + 1 == grid.height()) {
                grid.setCellType(x, y, CellType::solid);
                continue;
            }
            const double px = (static_cast<double>(x) + 0.5) * grid.cellWidth();
            const double py = (static_cast<double>(y) + 0.5) * grid.cellHeight();
            grid.setCellType(x, y, signedDistance(px, py) < 0.0 ? CellType::fluid : CellType::air);
        }
    }
}

struct ThetaDifference final {
    std::size_t compared = 0;
    std::size_t fallback = 0;
    double rms = 0.0;
    double maximum = 0.0;
};

ThetaDifference compareTheta(
    const GhostFluidThetaData& reference,
    const GhostFluidThetaData& actual) {
    ThetaDifference result;
    double squareSum = 0.0;
    const auto compare = [&](const auto& expectedValues,
                             const auto& expectedStates,
                             const auto& actualValues,
                             const auto& actualStates) {
        for (std::size_t index = 0; index < expectedValues.size(); ++index) {
            const bool expectedUsed = expectedStates[index] != GhostThetaState::unused;
            if (!expectedUsed) continue;
            if (actualStates[index] == GhostThetaState::fallback ||
                actualStates[index] == GhostThetaState::unused) {
                ++result.fallback;
                continue;
            }
            const double difference = std::abs(expectedValues[index] - actualValues[index]);
            squareSum += difference * difference;
            result.maximum = std::max(result.maximum, difference);
            ++result.compared;
        }
    };
    compare(reference.horizontalTheta, reference.horizontalState,
            actual.horizontalTheta, actual.horizontalState);
    compare(reference.verticalTheta, reference.verticalState,
            actual.verticalTheta, actual.verticalState);
    if (result.compared > 0) result.rms = std::sqrt(squareSum / result.compared);
    return result;
}

} // namespace

int runVofGeometryValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6);
    output << "VOF/PLIC geometry validation\n";
    const std::vector<std::pair<std::string, std::function<double(double, double)>>> cases{
        {"horizontal", [](double, double y) { return y - 0.47; }},
        {"tilted", [](double x, double y) { return (y - 0.35 - 0.30 * x) / std::hypot(1.0, 0.30); }},
        {"circle", circle(0.47, 0.52, 0.21)},
        {"ellipse", ellipse(0.48, 0.50, 0.26, 0.13)},
        {"thin-column", [](double x, double y) {
            return std::max(std::abs(x - 0.50) - 0.035, std::abs(y - 0.50) - 0.30);
        }},
        {"two-drops", [](double x, double y) {
            return std::min(std::hypot(x - 0.36, y - 0.50) - 0.11,
                            std::hypot(x - 0.64, y - 0.50) - 0.11);
        }},
        {"wall-contact", [](double x, double y) {
            return std::hypot(x - 0.08, y - 0.50) - 0.18;
        }},
    };
    bool passed = true;
    for (const auto& [name, shape] : cases) {
        for (const auto [width, height] : std::vector<std::pair<std::size_t, std::size_t>>{
                 {40, 30}, {80, 60}, {160, 120}}) {
            const auto start = Clock::now();
            const VolumeFractionField field = makeField(width, height, shape, 32);
            const GeometryMetrics metrics = geometryMetrics(field, shape);
            output << name << ' ' << width << 'x' << height
                   << " interface=" << metrics.interfaceCells
                   << " failed=" << metrics.failures
                   << " areaRebuildMax=" << metrics.maximumAreaError
                   << " normalMeanDeg=" << metrics.meanNormalErrorDegrees
                   << " normalP95Deg=" << metrics.p95NormalErrorDegrees
                   << " positionMaxH=" << metrics.maximumPositionErrorCells
                   << " ms=" << elapsedMilliseconds(start) << '\n';
            passed = passed && metrics.interfaceCells > 0 &&
                metrics.maximumAreaError < 1.0e-10 &&
                metrics.failures < std::max<std::size_t>(2, metrics.interfaceCells / 5);
        }
    }
    output << "VOF_GEOMETRY_RESULT " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

int runVofConservationValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6);
    output << "VOF conservative flux validation\n";
    bool passed = true;
    for (const std::size_t resolution : {40u, 80u, 160u}) {
        VolumeFractionField field = makeField(resolution, resolution, circle(0.50, 0.50, 0.18), 24);
        const double initialArea = field.totalArea();
        MacGrid velocity(resolution, resolution,
            static_cast<float>(1.0 / resolution), static_cast<float>(1.0 / resolution));
        velocity.fillCellTypes(CellType::fluid);
        setVelocity(velocity, [](double x, double y) {
            return Vector2{
                static_cast<float>(-0.35 * (y - 0.5)),
                static_cast<float>(0.35 * (x - 0.5)),
            };
        });
        const double dt = 0.20 / resolution;
        double maximumLedgerResidual = 0.0;
        std::size_t outOfBounds = 0;
        const auto start = Clock::now();
        for (std::size_t step = 0; step < 100; ++step) {
            const VolumeFractionLedger ledger = field.advectConservative(
                velocity, dt, (step & 1u) == 0);
            maximumLedgerResidual = std::max(
                maximumLedgerResidual, std::abs(ledger.conservationResidual));
            outOfBounds += ledger.outOfBoundsCount;
            require(ledger.accepted, "Conservative VOF rotation step was rejected.");
        }
        const double areaError = std::abs(field.totalArea() - initialArea);
        output << resolution << "x" << resolution
               << " initial=" << initialArea
               << " final=" << field.totalArea()
               << " areaError=" << areaError
               << " ledgerMax=" << maximumLedgerResidual
               << " bounds=" << outOfBounds
               << " ms=" << elapsedMilliseconds(start) << '\n';
        passed = passed && areaError < 5.0e-10 && maximumLedgerResidual < 5.0e-10 &&
            outOfBounds == 0;
    }

    VolumeFractionField base = makeField(40, 30, circle(0.4, 0.5, 0.15), 16);
    fluidlab::simulation::VolumeFractionTransaction transaction;
    transaction.initialize(base);
    MacGrid velocity(40, 30, 1.0f / 40.0f, 1.0f / 30.0f);
    velocity.fillCellTypes(CellType::fluid);
    setVelocity(velocity, [](double, double) { return Vector2{0.1f, 0.0f}; });
    const double committedArea = transaction.committed().totalArea();
    VolumeFractionField& candidate = transaction.prepareCandidate();
    require(candidate.advectConservative(velocity, 0.01).accepted,
        "VOF transaction candidate failed.");
    transaction.rejectCandidate();
    const bool rollbackPassed = !transaction.hasCandidate() &&
        std::abs(transaction.committed().totalArea() - committedArea) < 1.0e-14;
    output << "transactionRollback=" << (rollbackPassed ? "PASS" : "FAIL") << '\n';
    passed = passed && rollbackPassed;
    output << "VOF_CONSERVATION_RESULT " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

int runVofAdvectionValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6);
    output << "VOF advection convergence validation\n";
    bool passed = true;
    passed = runAdvectionCase(output, 40, 30, 40, 0.005) && passed;
    passed = runAdvectionCase(output, 80, 60, 80, 0.0025) && passed;
    passed = runAdvectionCase(output, 160, 120, 160, 0.00125) && passed;
    constexpr double displacement = 0.15 * 30.0 * 0.0025;
    passed = runFeatureTranslationCase(
        output,
        "thin-column",
        [](double x, double y) {
            return std::max(std::abs(x - 0.40) - 0.035, std::abs(y - 0.50) - 0.25);
        },
        [](double x, double y) {
            return std::max(
                std::abs(x - (0.40 + displacement)) - 0.035,
                std::abs(y - 0.50) - 0.25);
        },
        Vector2{0.15f, 0.0f},
        1) && passed;
    passed = runFeatureTranslationCase(
        output,
        "small-drop",
        circle(0.38, 0.50, 0.045),
        circle(0.38 + displacement, 0.50, 0.045),
        Vector2{0.15f, 0.0f},
        1) && passed;
    passed = runFeatureTranslationCase(
        output,
        "two-drops",
        [](double x, double y) {
            return std::min(std::hypot(x - 0.34, y - 0.50) - 0.075,
                            std::hypot(x - 0.62, y - 0.50) - 0.075);
        },
        [](double x, double y) {
            return std::min(std::hypot(x - (0.34 + displacement), y - 0.50) - 0.075,
                            std::hypot(x - (0.62 + displacement), y - 0.50) - 0.075);
        },
        Vector2{0.15f, 0.0f},
        2) && passed;
    passed = runFeatureTranslationCase(
        output,
        "wall-tangent",
        circle(0.10, 0.40, 0.08),
        circle(0.10, 0.40 + displacement, 0.08),
        Vector2{0.0f, 0.15f},
        1) && passed;
    output << "VOF_ADVECTION_RESULT " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

int runClsvofCouplingValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6);
    output << "CLSVOF one-way coupling validation\n";
    bool passed = true;
    for (const std::size_t resolution : {40u, 80u, 160u}) {
        const auto initialShape = circle(0.32, 0.50, 0.15);
        VolumeFractionField vof = makeField(resolution, resolution, initialShape, 24);
        LevelSetField levelSet = fluidlab::simulation::sampleAnalyticLevelSet(
            resolution, resolution, 1.0 / resolution, 1.0 / resolution, initialShape);
        MacGrid velocity(resolution, resolution,
            static_cast<float>(1.0 / resolution), static_cast<float>(1.0 / resolution));
        velocity.fillCellTypes(CellType::fluid);
        setVelocity(velocity, [](double, double) { return Vector2{0.15f, 0.0f}; });
        const double dt = 0.20 / resolution;
        for (std::size_t step = 0; step < resolution / 2; ++step) {
            require(vof.advectConservative(velocity, dt, (step & 1u) == 0).accepted,
                "CLSVOF VOF step failed.");
            levelSet = fluidlab::simulation::advectLevelSetLimitedMacCormack(
                levelSet,
                [](double, double) { return Vector2{0.15f, 0.0f}; },
                dt);
        }
        const double levelSetAreaBefore = fluidlab::simulation::estimateLevelSetArea(levelSet, 4).area;
        ClsvofCouplingSettings settings;
        settings.narrowBandWidth = 6.0 / resolution;
        settings.maximumCorrectionDistance = 1.5 / resolution;
        settings.redistanceAfterCorrection = false;
        const ClsvofCouplingReport report = fluidlab::simulation::constrainLevelSetToVolumeFraction(
            vof, levelSet, settings);
        const double beforeDifference = std::abs(levelSetAreaBefore - vof.totalArea());
        const double afterDifference = std::abs(report.levelSetAreaAfter - vof.totalArea());
        output << resolution << "x" << resolution
               << " vofArea=" << report.vofArea
               << " phiBefore=" << report.levelSetAreaBefore
               << " phiAfter=" << report.levelSetAreaAfter
               << " corrected=" << report.correctedCellCount
               << " plicFailure=" << report.plicFailureCount
               << " beforeDiff=" << beforeDifference
               << " afterDiff=" << afterDifference
               << " maxCorrection=" << report.maximumInterfaceCorrection << '\n';
        passed = passed && report.correctedCellCount > 0 && report.plicFailureCount == 0 &&
            afterDifference <= beforeDifference + 2.0 / (resolution * resolution);
    }
    output << "CLSVOF_COUPLING_RESULT " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

int runClsvofBenchmark(std::ostream& output) {
    output << std::fixed << std::setprecision(3);
    output << "CPU VOF/PLIC/CLSVOF benchmark\n";
    for (const auto [width, height] : std::vector<std::pair<std::size_t, std::size_t>>{
             {40, 30}, {80, 60}, {160, 120}, {64, 64}, {128, 128}, {192, 192}, {256, 256}}) {
        const auto shape = ellipse(0.45, 0.50, 0.24, 0.14);
        const auto initializationStart = Clock::now();
        VolumeFractionField vof = makeField(width, height, shape, 24);
        const double initializationMs = elapsedMilliseconds(initializationStart);

        const auto reconstructionStart = Clock::now();
        std::size_t interfaceCells = 0;
        std::size_t failures = 0;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const double fraction = vof.fraction(x, y);
                if (fraction > 0.0 && fraction < 1.0) {
                    ++interfaceCells;
                    failures += vof.reconstructCell(x, y).valid ? 0u : 1u;
                }
            }
        }
        const double reconstructionMs = elapsedMilliseconds(reconstructionStart);

        const auto heightNormalStart = Clock::now();
        std::size_t heightValid = 0;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                if (!(vof.fraction(x, y) > 0.0 && vof.fraction(x, y) < 1.0)) continue;
                bool valid = false;
                static_cast<void>(vof.estimateHeightFunctionNormal(x, y, &valid));
                heightValid += valid ? 1u : 0u;
            }
        }
        const double heightNormalMs = elapsedMilliseconds(heightNormalStart);

        MacGrid velocity(width, height,
            static_cast<float>(1.0 / width), static_cast<float>(1.0 / height));
        velocity.fillCellTypes(CellType::fluid);
        setVelocity(velocity, [](double, double) { return Vector2{0.10f, 0.0f}; });
        const auto advectionStart = Clock::now();
        const VolumeFractionLedger ledger = vof.advectConservative(
            velocity, 0.2 / std::max(width, height));
        const double advectionMs = elapsedMilliseconds(advectionStart);

        LevelSetField levelSet = fluidlab::simulation::sampleAnalyticLevelSet(
            width, height, 1.0 / width, 1.0 / height, shape);
        ClsvofCouplingSettings settings;
        settings.narrowBandWidth = 6.0 / std::max(width, height);
        const auto couplingStart = Clock::now();
        const ClsvofCouplingReport coupling =
            fluidlab::simulation::constrainLevelSetToVolumeFraction(vof, levelSet, settings);
        const double couplingMs = elapsedMilliseconds(couplingStart);
        LevelSetField segmentLevelSet = levelSet;
        const auto segmentCouplingStart = Clock::now();
        const ClsvofCouplingReport segmentCoupling =
            fluidlab::simulation::constrainLevelSetToPlicGeometry(
                vof, segmentLevelSet, settings, PlicNormalMethod::heightFunction);
        const double segmentCouplingMs = elapsedMilliseconds(segmentCouplingStart);
        classifyFromSignedDistance(velocity, shape);
        const auto thetaStart = Clock::now();
        const auto theta = fluidlab::simulation::buildPlicGhostFluidTheta(
            velocity, vof, PlicNormalMethod::heightFunction, 0.1);
        const double thetaMs = elapsedMilliseconds(thetaStart);
        const std::size_t storageBytes = vof.fractions().size() * sizeof(double) +
            vof.validMask().size() * sizeof(unsigned char);
        output << width << 'x' << height
               << " initMs=" << initializationMs
               << " plicMs=" << reconstructionMs
               << " heightMs=" << heightNormalMs
               << " advectMs=" << advectionMs
               << " coupleMs=" << couplingMs
               << " segmentCoupleMs=" << segmentCouplingMs
               << " thetaMs=" << thetaMs
               << " interface=" << interfaceCells
               << " failures=" << failures
               << " heightValid=" << heightValid
               << " limited=" << ledger.limitedFaceCount
               << " corrected=" << coupling.correctedCellCount
               << " segmentCorrected=" << segmentCoupling.correctedCellCount
               << " thetaFallback=" << theta.statistics.fallbackFaceCount
               << " bytes=" << storageBytes << '\n';
    }
    output << "CLSVOF_BENCHMARK_RESULT PASS\n";
    return 0;
}

int runVofFluxAudit(std::ostream& output) {
    output << std::scientific << std::setprecision(6)
           << "VOF flux limiter audit\n";
    bool invariantPassed = true;
    for (const auto [width, height] : std::vector<std::pair<std::size_t, std::size_t>>{
             {40, 30}, {80, 60}, {160, 120}}) {
        VolumeFractionField full(width, height, 1.0 / width, 1.0 / height);
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) full.setFraction(x, y, 1.0);
        }
        MacGrid velocity(width, height,
            static_cast<float>(1.0 / width), static_cast<float>(1.0 / height));
        velocity.fillCellTypes(CellType::fluid);
        setDiscreteDivergenceFreeVelocity(velocity, 0.02);
        const double divergence = maximumDiscreteDivergence(velocity);
        const auto fullLedger = full.advectConservative(
            velocity, 0.05 / std::max(width, height));
        double fullMaximumError = 0.0;
        for (double fraction : full.fractions()) {
            fullMaximumError = std::max(fullMaximumError, std::abs(fraction - 1.0));
        }

        VolumeFractionField interfaceField = makeField(
            width, height, circle(0.50, 0.50, 0.18), 24);
        const double initialArea = interfaceField.totalArea();
        VolumeFractionLedger aggregate;
        aggregate.minimumLimiterScale = 1.0;
        for (std::size_t step = 0; step < 40; ++step) {
            const auto ledger = interfaceField.advectConservative(
                velocity, 0.08 / std::max(width, height), (step & 1u) == 0);
            require(ledger.accepted, "VOF flux audit step failed.");
            aggregate.evaluatedFaceCount += ledger.evaluatedFaceCount;
            aggregate.nonZeroLiquidFluxFaceCount += ledger.nonZeroLiquidFluxFaceCount;
            aggregate.uniqueLimitedFaceCount += ledger.uniqueLimitedFaceCount;
            aggregate.repeatedLimitEventCount += ledger.repeatedLimitEventCount;
            aggregate.limiterIterationCount += ledger.limiterIterationCount;
            aggregate.donorInsufficientCount += ledger.donorInsufficientCount;
            aggregate.receiverCapacityCount += ledger.receiverCapacityCount;
            aggregate.coupledInOutCount += ledger.coupledInOutCount;
            aggregate.highCourantLimitedCount += ledger.highCourantLimitedCount;
            aggregate.divergentCellLimitedCount += ledger.divergentCellLimitedCount;
            aggregate.plicFailureCount += ledger.plicFailureCount;
            aggregate.nearBlockedFaceCount += ledger.nearBlockedFaceCount;
            aggregate.rawAbsoluteFlux += ledger.rawAbsoluteFlux;
            aggregate.finalAbsoluteFlux += ledger.finalAbsoluteFlux;
            aggregate.limiterScaleSamples.insert(
                aggregate.limiterScaleSamples.end(),
                ledger.limiterScaleSamples.begin(), ledger.limiterScaleSamples.end());
        }
        std::sort(aggregate.limiterScaleSamples.begin(), aggregate.limiterScaleSamples.end());
        double meanScale = 1.0;
        double p95Scale = 1.0;
        double minimumScale = 1.0;
        if (!aggregate.limiterScaleSamples.empty()) {
            minimumScale = aggregate.limiterScaleSamples.front();
            meanScale = std::accumulate(
                aggregate.limiterScaleSamples.begin(), aggregate.limiterScaleSamples.end(), 0.0) /
                aggregate.limiterScaleSamples.size();
            p95Scale = aggregate.limiterScaleSamples[static_cast<std::size_t>(
                0.95 * (aggregate.limiterScaleSamples.size() - 1))];
        }
        const double fluxReduction = aggregate.rawAbsoluteFlux > 0.0
            ? 1.0 - aggregate.finalAbsoluteFlux / aggregate.rawAbsoluteFlux : 0.0;
        output << width << 'x' << height
               << " divergence=" << divergence
               << " fullMaxError=" << fullMaximumError
               << " fullLimited=" << fullLedger.uniqueLimitedFaceCount
               << " areaError=" << interfaceField.totalArea() - initialArea << '\n'
               << "  faces evaluated/nonzero/unique/repeated="
               << aggregate.evaluatedFaceCount << '/' << aggregate.nonZeroLiquidFluxFaceCount
               << '/' << aggregate.uniqueLimitedFaceCount << '/'
               << aggregate.repeatedLimitEventCount
               << " iterations=" << aggregate.limiterIterationCount << '\n'
               << "  flux raw/final/reduction=" << aggregate.rawAbsoluteFlux << '/'
               << aggregate.finalAbsoluteFlux << '/' << fluxReduction
               << " scale min/mean/p95=" << minimumScale << '/' << meanScale << '/'
               << p95Scale << '\n'
               << "  reasons donor/capacity/coupled/highCfl/divergent/plic/blocked="
               << aggregate.donorInsufficientCount << '/' << aggregate.receiverCapacityCount
               << '/' << aggregate.coupledInOutCount << '/'
               << aggregate.highCourantLimitedCount << '/'
               << aggregate.divergentCellLimitedCount << '/' << aggregate.plicFailureCount
               << '/' << aggregate.nearBlockedFaceCount << '\n';
        invariantPassed = invariantPassed && fullLedger.accepted &&
            divergence < 2.0e-5 && fullMaximumError < 2.0e-10 &&
            std::abs(interfaceField.totalArea() - initialArea) < 1.0e-9;
    }
    output << "VOF_FLUX_AUDIT_INVARIANTS " << (invariantPassed ? "PASS" : "FAIL") << '\n';
    return invariantPassed ? 0 : 1;
}

int runVofShapeConvergenceValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6)
           << "VOF reversible shape convergence validation\n";
    bool invariantPassed = true;
    double previousL1 = std::numeric_limits<double>::infinity();
    for (const auto [width, height] : std::vector<std::pair<std::size_t, std::size_t>>{
             {40, 30}, {80, 60}, {160, 120}}) {
        const auto shape = ellipse(0.50, 0.50, 0.19, 0.12);
        const VolumeFractionField initial = makeField(width, height, shape, 32);
        VolumeFractionField field = initial;
        MacGrid velocity(width, height,
            static_cast<float>(1.0 / width), static_cast<float>(1.0 / height));
        velocity.fillCellTypes(CellType::fluid);
        const std::size_t halfSteps = 30;
        const double dt = 0.05 / std::max(width, height);
        std::size_t uniqueLimited = 0;
        std::size_t nonZeroFlux = 0;
        double finalFlux = 0.0;
        for (std::size_t step = 0; step < 2 * halfSteps; ++step) {
            setDiscreteDivergenceFreeVelocity(velocity, step < halfSteps ? 0.025 : -0.025);
            const auto ledger = field.advectConservative(velocity, dt, (step & 1u) == 0);
            require(ledger.accepted, "VOF reversible deformation step failed.");
            uniqueLimited += ledger.uniqueLimitedFaceCount;
            nonZeroFlux += ledger.nonZeroLiquidFluxFaceCount;
            finalFlux += ledger.finalAbsoluteFlux;
        }
        const FractionError error = compareFields(field, initial);
        const FieldMoments before = measureMoments(initial);
        const FieldMoments after = measureMoments(field);
        const double centroidError = std::hypot(
            after.centroidX - before.centroidX, after.centroidY - before.centroidY);
        const double momentError = std::hypot(
            after.secondX - before.secondX, after.secondY - before.secondY);
        const std::size_t components = occupiedComponents(field);
        const double limiterRate = nonZeroFlux > 0
            ? static_cast<double>(uniqueLimited) / nonZeroFlux : 0.0;
        output << width << 'x' << height
               << " areaError=" << after.area - before.area
               << " l1=" << error.l1Area
               << " centroidError=" << centroidError
               << " secondMomentError=" << momentError
               << " components=" << components
               << " mixed/tiny=" << after.mixedCells << '/' << after.tinyTailCells
               << " limiterRate=" << limiterRate
               << " transportedFlux=" << finalFlux << '\n';
        invariantPassed = invariantPassed &&
            std::abs(after.area - before.area) < 1.0e-9 && components == 1;
        if (std::isfinite(previousL1) && error.l1Area > previousL1 * 1.25) {
            output << "  convergenceWarning=shape L1 did not decrease\n";
        }
        previousL1 = error.l1Area;
    }
    output << "VOF_SHAPE_INVARIANTS " << (invariantPassed ? "PASS" : "FAIL") << '\n';
    return invariantPassed ? 0 : 1;
}

int runVofHeightFunctionValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6)
           << "VOF PLIC normal source validation\n";
    const std::vector<std::pair<std::string, std::function<double(double, double)>>> cases{
        {"horizontal", [](double, double y) { return y - 0.47; }},
        {"tilted", [](double x, double y) {
            return (y - 0.34 - 0.28 * x) / std::hypot(1.0, 0.28);
        }},
        {"circle", circle(0.49, 0.51, 0.22)},
        {"ellipse", ellipse(0.49, 0.51, 0.25, 0.13)},
        {"thin-column", [](double x, double y) {
            return std::max(std::abs(x - 0.5) - 0.025, std::abs(y - 0.5) - 0.28);
        }},
        {"small-drop", circle(0.50, 0.50, 0.045)},
        {"two-drops", [](double x, double y) {
            return std::min(std::hypot(x - 0.39, y - 0.50) - 0.085,
                            std::hypot(x - 0.61, y - 0.50) - 0.085);
        }},
        {"wall-contact", circle(0.07, 0.50, 0.15)},
    };
    bool operational = true;
    for (const auto& [name, shape] : cases) {
        for (const auto [width, height] : std::vector<std::pair<std::size_t, std::size_t>>{
                 {40, 30}, {80, 60}, {160, 120}}) {
            const VolumeFractionField field = makeField(width, height, shape, 40);
            const NormalStatistics gradient = normalStatistics(
                field, shape, PlicNormalMethod::fractionGradient);
            const NormalStatistics heightFunction = normalStatistics(
                field, shape, PlicNormalMethod::heightFunction);
            const double coverage = heightFunction.interfaceCount > 0
                ? static_cast<double>(heightFunction.validCount) /
                    heightFunction.interfaceCount : 0.0;
            output << name << ' ' << width << 'x' << height
                   << " interface=" << gradient.interfaceCount
                   << " gradient(mean/rms/p95/max)=" << gradient.mean << '/'
                   << gradient.rms << '/' << gradient.p95 << '/' << gradient.maximum
                   << " height(valid/invalid/coverage)=" << heightFunction.validCount << '/'
                   << heightFunction.invalidCount << '/' << coverage
                   << " height(mean/rms/p95/max)=" << heightFunction.mean << '/'
                   << heightFunction.rms << '/' << heightFunction.p95 << '/'
                   << heightFunction.maximum << '\n';
            operational = operational && gradient.interfaceCount > 0 &&
                gradient.validCount + gradient.invalidCount == gradient.interfaceCount &&
                heightFunction.validCount + heightFunction.invalidCount ==
                    heightFunction.interfaceCount;
        }
    }
    output << "VOF_HEIGHT_FUNCTION_ACCOUNTING " << (operational ? "PASS" : "FAIL") << '\n';
    return operational ? 0 : 1;
}

int runClsvofGeometryCouplingValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6)
           << "CLSVOF PLIC-segment geometry coupling validation\n";
    bool operational = true;
    for (const auto [width, height] : std::vector<std::pair<std::size_t, std::size_t>>{
             {40, 40}, {80, 80}, {160, 160}, {80, 60}}) {
        const auto initialShape = circle(0.33, 0.50, 0.15);
        const double displacement = 0.075;
        const auto exactShape = circle(0.33 + displacement, 0.50, 0.15);
        VolumeFractionField vof = makeField(width, height, initialShape, 32);
        LevelSetField transported = fluidlab::simulation::sampleAnalyticLevelSet(
            width, height, 1.0 / width, 1.0 / height, initialShape);
        MacGrid velocity(width, height,
            static_cast<float>(1.0 / width), static_cast<float>(1.0 / height));
        velocity.fillCellTypes(CellType::fluid);
        setVelocity(velocity, [](double, double) { return Vector2{0.15f, 0.0f}; });
        const std::size_t steps = 30;
        const double dt = displacement / (0.15 * steps);
        for (std::size_t step = 0; step < steps; ++step) {
            require(vof.advectConservative(velocity, dt, (step & 1u) == 0).accepted,
                "CLSVOF geometry VOF step failed.");
            transported = fluidlab::simulation::advectLevelSetLimitedMacCormack(
                transported, [](double, double) { return Vector2{0.15f, 0.0f}; }, dt);
        }
        LevelSetField center = transported;
        LevelSetField segments = transported;
        ClsvofCouplingSettings settings;
        settings.narrowBandWidth = 6.0 / std::max(width, height);
        settings.maximumCorrectionDistance = 1.5 / std::max(width, height);
        settings.redistanceAfterCorrection = false;
        const auto centerReport = fluidlab::simulation::constrainLevelSetToVolumeFraction(
            vof, center, settings);
        const auto segmentReport = fluidlab::simulation::constrainLevelSetToPlicGeometry(
            vof, segments, settings, PlicNormalMethod::heightFunction);
        const LevelSetField exact = fluidlab::simulation::sampleAnalyticLevelSet(
            width, height, 1.0 / width, 1.0 / height, exactShape);
        const auto phiError = [&](const LevelSetField& field) {
            double square = 0.0;
            std::size_t count = 0;
            const double band = 3.0 / std::max(width, height);
            for (std::size_t index = 0; index < field.values.size(); ++index) {
                if (std::abs(exact.values[index]) > band) continue;
                const double difference = field.values[index] - exact.values[index];
                square += difference * difference;
                ++count;
            }
            return count > 0 ? std::sqrt(square / count) : 0.0;
        };
        const double baseArea = fluidlab::simulation::estimateLevelSetArea(transported, 4).area;
        const double centerArea = fluidlab::simulation::estimateLevelSetArea(center, 4).area;
        const double segmentArea = fluidlab::simulation::estimateLevelSetArea(segments, 4).area;
        const double baseError = phiError(transported);
        const double centerError = phiError(center);
        const double segmentError = phiError(segments);
        output << width << 'x' << height
               << " vofArea=" << vof.totalArea()
               << " phiArea(base/center/segment)=" << baseArea << '/' << centerArea << '/'
               << segmentArea
               << " phiRms(base/center/segment)=" << baseError << '/' << centerError << '/'
               << segmentError << '\n'
               << "  center corrected/fail=" << centerReport.correctedCellCount << '/'
               << centerReport.plicFailureCount
               << " segment corrected/valid/incompatible/fallback="
               << segmentReport.correctedCellCount << '/' << segmentReport.validSegmentCount
               << '/' << segmentReport.incompatibleSegmentCount << '/'
               << segmentReport.geometryFallbackCount
               << " segmentImproved=" << (segmentError <= baseError ? "yes" : "no") << '\n';
        operational = operational && segmentReport.validSegmentCount > 0 &&
            std::isfinite(segmentArea) && std::isfinite(segmentError);
    }
    output << "CLSVOF_GEOMETRY_OPERATIONAL " << (operational ? "PASS" : "FAIL") << '\n';
    return operational ? 0 : 1;
}

int runPlicGhostFluidValidation(std::ostream& output) {
    using fluidlab::simulation::applyGhostFluidMatrix;
    using fluidlab::simulation::assembleGhostFluidSystem;
    using fluidlab::simulation::buildGhostFluidTheta;
    using fluidlab::simulation::buildPlicGhostFluidTheta;
    using fluidlab::simulation::solveGhostFluidSystemDirect;
    output << std::scientific << std::setprecision(6)
           << "PLIC theta and CPU ghost-fluid validation\n";
    const std::vector<std::pair<std::string, std::function<double(double, double)>>> cases{
        {"horizontal", [](double, double y) { return y - 0.61; }},
        {"tilted", [](double x, double y) {
            return (y - 0.40 - 0.20 * x) / std::hypot(1.0, 0.20);
        }},
        {"circle", circle(0.50, 0.48, 0.28)},
        {"wall", circle(0.18, 0.50, 0.23)},
    };
    bool consistencyPassed = true;
    for (const auto& [name, shape] : cases) {
        const std::size_t width = name == "tilted" ? 36u : 32u;
        const std::size_t height = name == "tilted" ? 24u : 32u;
        MacGrid grid(width, height,
            static_cast<float>(1.0 / width), static_cast<float>(1.0 / height));
        classifyFromSignedDistance(grid, shape);
        const LevelSetField levelSet = fluidlab::simulation::sampleAnalyticLevelSet(
            width, height, 1.0 / width, 1.0 / height, shape);
        const VolumeFractionField vof = makeField(width, height, shape, 48);
        const auto exactTheta = buildGhostFluidTheta(grid, levelSet, 0.1);
        const auto gradientTheta = buildPlicGhostFluidTheta(
            grid, vof, PlicNormalMethod::fractionGradient, 0.1);
        const auto heightTheta = buildPlicGhostFluidTheta(
            grid, vof, PlicNormalMethod::heightFunction, 0.1);
        const ThetaDifference gradientDifference = compareTheta(exactTheta, gradientTheta);
        const ThetaDifference heightDifference = compareTheta(exactTheta, heightTheta);
        const auto exactSystem = assembleGhostFluidSystem(grid, exactTheta, 1.0, 1.0 / 120.0);
        const auto gradientSystem = assembleGhostFluidSystem(grid, gradientTheta, 1.0, 1.0 / 120.0);
        const auto heightSystem = assembleGhostFluidSystem(grid, heightTheta, 1.0, 1.0 / 120.0);
        std::vector<double> manufactured(exactSystem.cellIndices.size(), 0.0);
        for (std::size_t index = 0; index < manufactured.size(); ++index) {
            const std::size_t cell = exactSystem.cellIndices[index];
            const double x = (cell % width + 0.5) / width;
            const double y = (cell / width + 0.5) / height;
            manufactured[index] = 0.5 + 0.2 * x - 0.15 * y + 0.08 * x * y;
        }
        auto gradientManufactured = gradientSystem;
        gradientManufactured.rightHandSide = applyGhostFluidMatrix(exactSystem, manufactured);
        const auto gradientSolve = solveGhostFluidSystemDirect(gradientManufactured);
        auto heightManufactured = heightSystem;
        heightManufactured.rightHandSide = gradientManufactured.rightHandSide;
        const auto heightSolve = solveGhostFluidSystemDirect(heightManufactured);
        double pressureMaximumError = 0.0;
        double heightPressureMaximumError = 0.0;
        if (gradientSolve.solved && gradientSolve.pressure.size() == manufactured.size()) {
            for (std::size_t index = 0; index < manufactured.size(); ++index) {
                pressureMaximumError = std::max(
                    pressureMaximumError,
                    std::abs(gradientSolve.pressure[index] - manufactured[index]));
            }
        } else {
            pressureMaximumError = std::numeric_limits<double>::infinity();
        }
        if (heightSolve.solved && heightSolve.pressure.size() == manufactured.size()) {
            for (std::size_t index = 0; index < manufactured.size(); ++index) {
                heightPressureMaximumError = std::max(
                    heightPressureMaximumError,
                    std::abs(heightSolve.pressure[index] - manufactured[index]));
            }
        } else {
            heightPressureMaximumError = std::numeric_limits<double>::infinity();
        }
        double symmetry = 0.0;
        const std::size_t n = gradientSystem.cellIndices.size();
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t column = row + 1; column < n; ++column) {
                symmetry = std::max(symmetry, std::abs(
                    gradientSystem.matrix[row * n + column] -
                    gradientSystem.matrix[column * n + row]));
            }
        }
        output << name
               << " thetaGradient(compared/fallback/rms/max)="
               << gradientDifference.compared << '/' << gradientDifference.fallback << '/'
               << gradientDifference.rms << '/' << gradientDifference.maximum
               << " thetaHeight(compared/fallback/rms/max)="
               << heightDifference.compared << '/' << heightDifference.fallback << '/'
               << heightDifference.rms << '/' << heightDifference.maximum << '\n'
               << "  unknowns=" << n
               << " symmetry=" << symmetry
               << " pressureMaxError(gradient/height)=" << pressureMaximumError << '/'
               << heightPressureMaximumError
               << " solveRms=" << gradientSolve.residual.rootMeanSquare
               << " solveMax=" << gradientSolve.residual.maximumAbsolute << '\n';
        consistencyPassed = consistencyPassed && gradientSolve.solved && heightSolve.solved &&
            symmetry < 1.0e-12 &&
            gradientSolve.residual.rootMeanSquare < 1.0e-8 &&
            std::isfinite(pressureMaximumError);
    }
    output << "PLIC_GHOST_FLUID_ALGEBRA " << (consistencyPassed ? "PASS" : "FAIL") << '\n';
    return consistencyPassed ? 0 : 1;
}

int runVofSplitConservationValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6)
           << "VOF split compression conservation validation\n";
    bool passed = true;
    for (const auto [width, height] : std::vector<std::pair<std::size_t, std::size_t>>{
             {40, 30}, {80, 60}, {64, 64}}) {
        MacGrid velocity(width, height, 1.0f / width, 1.0f / height);
        velocity.fillCellTypes(CellType::fluid);
        setSplitTestVelocity(velocity, 0.025);
        double maximumSpeed = 0.0;
        for (float value : velocity.horizontalVelocities()) maximumSpeed = std::max(maximumSpeed, std::abs(static_cast<double>(value)));
        for (float value : velocity.verticalVelocities()) maximumSpeed = std::max(maximumSpeed, std::abs(static_cast<double>(value)));
        const double dt = 0.60 * std::min(1.0 / width, 1.0 / height) / maximumSpeed;
        VolumeFractionField legacy = makeField(width, height, [](double, double) { return -1.0; }, 2);
        VolumeFractionField corrected = legacy;
        const auto oldLedger = legacy.advectConservative(velocity, dt, true);
        const auto newLedger = corrected.advectConservativeSplitExperimental(velocity, dt, true);
        double fullError = 0.0;
        for (double value : corrected.fractions()) fullError = std::max(fullError, std::abs(value - 1.0));
        output << width << 'x' << height
               << " dt=" << dt
               << " legacyLimited=" << oldLedger.uniqueLimitedFaceCount
               << " correctedLimited=" << newLedger.uniqueLimitedFaceCount
               << " compression(x/y/sum)=" << newLedger.xCompressionContribution << '/'
               << newLedger.yCompressionContribution << '/' << newLedger.compressionContribution
               << " fullError=" << fullError
               << " residual=" << newLedger.conservationResidual << '\n';
        passed = passed && oldLedger.accepted && newLedger.accepted &&
            newLedger.outOfBoundsCount == 0 && fullError < 1.0e-9 &&
            std::abs(newLedger.conservationResidual) < 1.0e-10 &&
            newLedger.uniqueLimitedFaceCount <= oldLedger.uniqueLimitedFaceCount;
    }
    output << "VOF_SPLIT_CONSERVATION_RESULT " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

int runVofSplitBoundednessValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6)
           << "VOF split boundedness and CFL validation\n";
    bool passed = true;
    for (double targetCourant : {0.20, 0.50, 0.75}) {
        for (bool xFirst : {true, false}) {
            constexpr std::size_t width = 80;
            constexpr std::size_t height = 60;
            VolumeFractionField field = makeField(width, height, circle(0.48, 0.51, 0.21), 32);
            MacGrid velocity(width, height, 1.0f / width, 1.0f / height);
            velocity.fillCellTypes(CellType::fluid);
            setSplitTestVelocity(velocity, 0.020);
            double maximumDirectionalRate = 0.0;
            for (float value : velocity.horizontalVelocities()) maximumDirectionalRate = std::max(maximumDirectionalRate, std::abs(static_cast<double>(value)) * width);
            for (float value : velocity.verticalVelocities()) maximumDirectionalRate = std::max(maximumDirectionalRate, std::abs(static_cast<double>(value)) * height);
            const double dt = targetCourant / maximumDirectionalRate;
            const double initialArea = field.totalArea();
            const auto ledger = field.advectConservativeSplitExperimental(velocity, dt, xFirst);
            output << "CFL=" << targetCourant << " order=" << (xFirst ? "xy" : "yx")
                   << " min/max=" << ledger.minimumFraction << '/' << ledger.maximumFraction
                   << " limited=" << ledger.uniqueLimitedFaceCount
                   << " weaken=" << (ledger.rawAbsoluteFlux > 0.0
                       ? 1.0 - ledger.finalAbsoluteFlux / ledger.rawAbsoluteFlux : 0.0)
                   << " areaError=" << field.totalArea() - initialArea
                   << " residual=" << ledger.conservationResidual << '\n';
            passed = passed && ledger.accepted && ledger.outOfBoundsCount == 0 &&
                ledger.minimumFraction >= -1.0e-10 && ledger.maximumFraction <= 1.0 + 1.0e-10 &&
                std::abs(ledger.conservationResidual) < 1.0e-10;
        }
    }
    output << "VOF_SPLIT_BOUNDEDNESS_RESULT " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

int runVofSplitShapeValidation(std::ostream& output) {
    output << std::scientific << std::setprecision(6)
           << "VOF legacy/corrected reversible shape validation\n";
    bool passed = true;
    for (const auto [width, height] : std::vector<std::pair<std::size_t, std::size_t>>{
             {40, 30}, {80, 60}, {160, 120}}) {
        const auto shape = circle(0.50, 0.50, 0.18);
        const VolumeFractionField initial = makeField(width, height, shape, 40);
        VolumeFractionField legacy = initial;
        VolumeFractionField corrected = initial;
        MacGrid velocity(width, height, 1.0f / width, 1.0f / height);
        velocity.fillCellTypes(CellType::fluid);
        const double dt = 0.12 / std::max(width, height);
        std::size_t legacyLimited = 0;
        std::size_t correctedLimited = 0;
        double legacyRaw = 0.0;
        double legacyFinal = 0.0;
        double correctedRaw = 0.0;
        double correctedFinal = 0.0;
        bool accepted = true;
        for (std::size_t step = 0; step < 80; ++step) {
            setSplitTestVelocity(velocity, 0.018, step < 40 ? 1.0 : -1.0);
            const bool xFirst = (step & 1u) == 0;
            const auto oldLedger = legacy.advectConservative(velocity, dt, xFirst);
            const auto newLedger = corrected.advectConservativeSplitExperimental(velocity, dt, xFirst);
            accepted = accepted && oldLedger.accepted && newLedger.accepted;
            legacyLimited += oldLedger.uniqueLimitedFaceCount;
            correctedLimited += newLedger.uniqueLimitedFaceCount;
            legacyRaw += oldLedger.rawAbsoluteFlux;
            legacyFinal += oldLedger.finalAbsoluteFlux;
            correctedRaw += newLedger.rawAbsoluteFlux;
            correctedFinal += newLedger.finalAbsoluteFlux;
        }
        const FractionError oldError = compareFields(legacy, initial);
        const FractionError newError = compareFields(corrected, initial);
        const auto oldDistribution = legacy.distribution();
        const auto newDistribution = corrected.distribution();
        output << width << 'x' << height
               << " L1(legacy/new)=" << oldError.l1Area << '/' << newError.l1Area
               << " limited=" << legacyLimited << '/' << correctedLimited
               << " weaken=" << (legacyRaw > 0.0 ? 1.0 - legacyFinal / legacyRaw : 0.0) << '/'
               << (correctedRaw > 0.0 ? 1.0 - correctedFinal / correctedRaw : 0.0)
               << " lowTail=" << oldDistribution.lowTail << '/' << newDistribution.lowTail
               << " areaError=" << legacy.totalArea() - initial.totalArea() << '/'
               << corrected.totalArea() - initial.totalArea() << '\n';
        passed = passed && accepted && std::abs(corrected.totalArea() - initial.totalArea()) < 1.0e-9 &&
            newError.l1Area < 0.15;
    }
    output << "VOF_SPLIT_SHAPE_RESULT " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

int runVofSplitBenchmark(std::ostream& output) {
    output << std::fixed << std::setprecision(3) << "VOF split CPU benchmark\n";
    bool passed = true;
    for (const auto [width, height] : std::vector<std::pair<std::size_t, std::size_t>>{
             {40, 30}, {80, 60}, {160, 120}, {64, 64}, {128, 128}, {192, 192}, {256, 256}}) {
        const VolumeFractionField initial = makeField(width, height, circle(0.5, 0.5, 0.2), 24);
        MacGrid velocity(width, height, 1.0f / width, 1.0f / height);
        velocity.fillCellTypes(CellType::fluid);
        setSplitTestVelocity(velocity, 0.015);
        const double dt = 0.10 / std::max(width, height);
        double oldMs = 0.0;
        double newMs = 0.0;
        std::size_t oldIterations = 0;
        std::size_t newIterations = 0;
        for (std::size_t repeat = 0; repeat < 5; ++repeat) {
            VolumeFractionField oldField = initial;
            auto start = Clock::now();
            const auto oldLedger = oldField.advectConservative(velocity, dt, true);
            oldMs += elapsedMilliseconds(start);
            VolumeFractionField newField = initial;
            start = Clock::now();
            const auto newLedger = newField.advectConservativeSplitExperimental(velocity, dt, true);
            newMs += elapsedMilliseconds(start);
            oldIterations += oldLedger.limiterIterationCount;
            newIterations += newLedger.limiterIterationCount;
            passed = passed && oldLedger.accepted && newLedger.accepted;
        }
        const std::size_t cells = width * height;
        const double bytes = static_cast<double>(cells * (sizeof(double) + 2 * sizeof(unsigned char)));
        output << width << 'x' << height
               << " legacyMs=" << oldMs / 5.0
               << " correctedMs=" << newMs / 5.0
               << " limiterIterations=" << oldIterations / 5 << '/' << newIterations / 5
               << " experimentalStateKiB=" << bytes / 1024.0 << '\n';
    }
    output << "VOF_SPLIT_BENCHMARK_RESULT " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

} // namespace fluidlab::diagnostics
