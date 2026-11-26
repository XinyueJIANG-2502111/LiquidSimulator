#include "InterfaceValidation.h"

#include "../simulation/mac/GhostFluidPressureProjection.h"
#include "../simulation/mac/InterfaceGeometry.h"
#include "../simulation/mac/LevelSetInterface.h"
#include "../simulation/mac/MarkerParticles.h"

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

using Phi = fluidlab::simulation::ScalarFieldFunction;
using fluidlab::simulation::CellType;
using fluidlab::simulation::LevelSetField;
using fluidlab::simulation::MacGrid;
using fluidlab::simulation::MarkerParticle;
using fluidlab::simulation::Vector2;

constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Shape final {
    const char* name = "";
    Phi phi;
};

struct InterfaceError final {
    std::size_t validNormals = 0;
    std::size_t invalidNormals = 0;
    std::size_t expectedCrossings = 0;
    std::size_t reconstructedCrossings = 0;
    double positionRms = 0.0;
    double positionMaximum = 0.0;
    double normalMean = 0.0;
    double normalRms = 0.0;
    double normalP95 = 0.0;
    double normalMaximum = 0.0;
    double orientationConsistency = 0.0;
};

[[nodiscard]] Vector2 analyticNormal(const Phi& phi, double x, double y, double h) {
    const double dx = phi(x + h, y) - phi(x - h, y);
    const double dy = phi(x, y + h) - phi(x, y - h);
    const double length = std::hypot(dx, dy);
    if (!(length > 1.0e-14) || !std::isfinite(length)) return {};
    return {static_cast<float>(dx / length), static_cast<float>(dy / length)};
}

template <typename SampleNormal>
[[nodiscard]] InterfaceError evaluateCrossings(
    std::size_t width,
    std::size_t height,
    double dx,
    double dy,
    const std::vector<double>& values,
    const std::vector<unsigned char>& confidence,
    const Phi& analytic,
    SampleNormal sampleNormal) {
    InterfaceError result;
    std::vector<double> angles;
    double squaredPosition = 0.0;
    double squaredAngle = 0.0;
    double dotSum = 0.0;
    const auto indexOf = [width](std::size_t x, std::size_t y) { return y * width + x; };
    const auto process = [&](std::size_t first, std::size_t second, double x, double y,
                             double firstValue, double secondValue) {
        if (!std::isfinite(firstValue) || !std::isfinite(secondValue) ||
            (firstValue < 0.0) == (secondValue < 0.0)) return;
        ++result.reconstructedCrossings;
        if (confidence[first] == 0u || confidence[second] == 0u) {
            ++result.invalidNormals;
            return;
        }
        const double distance = std::abs(analytic(x, y));
        squaredPosition += distance * distance;
        result.positionMaximum = std::max(result.positionMaximum, distance);
        const Vector2 expected = analyticNormal(analytic, x, y, 1.0e-5);
        const Vector2 actual = sampleNormal(x, y);
        const double expectedLength = std::hypot(expected.x, expected.y);
        const double actualLength = std::hypot(actual.x, actual.y);
        if (!(expectedLength > 0.5) || !(actualLength > 0.5)) {
            ++result.invalidNormals;
            return;
        }
        const double dot = std::clamp(
            static_cast<double>(expected.x) * actual.x +
            static_cast<double>(expected.y) * actual.y, -1.0, 1.0);
        const double angle = std::acos(dot) * 180.0 / kPi;
        angles.push_back(angle);
        squaredAngle += angle * angle;
        dotSum += dot;
        ++result.validNormals;
    };
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x + 1 < width; ++x) {
            const std::size_t first = indexOf(x, y);
            const std::size_t second = indexOf(x + 1, y);
            const double a = values[first];
            const double b = values[second];
            if (std::isfinite(a) && std::isfinite(b) && (a < 0.0) != (b < 0.0)) {
                const double theta = std::abs(a) / (std::abs(a) + std::abs(b));
                process(first, second, (x + 0.5 + theta) * dx, (y + 0.5) * dy, a, b);
            }
            const double analyticA = analytic((x + 0.5) * dx, (y + 0.5) * dy);
            const double analyticB = analytic((x + 1.5) * dx, (y + 0.5) * dy);
            if ((analyticA < 0.0) != (analyticB < 0.0)) ++result.expectedCrossings;
        }
    }
    for (std::size_t y = 0; y + 1 < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t first = indexOf(x, y);
            const std::size_t second = indexOf(x, y + 1);
            const double a = values[first];
            const double b = values[second];
            if (std::isfinite(a) && std::isfinite(b) && (a < 0.0) != (b < 0.0)) {
                const double theta = std::abs(a) / (std::abs(a) + std::abs(b));
                process(first, second, (x + 0.5) * dx, (y + 0.5 + theta) * dy, a, b);
            }
            const double analyticA = analytic((x + 0.5) * dx, (y + 0.5) * dy);
            const double analyticB = analytic((x + 0.5) * dx, (y + 1.5) * dy);
            if ((analyticA < 0.0) != (analyticB < 0.0)) ++result.expectedCrossings;
        }
    }
    const std::size_t positionCount = result.validNormals + result.invalidNormals;
    result.positionRms = std::sqrt(squaredPosition / std::max<std::size_t>(positionCount, 1));
    if (!angles.empty()) {
        std::sort(angles.begin(), angles.end());
        result.normalMean = std::accumulate(angles.begin(), angles.end(), 0.0) / angles.size();
        result.normalRms = std::sqrt(squaredAngle / angles.size());
        result.normalP95 = angles[static_cast<std::size_t>(
            std::floor(0.95 * static_cast<double>(angles.size() - 1)))];
        result.normalMaximum = angles.back();
        result.orientationConsistency = dotSum / angles.size();
    }
    return result;
}

[[nodiscard]] double referenceArea(const Phi& phi, std::size_t samples = 1024) {
    std::size_t inside = 0;
    for (std::size_t y = 0; y < samples; ++y) {
        for (std::size_t x = 0; x < samples; ++x) {
            if (phi((x + 0.5) / samples, (y + 0.5) / samples) <= 0.0) ++inside;
        }
    }
    return static_cast<double>(inside) / static_cast<double>(samples * samples);
}

[[nodiscard]] std::vector<MarkerParticle> seedParticles(
    const Phi& phi,
    std::size_t width,
    std::size_t height,
    std::size_t particlesPerCell,
    bool uneven = false) {
    const std::size_t columns = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<double>(particlesPerCell))));
    const std::size_t rows = (particlesPerCell + columns - 1) / columns;
    std::vector<MarkerParticle> particles;
    std::size_t serial = 0;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            if (phi((x + 0.5) / width, (y + 0.5) / height) > 0.0) continue;
            for (std::size_t p = 0; p < particlesPerCell; ++p, ++serial) {
                if (uneven && serial % 5 == 0) continue;
                const double px = (x + (p % columns + 0.5) / columns) / width;
                const double py = (y + (p / columns + 0.5) / rows) / height;
                if (phi(px, py) <= 0.0) {
                    particles.push_back({
                        {static_cast<float>(px), static_cast<float>(py)}, true});
                }
            }
        }
    }
    return particles;
}

[[nodiscard]] double binaryArea(const Phi& phi, std::size_t width, std::size_t height) {
    std::size_t count = 0;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            if (phi((x + 0.5) / width, (y + 0.5) / height) <= 0.0) ++count;
        }
    }
    return static_cast<double>(count) / static_cast<double>(width * height);
}

[[nodiscard]] double maximumSymmetryError(
    const fluidlab::simulation::GhostFluidLinearSystem& system) {
    const std::size_t count = system.cellIndices.size();
    double maximum = 0.0;
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column < count; ++column) {
            maximum = std::max(maximum, std::abs(
                system.matrix[row * count + column] -
                system.matrix[column * count + row]));
        }
    }
    return maximum;
}

[[nodiscard]] double minimumCholeskyPivot(
    const fluidlab::simulation::GhostFluidLinearSystem& system) {
    const std::size_t count = system.cellIndices.size();
    std::vector<double> lower(count * count, 0.0);
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            double value = system.matrix[row * count + column];
            for (std::size_t k = 0; k < column; ++k) {
                value -= lower[row * count + k] * lower[column * count + k];
            }
            if (row == column) {
                if (!(value > 0.0)) return value;
                lower[row * count + column] = std::sqrt(value);
                minimum = std::min(minimum, value);
            } else {
                lower[row * count + column] = value / lower[column * count + column];
            }
        }
    }
    return minimum;
}

void classifyFromPhi(MacGrid& grid, const Phi& phi, bool solidBoundary) {
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (solidBoundary && (x == 0 || y == 0 || x + 1 == grid.width())) {
                grid.setCellType(x, y, CellType::solid);
            } else {
                const double px = (x + 0.5) * grid.cellWidth();
                const double py = (y + 0.5) * grid.cellHeight();
                grid.setCellType(x, y, phi(px, py) <= 0.0 ? CellType::fluid : CellType::air);
            }
        }
    }
}

} // namespace

int runInterfaceReconstructionValidation(std::ostream& output) {
    using namespace fluidlab::simulation;
    const auto horizontal = [](double, double y) { return y - 0.58; };
    const auto inclined = [](double x, double y) {
        return (y - (0.24 + 0.43 * x)) / std::sqrt(1.0 + 0.43 * 0.43);
    };
    const auto circle = [](double x, double y) { return std::hypot(x - 0.52, y - 0.51) - 0.23; };
    const auto ellipse = [](double x, double y) {
        return std::sqrt(((x - 0.52) / 0.27) * ((x - 0.52) / 0.27) +
            ((y - 0.48) / 0.17) * ((y - 0.48) / 0.17)) - 1.0;
    };
    const auto narrow = [](double x, double y) {
        const double qx = std::abs(x - 0.48) - 0.045;
        const double qy = std::abs(y - 0.45) - 0.31;
        return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) +
            std::min(std::max(qx, qy), 0.0);
    };
    const auto wall = [](double x, double y) { return std::hypot(x - 0.08, y - 0.52) - 0.24; };
    const auto corner = [](double x, double y) { return std::hypot(x - 0.06, y - 0.07) - 0.21; };
    const auto separated = [](double x, double y) {
        return std::min(std::hypot(x - 0.34, y - 0.5) - 0.14,
            std::hypot(x - 0.68, y - 0.5) - 0.14);
    };
    const auto shifted = [](double, double y) { return y - 0.592333; };
    const std::vector<Shape> shapes{
        {"horizontal", horizontal}, {"inclined", inclined}, {"circle", circle},
        {"ellipse", ellipse}, {"narrow", narrow}, {"wall", wall},
        {"corner", corner}, {"separated", separated}, {"shifted", shifted},
    };

    output << "LiquidSimulator Phase 1.9B interface reconstruction validation\n"
           << "Analytic level set is a trusted geometry input, not a particle reconstruction.\n"
           << "Grid Shape PosRmsCells PosMaxCells NormalMean NormalP95 NormalMax "
              "BinaryAreaRel LevelAreaRel InvalidRate Crossings TimeMs\n";
    for (const std::size_t width : {40U, 80U, 160U}) {
        const std::size_t height = width * 3 / 4;
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        for (const Shape& shape : shapes) {
            const auto start = std::chrono::steady_clock::now();
            const LevelSetField field = sampleAnalyticLevelSet(width, height, dx, dy, shape.phi);
            const auto area = estimateLevelSetArea(field, 4);
            const InterfaceError error = evaluateCrossings(width, height, dx, dy,
                field.values, field.confidence, shape.phi,
                [&](double x, double y) { return field.normal(x, y); });
            const double reference = referenceArea(shape.phi);
            const double cellArea = binaryArea(shape.phi, width, height);
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            const double localH = std::min(dx, dy);
            const double invalidRate = static_cast<double>(error.invalidNormals) /
                std::max<std::size_t>(error.validNormals + error.invalidNormals, 1);
            output << width << 'x' << height << ' ' << shape.name << ' '
                   << std::scientific << std::setprecision(3)
                   << error.positionRms / localH << ' ' << error.positionMaximum / localH << ' '
                   << std::fixed << std::setprecision(3)
                   << error.normalMean << ' ' << error.normalP95 << ' '
                   << error.normalMaximum << ' ' << std::scientific
                   << std::abs(cellArea - reference) / std::max(reference, 1.0e-12) << ' '
                   << std::abs(area.area - reference) / std::max(reference, 1.0e-12) << ' '
                   << invalidRate << ' ' << error.reconstructedCrossings << '/'
                   << error.expectedCrossings << ' ' << std::fixed << milliseconds << '\n';
            if (std::string(shape.name) == "horizontal" ||
                std::string(shape.name) == "inclined") {
                require(error.normalMean < 10.0 && error.positionRms / localH < 0.25,
                    "trusted plane level-set geometry missed its accuracy target");
            }
        }
    }

    output << "Corrected particle-union normal baseline\n"
           << "PPC Uneven Valid Invalid Mean RMS P95 Max Orientation MissingCrossings AreaRel\n";
    for (const std::size_t particlesPerCell : {4U, 8U, 16U}) {
        for (const bool uneven : {false, true}) {
            constexpr std::size_t width = 80;
            constexpr std::size_t height = 60;
            const auto particles = seedParticles(horizontal, width, height, particlesPerCell, uneven);
            InterfaceGeometrySettings settings;
            settings.particleSupportRadius = static_cast<float>(
                0.78 / (width * std::sqrt(static_cast<double>(particlesPerCell))));
            settings.areaSamplesPerAxis = 2;
            const auto proxy = reconstructParticleInterface(width, height, 1.0f / width,
                1.0f / height, particles, settings);
            std::vector<double> values(proxy.signedDistanceProxy.begin(),
                proxy.signedDistanceProxy.end());
            std::vector<unsigned char> confidence(width * height, 1u);
            const InterfaceError error = evaluateCrossings(width, height, 1.0 / width,
                1.0 / height, values, confidence, horizontal,
                [&](double x, double y) {
                    const std::size_t ix = std::min(width - 1,
                        static_cast<std::size_t>(x * width));
                    const std::size_t iy = std::min(height - 1,
                        static_cast<std::size_t>(y * height));
                    return proxy.normals[iy * width + ix];
                });
            const std::size_t missing = error.expectedCrossings > error.reconstructedCrossings
                ? error.expectedCrossings - error.reconstructedCrossings : 0;
            const double reference = referenceArea(horizontal);
            output << particlesPerCell << ' ' << (uneven ? "yes" : "no") << ' '
                   << error.validNormals << ' ' << error.invalidNormals << ' '
                   << std::fixed << std::setprecision(3) << error.normalMean << ' '
                   << error.normalRms << ' ' << error.normalP95 << ' '
                   << error.normalMaximum << ' ' << error.orientationConsistency << ' '
                   << missing << ' ' << std::scientific
                   << std::abs(proxy.estimatedArea - reference) / reference << '\n';
        }
    }

    output << "Deterministic transport\n"
           << "Case PosRmsCells NormalMean AreaRel ReinitAreaDelta ReinitMaxDelta Invalid\n";
    {
        constexpr std::size_t width = 80;
        constexpr std::size_t height = 60;
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        LevelSetField translated = sampleAnalyticLevelSet(width, height, dx, dy, circle);
        constexpr double vx = 0.08;
        constexpr double vy = -0.04;
        constexpr double dt = 0.005;
        constexpr std::size_t steps = 40;
        for (std::size_t step = 0; step < steps; ++step) {
            translated = advectLevelSetRk2(translated,
                [](double, double) { return Vector2{static_cast<float>(vx), static_cast<float>(vy)}; }, dt);
        }
        const Phi expected = [](double x, double y) {
            return std::hypot(x - (0.52 + vx * dt * steps),
                y - (0.51 + vy * dt * steps)) - 0.23;
        };
        const InterfaceError before = evaluateCrossings(width, height, dx, dy,
            translated.values, translated.confidence, expected,
            [&](double x, double y) { return translated.normal(x, y); });
        const double areaBefore = estimateLevelSetArea(translated, 4).area;
        LevelSetField reinitialized = translated;
        const auto reinit = reinitializeLevelSet(reinitialized, 6.0 * std::max(dx, dy));
        const double areaAfter = estimateLevelSetArea(reinitialized, 4).area;
        output << "translation " << std::scientific << before.positionRms / std::min(dx, dy)
               << ' ' << std::fixed << before.normalMean << ' ' << std::scientific
               << std::abs(areaBefore - referenceArea(expected)) / referenceArea(expected) << ' '
               << std::abs(areaAfter - areaBefore) << ' ' << reinit.maximumInterfaceDisplacement
               << ' ' << reinit.invalidCellCount << '\n';
        require(before.positionRms / std::min(dx, dy) < 0.25,
            "translated level set exceeded its position target");
    }
    {
        constexpr std::size_t width = 80;
        constexpr std::size_t height = 60;
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        const Phi offCenterCircle = [](double x, double y) {
            return std::hypot(x - 0.63, y - 0.49) - 0.16;
        };
        LevelSetField rotated = sampleAnalyticLevelSet(width, height, dx, dy, offCenterCircle);
        constexpr double omega = 0.7;
        constexpr double dt = 0.005;
        constexpr std::size_t steps = 80;
        for (std::size_t step = 0; step < steps; ++step) {
            rotated = advectLevelSetRk2(rotated, [](double x, double y) {
                return Vector2{static_cast<float>(-omega * (y - 0.5)),
                    static_cast<float>(omega * (x - 0.5))};
            }, dt);
        }
        const double angle = omega * dt * steps;
        const double cx = 0.5 + std::cos(angle) * 0.13 - std::sin(angle) * -0.01;
        const double cy = 0.5 + std::sin(angle) * 0.13 + std::cos(angle) * -0.01;
        const Phi expected = [=](double x, double y) { return std::hypot(x - cx, y - cy) - 0.16; };
        const InterfaceError error = evaluateCrossings(width, height, dx, dy,
            rotated.values, rotated.confidence, expected,
            [&](double x, double y) { return rotated.normal(x, y); });
        const double area = estimateLevelSetArea(rotated, 4).area;
        output << "rotation " << std::scientific << error.positionRms / std::min(dx, dy)
               << ' ' << std::fixed << error.normalMean << ' ' << std::scientific
               << std::abs(area - referenceArea(expected)) / referenceArea(expected)
               << " 0 0 " << error.invalidNormals << '\n';
    }
    {
        constexpr std::size_t width = 80;
        constexpr std::size_t height = 60;
        LevelSetField stationary = sampleAnalyticLevelSet(width, height,
            1.0 / width, 1.0 / height, inclined);
        const std::vector<double> original = stationary.values;
        for (std::size_t step = 0; step < 100; ++step) {
            stationary = advectLevelSetRk2(stationary,
                [](double, double) { return Vector2{}; }, 0.01);
        }
        double maximum = 0.0;
        for (std::size_t i = 0; i < original.size(); ++i) {
            maximum = std::max(maximum, std::abs(original[i] - stationary.values[i]));
        }
        output << "stationary 0 0 0 0 " << maximum << " 0\n";
        require(maximum < 1.0e-12, "stationary level set changed during transport");
    }

    output << "Geometry cost\nGrid BuildMs AdvectMs ReinitMs ThetaMs BufferKiB "
              "ValidTheta InvalidTheta ReinitInvalid\n";
    for (const std::size_t width : {64U, 128U, 192U, 256U}) {
        const std::size_t height = width;
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        const auto buildStart = std::chrono::steady_clock::now();
        LevelSetField field = sampleAnalyticLevelSet(width, height, dx, dy, circle, 19);
        const double buildMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - buildStart).count();
        const auto advectStart = std::chrono::steady_clock::now();
        LevelSetField advected = advectLevelSetRk2(field,
            [](double, double) { return Vector2{0.04f, -0.02f}; }, 1.0 / 120.0);
        const double advectMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - advectStart).count();
        const auto reinitStart = std::chrono::steady_clock::now();
        const auto reinit = reinitializeLevelSet(advected, 6.0 * std::max(dx, dy));
        const double reinitMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - reinitStart).count();
        MacGrid grid(width, height, static_cast<float>(dx), static_cast<float>(dy));
        classifyFromPhi(grid, circle, false);
        const auto thetaStart = std::chrono::steady_clock::now();
        const auto theta = buildGhostFluidTheta(grid, field, 0.1);
        const double thetaMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - thetaStart).count();
        const std::size_t bytes = field.values.size() * sizeof(double) +
            field.confidence.size() * sizeof(unsigned char) +
            theta.horizontalTheta.size() * sizeof(double) +
            theta.verticalTheta.size() * sizeof(double) +
            theta.horizontalState.size() * sizeof(GhostThetaState) +
            theta.verticalState.size() * sizeof(GhostThetaState);
        output << width << 'x' << height << ' ' << std::fixed << std::setprecision(3)
               << buildMs << ' ' << advectMs << ' ' << reinitMs << ' ' << thetaMs << ' '
               << static_cast<double>(bytes) / 1024.0 << ' '
               << theta.statistics.validFaceCount + theta.statistics.clampedFaceCount << ' '
               << theta.statistics.invalidFaceCount << ' ' << reinit.invalidCellCount << '\n';
    }

    output << "Particle conclusion: current interior marker samples do not determine a unique "
              "subcell surface; the trusted level set therefore remains an independent reference.\n"
           << "Phase 1.9B interface reconstruction validation result: PASS\n";
    return 0;
}

int runGhostFluidCpuValidation(std::ostream& output) {
    using namespace fluidlab::simulation;
    output << "LiquidSimulator Phase 1.9B CPU ghost-fluid validation\n"
           << "Case Unknowns ValidTheta Clamped Fallback Symmetry MinPivot RMS MaxResidual "
              "Consistency PostDiv PressureError\n";

    const std::vector<std::pair<std::size_t, std::size_t>> grids{{12, 8}, {15, 9}, {18, 12}};
    for (const auto [width, height] : grids) {
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        const double surface = 0.63;
        const Phi phi = [=](double, double y) { return y - surface; };
        MacGrid grid(width, height, static_cast<float>(dx), static_cast<float>(dy));
        classifyFromPhi(grid, phi, true);
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 1; x < width; ++x) {
                grid.horizontalVelocity(x, y) = static_cast<float>(
                    0.03 * std::sin(0.37 * x + 0.21 * y));
            }
        }
        for (std::size_t y = 1; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                grid.verticalVelocity(x, y) = static_cast<float>(
                    0.025 * std::cos(0.29 * x - 0.18 * y));
            }
        }
        grid.enforceSolidBoundaries();
        grid.computeDivergence();
        const LevelSetField levelSet = sampleAnalyticLevelSet(width, height, dx, dy, phi, 7);
        const GhostFluidThetaData theta = buildGhostFluidTheta(grid, levelSet, 0.1);
        GhostFluidLinearSystem system = assembleGhostFluidSystem(grid, theta, 1.0, 1.0 / 120.0);
        const double symmetry = maximumSymmetryError(system);
        const double minimumPivot = minimumCholeskyPivot(system);

        std::vector<double> manufactured(system.cellIndices.size());
        for (std::size_t i = 0; i < manufactured.size(); ++i) {
            const std::size_t cell = system.cellIndices[i];
            const double x = (cell % width + 0.5) * dx;
            const double y = (cell / width + 0.5) * dy;
            manufactured[i] = 0.4 + 0.3 * x - 0.2 * y + 0.1 * x * y;
        }
        GhostFluidLinearSystem manufacturedSystem = system;
        manufacturedSystem.rightHandSide = applyGhostFluidMatrix(system, manufactured);
        const GhostFluidSolveResult manufacturedSolve =
            solveGhostFluidSystemDirect(manufacturedSystem);
        double pressureError = 0.0;
        for (std::size_t i = 0; i < manufactured.size(); ++i) {
            pressureError = std::max(pressureError,
                std::abs(manufacturedSolve.pressure[i] - manufactured[i]));
        }

        const GhostFluidSolveResult solve = solveGhostFluidSystemDirect(system);
        require(solve.solved, "ghost-fluid direct solve failed");
        writeGhostFluidPressure(grid, system, solve.pressure);
        const float projectionChange =
            applyGhostFluidPressureGradient(grid, theta, 1.0, 1.0 / 120.0);
        static_cast<void>(projectionChange);
        grid.enforceSolidBoundaries();
        grid.computeDivergence();
        double postDivergence = 0.0;
        for (const std::size_t cell : system.cellIndices) {
            postDivergence = std::max(postDivergence,
                std::abs(static_cast<double>(grid.divergence(cell % width, cell / width))));
        }

        MacGrid consistencyGrid(width, height, static_cast<float>(dx), static_cast<float>(dy));
        classifyFromPhi(consistencyGrid, phi, true);
        writeGhostFluidPressure(consistencyGrid, system, manufactured);
        const float consistencyChange =
            applyGhostFluidPressureGradient(consistencyGrid, theta, 1.0, 1.0 / 120.0);
        static_cast<void>(consistencyChange);
        consistencyGrid.computeDivergence();
        const std::vector<double> applied = applyGhostFluidMatrix(system, manufactured);
        double consistency = 0.0;
        for (std::size_t i = 0; i < applied.size(); ++i) {
            const std::size_t cell = system.cellIndices[i];
            const double expected = (1.0 / 120.0) * applied[i];
            consistency = std::max(consistency, std::abs(
                consistencyGrid.divergence(cell % width, cell / width) - expected));
        }
        output << width << 'x' << height << ' ' << system.cellIndices.size() << ' '
               << theta.statistics.validFaceCount << ' ' << theta.statistics.clampedFaceCount
               << ' ' << theta.statistics.fallbackFaceCount << ' '
               << std::scientific << symmetry << ' ' << minimumPivot << ' '
               << solve.residual.rootMeanSquare << ' ' << solve.residual.maximumAbsolute << ' '
               << consistency << ' ' << postDivergence << ' ' << pressureError << '\n';
        require(symmetry < 1.0e-12 && minimumPivot > 0.0 &&
                consistency < 5.0e-5 && postDivergence < 5.0e-5 && pressureError < 1.0e-10,
            "ghost-fluid matrix or projection consistency failed");
    }

    {
        constexpr std::size_t width = 13;
        constexpr std::size_t height = 9;
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        const Phi inclined = [](double x, double y) {
            return (y - (0.31 + 0.24 * x)) / std::sqrt(1.0 + 0.24 * 0.24);
        };
        MacGrid grid(width, height, static_cast<float>(dx), static_cast<float>(dy));
        classifyFromPhi(grid, inclined, true);
        grid.setCellType(6, 3, CellType::solid);
        grid.computeDivergence();
        const auto levelSet = sampleAnalyticLevelSet(width, height, dx, dy, inclined, 23);
        const auto theta = buildGhostFluidTheta(grid, levelSet, 0.1);
        auto system = assembleGhostFluidSystem(grid, theta, 1.0, 1.0 / 120.0);
        std::vector<double> manufactured(system.cellIndices.size());
        for (std::size_t i = 0; i < manufactured.size(); ++i) {
            const std::size_t cell = system.cellIndices[i];
            const double x = (cell % width + 0.5) * dx;
            const double y = (cell / width + 0.5) * dy;
            manufactured[i] = 0.2 + x * x + 0.3 * y + 0.1 * x * y;
        }
        system.rightHandSide = applyGhostFluidMatrix(system, manufactured);
        const auto solve = solveGhostFluidSystemDirect(system);
        double pressureError = 0.0;
        for (std::size_t i = 0; i < manufactured.size(); ++i) {
            pressureError = std::max(pressureError,
                std::abs(solve.pressure[i] - manufactured[i]));
        }
        output << "InclinedManufactured unknowns=" << manufactured.size()
               << " symmetry=" << maximumSymmetryError(system)
               << " min-pivot=" << minimumCholeskyPivot(system)
               << " pressure-error=" << pressureError
               << " solid-faces=" << theta.statistics.solidAdjacentFaceCount << '\n';
        require(solve.solved && pressureError < 1.0e-10 &&
                maximumSymmetryError(system) < 1.0e-12 &&
                minimumCholeskyPivot(system) > 0.0,
            "inclined ghost-fluid manufactured solution failed");
    }

    {
        constexpr std::size_t width = 16;
        constexpr std::size_t height = 12;
        constexpr double gravity = 9.81;
        constexpr double density = 1.0;
        constexpr double dt = 1.0 / 120.0;
        const double dx = 1.0 / width;
        const double dy = 1.0 / height;
        const double surface = 0.61;
        const Phi horizontal = [=](double, double y) { return y - surface; };
        auto buildHydroGrid = [&]() {
            MacGrid grid(width, height, static_cast<float>(dx), static_cast<float>(dy));
            classifyFromPhi(grid, horizontal, true);
            for (std::size_t y = 1; y < height; ++y) {
                for (std::size_t x = 1; x + 1 < width; ++x) {
                    const CellType below = grid.cellType(x, y - 1);
                    const CellType above = grid.cellType(x, y);
                    if (below != CellType::solid && above != CellType::solid &&
                        (below == CellType::fluid || above == CellType::fluid)) {
                        grid.verticalVelocity(x, y) = static_cast<float>(-gravity * dt);
                    }
                }
            }
            grid.enforceSolidBoundaries();
            grid.computeDivergence();
            return grid;
        };
        const LevelSetField levelSet = sampleAnalyticLevelSet(
            width, height, dx, dy, horizontal, 31);
        MacGrid ghostGrid = buildHydroGrid();
        const auto ghostTheta = buildGhostFluidTheta(ghostGrid, levelSet, 0.1);
        const auto ghostSystem = assembleGhostFluidSystem(
            ghostGrid, ghostTheta, density, dt);
        const auto ghostSolve = solveGhostFluidSystemDirect(ghostSystem);
        require(ghostSolve.solved, "ghost-fluid hydrostatic solve failed");
        double ghostPressureError = 0.0;
        for (std::size_t i = 0; i < ghostSolve.pressure.size(); ++i) {
            const std::size_t cell = ghostSystem.cellIndices[i];
            const double y = (cell / width + 0.5) * dy;
            ghostPressureError = std::max(ghostPressureError,
                std::abs(ghostSolve.pressure[i] - density * gravity * (surface - y)));
        }
        writeGhostFluidPressure(ghostGrid, ghostSystem, ghostSolve.pressure);
        const float ghostChange = applyGhostFluidPressureGradient(
            ghostGrid, ghostTheta, density, dt);
        static_cast<void>(ghostChange);
        ghostGrid.enforceSolidBoundaries();
        double ghostMaximumVelocity = 0.0;
        for (const float velocity : ghostGrid.verticalVelocities()) {
            ghostMaximumVelocity = std::max(ghostMaximumVelocity,
                std::abs(static_cast<double>(velocity)));
        }

        MacGrid cellGrid = buildHydroGrid();
        LevelSetField cellLevelSet = buildClassificationLevelSet(cellGrid);
        auto cellTheta = buildGhostFluidTheta(cellGrid, cellLevelSet, 1.0);
        const auto cellSystem = assembleGhostFluidSystem(cellGrid, cellTheta, density, dt);
        const auto cellSolve = solveGhostFluidSystemDirect(cellSystem);
        double cellPressureError = 0.0;
        for (std::size_t i = 0; i < cellSolve.pressure.size(); ++i) {
            const std::size_t cell = cellSystem.cellIndices[i];
            const double y = (cell / width + 0.5) * dy;
            cellPressureError = std::max(cellPressureError,
                std::abs(cellSolve.pressure[i] - density * gravity * (surface - y)));
        }
        output << "Hydrostatic ghost-pressure-error=" << std::scientific
               << ghostPressureError << " cell-pressure-error=" << cellPressureError
               << " ghost-max-speed=" << ghostMaximumVelocity << '\n';
        require(ghostPressureError < cellPressureError && ghostMaximumVelocity < 1.0e-5,
            "ghost-fluid hydrostatic reference did not improve the interface pressure");
    }

    {
        constexpr std::size_t width = 10;
        constexpr std::size_t height = 8;
        const Phi horizontal = [](double, double y) { return y - 0.52; };
        MacGrid grid(width, height, 1.0f / width, 1.0f / height);
        classifyFromPhi(grid, horizontal, false);
        LevelSetField unreliable = sampleAnalyticLevelSet(
            width, height, 1.0 / width, 1.0 / height, horizontal, 37);
        for (std::size_t x = 0; x < width; ++x) {
            unreliable.confidence[4 * width + x] = 0u;
        }
        const auto theta = buildGhostFluidTheta(grid, unreliable, 0.1, true);
        output << "InvalidTheta fallback=" << theta.statistics.fallbackFaceCount
               << " invalid=" << theta.statistics.invalidFaceCount << '\n';
        require(theta.statistics.fallbackFaceCount != 0,
            "invalid theta did not use the explicit cell-boundary fallback");
    }

    output << "Theta clamp study\nTheta Raw Clamped Valid ClampedCount Fallback\n";
    for (const double minimumTheta : {0.05, 0.1, 0.2}) {
        constexpr std::size_t width = 10;
        constexpr std::size_t height = 8;
        const double dy = 1.0 / height;
        const double surface = (4.5 + 0.02) * dy;
        const Phi phi = [=](double, double y) { return y - surface; };
        MacGrid grid(width, height, 1.0f / width, 1.0f / height);
        classifyFromPhi(grid, phi, false);
        const LevelSetField levelSet = sampleAnalyticLevelSet(width, height,
            1.0 / width, 1.0 / height, phi, 11);
        const GhostFluidThetaData theta = buildGhostFluidTheta(grid, levelSet, minimumTheta);
        double observed = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t y = 1; y < height; ++y) {
            const std::size_t face = y * width + width / 2;
            if (theta.verticalState[face] != GhostThetaState::unused) {
                observed = theta.verticalTheta[face];
                break;
            }
        }
        output << minimumTheta << " 0.02 " << observed << ' '
               << theta.statistics.validFaceCount << ' ' << theta.statistics.clampedFaceCount
               << ' ' << theta.statistics.fallbackFaceCount << '\n';
        require(observed >= minimumTheta, "theta clamp was not applied");
    }

    {
        MacGrid closed(8, 6, 1.0f / 8.0f, 1.0f / 6.0f);
        closed.fillCellTypes(CellType::fluid);
        closed.computeDivergence();
        const LevelSetField phi = sampleAnalyticLevelSet(8, 6, 1.0 / 8.0, 1.0 / 6.0,
            [](double, double) { return -1.0; });
        const auto theta = buildGhostFluidTheta(closed, phi, 0.1);
        const auto system = assembleGhostFluidSystem(closed, theta, 1.0, 1.0 / 120.0);
        std::vector<double> ones(system.cellIndices.size(), 1.0);
        const auto nullResult = applyGhostFluidMatrix(system, ones);
        double nullMaximum = 0.0;
        for (const double value : nullResult) nullMaximum = std::max(nullMaximum, std::abs(value));
        const auto solve = solveGhostFluidSystemDirect(system);
        output << "PureNeumann components=" << system.pureNeumannComponentCount
               << " null-max=" << nullMaximum << " solved=" << solve.solved << '\n';
        require(system.pureNeumannComponentCount == 1 && nullMaximum < 1.0e-10 && solve.solved,
            "pure Neumann null space handling failed");
    }
    {
        MacGrid split(9, 6, 1.0f / 9.0f, 1.0f / 6.0f);
        split.fillCellTypes(CellType::solid);
        for (std::size_t y = 1; y < 5; ++y) {
            for (std::size_t x = 1; x < 4; ++x) split.setCellType(x, y, CellType::fluid);
            for (std::size_t x = 5; x < 8; ++x) split.setCellType(x, y, CellType::fluid);
        }
        split.computeDivergence();
        const LevelSetField phi = sampleAnalyticLevelSet(9, 6, 1.0 / 9.0, 1.0 / 6.0,
            [](double, double) { return -1.0; });
        const auto theta = buildGhostFluidTheta(split, phi, 0.1);
        const auto system = assembleGhostFluidSystem(split, theta, 1.0, 1.0 / 120.0);
        output << "TwoNeumann components=" << system.pureNeumannComponentCount
               << " supported=" << system.supported << '\n';
        require(system.pureNeumannComponentCount == 2 && !system.supported,
            "multiple pure Neumann components were not rejected");
    }

    output << "Version design: classification, interface, theta, fine operator, and MG hierarchy "
              "must have independent monotonically increasing versions. Theta changes invalidate "
              "the fine operator and every Galerkin level even when cell classification is unchanged.\n"
           << "GPU ghost-fluid experiment: not implemented because current marker particles do not "
              "provide a reliable dynamic subcell interface source.\n"
           << "Phase 1.9B CPU ghost-fluid validation result: PASS\n";
    return 0;
}

} // namespace fluidlab::diagnostics
