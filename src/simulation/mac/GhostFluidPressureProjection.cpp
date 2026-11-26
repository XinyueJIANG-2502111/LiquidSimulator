#include "GhostFluidPressureProjection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace fluidlab::simulation {
namespace {

[[nodiscard]] std::size_t cellIndex(
    std::size_t width, std::size_t x, std::size_t y) noexcept {
    return y * width + x;
}

[[nodiscard]] std::size_t horizontalIndex(
    std::size_t width, std::size_t x, std::size_t y) noexcept {
    return y * (width + 1) + x;
}

[[nodiscard]] std::size_t verticalIndex(
    std::size_t width, std::size_t x, std::size_t y) noexcept {
    return y * width + x;
}

[[nodiscard]] bool isFluid(CellType type) noexcept {
    return type == CellType::fluid;
}

[[nodiscard]] bool isFluidAir(CellType first, CellType second) noexcept {
    return (first == CellType::fluid && second == CellType::air) ||
        (first == CellType::air && second == CellType::fluid);
}

struct ThetaResult final {
    double value = 1.0;
    GhostThetaState state = GhostThetaState::fallback;
};

[[nodiscard]] ThetaResult computeTheta(
    double fluidPhi,
    double airPhi,
    unsigned char fluidConfidence,
    unsigned char airConfidence,
    double minimumTheta,
    bool fallbackToCellBoundary) noexcept {
    if (fluidConfidence != 0u && airConfidence != 0u &&
        std::isfinite(fluidPhi) && std::isfinite(airPhi) &&
        fluidPhi <= 0.0 && airPhi >= 0.0) {
        const double denominator = std::abs(fluidPhi) + std::abs(airPhi);
        if (denominator > 1.0e-14) {
            const double raw = std::abs(fluidPhi) / denominator;
            if (raw > 0.0 && raw <= 1.0) {
                if (raw < minimumTheta) {
                    return {minimumTheta, GhostThetaState::clamped};
                }
                return {raw, GhostThetaState::valid};
            }
        }
    }
    return fallbackToCellBoundary
        ? ThetaResult{1.0, GhostThetaState::fallback}
        : ThetaResult{std::numeric_limits<double>::quiet_NaN(), GhostThetaState::fallback};
}

void recordTheta(GhostThetaStatistics& statistics, GhostThetaState state) noexcept {
    if (state == GhostThetaState::valid) ++statistics.validFaceCount;
    else if (state == GhostThetaState::clamped) ++statistics.clampedFaceCount;
    else if (state == GhostThetaState::fallback) {
        ++statistics.invalidFaceCount;
        ++statistics.fallbackFaceCount;
    }
}

[[nodiscard]] std::vector<double> solveDense(
    std::vector<double> matrix,
    std::vector<double> rightHandSide,
    std::size_t dimension) {
    for (std::size_t column = 0; column < dimension; ++column) {
        std::size_t pivot = column;
        double pivotMagnitude = std::abs(matrix[column * dimension + column]);
        for (std::size_t row = column + 1; row < dimension; ++row) {
            const double magnitude = std::abs(matrix[row * dimension + column]);
            if (magnitude > pivotMagnitude) {
                pivot = row;
                pivotMagnitude = magnitude;
            }
        }
        if (!(pivotMagnitude > 1.0e-13) || !std::isfinite(pivotMagnitude)) return {};
        if (pivot != column) {
            for (std::size_t entry = column; entry < dimension; ++entry) {
                std::swap(matrix[column * dimension + entry],
                    matrix[pivot * dimension + entry]);
            }
            std::swap(rightHandSide[column], rightHandSide[pivot]);
        }
        const double inversePivot = 1.0 / matrix[column * dimension + column];
        for (std::size_t row = column + 1; row < dimension; ++row) {
            const double factor = matrix[row * dimension + column] * inversePivot;
            if (factor == 0.0) continue;
            matrix[row * dimension + column] = 0.0;
            for (std::size_t entry = column + 1; entry < dimension; ++entry) {
                matrix[row * dimension + entry] -=
                    factor * matrix[column * dimension + entry];
            }
            rightHandSide[row] -= factor * rightHandSide[column];
        }
    }
    std::vector<double> solution(dimension, 0.0);
    for (std::size_t reverse = 0; reverse < dimension; ++reverse) {
        const std::size_t row = dimension - 1 - reverse;
        double value = rightHandSide[row];
        for (std::size_t column = row + 1; column < dimension; ++column) {
            value -= matrix[row * dimension + column] * solution[column];
        }
        solution[row] = value / matrix[row * dimension + row];
    }
    return solution;
}

} // namespace

GhostFluidThetaData buildGhostFluidTheta(
    const MacGrid& grid,
    const LevelSetField& interfaceField,
    double minimumTheta,
    bool fallbackToCellBoundary) {
    if (interfaceField.width != grid.width() || interfaceField.height != grid.height() ||
        !(minimumTheta > 0.0) || minimumTheta > 1.0) {
        throw std::invalid_argument("Ghost-fluid theta input is incompatible with the grid.");
    }
    GhostFluidThetaData result;
    result.width = grid.width();
    result.height = grid.height();
    result.minimumTheta = minimumTheta;
    result.interfaceVersion = interfaceField.sourceVersion;
    result.horizontalTheta.assign((result.width + 1) * result.height,
        std::numeric_limits<double>::quiet_NaN());
    result.verticalTheta.assign(result.width * (result.height + 1),
        std::numeric_limits<double>::quiet_NaN());
    result.horizontalState.assign(result.horizontalTheta.size(), GhostThetaState::unused);
    result.verticalState.assign(result.verticalTheta.size(), GhostThetaState::unused);

    for (std::size_t y = 0; y < result.height; ++y) {
        for (std::size_t x = 1; x < result.width; ++x) {
            const CellType left = grid.cellType(x - 1, y);
            const CellType right = grid.cellType(x, y);
            if (left == CellType::solid || right == CellType::solid) {
                ++result.statistics.solidAdjacentFaceCount;
                continue;
            }
            if (!isFluidAir(left, right)) continue;
            const std::size_t leftCell = cellIndex(result.width, x - 1, y);
            const std::size_t rightCell = cellIndex(result.width, x, y);
            const bool leftFluid = left == CellType::fluid;
            const std::size_t fluidCell = leftFluid ? leftCell : rightCell;
            const std::size_t airCell = leftFluid ? rightCell : leftCell;
            const ThetaResult theta = computeTheta(
                interfaceField.values[fluidCell], interfaceField.values[airCell],
                interfaceField.confidence[fluidCell], interfaceField.confidence[airCell],
                minimumTheta, fallbackToCellBoundary);
            const std::size_t face = horizontalIndex(result.width, x, y);
            result.horizontalTheta[face] = theta.value;
            result.horizontalState[face] = theta.state;
            recordTheta(result.statistics, theta.state);
        }
    }
    for (std::size_t y = 1; y < result.height; ++y) {
        for (std::size_t x = 0; x < result.width; ++x) {
            const CellType below = grid.cellType(x, y - 1);
            const CellType above = grid.cellType(x, y);
            if (below == CellType::solid || above == CellType::solid) {
                ++result.statistics.solidAdjacentFaceCount;
                continue;
            }
            if (!isFluidAir(below, above)) continue;
            const std::size_t belowCell = cellIndex(result.width, x, y - 1);
            const std::size_t aboveCell = cellIndex(result.width, x, y);
            const bool belowFluid = below == CellType::fluid;
            const std::size_t fluidCell = belowFluid ? belowCell : aboveCell;
            const std::size_t airCell = belowFluid ? aboveCell : belowCell;
            const ThetaResult theta = computeTheta(
                interfaceField.values[fluidCell], interfaceField.values[airCell],
                interfaceField.confidence[fluidCell], interfaceField.confidence[airCell],
                minimumTheta, fallbackToCellBoundary);
            const std::size_t face = verticalIndex(result.width, x, y);
            result.verticalTheta[face] = theta.value;
            result.verticalState[face] = theta.state;
            recordTheta(result.statistics, theta.state);
        }
    }
    return result;
}

GhostFluidLinearSystem assembleGhostFluidSystem(
    const MacGrid& grid,
    const GhostFluidThetaData& theta,
    double density,
    double timeStep) {
    if (theta.width != grid.width() || theta.height != grid.height() ||
        !(density > 0.0) || !(timeStep > 0.0)) {
        throw std::invalid_argument("Ghost-fluid system input is invalid.");
    }
    GhostFluidLinearSystem system;
    system.width = grid.width();
    system.height = grid.height();
    system.unknownIndex.assign(system.width * system.height, -1);
    for (std::size_t y = 0; y < system.height; ++y) {
        for (std::size_t x = 0; x < system.width; ++x) {
            if (!isFluid(grid.cellType(x, y))) continue;
            system.unknownIndex[cellIndex(system.width, x, y)] =
                static_cast<int>(system.cellIndices.size());
            system.cellIndices.push_back(cellIndex(system.width, x, y));
        }
    }
    const std::size_t count = system.cellIndices.size();
    system.matrix.assign(count * count, 0.0);
    system.rightHandSide.assign(count, 0.0);
    const double inverseDxSquared = 1.0 /
        (static_cast<double>(grid.cellWidth()) * grid.cellWidth());
    const double inverseDySquared = 1.0 /
        (static_cast<double>(grid.cellHeight()) * grid.cellHeight());

    for (std::size_t row = 0; row < count; ++row) {
        const std::size_t cell = system.cellIndices[row];
        const std::size_t x = cell % system.width;
        const std::size_t y = cell / system.width;
        system.rightHandSide[row] = -density / timeStep * grid.divergence(x, y);
        const auto addNeighbor = [&](std::size_t nx, std::size_t ny, double coefficient,
                                     double interfaceTheta) {
            const CellType type = grid.cellType(nx, ny);
            if (type == CellType::fluid) {
                const int column = system.unknownIndex[cellIndex(system.width, nx, ny)];
                system.matrix[row * count + row] += coefficient;
                system.matrix[row * count + static_cast<std::size_t>(column)] -= coefficient;
            } else if (type == CellType::air) {
                const double safeTheta = std::isfinite(interfaceTheta) && interfaceTheta > 0.0
                    ? interfaceTheta : 1.0;
                system.matrix[row * count + row] += coefficient / safeTheta;
                system.hasDirichletBoundary = true;
            }
        };
        if (x > 0) addNeighbor(x - 1, y, inverseDxSquared,
            theta.horizontalTheta[horizontalIndex(system.width, x, y)]);
        if (x + 1 < system.width) addNeighbor(x + 1, y, inverseDxSquared,
            theta.horizontalTheta[horizontalIndex(system.width, x + 1, y)]);
        if (y > 0) addNeighbor(x, y - 1, inverseDySquared,
            theta.verticalTheta[verticalIndex(system.width, x, y)]);
        if (y + 1 < system.height) addNeighbor(x, y + 1, inverseDySquared,
            theta.verticalTheta[verticalIndex(system.width, x, y + 1)]);
    }

    std::vector<unsigned char> visited(count, 0u);
    for (std::size_t start = 0; start < count; ++start) {
        if (visited[start] != 0u) continue;
        bool touchesAir = false;
        std::vector<std::size_t> component;
        std::queue<std::size_t> pending;
        pending.push(start);
        visited[start] = 1u;
        while (!pending.empty()) {
            const std::size_t row = pending.front();
            pending.pop();
            component.push_back(row);
            const std::size_t cell = system.cellIndices[row];
            const std::size_t x = cell % system.width;
            const std::size_t y = cell / system.width;
            const auto inspect = [&](std::size_t nx, std::size_t ny) {
                const CellType type = grid.cellType(nx, ny);
                if (type == CellType::air) touchesAir = true;
                if (type != CellType::fluid) return;
                const std::size_t neighbor = static_cast<std::size_t>(
                    system.unknownIndex[cellIndex(system.width, nx, ny)]);
                if (visited[neighbor] == 0u) {
                    visited[neighbor] = 1u;
                    pending.push(neighbor);
                }
            };
            if (x > 0) inspect(x - 1, y);
            if (x + 1 < system.width) inspect(x + 1, y);
            if (y > 0) inspect(x, y - 1);
            if (y + 1 < system.height) inspect(x, y + 1);
        }
        if (!touchesAir) {
            ++system.pureNeumannComponentCount;
            if (system.pureNeumannComponentCount == 1) {
                system.pureNeumannUnknowns = std::move(component);
            }
        }
    }
    system.supported = system.pureNeumannComponentCount <= 1;
    return system;
}

std::vector<double> applyGhostFluidMatrix(
    const GhostFluidLinearSystem& system,
    const std::vector<double>& pressure) {
    if (pressure.size() != system.cellIndices.size()) {
        throw std::invalid_argument("Ghost-fluid matrix vector size is invalid.");
    }
    const std::size_t count = pressure.size();
    std::vector<double> result(count, 0.0);
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column < count; ++column) {
            result[row] += system.matrix[row * count + column] * pressure[column];
        }
    }
    return result;
}

GhostFluidSolveResult solveGhostFluidSystemDirect(
    const GhostFluidLinearSystem& system) {
    GhostFluidSolveResult result;
    if (!system.supported) return result;
    const std::size_t count = system.cellIndices.size();
    if (count == 0) {
        result.solved = true;
        return result;
    }
    std::vector<double> rightHandSide = system.rightHandSide;
    if (system.pureNeumannComponentCount == 1) {
        double mean = 0.0;
        for (const std::size_t index : system.pureNeumannUnknowns) {
            mean += rightHandSide[index];
        }
        mean /= static_cast<double>(system.pureNeumannUnknowns.size());
        result.compatibleRightHandSide = std::abs(mean) <= 1.0e-8;
        for (const std::size_t index : system.pureNeumannUnknowns) {
            rightHandSide[index] -= mean;
        }
        const std::size_t augmented = count + 1;
        std::vector<double> matrix(augmented * augmented, 0.0);
        std::vector<double> rhs(augmented, 0.0);
        for (std::size_t row = 0; row < count; ++row) {
            for (std::size_t column = 0; column < count; ++column) {
                matrix[row * augmented + column] = system.matrix[row * count + column];
            }
            rhs[row] = rightHandSide[row];
        }
        for (const std::size_t index : system.pureNeumannUnknowns) {
            matrix[index * augmented + count] = 1.0;
            matrix[count * augmented + index] = 1.0;
        }
        std::vector<double> augmentedSolution = solveDense(matrix, rhs, augmented);
        if (augmentedSolution.empty()) return result;
        result.pressure.assign(augmentedSolution.begin(), augmentedSolution.begin() + count);
    } else {
        result.pressure = solveDense(system.matrix, rightHandSide, count);
        if (result.pressure.empty()) return result;
    }
    const std::vector<double> applied = applyGhostFluidMatrix(system, result.pressure);
    double squared = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double value = applied[i] - rightHandSide[i];
        result.residual.maximumAbsolute = std::max(
            result.residual.maximumAbsolute, static_cast<float>(std::abs(value)));
        squared += value * value;
    }
    result.residual.rootMeanSquare = static_cast<float>(std::sqrt(squared / count));
    result.solved = std::isfinite(result.residual.rootMeanSquare) &&
        std::isfinite(result.residual.maximumAbsolute);
    return result;
}

void writeGhostFluidPressure(
    MacGrid& grid,
    const GhostFluidLinearSystem& system,
    const std::vector<double>& pressure) noexcept {
    grid.clearPressure();
    for (std::size_t index = 0; index < pressure.size(); ++index) {
        const std::size_t cell = system.cellIndices[index];
        grid.pressure(cell % system.width, cell / system.width) =
            static_cast<float>(pressure[index]);
    }
}

float applyGhostFluidPressureGradient(
    MacGrid& grid,
    const GhostFluidThetaData& theta,
    double density,
    double timeStep) noexcept {
    float maximumChange = 0.0f;
    const double scaleX = timeStep / (density * grid.cellWidth());
    const double scaleY = timeStep / (density * grid.cellHeight());
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 1; x < grid.width(); ++x) {
            const CellType left = grid.cellType(x - 1, y);
            const CellType right = grid.cellType(x, y);
            float& velocity = grid.horizontalVelocity(x, y);
            const float before = velocity;
            if (left == CellType::solid || right == CellType::solid) {
                velocity = 0.0f;
            } else if (isFluid(left) && isFluid(right)) {
                velocity -= static_cast<float>(scaleX *
                    (grid.pressure(x, y) - grid.pressure(x - 1, y)));
            } else if (isFluidAir(left, right)) {
                const double faceTheta = theta.horizontalTheta[horizontalIndex(grid.width(), x, y)];
                const double safeTheta = std::isfinite(faceTheta) && faceTheta > 0.0
                    ? faceTheta : 1.0;
                const double gradient = left == CellType::fluid
                    ? -grid.pressure(x - 1, y) / safeTheta
                    : grid.pressure(x, y) / safeTheta;
                velocity -= static_cast<float>(scaleX * gradient);
            }
            maximumChange = std::max(maximumChange, std::abs(velocity - before));
        }
    }
    for (std::size_t y = 1; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const CellType below = grid.cellType(x, y - 1);
            const CellType above = grid.cellType(x, y);
            float& velocity = grid.verticalVelocity(x, y);
            const float before = velocity;
            if (below == CellType::solid || above == CellType::solid) {
                velocity = 0.0f;
            } else if (isFluid(below) && isFluid(above)) {
                velocity -= static_cast<float>(scaleY *
                    (grid.pressure(x, y) - grid.pressure(x, y - 1)));
            } else if (isFluidAir(below, above)) {
                const double faceTheta = theta.verticalTheta[verticalIndex(grid.width(), x, y)];
                const double safeTheta = std::isfinite(faceTheta) && faceTheta > 0.0
                    ? faceTheta : 1.0;
                const double gradient = below == CellType::fluid
                    ? -grid.pressure(x, y - 1) / safeTheta
                    : grid.pressure(x, y) / safeTheta;
                velocity -= static_cast<float>(scaleY * gradient);
            }
            maximumChange = std::max(maximumChange, std::abs(velocity - before));
        }
    }
    return maximumChange;
}

} // namespace fluidlab::simulation
