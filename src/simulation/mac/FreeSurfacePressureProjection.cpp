#include "FreeSurfacePressureProjection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace fluidlab::simulation {
namespace {

[[nodiscard]] bool isFluid(CellType type) noexcept {
    return type == CellType::fluid;
}

} // namespace

FreeSurfacePressureProjector::FreeSurfacePressureProjector(PressureSolverSettings settings)
    : m_settings(settings) {
    if (!std::isfinite(settings.density) || settings.density <= 0.0f ||
        !std::isfinite(settings.timeStep) || settings.timeStep <= 0.0f ||
        !std::isfinite(settings.residualTolerance) || settings.residualTolerance < 0.0f ||
        !std::isfinite(settings.relaxationFactor) || settings.relaxationFactor <= 0.0f ||
        settings.relaxationFactor >= 2.0f || settings.maximumIterations == 0 ||
        settings.residualCheckInterval == 0) {
        throw std::invalid_argument("Invalid free-surface pressure solver settings.");
    }
}

FreeSurfaceProjectionReport FreeSurfacePressureProjector::project(MacGrid& grid) const {
    using Clock = std::chrono::steady_clock;
    const auto totalStart = Clock::now();

    grid.enforceSolidBoundaries();
    grid.computeDivergence();
    const DomainInfo domain = inspectDomain(grid);

    FreeSurfaceProjectionReport result;
    result.fluidCellCount = domain.fluidCellCount;
    result.fluidAirFaceCount = domain.fluidAirFaceCount;
    result.pressureReferenceFixedByAir = domain.fluidAirFaceCount != 0;
    ProjectionReport& report = result.projection;
    report.divergenceBefore = fluidDivergenceStatistics(grid);
    const auto divergenceEnd = Clock::now();

    PressureSolveReport& solve = report.pressureSolve;
    solve.warmStartUsed = m_settings.warmStartEnabled;
    solve.residualTolerance = m_settings.residualTolerance;
    if (!m_settings.warmStartEnabled) {
        grid.clearPressure();
    }
    clearNonFluidPressure(grid);
    if (!result.pressureReferenceFixedByAir) {
        solve.initialMaximumAbsolutePressure = normalizeClosedFluidPressure(grid);
        ++solve.pressureMeanRemovalCount;
    } else {
        solve.initialMaximumAbsolutePressure = maximumAbsoluteFluidPressure(grid);
    }

    solve.initialResidual = computeResidual(grid);
    solve.finalResidual = solve.initialResidual;
    if (solve.initialResidual.rootMeanSquare <= m_settings.residualTolerance) {
        solve.finalMaximumAbsolutePressure = solve.initialMaximumAbsolutePressure;
        solve.converged = true;
        solve.termination = PressureSolveTermination::initialResidualWithinTolerance;
    } else {
        const float inverseDxSquared = 1.0f / (grid.cellWidth() * grid.cellWidth());
        const float inverseDySquared = 1.0f / (grid.cellHeight() * grid.cellHeight());
        const float rightHandScale = m_settings.density / m_settings.timeStep;

        for (std::size_t iteration = 1; iteration <= m_settings.maximumIterations; ++iteration) {
            for (std::size_t parity = 0; parity < 2; ++parity) {
                for (std::size_t y = 0; y < grid.height(); ++y) {
                    for (std::size_t x = 0; x < grid.width(); ++x) {
                        if (((x + y) & 1U) != parity || !isFluid(grid.cellType(x, y))) {
                            continue;
                        }

                        float neighborSum = 0.0f;
                        float diagonal = 0.0f;
                        const auto addNeighbor = [&](std::size_t nx, std::size_t ny, float coefficient) {
                            const CellType type = grid.cellType(nx, ny);
                            if (type == CellType::fluid) {
                                neighborSum += coefficient * grid.pressure(nx, ny);
                                diagonal += coefficient;
                            } else if (type == CellType::air) {
                                // Atmospheric gauge pressure is zero at a cell-level interface.
                                diagonal += coefficient;
                            }
                        };
                        if (x > 0) addNeighbor(x - 1, y, inverseDxSquared);
                        if (x + 1 < grid.width()) addNeighbor(x + 1, y, inverseDxSquared);
                        if (y > 0) addNeighbor(x, y - 1, inverseDySquared);
                        if (y + 1 < grid.height()) addNeighbor(x, y + 1, inverseDySquared);

                        if (diagonal == 0.0f) {
                            grid.pressure(x, y) = 0.0f;
                            continue;
                        }
                        const float rhs = rightHandScale * grid.divergence(x, y);
                        const float candidate = (neighborSum - rhs) / diagonal;
                        float& pressure = grid.pressure(x, y);
                        pressure += m_settings.relaxationFactor * (candidate - pressure);
                    }
                }
            }

            if (iteration % m_settings.residualCheckInterval == 0 ||
                iteration == m_settings.maximumIterations) {
                solve.finalResidual = computeResidual(grid);
                solve.iterations = iteration;
                if (solve.finalResidual.rootMeanSquare <= m_settings.residualTolerance) {
                    solve.converged = true;
                    solve.termination = PressureSolveTermination::iterativeResidualWithinTolerance;
                    break;
                }
            }
        }
        if (!result.pressureReferenceFixedByAir) {
            solve.finalMaximumAbsolutePressure = normalizeClosedFluidPressure(grid);
            ++solve.pressureMeanRemovalCount;
        } else {
            solve.finalMaximumAbsolutePressure = maximumAbsoluteFluidPressure(grid);
        }
    }
    const auto solveEnd = Clock::now();

    report.maximumVelocityChange = applyPressureGradient(grid);
    grid.enforceSolidBoundaries();
    grid.computeDivergence();
    report.divergenceAfter = fluidDivergenceStatistics(grid);
    const auto totalEnd = Clock::now();
    report.timings.divergenceBeforeMilliseconds =
        std::chrono::duration<double, std::milli>(divergenceEnd - totalStart).count();
    report.timings.pressureSolveMilliseconds =
        std::chrono::duration<double, std::milli>(solveEnd - divergenceEnd).count();
    report.timings.pressureApplicationMilliseconds =
        std::chrono::duration<double, std::milli>(totalEnd - solveEnd).count();
    report.timings.totalMilliseconds =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    return result;
}

const PressureSolverSettings& FreeSurfacePressureProjector::settings() const noexcept {
    return m_settings;
}

FreeSurfacePressureProjector::DomainInfo FreeSurfacePressureProjector::inspectDomain(
    const MacGrid& grid) const noexcept {
    DomainInfo info;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (!isFluid(grid.cellType(x, y))) continue;
            ++info.fluidCellCount;
            if (x > 0 && grid.cellType(x - 1, y) == CellType::air) ++info.fluidAirFaceCount;
            if (x + 1 < grid.width() && grid.cellType(x + 1, y) == CellType::air) ++info.fluidAirFaceCount;
            if (y > 0 && grid.cellType(x, y - 1) == CellType::air) ++info.fluidAirFaceCount;
            if (y + 1 < grid.height() && grid.cellType(x, y + 1) == CellType::air) ++info.fluidAirFaceCount;
        }
    }
    return info;
}

DivergenceStatistics FreeSurfacePressureProjector::fluidDivergenceStatistics(
    const MacGrid& grid) const noexcept {
    DivergenceStatistics statistics;
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (!isFluid(grid.cellType(x, y))) continue;
            const float value = std::abs(grid.divergence(x, y));
            statistics.maximumAbsolute = std::max(statistics.maximumAbsolute, value);
            sum += value;
            ++count;
        }
    }
    statistics.meanAbsolute = count != 0 ? static_cast<float>(sum / count) : 0.0f;
    return statistics;
}

PressureResidual FreeSurfacePressureProjector::computeResidual(const MacGrid& grid) const noexcept {
    const float ix2 = 1.0f / (grid.cellWidth() * grid.cellWidth());
    const float iy2 = 1.0f / (grid.cellHeight() * grid.cellHeight());
    const float rhsScale = m_settings.density / m_settings.timeStep;
    PressureResidual result;
    double squared = 0.0;
    std::size_t count = 0;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            if (!isFluid(grid.cellType(x, y))) continue;
            const float center = grid.pressure(x, y);
            float laplacian = 0.0f;
            const auto add = [&](std::size_t nx, std::size_t ny, float coefficient) {
                const CellType type = grid.cellType(nx, ny);
                if (type == CellType::fluid) laplacian += coefficient * (grid.pressure(nx, ny) - center);
                else if (type == CellType::air) laplacian += coefficient * (0.0f - center);
            };
            if (x > 0) add(x - 1, y, ix2);
            if (x + 1 < grid.width()) add(x + 1, y, ix2);
            if (y > 0) add(x, y - 1, iy2);
            if (y + 1 < grid.height()) add(x, y + 1, iy2);
            const float value = laplacian - rhsScale * grid.divergence(x, y);
            result.maximumAbsolute = std::max(result.maximumAbsolute, std::abs(value));
            squared += static_cast<double>(value) * value;
            ++count;
        }
    }
    result.rootMeanSquare = count != 0 ? static_cast<float>(std::sqrt(squared / count)) : 0.0f;
    return result;
}

float FreeSurfacePressureProjector::normalizeClosedFluidPressure(MacGrid& grid) const noexcept {
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t y = 0; y < grid.height(); ++y) for (std::size_t x = 0; x < grid.width(); ++x) {
        if (isFluid(grid.cellType(x, y))) { sum += grid.pressure(x, y); ++count; }
    }
    const float mean = count != 0 ? static_cast<float>(sum / count) : 0.0f;
    float maximum = 0.0f;
    for (std::size_t y = 0; y < grid.height(); ++y) for (std::size_t x = 0; x < grid.width(); ++x) {
        if (isFluid(grid.cellType(x, y))) {
            grid.pressure(x, y) -= mean;
            maximum = std::max(maximum, std::abs(grid.pressure(x, y)));
        }
    }
    return maximum;
}

float FreeSurfacePressureProjector::maximumAbsoluteFluidPressure(const MacGrid& grid) const noexcept {
    float maximum = 0.0f;
    for (std::size_t y = 0; y < grid.height(); ++y) for (std::size_t x = 0; x < grid.width(); ++x) {
        if (isFluid(grid.cellType(x, y))) maximum = std::max(maximum, std::abs(grid.pressure(x, y)));
    }
    return maximum;
}

float FreeSurfacePressureProjector::applyPressureGradient(MacGrid& grid) const noexcept {
    const float sx = m_settings.timeStep / (m_settings.density * grid.cellWidth());
    const float sy = m_settings.timeStep / (m_settings.density * grid.cellHeight());
    float maximum = 0.0f;
    for (std::size_t y = 0; y < grid.height(); ++y) for (std::size_t x = 1; x < grid.width(); ++x) {
        const CellType left = grid.cellType(x - 1, y), right = grid.cellType(x, y);
        float& velocity = grid.horizontalVelocity(x, y);
        const float before = velocity;
        if (left == CellType::solid || right == CellType::solid) velocity = 0.0f;
        else if (isFluid(left) || isFluid(right)) {
            const float pl = isFluid(left) ? grid.pressure(x - 1, y) : 0.0f;
            const float pr = isFluid(right) ? grid.pressure(x, y) : 0.0f;
            velocity -= sx * (pr - pl);
        }
        maximum = std::max(maximum, std::abs(velocity - before));
    }
    for (std::size_t y = 1; y < grid.height(); ++y) for (std::size_t x = 0; x < grid.width(); ++x) {
        const CellType below = grid.cellType(x, y - 1), above = grid.cellType(x, y);
        float& velocity = grid.verticalVelocity(x, y);
        const float before = velocity;
        if (below == CellType::solid || above == CellType::solid) velocity = 0.0f;
        else if (isFluid(below) || isFluid(above)) {
            const float pb = isFluid(below) ? grid.pressure(x, y - 1) : 0.0f;
            const float pa = isFluid(above) ? grid.pressure(x, y) : 0.0f;
            velocity -= sy * (pa - pb);
        }
        maximum = std::max(maximum, std::abs(velocity - before));
    }
    return maximum;
}

void FreeSurfacePressureProjector::clearNonFluidPressure(MacGrid& grid) const noexcept {
    for (std::size_t y = 0; y < grid.height(); ++y) for (std::size_t x = 0; x < grid.width(); ++x) {
        if (!isFluid(grid.cellType(x, y))) grid.pressure(x, y) = 0.0f;
    }
}

} // namespace fluidlab::simulation
