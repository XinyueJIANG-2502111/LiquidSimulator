#include "PressureProjection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace fluidlab::simulation {

PressureProjector::PressureProjector(PressureSolverSettings settings)
    : m_settings(settings) {
    if (!std::isfinite(settings.density) || settings.density <= 0.0f) {
        throw std::invalid_argument("Fluid density must be a positive finite value.");
    }
    if (!std::isfinite(settings.timeStep) || settings.timeStep <= 0.0f) {
        throw std::invalid_argument("Projection timestep must be a positive finite value.");
    }
    if (!std::isfinite(settings.residualTolerance) || settings.residualTolerance < 0.0f) {
        throw std::invalid_argument("Pressure residual tolerance must be finite and non-negative.");
    }
    if (!std::isfinite(settings.relaxationFactor) ||
        settings.relaxationFactor <= 0.0f || settings.relaxationFactor >= 2.0f) {
        throw std::invalid_argument("SOR relaxation factor must lie in the open interval (0, 2).");
    }
    if (settings.maximumIterations == 0 || settings.residualCheckInterval == 0) {
        throw std::invalid_argument("Pressure iteration counts must be greater than zero.");
    }
}

ProjectionReport PressureProjector::project(MacGrid& grid) const {
    using Clock = std::chrono::steady_clock;
    const auto totalStart = Clock::now();

    grid.enforceClosedBoundary();
    grid.computeDivergence();

    ProjectionReport report;
    report.divergenceBefore = grid.divergenceStatistics();
    const auto divergenceEnd = Clock::now();
    report.pressureSolve = solvePressure(grid);
    const auto solveEnd = Clock::now();
    report.maximumVelocityChange = applyPressureGradient(grid);
    grid.enforceClosedBoundary();
    grid.computeDivergence();
    report.divergenceAfter = grid.divergenceStatistics();
    const auto totalEnd = Clock::now();
    report.timings.divergenceBeforeMilliseconds =
        std::chrono::duration<double, std::milli>(divergenceEnd - totalStart).count();
    report.timings.pressureSolveMilliseconds =
        std::chrono::duration<double, std::milli>(solveEnd - divergenceEnd).count();
    report.timings.pressureApplicationMilliseconds =
        std::chrono::duration<double, std::milli>(totalEnd - solveEnd).count();
    report.timings.totalMilliseconds =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    return report;
}

const PressureSolverSettings& PressureProjector::settings() const noexcept {
    return m_settings;
}

PressureSolveReport PressureProjector::solvePressure(MacGrid& grid) const {
    if (!m_settings.warmStartEnabled) {
        grid.clearPressure();
    }

    PressureSolveReport report;
    report.warmStartUsed = m_settings.warmStartEnabled;
    report.residualTolerance = m_settings.residualTolerance;
    if (m_settings.warmStartEnabled) {
        // Normalize the retained guess once so long runs cannot accumulate a
        // large null-space offset that does not contribute to velocity.
        report.initialMaximumAbsolutePressure = removePressureMean(grid);
        ++report.pressureMeanRemovalCount;
    }
    report.initialResidual = computeResidual(grid);
    report.finalResidual = report.initialResidual;
    if (report.initialResidual.rootMeanSquare <= m_settings.residualTolerance) {
        if (!m_settings.warmStartEnabled) {
            report.finalMaximumAbsolutePressure = removePressureMean(grid);
            ++report.pressureMeanRemovalCount;
        } else {
            report.finalMaximumAbsolutePressure = report.initialMaximumAbsolutePressure;
        }
        report.converged = true;
        report.termination = PressureSolveTermination::initialResidualWithinTolerance;
        return report;
    }

    const float inverseDxSquared = 1.0f / (grid.cellWidth() * grid.cellWidth());
    const float inverseDySquared = 1.0f / (grid.cellHeight() * grid.cellHeight());
    const float rightHandScale = m_settings.density / m_settings.timeStep;

    // Red-black Gauss-Seidel permits an in-place update while preserving a
    // deterministic ordering. SOR accelerates the low-frequency convergence
    // that dominates a closed 64 x 64 pressure solve.
    float latestMaximumAbsolutePressure = report.initialMaximumAbsolutePressure;
    for (std::size_t iteration = 1; iteration <= m_settings.maximumIterations; ++iteration) {
        for (std::size_t parity = 0; parity < 2; ++parity) {
            for (std::size_t y = 0; y < grid.height(); ++y) {
                for (std::size_t x = 0; x < grid.width(); ++x) {
                    if (((x + y) & 1U) != parity) {
                        continue;
                    }

                    float neighborSum = 0.0f;
                    float diagonal = 0.0f;
                    if (x > 0) {
                        neighborSum += inverseDxSquared * grid.pressure(x - 1, y);
                        diagonal += inverseDxSquared;
                    }
                    if (x + 1 < grid.width()) {
                        neighborSum += inverseDxSquared * grid.pressure(x + 1, y);
                        diagonal += inverseDxSquared;
                    }
                    if (y > 0) {
                        neighborSum += inverseDySquared * grid.pressure(x, y - 1);
                        diagonal += inverseDySquared;
                    }
                    if (y + 1 < grid.height()) {
                        neighborSum += inverseDySquared * grid.pressure(x, y + 1);
                        diagonal += inverseDySquared;
                    }

                    // Missing neighbors use a zero normal pressure derivative.
                    // Algebraically this removes their coefficient from both the
                    // neighbor sum and diagonal, matching a fixed wall face.
                    const float rightHandSide = rightHandScale * grid.divergence(x, y);
                    const float gaussSeidelValue = (neighborSum - rightHandSide) / diagonal;
                    float& pressure = grid.pressure(x, y);
                    pressure += m_settings.relaxationFactor * (gaussSeidelValue - pressure);
                }
            }
        }

        if (m_settings.pressureMeanRemovalInterval != 0 &&
            iteration % m_settings.pressureMeanRemovalInterval == 0) {
            // Pure Neumann pressure is unique only up to an additive constant.
            // Periodic normalization controls null-space drift without changing
            // pressure gradients or any physical cell equation.
            latestMaximumAbsolutePressure = removePressureMean(grid);
            ++report.pressureMeanRemovalCount;
        }

        if (iteration % m_settings.residualCheckInterval == 0 || iteration == m_settings.maximumIterations) {
            report.finalResidual = computeResidual(grid);
            report.iterations = iteration;
            if (report.finalResidual.rootMeanSquare <= m_settings.residualTolerance) {
                report.converged = true;
                report.termination = PressureSolveTermination::iterativeResidualWithinTolerance;
                break;
            }
        }
    }

    if (m_settings.pressureMeanRemovalInterval == 0 ||
        report.iterations % m_settings.pressureMeanRemovalInterval != 0) {
        latestMaximumAbsolutePressure = removePressureMean(grid);
        ++report.pressureMeanRemovalCount;
    }
    report.finalMaximumAbsolutePressure = latestMaximumAbsolutePressure;
    return report;
}

PressureResidual PressureProjector::computeResidual(const MacGrid& grid) const noexcept {
    const float inverseDxSquared = 1.0f / (grid.cellWidth() * grid.cellWidth());
    const float inverseDySquared = 1.0f / (grid.cellHeight() * grid.cellHeight());
    const float rightHandScale = m_settings.density / m_settings.timeStep;

    PressureResidual residual;
    double squaredSum = 0.0;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const float center = grid.pressure(x, y);
            float laplacian = 0.0f;
            if (x > 0) {
                laplacian += inverseDxSquared * (grid.pressure(x - 1, y) - center);
            }
            if (x + 1 < grid.width()) {
                laplacian += inverseDxSquared * (grid.pressure(x + 1, y) - center);
            }
            if (y > 0) {
                laplacian += inverseDySquared * (grid.pressure(x, y - 1) - center);
            }
            if (y + 1 < grid.height()) {
                laplacian += inverseDySquared * (grid.pressure(x, y + 1) - center);
            }

            const float value = laplacian - rightHandScale * grid.divergence(x, y);
            residual.maximumAbsolute = std::max(residual.maximumAbsolute, std::abs(value));
            squaredSum += static_cast<double>(value) * static_cast<double>(value);
        }
    }

    const double cellCount = static_cast<double>(grid.width() * grid.height());
    residual.rootMeanSquare = static_cast<float>(std::sqrt(squaredSum / cellCount));
    return residual;
}

float PressureProjector::removePressureMean(MacGrid& grid) const noexcept {
    double sum = 0.0;
    for (const float value : grid.pressures()) {
        sum += value;
    }
    const float mean = static_cast<float>(sum / static_cast<double>(grid.width() * grid.height()));
    float maximumAbsolute = 0.0f;
    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            grid.pressure(x, y) -= mean;
            maximumAbsolute = std::max(maximumAbsolute, std::abs(grid.pressure(x, y)));
        }
    }
    return maximumAbsolute;
}

float PressureProjector::applyPressureGradient(MacGrid& grid) const noexcept {
    const float scaleX = m_settings.timeStep / (m_settings.density * grid.cellWidth());
    const float scaleY = m_settings.timeStep / (m_settings.density * grid.cellHeight());
    float maximumVelocityChange = 0.0f;

    for (std::size_t y = 0; y < grid.height(); ++y) {
        for (std::size_t faceX = 1; faceX < grid.width(); ++faceX) {
            const float gradient = grid.pressure(faceX, y) - grid.pressure(faceX - 1, y);
            float& velocity = grid.horizontalVelocity(faceX, y);
            const float previousVelocity = velocity;
            velocity -= scaleX * gradient;
            maximumVelocityChange = std::max(maximumVelocityChange, std::abs(velocity - previousVelocity));
        }
    }
    for (std::size_t faceY = 1; faceY < grid.height(); ++faceY) {
        for (std::size_t x = 0; x < grid.width(); ++x) {
            const float gradient = grid.pressure(x, faceY) - grid.pressure(x, faceY - 1);
            float& velocity = grid.verticalVelocity(x, faceY);
            const float previousVelocity = velocity;
            velocity -= scaleY * gradient;
            maximumVelocityChange = std::max(maximumVelocityChange, std::abs(velocity - previousVelocity));
        }
    }
    return maximumVelocityChange;
}

} // namespace fluidlab::simulation
