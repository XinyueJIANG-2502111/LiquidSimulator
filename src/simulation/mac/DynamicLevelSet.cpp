#include "DynamicLevelSet.h"

#include "VelocityAdvection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fluidlab::simulation {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::size_t indexOf(std::size_t width, std::size_t x, std::size_t y) noexcept {
    return y * width + x;
}

[[nodiscard]] bool adjacentToSolid(const MacGrid& grid, std::size_t x, std::size_t y) noexcept {
    return (x > 0 && grid.cellType(x - 1, y) == CellType::solid) ||
        (x + 1 < grid.width() && grid.cellType(x + 1, y) == CellType::solid) ||
        (y > 0 && grid.cellType(x, y - 1) == CellType::solid) ||
        (y + 1 < grid.height() && grid.cellType(x, y + 1) == CellType::solid);
}

} // namespace

ParticleLevelSetConsistency diagnoseParticleLevelSetConsistency(
    const MacGrid& grid,
    const MarkerParticleSystem& particles,
    const LevelSetField& field,
    double interfaceTolerance) {
    ParticleLevelSetConsistency result;
    if (field.width != grid.width() || field.height != grid.height() ||
        !(interfaceTolerance >= 0.0)) {
        return result;
    }
    std::vector<unsigned int> particleCount(field.width * field.height, 0u);
    for (const MarkerParticle& particle : particles.particles()) {
        if (!particle.valid || !std::isfinite(particle.position.x) ||
            !std::isfinite(particle.position.y)) continue;
        ++result.validParticleCount;
        const double phi = field.sample(particle.position.x, particle.position.y);
        if (std::isfinite(phi) && phi > 0.0) {
            ++result.particleInPositivePhiCount;
            result.maximumParticlePositiveDistance = std::max(
                result.maximumParticlePositiveDistance, phi);
            if (phi > interfaceTolerance) ++result.farPositiveParticleCount;
        }
        const std::size_t x = std::min(grid.width() - 1,
            static_cast<std::size_t>(std::max(0.0f, particle.position.x) / grid.cellWidth()));
        const std::size_t y = std::min(grid.height() - 1,
            static_cast<std::size_t>(std::max(0.0f, particle.position.y) / grid.cellHeight()));
        ++particleCount[indexOf(field.width, x, y)];
    }

    double boundaryDistanceSum = 0.0;
    std::size_t boundaryDistanceCount = 0;
    const double band = 1.5 * std::max(field.cellWidth, field.cellHeight);
    for (std::size_t y = 0; y < field.height; ++y) {
        for (std::size_t x = 0; x < field.width; ++x) {
            const std::size_t index = indexOf(field.width, x, y);
            const double phi = field.values[index];
            if (!std::isfinite(phi)) continue;
            if (phi < -interfaceTolerance && particleCount[index] == 0u &&
                grid.cellType(x, y) != CellType::solid) {
                ++result.negativeCellWithoutParticleCount;
            }
            const bool classifiedFluid = grid.cellType(x, y) == CellType::fluid;
            if (grid.cellType(x, y) != CellType::solid && classifiedFluid != (phi <= 0.0)) {
                ++result.classificationSignMismatchCount;
                boundaryDistanceSum += std::abs(phi);
                ++boundaryDistanceCount;
            }
            if (std::abs(phi) <= band && field.confidence[index] == 0u) {
                ++result.lowConfidenceInterfaceCellCount;
            }
            if (std::abs(phi) <= band && adjacentToSolid(grid, x, y) &&
                field.confidence[index] == 0u) {
                ++result.solidAdjacentConflictCount;
            }
        }
    }
    result.meanClassificationInterfaceDistance = boundaryDistanceSum /
        std::max<std::size_t>(boundaryDistanceCount, 1);
    return result;
}

ParticleCorrectionReport applyConstrainedParticleCorrection(
    const MacGrid& grid,
    const MarkerParticleSystem& particles,
    LevelSetField& field,
    const ParticleCorrectionSettings& settings) {
    ParticleCorrectionReport result;
    result.areaBefore = estimateLevelSetArea(field, 2).area;
    if (!(settings.interfaceTolerance >= 0.0) ||
        !(settings.maximumCorrectionDistance > 0.0) ||
        !(settings.correctionDepth > 0.0)) {
        result.areaAfter = result.areaBefore;
        return result;
    }
    for (const MarkerParticle& particle : particles.particles()) {
        if (!particle.valid || !std::isfinite(particle.position.x) ||
            !std::isfinite(particle.position.y)) continue;
        const double phi = field.sample(particle.position.x, particle.position.y);
        if (!std::isfinite(phi) || phi <= settings.interfaceTolerance) continue;
        ++result.proposedCount;
        if (phi > settings.maximumCorrectionDistance) {
            ++result.rejectedFarCount;
            continue;
        }
        const std::size_t x = std::min(grid.width() - 1,
            static_cast<std::size_t>(particle.position.x / grid.cellWidth()));
        const std::size_t y = std::min(grid.height() - 1,
            static_cast<std::size_t>(particle.position.y / grid.cellHeight()));
        if (grid.cellType(x, y) == CellType::solid || adjacentToSolid(grid, x, y)) {
            ++result.rejectedSolidCount;
            continue;
        }
        if (!settings.enabled) continue;
        const std::size_t index = indexOf(field.width, x, y);
        const double corrected = -std::min(settings.correctionDepth,
            settings.maximumCorrectionDistance - phi + settings.correctionDepth);
        result.maximumValueChange = std::max(
            result.maximumValueChange, std::abs(corrected - field.values[index]));
        field.values[index] = std::min(field.values[index], corrected);
        field.confidence[index] = 1u;
        ++result.appliedCount;
    }
    if (result.appliedCount > 0) ++field.sourceVersion;
    result.areaAfter = estimateLevelSetArea(field, 2).area;
    return result;
}

void DynamicLevelSetTracker::initialize(LevelSetField initialField, double simulationTime) {
    if (!std::isfinite(simulationTime) || initialField.values.empty() ||
        initialField.values.size() != initialField.width * initialField.height) {
        throw std::invalid_argument("Dynamic level set requires a valid initial field and time.");
    }
    m_committed.field = std::move(initialField);
    m_committed.simulationTime = simulationTime;
    m_committed.valid = true;
    m_committed.versions.geometry = std::max<std::size_t>(
        m_committed.field.sourceVersion, 1);
    m_committed.versions.theta = 0;
    m_committed.versions.fineOperator = 0;
    m_committed.versions.multigridHierarchy = 0;
    m_candidate = {};
    m_hasCandidate = false;
}

DynamicLevelSetAdvanceReport DynamicLevelSetTracker::prepareCandidate(
    const MacGrid& velocityGrid,
    const MarkerParticleSystem& particles,
    double timeStep,
    const LevelSetRedistanceSettings& redistanceSettings,
    const ParticleCorrectionSettings& correctionSettings,
    double thetaMinimum) {
    if (!m_committed.valid || m_hasCandidate || !std::isfinite(timeStep) || timeStep <= 0.0 ||
        velocityGrid.width() != m_committed.field.width ||
        velocityGrid.height() != m_committed.field.height) {
        throw std::logic_error("Dynamic level-set candidate cannot be prepared in the current state.");
    }
    DynamicLevelSetAdvanceReport report;
    auto start = Clock::now();
    m_candidate = m_committed;
    m_candidate.field = advectLevelSetRk2(m_committed.field,
        [&](double x, double y) {
            return MacVelocitySampler::sampleVelocity(velocityGrid,
                {static_cast<float>(x), static_cast<float>(y)});
        }, timeStep);
    report.advectionMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();

    for (std::size_t y = 0; y < velocityGrid.height(); ++y) {
        for (std::size_t x = 0; x < velocityGrid.width(); ++x) {
            if (velocityGrid.cellType(x, y) == CellType::solid) {
                m_candidate.field.confidence[indexOf(velocityGrid.width(), x, y)] = 0u;
            }
        }
    }

    start = Clock::now();
    report.redistance = redistanceLevelSetNarrowBand(
        m_candidate.field, redistanceSettings);
    report.redistanceMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();

    const double tolerance = correctionSettings.interfaceTolerance > 0.0
        ? correctionSettings.interfaceTolerance
        : std::max(m_candidate.field.cellWidth, m_candidate.field.cellHeight);
    start = Clock::now();
    report.consistency = diagnoseParticleLevelSetConsistency(
        velocityGrid, particles, m_candidate.field, tolerance);
    report.consistencyMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();

    start = Clock::now();
    report.correction = applyConstrainedParticleCorrection(
        velocityGrid, particles, m_candidate.field, correctionSettings);
    report.correctionMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();

    start = Clock::now();
    const GhostFluidThetaData theta = buildGhostFluidTheta(
        velocityGrid, m_candidate.field, thetaMinimum, true);
    report.theta = theta.statistics;
    report.thetaMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();

    m_candidate.simulationTime = m_committed.simulationTime + timeStep;
    m_candidate.versions.geometry = m_committed.versions.geometry + 1;
    m_candidate.versions.theta = m_committed.versions.theta + 1;
    m_candidate.versions.fineOperator = 0;
    m_candidate.versions.multigridHierarchy = 0;
    m_candidate.valid = std::all_of(m_candidate.field.values.begin(),
        m_candidate.field.values.end(), [](double value) { return std::isfinite(value); });
    m_hasCandidate = true;
    return report;
}

void DynamicLevelSetTracker::acceptCandidate() {
    if (!m_hasCandidate || !m_candidate.valid) {
        throw std::logic_error("Only a valid dynamic level-set candidate can be accepted.");
    }
    m_committed = std::move(m_candidate);
    m_candidate = {};
    m_hasCandidate = false;
}

void DynamicLevelSetTracker::rejectCandidate() noexcept {
    m_candidate = {};
    m_hasCandidate = false;
}

const DynamicInterfaceSnapshot& DynamicLevelSetTracker::committed() const noexcept {
    return m_committed;
}

const DynamicInterfaceSnapshot& DynamicLevelSetTracker::candidate() const noexcept {
    return m_candidate;
}

bool DynamicLevelSetTracker::hasCandidate() const noexcept { return m_hasCandidate; }

} // namespace fluidlab::simulation
