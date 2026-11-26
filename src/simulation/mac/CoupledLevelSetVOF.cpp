#include "CoupledLevelSetVOF.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fluidlab::simulation {

ClsvofCouplingReport constrainLevelSetToVolumeFraction(
    const VolumeFractionField& volumeFraction,
    LevelSetField& levelSet,
    const ClsvofCouplingSettings& settings) {
    if (volumeFraction.width() != levelSet.width ||
        volumeFraction.height() != levelSet.height ||
        std::abs(volumeFraction.cellWidth() - levelSet.cellWidth) > 1.0e-12 ||
        std::abs(volumeFraction.cellHeight() - levelSet.cellHeight) > 1.0e-12) {
        throw std::invalid_argument("CLSVOF fields must have matching layouts.");
    }

    ClsvofCouplingReport report;
    report.vofArea = volumeFraction.totalArea();
    report.levelSetAreaBefore = estimateLevelSetArea(levelSet, 4).area;
    double squaredCorrection = 0.0;
    const double defaultMaximumCorrection =
        1.5 * std::hypot(levelSet.cellWidth, levelSet.cellHeight);
    const double maximumCorrection = settings.maximumCorrectionDistance > 0.0
        ? settings.maximumCorrectionDistance
        : defaultMaximumCorrection;

    for (std::size_t y = 0; y < levelSet.height; ++y) {
        for (std::size_t x = 0; x < levelSet.width; ++x) {
            if (!volumeFraction.isValid(x, y)) {
                continue;
            }
            const double fraction = volumeFraction.fraction(x, y);
            if (!(fraction > settings.minimumInterfaceFraction &&
                  fraction < settings.maximumInterfaceFraction)) {
                continue;
            }
            ++report.interfaceCellCount;
            const double centerX = (static_cast<double>(x) + 0.5) * levelSet.cellWidth;
            const double centerY = (static_cast<double>(y) + 0.5) * levelSet.cellHeight;
            const Vector2 levelSetNormal = levelSet.normal(centerX, centerY);
            Vector2d trustedNormal{
                static_cast<double>(levelSetNormal.x),
                static_cast<double>(levelSetNormal.y),
            };
            if (std::hypot(trustedNormal.x, trustedNormal.y) <= 1.0e-8) {
                trustedNormal = volumeFraction.estimateNormal(x, y);
                ++report.lowConfidenceCount;
            }
            const PlicReconstruction plic = reconstructPlic(
                fraction,
                trustedNormal,
                levelSet.cellWidth,
                levelSet.cellHeight);
            if (!plic.valid) {
                ++report.plicFailureCount;
                continue;
            }
            const double localCenterX = 0.5 * levelSet.cellWidth;
            const double localCenterY = 0.5 * levelSet.cellHeight;
            const double targetValue =
                plic.normal.x * localCenterX + plic.normal.y * localCenterY - plic.alpha;
            const std::size_t offset = y * levelSet.width + x;
            const double correction = targetValue - levelSet.values[offset];
            if (!std::isfinite(correction) || std::abs(correction) > maximumCorrection) {
                ++report.rejectedLargeCorrectionCount;
                levelSet.confidence[offset] = 0;
                continue;
            }
            levelSet.values[offset] = targetValue;
            levelSet.confidence[offset] = 1;
            ++report.correctedCellCount;
            report.maximumInterfaceCorrection = std::max(
                report.maximumInterfaceCorrection, std::abs(correction));
            squaredCorrection += correction * correction;
        }
    }

    if (report.correctedCellCount > 0) {
        report.rmsInterfaceCorrection = std::sqrt(
            squaredCorrection / static_cast<double>(report.correctedCellCount));
    }
    if (settings.redistanceAfterCorrection && report.correctedCellCount > 0) {
        LevelSetRedistanceSettings redistance;
        redistance.narrowBandWidth = settings.narrowBandWidth > 0.0
            ? settings.narrowBandWidth
            : 6.0 * std::max(levelSet.cellWidth, levelSet.cellHeight);
        redistance.areaSamplesPerAxis = 4;
        redistance.preserveInterfaceCrossings = true;
        report.redistance = redistanceLevelSetNarrowBand(levelSet, redistance);
    }
    report.levelSetAreaAfter = estimateLevelSetArea(levelSet, 4).area;
    ++levelSet.sourceVersion;
    return report;
}

namespace {

double pointSegmentDistance(
    double px,
    double py,
    const PlicInterfaceSegment& segment,
    double originX,
    double originY) noexcept {
    const double ax = originX + segment.first.x;
    const double ay = originY + segment.first.y;
    const double bx = originX + segment.second.x;
    const double by = originY + segment.second.y;
    const double dx = bx - ax;
    const double dy = by - ay;
    const double lengthSquared = dx * dx + dy * dy;
    const double t = lengthSquared > 0.0
        ? std::clamp(((px - ax) * dx + (py - ay) * dy) / lengthSquared, 0.0, 1.0)
        : 0.0;
    return std::hypot(px - (ax + t * dx), py - (ay + t * dy));
}

} // namespace

ClsvofCouplingReport constrainLevelSetToPlicGeometry(
    const VolumeFractionField& volumeFraction,
    LevelSetField& levelSet,
    const ClsvofCouplingSettings& settings,
    PlicNormalMethod normalMethod) {
    if (volumeFraction.width() != levelSet.width ||
        volumeFraction.height() != levelSet.height) {
        throw std::invalid_argument("PLIC coupling fields must have matching layouts.");
    }
    struct SegmentRecord final {
        PlicReconstruction plic;
        PlicInterfaceSegment segment;
        std::size_t x = 0;
        std::size_t y = 0;
        bool compatible = false;
    };
    std::vector<SegmentRecord> records;
    ClsvofCouplingReport report;
    report.vofArea = volumeFraction.totalArea();
    report.levelSetAreaBefore = estimateLevelSetArea(levelSet, 4).area;
    for (std::size_t y = 0; y < volumeFraction.height(); ++y) {
        for (std::size_t x = 0; x < volumeFraction.width(); ++x) {
            const double fraction = volumeFraction.fraction(x, y);
            if (!volumeFraction.isValid(x, y) ||
                !(fraction > settings.minimumInterfaceFraction &&
                  fraction < settings.maximumInterfaceFraction)) continue;
            ++report.interfaceCellCount;
            bool heightValid = false;
            Vector2d normal = normalMethod == PlicNormalMethod::heightFunction
                ? volumeFraction.estimateHeightFunctionNormal(x, y, &heightValid)
                : volumeFraction.estimateNormal(x, y);
            if (normalMethod == PlicNormalMethod::heightFunction && !heightValid) {
                normal = volumeFraction.estimateNormal(x, y);
                ++report.geometryFallbackCount;
            }
            const PlicReconstruction plic = reconstructPlic(
                fraction, normal, volumeFraction.cellWidth(), volumeFraction.cellHeight());
            const PlicInterfaceSegment segment = plicInterfaceSegment(
                plic, volumeFraction.cellWidth(), volumeFraction.cellHeight());
            if (!plic.valid || !segment.valid) {
                ++report.plicFailureCount;
                continue;
            }
            records.push_back({plic, segment, x, y, false});
        }
    }
    const double neighborDistance = 2.25 * std::hypot(
        volumeFraction.cellWidth(), volumeFraction.cellHeight());
    for (SegmentRecord& record : records) {
        for (const SegmentRecord& other : records) {
            if (&record == &other) continue;
            const std::ptrdiff_t dx = static_cast<std::ptrdiff_t>(record.x) -
                static_cast<std::ptrdiff_t>(other.x);
            const std::ptrdiff_t dy = static_cast<std::ptrdiff_t>(record.y) -
                static_cast<std::ptrdiff_t>(other.y);
            if (std::abs(dx) > 1 || std::abs(dy) > 1) continue;
            const double normalDot = record.plic.normal.x * other.plic.normal.x +
                record.plic.normal.y * other.plic.normal.y;
            const double rOriginX = record.x * volumeFraction.cellWidth();
            const double rOriginY = record.y * volumeFraction.cellHeight();
            const double oOriginX = other.x * volumeFraction.cellWidth();
            const double oOriginY = other.y * volumeFraction.cellHeight();
            const double endpointDistance = std::min({
                std::hypot(rOriginX + record.segment.first.x - oOriginX - other.segment.first.x,
                           rOriginY + record.segment.first.y - oOriginY - other.segment.first.y),
                std::hypot(rOriginX + record.segment.first.x - oOriginX - other.segment.second.x,
                           rOriginY + record.segment.first.y - oOriginY - other.segment.second.y),
                std::hypot(rOriginX + record.segment.second.x - oOriginX - other.segment.first.x,
                           rOriginY + record.segment.second.y - oOriginY - other.segment.first.y),
                std::hypot(rOriginX + record.segment.second.x - oOriginX - other.segment.second.x,
                           rOriginY + record.segment.second.y - oOriginY - other.segment.second.y),
            });
            if (normalDot > 0.5 && endpointDistance <= neighborDistance) {
                record.compatible = true;
                break;
            }
        }
        if (record.compatible) ++report.validSegmentCount;
        else ++report.incompatibleSegmentCount;
    }

    const double maximumCorrection = settings.maximumCorrectionDistance > 0.0
        ? settings.maximumCorrectionDistance
        : 1.5 * std::hypot(levelSet.cellWidth, levelSet.cellHeight);
    std::vector<double> updated = levelSet.values;
    double squaredCorrection = 0.0;
    for (std::size_t y = 0; y < levelSet.height; ++y) {
        for (std::size_t x = 0; x < levelSet.width; ++x) {
            const double px = (static_cast<double>(x) + 0.5) * levelSet.cellWidth;
            const double py = (static_cast<double>(y) + 0.5) * levelSet.cellHeight;
            double nearest = std::numeric_limits<double>::infinity();
            double signedDistance = 0.0;
            for (const SegmentRecord& record : records) {
                if (!record.compatible ||
                    std::abs(static_cast<std::ptrdiff_t>(x) -
                        static_cast<std::ptrdiff_t>(record.x)) > 2 ||
                    std::abs(static_cast<std::ptrdiff_t>(y) -
                        static_cast<std::ptrdiff_t>(record.y)) > 2) continue;
                const double originX = record.x * levelSet.cellWidth;
                const double originY = record.y * levelSet.cellHeight;
                const double distance = pointSegmentDistance(
                    px, py, record.segment, originX, originY);
                if (distance < nearest) {
                    nearest = distance;
                    const double lineValue = record.plic.normal.x * (px - originX) +
                        record.plic.normal.y * (py - originY) - record.plic.alpha;
                    signedDistance = std::copysign(distance, lineValue);
                }
            }
            if (!std::isfinite(nearest) || nearest > maximumCorrection) continue;
            const std::size_t offset = y * levelSet.width + x;
            const double correction = signedDistance - levelSet.values[offset];
            if (!std::isfinite(correction) || std::abs(correction) > maximumCorrection) {
                ++report.rejectedLargeCorrectionCount;
                continue;
            }
            updated[offset] = signedDistance;
            levelSet.confidence[offset] = 1;
            ++report.correctedCellCount;
            squaredCorrection += correction * correction;
            report.maximumInterfaceCorrection = std::max(
                report.maximumInterfaceCorrection, std::abs(correction));
        }
    }
    levelSet.values.swap(updated);
    if (report.correctedCellCount > 0) {
        report.rmsInterfaceCorrection = std::sqrt(
            squaredCorrection / report.correctedCellCount);
    }
    if (settings.redistanceAfterCorrection && report.correctedCellCount > 0) {
        LevelSetRedistanceSettings redistance;
        redistance.narrowBandWidth = settings.narrowBandWidth > 0.0
            ? settings.narrowBandWidth
            : 6.0 * std::max(levelSet.cellWidth, levelSet.cellHeight);
        redistance.areaSamplesPerAxis = 4;
        redistance.preserveInterfaceCrossings = true;
        report.redistance = redistanceLevelSetNarrowBand(levelSet, redistance);
    }
    report.levelSetAreaAfter = estimateLevelSetArea(levelSet, 4).area;
    ++levelSet.sourceVersion;
    return report;
}

GhostFluidThetaData buildPlicGhostFluidTheta(
    const MacGrid& grid,
    const VolumeFractionField& volumeFraction,
    PlicNormalMethod normalMethod,
    double minimumTheta,
    bool fallbackToCellBoundary) {
    if (grid.width() != volumeFraction.width() || grid.height() != volumeFraction.height() ||
        !(minimumTheta > 0.0) || minimumTheta > 1.0) {
        throw std::invalid_argument("PLIC theta input is incompatible with the grid.");
    }
    GhostFluidThetaData result;
    result.width = grid.width();
    result.height = grid.height();
    result.minimumTheta = minimumTheta;
    result.interfaceVersion = volumeFraction.version();
    result.horizontalTheta.assign((result.width + 1) * result.height,
        std::numeric_limits<double>::quiet_NaN());
    result.verticalTheta.assign(result.width * (result.height + 1),
        std::numeric_limits<double>::quiet_NaN());
    result.horizontalState.assign(result.horizontalTheta.size(), GhostThetaState::unused);
    result.verticalState.assign(result.verticalTheta.size(), GhostThetaState::unused);

    const auto faceTheta = [&](std::size_t fluidX, std::size_t fluidY,
                               std::size_t airX, std::size_t airY) {
        const std::array<std::pair<std::size_t, std::size_t>, 2> cells{{
            {fluidX, fluidY}, {airX, airY}}};
        for (const auto [cellX, cellY] : cells) {
            const double fraction = volumeFraction.fraction(cellX, cellY);
            if (!(fraction > 1.0e-6 && fraction < 1.0 - 1.0e-6)) continue;
            bool heightValid = false;
            Vector2d normal = normalMethod == PlicNormalMethod::heightFunction
                ? volumeFraction.estimateHeightFunctionNormal(cellX, cellY, &heightValid)
                : volumeFraction.estimateNormal(cellX, cellY);
            if (normalMethod == PlicNormalMethod::heightFunction && !heightValid) continue;
            const PlicReconstruction plic = reconstructPlic(
                fraction, normal, volumeFraction.cellWidth(), volumeFraction.cellHeight());
            const PlicInterfaceSegment segment = plicInterfaceSegment(
                plic, volumeFraction.cellWidth(), volumeFraction.cellHeight());
            if (!plic.valid || !segment.valid) continue;
            const double originX = cellX * volumeFraction.cellWidth();
            const double originY = cellY * volumeFraction.cellHeight();
            const double fx = (fluidX + 0.5) * volumeFraction.cellWidth();
            const double fy = (fluidY + 0.5) * volumeFraction.cellHeight();
            const double ax = (airX + 0.5) * volumeFraction.cellWidth();
            const double ay = (airY + 0.5) * volumeFraction.cellHeight();
            const double denominator = plic.normal.x * (ax - fx) +
                plic.normal.y * (ay - fy);
            if (std::abs(denominator) <= 1.0e-14) continue;
            const double globalAlpha = plic.alpha + plic.normal.x * originX +
                plic.normal.y * originY;
            const double raw = (globalAlpha - plic.normal.x * fx - plic.normal.y * fy) /
                denominator;
            if (!(raw > 0.0 && raw <= 1.0) || !std::isfinite(raw)) continue;
            const double ix = fx + raw * (ax - fx);
            const double iy = fy + raw * (ay - fy);
            if (pointSegmentDistance(ix, iy, segment, originX, originY) > 1.0e-8) continue;
            return std::pair<double, GhostThetaState>{
                std::max(raw, minimumTheta),
                raw < minimumTheta ? GhostThetaState::clamped : GhostThetaState::valid};
        }
        return std::pair<double, GhostThetaState>{
            fallbackToCellBoundary ? 1.0 : std::numeric_limits<double>::quiet_NaN(),
            GhostThetaState::fallback};
    };
    const auto record = [&](GhostThetaState state) {
        if (state == GhostThetaState::valid) ++result.statistics.validFaceCount;
        else if (state == GhostThetaState::clamped) ++result.statistics.clampedFaceCount;
        else {
            ++result.statistics.invalidFaceCount;
            ++result.statistics.fallbackFaceCount;
        }
    };
    for (std::size_t y = 0; y < result.height; ++y) {
        for (std::size_t x = 1; x < result.width; ++x) {
            const CellType left = grid.cellType(x - 1, y);
            const CellType right = grid.cellType(x, y);
            if (left == CellType::solid || right == CellType::solid) {
                ++result.statistics.solidAdjacentFaceCount;
                continue;
            }
            if (!((left == CellType::fluid && right == CellType::air) ||
                  (left == CellType::air && right == CellType::fluid))) continue;
            const bool leftFluid = left == CellType::fluid;
            const auto [theta, state] = faceTheta(
                leftFluid ? x - 1 : x, y, leftFluid ? x : x - 1, y);
            const std::size_t face = y * (result.width + 1) + x;
            result.horizontalTheta[face] = theta;
            result.horizontalState[face] = state;
            record(state);
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
            if (!((below == CellType::fluid && above == CellType::air) ||
                  (below == CellType::air && above == CellType::fluid))) continue;
            const bool belowFluid = below == CellType::fluid;
            const auto [theta, state] = faceTheta(
                x, belowFluid ? y - 1 : y, x, belowFluid ? y : y - 1);
            const std::size_t face = y * result.width + x;
            result.verticalTheta[face] = theta;
            result.verticalState[face] = state;
            record(state);
        }
    }
    return result;
}

void CoupledLevelSetVOF::initialize(
    VolumeFractionField volumeFraction,
    LevelSetField levelSet,
    double physicalTime) {
    if (volumeFraction.width() != levelSet.width || volumeFraction.height() != levelSet.height) {
        throw std::invalid_argument("CLSVOF initialization layouts do not match.");
    }
    m_committed.volumeFraction = std::move(volumeFraction);
    m_committed.levelSet = std::move(levelSet);
    m_committed.geometryVersion = 1;
    m_committed.physicalTime = physicalTime;
    m_committed.valid = true;
    m_candidate = {};
    m_hasCandidate = false;
}

ClsvofCouplingReport CoupledLevelSetVOF::prepareCandidate(
    const VolumeFractionField& advectedVolumeFraction,
    const LevelSetField& advectedLevelSet,
    const ClsvofCouplingSettings& settings) {
    if (!m_committed.valid || m_hasCandidate) {
        throw std::logic_error("CLSVOF candidate state is not available.");
    }
    m_candidate.volumeFraction = advectedVolumeFraction;
    m_candidate.levelSet = advectedLevelSet;
    m_candidate.geometryVersion = m_committed.geometryVersion + 1;
    m_candidate.physicalTime = advectedVolumeFraction.physicalTime();
    m_candidate.valid = true;
    m_hasCandidate = true;
    return constrainLevelSetToVolumeFraction(
        m_candidate.volumeFraction,
        m_candidate.levelSet,
        settings);
}

void CoupledLevelSetVOF::acceptCandidate() {
    if (!m_hasCandidate) {
        throw std::logic_error("No CLSVOF candidate is available to accept.");
    }
    m_committed = std::move(m_candidate);
    m_candidate = {};
    m_hasCandidate = false;
}

void CoupledLevelSetVOF::rejectCandidate() noexcept {
    m_candidate = {};
    m_hasCandidate = false;
}

void CoupledLevelSetVOF::reset() {
    m_committed = {};
    rejectCandidate();
}

const CoupledInterfaceState& CoupledLevelSetVOF::committed() const noexcept {
    return m_committed;
}

const CoupledInterfaceState& CoupledLevelSetVOF::candidate() const noexcept {
    return m_candidate;
}

bool CoupledLevelSetVOF::hasCandidate() const noexcept { return m_hasCandidate; }

} // namespace fluidlab::simulation
