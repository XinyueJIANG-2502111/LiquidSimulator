#pragma once

#include "MacGrid.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace fluidlab::simulation {

struct Vector2d final {
    double x = 0.0;
    double y = 0.0;
};

struct PlicReconstruction final {
    Vector2d normal;
    double alpha = 0.0;
    double reconstructedArea = 0.0;
    bool valid = false;
};

enum class PlicNormalMethod : unsigned char {
    fractionGradient,
    heightFunction,
};

enum class VolumeFractionAdvectionMode : unsigned char {
    legacySplit,
    conservativeSplitExperimental,
};

struct VolumeFractionDistribution final {
    double tolerance = 1.0e-6;
    std::size_t strictNonZero = 0;
    std::size_t air = 0;
    std::size_t lowTail = 0;
    std::size_t lowPartial = 0;
    std::size_t centralMixed = 0;
    std::size_t highPartial = 0;
    std::size_t nearFull = 0;
    std::size_t full = 0;
    std::size_t solid = 0;

    [[nodiscard]] std::size_t mixed() const noexcept {
        return lowTail + lowPartial + centralMixed + highPartial + nearFull;
    }
};

struct PlicInterfaceSegment final {
    Vector2d first;
    Vector2d second;
    bool valid = false;
};

struct LimitedFluxEvent final {
    std::size_t faceX = 0;
    std::size_t faceY = 0;
    bool xDirection = false;
    bool positiveDirection = false;
    double rawFlux = 0.0;
    double finalFlux = 0.0;
    double donorFraction = 0.0;
    double receiverFraction = 0.0;
    double faceVelocity = 0.0;
    double localDivergence = 0.0;
    Vector2d plicNormal;
};

struct VolumeFractionLedger final {
    double areaBefore = 0.0;
    double areaAfter = 0.0;
    double boundaryFlux = 0.0;
    double solidBoundaryFlux = 0.0;
    double conservationResidual = 0.0;
    double compressionContribution = 0.0;
    double xCompressionContribution = 0.0;
    double yCompressionContribution = 0.0;
    double maximumXCourant = 0.0;
    double maximumYCourant = 0.0;
    double minimumFraction = 0.0;
    double maximumFraction = 0.0;
    std::size_t outOfBoundsCount = 0;
    std::size_t limitedFaceCount = 0;
    std::size_t evaluatedFaceCount = 0;
    std::size_t nonZeroLiquidFluxFaceCount = 0;
    std::size_t uniqueLimitedFaceCount = 0;
    std::size_t repeatedLimitEventCount = 0;
    std::size_t limiterIterationCount = 0;
    std::size_t positiveLimitedFaceCount = 0;
    std::size_t negativeLimitedFaceCount = 0;
    std::size_t xLimitedFaceCount = 0;
    std::size_t yLimitedFaceCount = 0;
    std::size_t donorInsufficientCount = 0;
    std::size_t receiverCapacityCount = 0;
    std::size_t coupledInOutCount = 0;
    std::size_t highCourantLimitedCount = 0;
    std::size_t divergentCellLimitedCount = 0;
    std::size_t solidAdjacentFaceCount = 0;
    std::size_t nearBlockedFaceCount = 0;
    double rawAbsoluteFlux = 0.0;
    double finalAbsoluteFlux = 0.0;
    double minimumLimiterScale = 1.0;
    double meanLimiterScale = 1.0;
    double p95LimiterScale = 1.0;
    std::size_t plicFailureCount = 0;
    std::size_t xFaceFluxCount = 0;
    std::size_t yFaceFluxCount = 0;
    bool accepted = false;
    VolumeFractionAdvectionMode mode = VolumeFractionAdvectionMode::legacySplit;
    std::vector<double> limiterScaleSamples;
    std::vector<LimitedFluxEvent> strongestLimitedFaces;
};

using VolumeFractionSignedDistance = std::function<double(double, double)>;

class VolumeFractionField final {
public:
    VolumeFractionField() = default;
    VolumeFractionField(
        std::size_t width,
        std::size_t height,
        double cellWidth,
        double cellHeight);

    [[nodiscard]] std::size_t width() const noexcept;
    [[nodiscard]] std::size_t height() const noexcept;
    [[nodiscard]] double cellWidth() const noexcept;
    [[nodiscard]] double cellHeight() const noexcept;
    [[nodiscard]] double physicalTime() const noexcept;
    [[nodiscard]] std::size_t version() const noexcept;

    [[nodiscard]] double fraction(std::size_t x, std::size_t y) const noexcept;
    void setFraction(std::size_t x, std::size_t y, double value) noexcept;
    [[nodiscard]] bool isValid(std::size_t x, std::size_t y) const noexcept;
    void setValid(std::size_t x, std::size_t y, bool valid) noexcept;
    [[nodiscard]] double cellArea(std::size_t x, std::size_t y) const noexcept;

    void initializeFromSignedDistance(
        const VolumeFractionSignedDistance& signedDistance,
        std::size_t samplesPerAxis = 16);
    void applyWholeCellSolidMask(const MacGrid& grid);

    [[nodiscard]] double totalArea() const noexcept;
    [[nodiscard]] Vector2d estimateNormal(std::size_t x, std::size_t y) const noexcept;
    [[nodiscard]] Vector2d estimateHeightFunctionNormal(
        std::size_t x,
        std::size_t y,
        bool* valid = nullptr) const noexcept;
    [[nodiscard]] PlicReconstruction reconstructCell(
        std::size_t x,
        std::size_t y,
        const Vector2d* trustedNormal = nullptr,
        PlicNormalMethod normalMethod = PlicNormalMethod::fractionGradient) const noexcept;

    [[nodiscard]] VolumeFractionLedger advectConservative(
        const MacGrid& velocity,
        double timeStep,
        bool xDirectionFirst = true);
    [[nodiscard]] VolumeFractionLedger advectConservativeSplitExperimental(
        const MacGrid& velocity,
        double timeStep,
        bool xDirectionFirst = true);
    [[nodiscard]] VolumeFractionDistribution distribution(
        double tolerance = 1.0e-6) const noexcept;

    void setPhysicalTime(double value) noexcept;
    void incrementVersion() noexcept;

    [[nodiscard]] const std::vector<double>& fractions() const noexcept;
    [[nodiscard]] const std::vector<unsigned char>& validMask() const noexcept;

private:
    [[nodiscard]] std::size_t index(std::size_t x, std::size_t y) const noexcept;
    [[nodiscard]] VolumeFractionLedger advectAxis(
        const MacGrid& velocity,
        double timeStep,
        bool xDirection,
        const std::vector<unsigned char>* compressionCoefficient);
    [[nodiscard]] VolumeFractionLedger advectSplit(
        const MacGrid& velocity,
        double timeStep,
        bool xDirectionFirst,
        VolumeFractionAdvectionMode mode);

    std::size_t m_width = 0;
    std::size_t m_height = 0;
    double m_cellWidth = 0.0;
    double m_cellHeight = 0.0;
    double m_physicalTime = 0.0;
    std::size_t m_version = 0;
    std::vector<double> m_fraction;
    std::vector<unsigned char> m_valid;
};

class VolumeFractionTransaction final {
public:
    void initialize(VolumeFractionField field);
    [[nodiscard]] VolumeFractionField& prepareCandidate();
    void acceptCandidate();
    void rejectCandidate() noexcept;
    void reset();

    [[nodiscard]] const VolumeFractionField& committed() const noexcept;
    [[nodiscard]] const VolumeFractionField& candidate() const noexcept;
    [[nodiscard]] bool hasCandidate() const noexcept;

private:
    VolumeFractionField m_committed;
    VolumeFractionField m_candidate;
    bool m_hasCandidate = false;
};

[[nodiscard]] PlicReconstruction reconstructPlic(
    double fraction,
    Vector2d normal,
    double cellWidth,
    double cellHeight) noexcept;

[[nodiscard]] double plicLiquidAreaInBox(
    const PlicReconstruction& reconstruction,
    double minimumX,
    double minimumY,
    double maximumX,
    double maximumY) noexcept;

[[nodiscard]] PlicInterfaceSegment plicInterfaceSegment(
    const PlicReconstruction& reconstruction,
    double cellWidth,
    double cellHeight) noexcept;

} // namespace fluidlab::simulation
