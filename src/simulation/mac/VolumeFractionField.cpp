#include "VolumeFractionField.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fluidlab::simulation {
namespace {

constexpr double kFractionTolerance = 1.0e-10;
constexpr double kNormalTolerance = 1.0e-14;

struct PolygonPoint final {
    double x = 0.0;
    double y = 0.0;
};

double polygonArea(const std::vector<PolygonPoint>& polygon) noexcept {
    if (polygon.size() < 3) {
        return 0.0;
    }
    double twiceArea = 0.0;
    for (std::size_t index = 0; index < polygon.size(); ++index) {
        const PolygonPoint& a = polygon[index];
        const PolygonPoint& b = polygon[(index + 1) % polygon.size()];
        twiceArea += a.x * b.y - a.y * b.x;
    }
    return 0.5 * std::abs(twiceArea);
}

std::vector<PolygonPoint> clipHalfPlane(
    const std::vector<PolygonPoint>& input,
    Vector2d normal,
    double alpha) {
    std::vector<PolygonPoint> output;
    if (input.empty()) {
        return output;
    }
    output.reserve(input.size() + 2);
    const auto signedValue = [&](const PolygonPoint& point) {
        return normal.x * point.x + normal.y * point.y - alpha;
    };
    for (std::size_t index = 0; index < input.size(); ++index) {
        const PolygonPoint& current = input[index];
        const PolygonPoint& next = input[(index + 1) % input.size()];
        const double currentValue = signedValue(current);
        const double nextValue = signedValue(next);
        const bool currentInside = currentValue <= 0.0;
        const bool nextInside = nextValue <= 0.0;
        if (currentInside) {
            output.push_back(current);
        }
        if (currentInside != nextInside) {
            const double denominator = currentValue - nextValue;
            if (std::abs(denominator) > std::numeric_limits<double>::epsilon()) {
                const double t = std::clamp(currentValue / denominator, 0.0, 1.0);
                output.push_back({
                    current.x + t * (next.x - current.x),
                    current.y + t * (next.y - current.y),
                });
            }
        }
    }
    return output;
}

double clippedRectangleArea(
    double minimumX,
    double minimumY,
    double maximumX,
    double maximumY,
    Vector2d normal,
    double alpha) noexcept {
    if (!(maximumX > minimumX) || !(maximumY > minimumY)) {
        return 0.0;
    }
    std::vector<PolygonPoint> rectangle{
        {minimumX, minimumY},
        {maximumX, minimumY},
        {maximumX, maximumY},
        {minimumX, maximumY},
    };
    return polygonArea(clipHalfPlane(rectangle, normal, alpha));
}

double cornerProjectionMinimum(Vector2d normal, double width, double height) noexcept {
    return std::min({
        0.0,
        normal.x * width,
        normal.y * height,
        normal.x * width + normal.y * height,
    });
}

double cornerProjectionMaximum(Vector2d normal, double width, double height) noexcept {
    return std::max({
        0.0,
        normal.x * width,
        normal.y * height,
        normal.x * width + normal.y * height,
    });
}

} // namespace

VolumeFractionField::VolumeFractionField(
    std::size_t width,
    std::size_t height,
    double cellWidth,
    double cellHeight)
    : m_width(width),
      m_height(height),
      m_cellWidth(cellWidth),
      m_cellHeight(cellHeight),
      m_fraction(width * height, 0.0),
      m_valid(width * height, 1) {
    if (width == 0 || height == 0 || !(cellWidth > 0.0) || !(cellHeight > 0.0)) {
        throw std::invalid_argument("Volume fraction grid dimensions must be positive.");
    }
}

std::size_t VolumeFractionField::width() const noexcept { return m_width; }
std::size_t VolumeFractionField::height() const noexcept { return m_height; }
double VolumeFractionField::cellWidth() const noexcept { return m_cellWidth; }
double VolumeFractionField::cellHeight() const noexcept { return m_cellHeight; }
double VolumeFractionField::physicalTime() const noexcept { return m_physicalTime; }
std::size_t VolumeFractionField::version() const noexcept { return m_version; }

std::size_t VolumeFractionField::index(std::size_t x, std::size_t y) const noexcept {
    return y * m_width + x;
}

double VolumeFractionField::fraction(std::size_t x, std::size_t y) const noexcept {
    return m_fraction[index(x, y)];
}

void VolumeFractionField::setFraction(std::size_t x, std::size_t y, double value) noexcept {
    m_fraction[index(x, y)] = value;
}

bool VolumeFractionField::isValid(std::size_t x, std::size_t y) const noexcept {
    return m_valid[index(x, y)] != 0;
}

void VolumeFractionField::setValid(std::size_t x, std::size_t y, bool valid) noexcept {
    const std::size_t offset = index(x, y);
    m_valid[offset] = valid ? 1 : 0;
    if (!valid) {
        m_fraction[offset] = 0.0;
    }
}

double VolumeFractionField::cellArea(std::size_t x, std::size_t y) const noexcept {
    return isValid(x, y) ? m_cellWidth * m_cellHeight : 0.0;
}

void VolumeFractionField::initializeFromSignedDistance(
    const VolumeFractionSignedDistance& signedDistance,
    std::size_t samplesPerAxis) {
    if (!signedDistance || samplesPerAxis == 0) {
        throw std::invalid_argument("VOF initialization requires a signed distance and samples.");
    }
    const double inverseSamples = 1.0 / static_cast<double>(samplesPerAxis * samplesPerAxis);
    for (std::size_t y = 0; y < m_height; ++y) {
        for (std::size_t x = 0; x < m_width; ++x) {
            if (!isValid(x, y)) {
                continue;
            }
            std::size_t liquidSamples = 0;
            for (std::size_t sampleY = 0; sampleY < samplesPerAxis; ++sampleY) {
                for (std::size_t sampleX = 0; sampleX < samplesPerAxis; ++sampleX) {
                    const double px = (static_cast<double>(x) +
                        (static_cast<double>(sampleX) + 0.5) / samplesPerAxis) * m_cellWidth;
                    const double py = (static_cast<double>(y) +
                        (static_cast<double>(sampleY) + 0.5) / samplesPerAxis) * m_cellHeight;
                    liquidSamples += signedDistance(px, py) <= 0.0 ? 1u : 0u;
                }
            }
            m_fraction[index(x, y)] = static_cast<double>(liquidSamples) * inverseSamples;
        }
    }
    ++m_version;
}

void VolumeFractionField::applyWholeCellSolidMask(const MacGrid& grid) {
    if (grid.width() != m_width || grid.height() != m_height) {
        throw std::invalid_argument("Solid mask and VOF layouts do not match.");
    }
    for (std::size_t y = 0; y < m_height; ++y) {
        for (std::size_t x = 0; x < m_width; ++x) {
            setValid(x, y, grid.cellType(x, y) != CellType::solid);
        }
    }
}

double VolumeFractionField::totalArea() const noexcept {
    double area = 0.0;
    for (std::size_t y = 0; y < m_height; ++y) {
        for (std::size_t x = 0; x < m_width; ++x) {
            area += fraction(x, y) * cellArea(x, y);
        }
    }
    return area;
}

Vector2d VolumeFractionField::estimateNormal(std::size_t x, std::size_t y) const noexcept {
    const auto sample = [&](std::ptrdiff_t sx, std::ptrdiff_t sy) {
        const std::size_t cx = static_cast<std::size_t>(std::clamp<std::ptrdiff_t>(
            sx, 0, static_cast<std::ptrdiff_t>(m_width) - 1));
        const std::size_t cy = static_cast<std::size_t>(std::clamp<std::ptrdiff_t>(
            sy, 0, static_cast<std::ptrdiff_t>(m_height) - 1));
        return isValid(cx, cy) ? fraction(cx, cy) : fraction(x, y);
    };
    const std::ptrdiff_t px = static_cast<std::ptrdiff_t>(x);
    const std::ptrdiff_t py = static_cast<std::ptrdiff_t>(y);
    const double gradientX = (sample(px + 1, py) - sample(px - 1, py)) / (2.0 * m_cellWidth);
    const double gradientY = (sample(px, py + 1) - sample(px, py - 1)) / (2.0 * m_cellHeight);
    const double length = std::hypot(gradientX, gradientY);
    if (!(length > kNormalTolerance) || !std::isfinite(length)) {
        return {};
    }
    // F increases toward liquid, while the interface normal points toward air.
    return {-gradientX / length, -gradientY / length};
}

Vector2d VolumeFractionField::estimateHeightFunctionNormal(
    std::size_t x,
    std::size_t y,
    bool* valid) const noexcept {
    if (valid != nullptr) {
        *valid = false;
    }
    if (!isValid(x, y) || !(fraction(x, y) > 0.0 && fraction(x, y) < 1.0)) {
        return {};
    }
    const Vector2d gradientNormal = estimateNormal(x, y);
    if (std::hypot(gradientNormal.x, gradientNormal.y) <= kNormalTolerance) {
        return {};
    }
    constexpr std::ptrdiff_t radius = 3;
    const auto inBounds = [&](std::ptrdiff_t sx, std::ptrdiff_t sy) {
        return sx >= 0 && sy >= 0 && sx < static_cast<std::ptrdiff_t>(m_width) &&
            sy < static_cast<std::ptrdiff_t>(m_height) &&
            isValid(static_cast<std::size_t>(sx), static_cast<std::size_t>(sy));
    };
    const auto columnHeight = [&](std::ptrdiff_t sx, double& height, bool& liquidAtLowSide) {
        bool hasLiquid = false;
        bool hasAir = false;
        std::size_t partialCount = 0;
        std::size_t phaseChanges = 0;
        bool previousLiquid = false;
        bool havePrevious = false;
        bool firstLiquid = false;
        bool lastLiquid = false;
        height = 0.0;
        for (std::ptrdiff_t offset = -radius; offset <= radius; ++offset) {
            const std::ptrdiff_t sy = static_cast<std::ptrdiff_t>(y) + offset;
            if (!inBounds(sx, sy)) return false;
            const double value = fraction(static_cast<std::size_t>(sx), static_cast<std::size_t>(sy));
            height += value * m_cellHeight;
            hasLiquid = hasLiquid || value >= 1.0 - 1.0e-6;
            hasAir = hasAir || value <= 1.0e-6;
            partialCount += value > 1.0e-6 && value < 1.0 - 1.0e-6 ? 1u : 0u;
            const bool currentLiquid = value >= 0.5;
            if (!havePrevious) {
                firstLiquid = currentLiquid;
                havePrevious = true;
            } else if (currentLiquid != previousLiquid) {
                ++phaseChanges;
            }
            previousLiquid = currentLiquid;
            lastLiquid = currentLiquid;
        }
        liquidAtLowSide = firstLiquid;
        return hasLiquid && hasAir && firstLiquid != lastLiquid &&
            phaseChanges == 1 && partialCount <= 2;
    };
    const auto rowWidth = [&](std::ptrdiff_t sy, double& width, bool& liquidAtLowSide) {
        bool hasLiquid = false;
        bool hasAir = false;
        std::size_t partialCount = 0;
        std::size_t phaseChanges = 0;
        bool previousLiquid = false;
        bool havePrevious = false;
        bool firstLiquid = false;
        bool lastLiquid = false;
        width = 0.0;
        for (std::ptrdiff_t offset = -radius; offset <= radius; ++offset) {
            const std::ptrdiff_t sx = static_cast<std::ptrdiff_t>(x) + offset;
            if (!inBounds(sx, sy)) return false;
            const double value = fraction(static_cast<std::size_t>(sx), static_cast<std::size_t>(sy));
            width += value * m_cellWidth;
            hasLiquid = hasLiquid || value >= 1.0 - 1.0e-6;
            hasAir = hasAir || value <= 1.0e-6;
            partialCount += value > 1.0e-6 && value < 1.0 - 1.0e-6 ? 1u : 0u;
            const bool currentLiquid = value >= 0.5;
            if (!havePrevious) {
                firstLiquid = currentLiquid;
                havePrevious = true;
            } else if (currentLiquid != previousLiquid) {
                ++phaseChanges;
            }
            previousLiquid = currentLiquid;
            lastLiquid = currentLiquid;
        }
        liquidAtLowSide = firstLiquid;
        return hasLiquid && hasAir && firstLiquid != lastLiquid &&
            phaseChanges == 1 && partialCount <= 2;
    };

    Vector2d normal;
    bool heightValid = false;
    if (std::abs(gradientNormal.y) >= std::abs(gradientNormal.x)) {
        double left = 0.0, center = 0.0, right = 0.0;
        bool leftLow = false, centerLow = false, rightLow = false;
        heightValid = x > 0 && x + 1 < m_width &&
            columnHeight(static_cast<std::ptrdiff_t>(x) - 1, left, leftLow) &&
            columnHeight(static_cast<std::ptrdiff_t>(x), center, centerLow) &&
            columnHeight(static_cast<std::ptrdiff_t>(x) + 1, right, rightLow) &&
            leftLow == centerLow && centerLow == rightLow;
        if (heightValid) {
            static_cast<void>(center);
            const double slope = (right - left) / (2.0 * m_cellWidth);
            normal = {-slope, centerLow ? 1.0 : -1.0};
        }
    } else {
        double below = 0.0, center = 0.0, above = 0.0;
        bool belowLow = false, centerLow = false, aboveLow = false;
        heightValid = y > 0 && y + 1 < m_height &&
            rowWidth(static_cast<std::ptrdiff_t>(y) - 1, below, belowLow) &&
            rowWidth(static_cast<std::ptrdiff_t>(y), center, centerLow) &&
            rowWidth(static_cast<std::ptrdiff_t>(y) + 1, above, aboveLow) &&
            belowLow == centerLow && centerLow == aboveLow;
        if (heightValid) {
            static_cast<void>(center);
            const double slope = (above - below) / (2.0 * m_cellHeight);
            normal = {centerLow ? 1.0 : -1.0, -slope};
        }
    }
    const double length = std::hypot(normal.x, normal.y);
    if (!heightValid || !(length > kNormalTolerance) || !std::isfinite(length)) {
        return {};
    }
    normal.x /= length;
    normal.y /= length;
    if (normal.x * gradientNormal.x + normal.y * gradientNormal.y < 0.0) {
        normal.x = -normal.x;
        normal.y = -normal.y;
    }
    if (valid != nullptr) {
        *valid = true;
    }
    return normal;
}

PlicReconstruction VolumeFractionField::reconstructCell(
    std::size_t x,
    std::size_t y,
    const Vector2d* trustedNormal,
    PlicNormalMethod normalMethod) const noexcept {
    if (!isValid(x, y)) {
        return {};
    }
    bool heightValid = false;
    const Vector2d estimated = normalMethod == PlicNormalMethod::heightFunction
        ? estimateHeightFunctionNormal(x, y, &heightValid)
        : estimateNormal(x, y);
    return reconstructPlic(
        fraction(x, y),
        trustedNormal != nullptr ? *trustedNormal : estimated,
        m_cellWidth,
        m_cellHeight);
}

PlicReconstruction reconstructPlic(
    double fraction,
    Vector2d normal,
    double cellWidth,
    double cellHeight) noexcept {
    PlicReconstruction result;
    if (!std::isfinite(fraction) || !std::isfinite(normal.x) || !std::isfinite(normal.y) ||
        !(cellWidth > 0.0) || !(cellHeight > 0.0)) {
        return result;
    }
    const double length = std::hypot(normal.x, normal.y);
    if (!(length > kNormalTolerance)) {
        return result;
    }
    result.normal = {normal.x / length, normal.y / length};
    const double fullArea = cellWidth * cellHeight;
    const double targetArea = std::clamp(fraction, 0.0, 1.0) * fullArea;
    double lower = cornerProjectionMinimum(result.normal, cellWidth, cellHeight);
    double upper = cornerProjectionMaximum(result.normal, cellWidth, cellHeight);
    if (fraction <= 0.0) {
        result.alpha = lower;
        result.reconstructedArea = 0.0;
        result.valid = true;
        return result;
    }
    if (fraction >= 1.0) {
        result.alpha = upper;
        result.reconstructedArea = fullArea;
        result.valid = true;
        return result;
    }
    for (std::size_t iteration = 0; iteration < 64; ++iteration) {
        const double midpoint = 0.5 * (lower + upper);
        const double area = clippedRectangleArea(
            0.0, 0.0, cellWidth, cellHeight, result.normal, midpoint);
        if (area < targetArea) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    result.alpha = 0.5 * (lower + upper);
    result.reconstructedArea = clippedRectangleArea(
        0.0, 0.0, cellWidth, cellHeight, result.normal, result.alpha);
    result.valid = std::isfinite(result.alpha) &&
        std::abs(result.reconstructedArea - targetArea) <=
            std::max(1.0e-13, fullArea * 1.0e-10);
    return result;
}

double plicLiquidAreaInBox(
    const PlicReconstruction& reconstruction,
    double minimumX,
    double minimumY,
    double maximumX,
    double maximumY) noexcept {
    if (!reconstruction.valid) {
        return 0.0;
    }
    return clippedRectangleArea(
        minimumX,
        minimumY,
        maximumX,
        maximumY,
        reconstruction.normal,
        reconstruction.alpha);
}

PlicInterfaceSegment plicInterfaceSegment(
    const PlicReconstruction& reconstruction,
    double cellWidth,
    double cellHeight) noexcept {
    PlicInterfaceSegment result;
    if (!reconstruction.valid || !(cellWidth > 0.0) || !(cellHeight > 0.0)) {
        return result;
    }
    std::vector<Vector2d> intersections;
    const auto add = [&](double x, double y) {
        if (x < -1.0e-12 || x > cellWidth + 1.0e-12 ||
            y < -1.0e-12 || y > cellHeight + 1.0e-12) return;
        const Vector2d point{std::clamp(x, 0.0, cellWidth), std::clamp(y, 0.0, cellHeight)};
        for (const Vector2d& existing : intersections) {
            if (std::hypot(existing.x - point.x, existing.y - point.y) < 1.0e-10) return;
        }
        intersections.push_back(point);
    };
    if (std::abs(reconstruction.normal.y) > kNormalTolerance) {
        add(0.0, reconstruction.alpha / reconstruction.normal.y);
        add(cellWidth,
            (reconstruction.alpha - reconstruction.normal.x * cellWidth) /
                reconstruction.normal.y);
    }
    if (std::abs(reconstruction.normal.x) > kNormalTolerance) {
        add(reconstruction.alpha / reconstruction.normal.x, 0.0);
        add((reconstruction.alpha - reconstruction.normal.y * cellHeight) /
                reconstruction.normal.x,
            cellHeight);
    }
    if (intersections.size() < 2) {
        return result;
    }
    double maximumDistance = -1.0;
    for (std::size_t a = 0; a < intersections.size(); ++a) {
        for (std::size_t b = a + 1; b < intersections.size(); ++b) {
            const double distance = std::hypot(
                intersections[a].x - intersections[b].x,
                intersections[a].y - intersections[b].y);
            if (distance > maximumDistance) {
                maximumDistance = distance;
                result.first = intersections[a];
                result.second = intersections[b];
            }
        }
    }
    result.valid = maximumDistance > 1.0e-12;
    return result;
}

VolumeFractionLedger VolumeFractionField::advectAxis(
    const MacGrid& velocity,
    double timeStep,
    bool xDirection,
    const std::vector<unsigned char>* compressionCoefficient) {
    VolumeFractionLedger ledger;
    ledger.areaBefore = totalArea();
    const double fullCellArea = m_cellWidth * m_cellHeight;
    struct FaceTransfer final {
        std::size_t donor = 0;
        std::size_t receiver = 0;
        double amount = 0.0;
        double rawAmount = 0.0;
        double donorFraction = 0.0;
        double receiverFraction = 0.0;
        double speed = 0.0;
        double localDivergence = 0.0;
        Vector2d normal;
        std::size_t faceX = 0;
        std::size_t faceY = 0;
        std::size_t scaleEventCount = 0;
        bool positiveDirection = false;
        bool donorLimited = false;
        bool receiverLimited = false;
        bool coupled = false;
    };
    std::vector<FaceTransfer> transfers;
    transfers.reserve((xDirection ? (m_width - 1) * m_height : m_width * (m_height - 1)));

    const auto addTransfer = [&](std::size_t donorX,
                                 std::size_t donorY,
                                 std::size_t receiverX,
                                 std::size_t receiverY,
                                 double speed,
                                 bool fromLowSide,
                                 std::size_t faceX,
                                 std::size_t faceY) {
        ++ledger.evaluatedFaceCount;
        if (!isValid(donorX, donorY) || !isValid(receiverX, receiverY)) {
            ++ledger.solidAdjacentFaceCount;
            return;
        }
        if (speed == 0.0) {
            return;
        }
        const double donorFraction = fraction(donorX, donorY);
        if (!(donorFraction > 0.0)) {
            return;
        }
        const double extent = xDirection ? m_cellWidth : m_cellHeight;
        const double swept = std::min(std::abs(speed) * timeStep, extent);
        PlicReconstruction reconstruction = reconstructCell(donorX, donorY);
        double amount = 0.0;
        if (donorFraction >= 1.0 - kFractionTolerance) {
            amount = swept * (xDirection ? m_cellHeight : m_cellWidth);
        } else if (reconstruction.valid) {
            if (xDirection) {
                const double minimumX = fromLowSide ? 0.0 : m_cellWidth - swept;
                const double maximumX = fromLowSide ? swept : m_cellWidth;
                amount = plicLiquidAreaInBox(
                    reconstruction, minimumX, 0.0, maximumX, m_cellHeight);
            } else {
                const double minimumY = fromLowSide ? 0.0 : m_cellHeight - swept;
                const double maximumY = fromLowSide ? swept : m_cellHeight;
                amount = plicLiquidAreaInBox(
                    reconstruction, 0.0, minimumY, m_cellWidth, maximumY);
            }
        } else {
            ++ledger.plicFailureCount;
            amount = donorFraction * swept * (xDirection ? m_cellHeight : m_cellWidth);
        }
        if (amount > 0.0) {
            const double divergence =
                (velocity.horizontalVelocity(donorX + 1, donorY) -
                    velocity.horizontalVelocity(donorX, donorY)) / m_cellWidth +
                (velocity.verticalVelocity(donorX, donorY + 1) -
                    velocity.verticalVelocity(donorX, donorY)) / m_cellHeight;
            transfers.push_back({
                index(donorX, donorY),
                index(receiverX, receiverY),
                amount,
                amount,
                donorFraction,
                fraction(receiverX, receiverY),
                speed,
                divergence,
                reconstruction.valid ? reconstruction.normal : Vector2d{},
                faceX,
                faceY,
                0,
                speed > 0.0,
                false,
                false,
                false,
            });
            ++ledger.nonZeroLiquidFluxFaceCount;
            ledger.rawAbsoluteFlux += amount;
        }
    };

    if (xDirection) {
        for (std::size_t y = 0; y < m_height; ++y) {
            for (std::size_t faceX = 1; faceX < m_width; ++faceX) {
                const double speed = velocity.horizontalVelocity(faceX, y);
                if (speed > 0.0) {
                    addTransfer(faceX - 1, y, faceX, y, speed, false, faceX, y);
                } else if (speed < 0.0) {
                    addTransfer(faceX, y, faceX - 1, y, speed, true, faceX, y);
                } else {
                    ++ledger.evaluatedFaceCount;
                }
            }
        }
        ledger.xFaceFluxCount = transfers.size();
    } else {
        for (std::size_t faceY = 1; faceY < m_height; ++faceY) {
            for (std::size_t x = 0; x < m_width; ++x) {
                const double speed = velocity.verticalVelocity(x, faceY);
                if (speed > 0.0) {
                    addTransfer(x, faceY - 1, x, faceY, speed, false, x, faceY);
                } else if (speed < 0.0) {
                    addTransfer(x, faceY, x, faceY - 1, speed, true, x, faceY);
                } else {
                    ++ledger.evaluatedFaceCount;
                }
            }
        }
        ledger.yFaceFluxCount = transfers.size();
    }

    std::vector<double> compressionVolume(m_fraction.size(), 0.0);
    if (compressionCoefficient != nullptr) {
        for (std::size_t y = 0; y < m_height; ++y) {
            for (std::size_t x = 0; x < m_width; ++x) {
                const std::size_t offset = index(x, y);
                if (m_valid[offset] == 0 || (*compressionCoefficient)[offset] == 0) {
                    continue;
                }
                const double directionalDivergence = xDirection
                    ? (velocity.horizontalVelocity(x + 1, y) -
                        velocity.horizontalVelocity(x, y)) / m_cellWidth
                    : (velocity.verticalVelocity(x, y + 1) -
                        velocity.verticalVelocity(x, y)) / m_cellHeight;
                compressionVolume[offset] =
                    timeStep * directionalDivergence * fullCellArea;
                ledger.compressionContribution += compressionVolume[offset];
            }
        }
        if (xDirection) {
            ledger.xCompressionContribution = ledger.compressionContribution;
        } else {
            ledger.yCompressionContribution = ledger.compressionContribution;
        }
    }

    // Limit shared face transfers using the complete in/out balance. A full
    // cell may accept volume when it also exports volume in the same split.
    // Scaling the shared transfer preserves conservation on both cells.
    const double volumeTolerance = kFractionTolerance * fullCellArea;
    // A bound violation can propagate along an entire split-direction chain.
    // The previous fixed 64 iterations rejected otherwise valid uniform-field
    // tests once a row or column exceeded that length.
    const std::size_t limiterIterationLimit = std::max<std::size_t>(
        64, 2 * (xDirection ? m_width : m_height) + 8);
    for (std::size_t iteration = 0; iteration < limiterIterationLimit; ++iteration) {
        std::vector<double> requestedOut(m_fraction.size(), 0.0);
        std::vector<double> requestedIn(m_fraction.size(), 0.0);
        for (const FaceTransfer& transfer : transfers) {
            requestedOut[transfer.donor] += transfer.amount;
            requestedIn[transfer.receiver] += transfer.amount;
        }
        std::vector<double> outgoingScale(m_fraction.size(), 1.0);
        std::vector<double> incomingScale(m_fraction.size(), 1.0);
        bool needsScaling = false;
        for (std::size_t offset = 0; offset < m_fraction.size(); ++offset) {
            const double currentVolume =
                m_fraction[offset] * fullCellArea + compressionVolume[offset];
            const double proposedVolume = currentVolume + requestedIn[offset] - requestedOut[offset];
            if (proposedVolume < -volumeTolerance && requestedOut[offset] > 0.0) {
                outgoingScale[offset] = std::clamp(
                    (currentVolume + requestedIn[offset]) / requestedOut[offset], 0.0, 1.0);
                needsScaling = true;
            } else if (proposedVolume > fullCellArea + volumeTolerance &&
                       requestedIn[offset] > 0.0) {
                incomingScale[offset] = std::clamp(
                    (fullCellArea - currentVolume + requestedOut[offset]) / requestedIn[offset],
                    0.0,
                    1.0);
                needsScaling = true;
            }
        }
        if (!needsScaling) {
            break;
        }
        ++ledger.limiterIterationCount;
        for (FaceTransfer& transfer : transfers) {
            const double scale = std::min(
                outgoingScale[transfer.donor], incomingScale[transfer.receiver]);
            if (scale < 1.0) {
                if (transfer.scaleEventCount == 0) {
                    ++ledger.uniqueLimitedFaceCount;
                    ledger.xLimitedFaceCount += xDirection ? 1u : 0u;
                    ledger.yLimitedFaceCount += xDirection ? 0u : 1u;
                    ledger.positiveLimitedFaceCount += transfer.positiveDirection ? 1u : 0u;
                    ledger.negativeLimitedFaceCount += transfer.positiveDirection ? 0u : 1u;
                    const double courant = std::abs(transfer.speed) * timeStep /
                        (xDirection ? m_cellWidth : m_cellHeight);
                    ledger.highCourantLimitedCount += courant > 0.5 ? 1u : 0u;
                    ledger.divergentCellLimitedCount +=
                        std::abs(transfer.localDivergence) > 1.0e-4 ? 1u : 0u;
                } else {
                    ++ledger.repeatedLimitEventCount;
                }
                const bool donorLimited = outgoingScale[transfer.donor] <=
                    incomingScale[transfer.receiver] && outgoingScale[transfer.donor] < 1.0;
                const bool receiverLimited = incomingScale[transfer.receiver] <=
                    outgoingScale[transfer.donor] && incomingScale[transfer.receiver] < 1.0;
                transfer.donorLimited = transfer.donorLimited || donorLimited;
                transfer.receiverLimited = transfer.receiverLimited || receiverLimited;
                transfer.coupled = transfer.coupled ||
                    (requestedIn[transfer.donor] > 0.0 && requestedOut[transfer.donor] > 0.0) ||
                    (requestedIn[transfer.receiver] > 0.0 && requestedOut[transfer.receiver] > 0.0);
                transfer.amount *= scale;
                ++transfer.scaleEventCount;
                ++ledger.limitedFaceCount;
            }
        }
    }

    for (const FaceTransfer& transfer : transfers) {
        ledger.finalAbsoluteFlux += transfer.amount;
        if (transfer.scaleEventCount == 0 || !(transfer.rawAmount > 0.0)) continue;
        const double finalScale = transfer.amount / transfer.rawAmount;
        ledger.limiterScaleSamples.push_back(finalScale);
        ledger.minimumLimiterScale = std::min(ledger.minimumLimiterScale, finalScale);
        ledger.nearBlockedFaceCount += finalScale < 0.01 ? 1u : 0u;
        ledger.donorInsufficientCount += transfer.donorLimited ? 1u : 0u;
        ledger.receiverCapacityCount += transfer.receiverLimited ? 1u : 0u;
        ledger.coupledInOutCount += transfer.coupled ? 1u : 0u;
        ledger.strongestLimitedFaces.push_back({
            transfer.faceX,
            transfer.faceY,
            xDirection,
            transfer.positiveDirection,
            transfer.rawAmount,
            transfer.amount,
            transfer.donorFraction,
            transfer.receiverFraction,
            transfer.speed,
            transfer.localDivergence,
            transfer.normal,
        });
    }
    if (!ledger.limiterScaleSamples.empty()) {
        std::sort(ledger.limiterScaleSamples.begin(), ledger.limiterScaleSamples.end());
        double scaleSum = 0.0;
        for (double scale : ledger.limiterScaleSamples) scaleSum += scale;
        ledger.meanLimiterScale = scaleSum / ledger.limiterScaleSamples.size();
        ledger.p95LimiterScale = ledger.limiterScaleSamples[static_cast<std::size_t>(
            0.95 * static_cast<double>(ledger.limiterScaleSamples.size() - 1))];
        std::sort(
            ledger.strongestLimitedFaces.begin(),
            ledger.strongestLimitedFaces.end(),
            [](const LimitedFluxEvent& a, const LimitedFluxEvent& b) {
                return a.finalFlux / a.rawFlux < b.finalFlux / b.rawFlux;
            });
        if (ledger.strongestLimitedFaces.size() > 16) {
            ledger.strongestLimitedFaces.resize(16);
        }
    }

    std::vector<double> updated = m_fraction;
    for (std::size_t offset = 0; offset < updated.size(); ++offset) {
        updated[offset] += compressionVolume[offset] / fullCellArea;
    }
    for (const FaceTransfer& transfer : transfers) {
        updated[transfer.donor] -= transfer.amount / fullCellArea;
        updated[transfer.receiver] += transfer.amount / fullCellArea;
    }
    ledger.minimumFraction = std::numeric_limits<double>::infinity();
    ledger.maximumFraction = -std::numeric_limits<double>::infinity();
    for (std::size_t offset = 0; offset < updated.size(); ++offset) {
        if (m_valid[offset] == 0) {
            updated[offset] = 0.0;
            continue;
        }
        ledger.minimumFraction = std::min(ledger.minimumFraction, updated[offset]);
        ledger.maximumFraction = std::max(ledger.maximumFraction, updated[offset]);
        if (updated[offset] < -kFractionTolerance || updated[offset] > 1.0 + kFractionTolerance ||
            !std::isfinite(updated[offset])) {
            ++ledger.outOfBoundsCount;
        }
    }
    if (ledger.outOfBoundsCount == 0) {
        for (double& value : updated) {
            if (value < 0.0 && value >= -kFractionTolerance) {
                value = 0.0;
            } else if (value > 1.0 && value <= 1.0 + kFractionTolerance) {
                value = 1.0;
            }
        }
        m_fraction.swap(updated);
        ledger.accepted = true;
    }
    ledger.areaAfter = totalArea();
    ledger.conservationResidual = ledger.areaAfter - ledger.areaBefore + ledger.boundaryFlux -
        ledger.compressionContribution;
    return ledger;
}

VolumeFractionLedger VolumeFractionField::advectConservative(
    const MacGrid& velocity,
    double timeStep,
    bool xDirectionFirst) {
    return advectSplit(
        velocity, timeStep, xDirectionFirst, VolumeFractionAdvectionMode::legacySplit);
}

VolumeFractionLedger VolumeFractionField::advectConservativeSplitExperimental(
    const MacGrid& velocity,
    double timeStep,
    bool xDirectionFirst) {
    return advectSplit(
        velocity,
        timeStep,
        xDirectionFirst,
        VolumeFractionAdvectionMode::conservativeSplitExperimental);
}

VolumeFractionLedger VolumeFractionField::advectSplit(
    const MacGrid& velocity,
    double timeStep,
    bool xDirectionFirst,
    VolumeFractionAdvectionMode mode) {
    if (velocity.width() != m_width || velocity.height() != m_height ||
        std::abs(static_cast<double>(velocity.cellWidth()) - m_cellWidth) > 1.0e-7 ||
        std::abs(static_cast<double>(velocity.cellHeight()) - m_cellHeight) > 1.0e-7 ||
        !(timeStep > 0.0)) {
        throw std::invalid_argument("VOF advection layout or time step is invalid.");
    }
    const double maximumXCourant = [&]() {
        double value = 0.0;
        for (float speed : velocity.horizontalVelocities()) {
            value = std::max(value, std::abs(static_cast<double>(speed)) * timeStep / m_cellWidth);
        }
        return value;
    }();
    const double maximumYCourant = [&]() {
        double value = 0.0;
        for (float speed : velocity.verticalVelocities()) {
            value = std::max(value, std::abs(static_cast<double>(speed)) * timeStep / m_cellHeight);
        }
        return value;
    }();
    if (maximumXCourant > 1.0 + 1.0e-9 || maximumYCourant > 1.0 + 1.0e-9) {
        throw std::invalid_argument("VOF advection requires directional CFL no greater than one.");
    }

    const double areaBefore = totalArea();
    const std::vector<double> fractionBefore = m_fraction;
    const double physicalTimeBefore = m_physicalTime;
    const std::size_t versionBefore = m_version;
    std::vector<unsigned char> compressionCoefficient;
    const std::vector<unsigned char>* coefficient = nullptr;
    if (mode == VolumeFractionAdvectionMode::conservativeSplitExperimental) {
        compressionCoefficient.resize(m_fraction.size(), 0);
        for (std::size_t offset = 0; offset < m_fraction.size(); ++offset) {
            compressionCoefficient[offset] =
                m_valid[offset] != 0 && m_fraction[offset] >= 0.5 ? 1u : 0u;
        }
        coefficient = &compressionCoefficient;
    }
    VolumeFractionLedger first =
        advectAxis(velocity, timeStep, xDirectionFirst, coefficient);
    first.mode = mode;
    first.maximumXCourant = maximumXCourant;
    first.maximumYCourant = maximumYCourant;
    if (!first.accepted) {
        m_fraction = fractionBefore;
        return first;
    }
    const VolumeFractionLedger second =
        advectAxis(velocity, timeStep, !xDirectionFirst, coefficient);
    VolumeFractionLedger combined = second;
    combined.mode = mode;
    combined.areaBefore = areaBefore;
    combined.maximumXCourant = maximumXCourant;
    combined.maximumYCourant = maximumYCourant;
    combined.xFaceFluxCount += first.xFaceFluxCount;
    combined.yFaceFluxCount += first.yFaceFluxCount;
    combined.limitedFaceCount += first.limitedFaceCount;
    combined.evaluatedFaceCount += first.evaluatedFaceCount;
    combined.nonZeroLiquidFluxFaceCount += first.nonZeroLiquidFluxFaceCount;
    combined.uniqueLimitedFaceCount += first.uniqueLimitedFaceCount;
    combined.repeatedLimitEventCount += first.repeatedLimitEventCount;
    combined.limiterIterationCount += first.limiterIterationCount;
    combined.positiveLimitedFaceCount += first.positiveLimitedFaceCount;
    combined.negativeLimitedFaceCount += first.negativeLimitedFaceCount;
    combined.xLimitedFaceCount += first.xLimitedFaceCount;
    combined.yLimitedFaceCount += first.yLimitedFaceCount;
    combined.donorInsufficientCount += first.donorInsufficientCount;
    combined.receiverCapacityCount += first.receiverCapacityCount;
    combined.coupledInOutCount += first.coupledInOutCount;
    combined.highCourantLimitedCount += first.highCourantLimitedCount;
    combined.divergentCellLimitedCount += first.divergentCellLimitedCount;
    combined.solidAdjacentFaceCount += first.solidAdjacentFaceCount;
    combined.nearBlockedFaceCount += first.nearBlockedFaceCount;
    combined.rawAbsoluteFlux += first.rawAbsoluteFlux;
    combined.finalAbsoluteFlux += first.finalAbsoluteFlux;
    combined.plicFailureCount += first.plicFailureCount;
    combined.outOfBoundsCount += first.outOfBoundsCount;
    combined.boundaryFlux += first.boundaryFlux;
    combined.solidBoundaryFlux += first.solidBoundaryFlux;
    combined.compressionContribution += first.compressionContribution;
    combined.xCompressionContribution += first.xCompressionContribution;
    combined.yCompressionContribution += first.yCompressionContribution;
    combined.areaAfter = totalArea();
    combined.conservationResidual = combined.areaAfter - areaBefore + combined.boundaryFlux -
        combined.compressionContribution;
    combined.accepted = first.accepted && second.accepted;
    combined.limiterScaleSamples.insert(
        combined.limiterScaleSamples.end(),
        first.limiterScaleSamples.begin(),
        first.limiterScaleSamples.end());
    combined.strongestLimitedFaces.insert(
        combined.strongestLimitedFaces.end(),
        first.strongestLimitedFaces.begin(),
        first.strongestLimitedFaces.end());
    if (!combined.limiterScaleSamples.empty()) {
        std::sort(combined.limiterScaleSamples.begin(), combined.limiterScaleSamples.end());
        double scaleSum = 0.0;
        for (double scale : combined.limiterScaleSamples) scaleSum += scale;
        combined.minimumLimiterScale = combined.limiterScaleSamples.front();
        combined.meanLimiterScale = scaleSum / combined.limiterScaleSamples.size();
        combined.p95LimiterScale = combined.limiterScaleSamples[static_cast<std::size_t>(
            0.95 * static_cast<double>(combined.limiterScaleSamples.size() - 1))];
        std::sort(
            combined.strongestLimitedFaces.begin(),
            combined.strongestLimitedFaces.end(),
            [](const LimitedFluxEvent& a, const LimitedFluxEvent& b) {
                return a.finalFlux / a.rawFlux < b.finalFlux / b.rawFlux;
            });
        if (combined.strongestLimitedFaces.size() > 16) {
            combined.strongestLimitedFaces.resize(16);
        }
    }
    if (combined.accepted) {
        m_physicalTime += timeStep;
        ++m_version;
    } else {
        m_fraction = fractionBefore;
        m_physicalTime = physicalTimeBefore;
        m_version = versionBefore;
        combined.areaAfter = areaBefore;
        combined.conservationResidual = 0.0;
    }
    return combined;
}

VolumeFractionDistribution VolumeFractionField::distribution(double tolerance) const noexcept {
    VolumeFractionDistribution result;
    result.tolerance = std::max(0.0, tolerance);
    for (std::size_t offset = 0; offset < m_fraction.size(); ++offset) {
        if (m_valid[offset] == 0) {
            ++result.solid;
            continue;
        }
        const double value = m_fraction[offset];
        result.strictNonZero += value > 0.0 ? 1u : 0u;
        if (value <= result.tolerance) {
            ++result.air;
        } else if (value < 0.01) {
            ++result.lowTail;
        } else if (value < 0.1) {
            ++result.lowPartial;
        } else if (value <= 0.9) {
            ++result.centralMixed;
        } else if (value < 0.99) {
            ++result.highPartial;
        } else if (value < 1.0 - result.tolerance) {
            ++result.nearFull;
        } else {
            ++result.full;
        }
    }
    return result;
}

void VolumeFractionField::setPhysicalTime(double value) noexcept { m_physicalTime = value; }
void VolumeFractionField::incrementVersion() noexcept { ++m_version; }
const std::vector<double>& VolumeFractionField::fractions() const noexcept { return m_fraction; }
const std::vector<unsigned char>& VolumeFractionField::validMask() const noexcept { return m_valid; }

void VolumeFractionTransaction::initialize(VolumeFractionField field) {
    m_committed = std::move(field);
    m_candidate = {};
    m_hasCandidate = false;
}

VolumeFractionField& VolumeFractionTransaction::prepareCandidate() {
    m_candidate = m_committed;
    m_hasCandidate = true;
    return m_candidate;
}

void VolumeFractionTransaction::acceptCandidate() {
    if (!m_hasCandidate) {
        throw std::logic_error("No VOF candidate is available to accept.");
    }
    m_committed = std::move(m_candidate);
    m_candidate = {};
    m_hasCandidate = false;
}

void VolumeFractionTransaction::rejectCandidate() noexcept {
    m_candidate = {};
    m_hasCandidate = false;
}

void VolumeFractionTransaction::reset() {
    m_committed = {};
    rejectCandidate();
}

const VolumeFractionField& VolumeFractionTransaction::committed() const noexcept { return m_committed; }
const VolumeFractionField& VolumeFractionTransaction::candidate() const noexcept { return m_candidate; }
bool VolumeFractionTransaction::hasCandidate() const noexcept { return m_hasCandidate; }

} // namespace fluidlab::simulation
