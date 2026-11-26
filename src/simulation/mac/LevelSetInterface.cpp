#include "LevelSetInterface.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace fluidlab::simulation {
namespace {

[[nodiscard]] std::size_t indexOf(
    std::size_t width, std::size_t x, std::size_t y) noexcept {
    return y * width + x;
}

[[nodiscard]] double clampCoordinate(double value, double minimum, double maximum) noexcept {
    return std::min(std::max(value, minimum), maximum);
}

struct Point final {
    double x = 0.0;
    double y = 0.0;
};

struct Segment final {
    Point a;
    Point b;
};

[[nodiscard]] double pointSegmentDistance(double x, double y, const Segment& segment) noexcept {
    const double dx = segment.b.x - segment.a.x;
    const double dy = segment.b.y - segment.a.y;
    const double lengthSquared = dx * dx + dy * dy;
    if (!(lengthSquared > 0.0)) return std::hypot(x - segment.a.x, y - segment.a.y);
    const double t = std::clamp(
        ((x - segment.a.x) * dx + (y - segment.a.y) * dy) / lengthSquared, 0.0, 1.0);
    return std::hypot(x - (segment.a.x + t * dx), y - (segment.a.y + t * dy));
}

[[nodiscard]] std::size_t negativeComponentCount(const LevelSetField& field) {
    const std::size_t count = field.width * field.height;
    std::vector<unsigned char> visited(count, 0u);
    std::queue<std::size_t> pending;
    std::size_t components = 0;
    for (std::size_t start = 0; start < count; ++start) {
        if (visited[start] != 0u || !std::isfinite(field.values[start]) || field.values[start] > 0.0) {
            continue;
        }
        ++components;
        visited[start] = 1u;
        pending.push(start);
        while (!pending.empty()) {
            const std::size_t current = pending.front();
            pending.pop();
            const std::size_t x = current % field.width;
            const std::size_t y = current / field.width;
            const auto visit = [&](std::size_t nx, std::size_t ny) {
                const std::size_t neighbor = indexOf(field.width, nx, ny);
                if (visited[neighbor] == 0u && std::isfinite(field.values[neighbor]) &&
                    field.values[neighbor] <= 0.0) {
                    visited[neighbor] = 1u;
                    pending.push(neighbor);
                }
            };
            if (x > 0) visit(x - 1, y);
            if (x + 1 < field.width) visit(x + 1, y);
            if (y > 0) visit(x, y - 1);
            if (y + 1 < field.height) visit(x, y + 1);
        }
    }
    return components;
}

[[nodiscard]] std::vector<Point> edgeCrossings(const LevelSetField& field) {
    std::vector<Point> crossings;
    crossings.reserve(field.width * field.height / 2);
    for (std::size_t y = 0; y < field.height; ++y) {
        for (std::size_t x = 0; x + 1 < field.width; ++x) {
            const double a = field.value(x, y);
            const double b = field.value(x + 1, y);
            if (!std::isfinite(a) || !std::isfinite(b) || (a < 0.0) == (b < 0.0)) continue;
            const double theta = std::abs(a) / (std::abs(a) + std::abs(b));
            crossings.push_back({(static_cast<double>(x) + 0.5 + theta) * field.cellWidth,
                (static_cast<double>(y) + 0.5) * field.cellHeight});
        }
    }
    for (std::size_t y = 0; y + 1 < field.height; ++y) {
        for (std::size_t x = 0; x < field.width; ++x) {
            const double a = field.value(x, y);
            const double b = field.value(x, y + 1);
            if (!std::isfinite(a) || !std::isfinite(b) || (a < 0.0) == (b < 0.0)) continue;
            const double theta = std::abs(a) / (std::abs(a) + std::abs(b));
            crossings.push_back({(static_cast<double>(x) + 0.5) * field.cellWidth,
                (static_cast<double>(y) + 0.5 + theta) * field.cellHeight});
        }
    }
    return crossings;
}

[[nodiscard]] std::vector<Segment> extractZeroSegments(const LevelSetField& field) {
    std::vector<Segment> segments;
    segments.reserve(field.width * field.height / 2);
    const auto crossing = [](Point a, Point b, double va, double vb) {
        const double theta = std::abs(va) / (std::abs(va) + std::abs(vb));
        return Point{a.x + theta * (b.x - a.x), a.y + theta * (b.y - a.y)};
    };
    for (std::size_t y = 0; y + 1 < field.height; ++y) {
        for (std::size_t x = 0; x + 1 < field.width; ++x) {
            const Point p[4]{
                {(x + 0.5) * field.cellWidth, (y + 0.5) * field.cellHeight},
                {(x + 1.5) * field.cellWidth, (y + 0.5) * field.cellHeight},
                {(x + 1.5) * field.cellWidth, (y + 1.5) * field.cellHeight},
                {(x + 0.5) * field.cellWidth, (y + 1.5) * field.cellHeight}};
            const double v[4]{field.value(x, y), field.value(x + 1, y),
                field.value(x + 1, y + 1), field.value(x, y + 1)};
            if (!std::all_of(std::begin(v), std::end(v), [](double value) {
                    return std::isfinite(value);
                })) continue;
            Point hits[4];
            std::size_t hitCount = 0;
            for (std::size_t edge = 0; edge < 4; ++edge) {
                const std::size_t next = (edge + 1) % 4;
                if ((v[edge] < 0.0) != (v[next] < 0.0)) {
                    hits[hitCount++] = crossing(p[edge], p[next], v[edge], v[next]);
                }
            }
            if (hitCount == 2) {
                segments.push_back({hits[0], hits[1]});
            } else if (hitCount == 4) {
                const double center = 0.25 * (v[0] + v[1] + v[2] + v[3]);
                const bool connectFirst = (center < 0.0) == (v[0] < 0.0);
                if (connectFirst) {
                    segments.push_back({hits[0], hits[3]});
                    segments.push_back({hits[1], hits[2]});
                } else {
                    segments.push_back({hits[0], hits[1]});
                    segments.push_back({hits[2], hits[3]});
                }
            }
        }
    }
    return segments;
}

} // namespace

double LevelSetField::value(std::size_t x, std::size_t y) const noexcept {
    return values[indexOf(width, x, y)];
}

double LevelSetField::sample(double x, double y) const noexcept {
    if (width == 0 || height == 0 || values.size() != width * height) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double gx = clampCoordinate(x / cellWidth - 0.5, 0.0,
        static_cast<double>(width - 1));
    const double gy = clampCoordinate(y / cellHeight - 0.5, 0.0,
        static_cast<double>(height - 1));
    const auto x0 = static_cast<std::size_t>(std::floor(gx));
    const auto y0 = static_cast<std::size_t>(std::floor(gy));
    const std::size_t x1 = std::min(x0 + 1, width - 1);
    const std::size_t y1 = std::min(y0 + 1, height - 1);
    const double tx = gx - static_cast<double>(x0);
    const double ty = gy - static_cast<double>(y0);
    const double a = value(x0, y0) * (1.0 - tx) + value(x1, y0) * tx;
    const double b = value(x0, y1) * (1.0 - tx) + value(x1, y1) * tx;
    return a * (1.0 - ty) + b * ty;
}

Vector2 LevelSetField::normal(double x, double y) const noexcept {
    const double epsilonX = 0.5 * cellWidth;
    const double epsilonY = 0.5 * cellHeight;
    const double nx = (sample(x + epsilonX, y) - sample(x - epsilonX, y)) /
        (2.0 * epsilonX);
    const double ny = (sample(x, y + epsilonY) - sample(x, y - epsilonY)) /
        (2.0 * epsilonY);
    const double length = std::hypot(nx, ny);
    if (!std::isfinite(length) || length <= 1.0e-12) return {};
    return {static_cast<float>(nx / length), static_cast<float>(ny / length)};
}

LevelSetField sampleAnalyticLevelSet(
    std::size_t width,
    std::size_t height,
    double cellWidth,
    double cellHeight,
    const ScalarFieldFunction& field,
    std::size_t sourceVersion) {
    if (width < 2 || height < 2 || !(cellWidth > 0.0) || !(cellHeight > 0.0) || !field) {
        throw std::invalid_argument("Level-set sampling requires a valid grid and field.");
    }
    LevelSetField result;
    result.width = width;
    result.height = height;
    result.cellWidth = cellWidth;
    result.cellHeight = cellHeight;
    result.values.resize(width * height);
    result.confidence.assign(width * height, 1u);
    result.sourceVersion = sourceVersion;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const double px = (static_cast<double>(x) + 0.5) * cellWidth;
            const double py = (static_cast<double>(y) + 0.5) * cellHeight;
            const double value = field(px, py);
            result.values[indexOf(width, x, y)] = value;
            if (!std::isfinite(value)) result.confidence[indexOf(width, x, y)] = 0u;
        }
    }
    return result;
}

LevelSetField buildClassificationLevelSet(const MacGrid& grid) {
    const double halfCell = 0.5 * std::min(grid.cellWidth(), grid.cellHeight());
    LevelSetField result;
    result.width = grid.width();
    result.height = grid.height();
    result.cellWidth = grid.cellWidth();
    result.cellHeight = grid.cellHeight();
    result.values.resize(result.width * result.height, halfCell);
    result.confidence.assign(result.width * result.height, 1u);
    result.sourceVersion = 1;
    for (std::size_t y = 0; y < result.height; ++y) {
        for (std::size_t x = 0; x < result.width; ++x) {
            const std::size_t index = indexOf(result.width, x, y);
            if (grid.cellType(x, y) == CellType::fluid) result.values[index] = -halfCell;
            if (grid.cellType(x, y) == CellType::solid) {
                result.values[index] = std::numeric_limits<double>::quiet_NaN();
                result.confidence[index] = 0u;
            }
        }
    }
    return result;
}

LevelSetField advectLevelSetRk2(
    const LevelSetField& source,
    const VelocityFieldFunction& velocity,
    double timeStep) {
    if (!velocity || !std::isfinite(timeStep)) {
        throw std::invalid_argument("Level-set advection requires a finite step and velocity.");
    }
    LevelSetField result = source;
    ++result.sourceVersion;
    const double maximumX = source.width * source.cellWidth;
    const double maximumY = source.height * source.cellHeight;
    for (std::size_t y = 0; y < source.height; ++y) {
        for (std::size_t x = 0; x < source.width; ++x) {
            const std::size_t index = indexOf(source.width, x, y);
            const double px = (static_cast<double>(x) + 0.5) * source.cellWidth;
            const double py = (static_cast<double>(y) + 0.5) * source.cellHeight;
            const Vector2 first = velocity(px, py);
            const double middleX = px - 0.5 * timeStep * first.x;
            const double middleY = py - 0.5 * timeStep * first.y;
            const Vector2 middle = velocity(middleX, middleY);
            const double departureX = px - timeStep * middle.x;
            const double departureY = py - timeStep * middle.y;
            const bool inside = departureX >= 0.5 * source.cellWidth &&
                departureX <= maximumX - 0.5 * source.cellWidth &&
                departureY >= 0.5 * source.cellHeight &&
                departureY <= maximumY - 0.5 * source.cellHeight;
            result.values[index] = source.sample(departureX, departureY);
            result.confidence[index] = inside && std::isfinite(result.values[index]) ? 1u : 0u;
        }
    }
    return result;
}

LevelSetField advectLevelSetLimitedMacCormack(
    const LevelSetField& source,
    const VelocityFieldFunction& velocity,
    double timeStep,
    LevelSetMacCormackReport* report) {
    if (!velocity || !std::isfinite(timeStep)) {
        throw std::invalid_argument("Level-set advection requires a finite step and velocity.");
    }
    LevelSetMacCormackReport localReport;
    const LevelSetField forward = advectLevelSetRk2(source, velocity, timeStep);
    const LevelSetField backward = advectLevelSetRk2(forward, velocity, -timeStep);
    LevelSetField result = forward;
    const double minimumX = 0.5 * source.cellWidth;
    const double minimumY = 0.5 * source.cellHeight;
    const double maximumX = (static_cast<double>(source.width) - 0.5) * source.cellWidth;
    const double maximumY = (static_cast<double>(source.height) - 0.5) * source.cellHeight;
    for (std::size_t y = 0; y < source.height; ++y) {
        for (std::size_t x = 0; x < source.width; ++x) {
            const std::size_t index = indexOf(source.width, x, y);
            const double px = (static_cast<double>(x) + 0.5) * source.cellWidth;
            const double py = (static_cast<double>(y) + 0.5) * source.cellHeight;
            const Vector2 first = velocity(px, py);
            const Vector2 middle = velocity(
                px - 0.5 * timeStep * first.x,
                py - 0.5 * timeStep * first.y);
            const double departureX = px - timeStep * middle.x;
            const double departureY = py - timeStep * middle.y;
            const bool inside = departureX >= minimumX && departureX <= maximumX &&
                departureY >= minimumY && departureY <= maximumY;
            const double qx = std::clamp(departureX / source.cellWidth - 0.5,
                0.0, static_cast<double>(source.width - 1));
            const double qy = std::clamp(departureY / source.cellHeight - 0.5,
                0.0, static_cast<double>(source.height - 1));
            const std::size_t x0 = static_cast<std::size_t>(std::floor(qx));
            const std::size_t y0 = static_cast<std::size_t>(std::floor(qy));
            const std::size_t x1 = std::min(x0 + 1, source.width - 1);
            const std::size_t y1 = std::min(y0 + 1, source.height - 1);
            const std::array<std::size_t, 4> stencil{
                indexOf(source.width, x0, y0), indexOf(source.width, x1, y0),
                indexOf(source.width, x0, y1), indexOf(source.width, x1, y1)};
            bool reliable = inside && forward.confidence[index] != 0u &&
                backward.confidence[index] != 0u;
            double minimum = std::numeric_limits<double>::infinity();
            double maximum = -std::numeric_limits<double>::infinity();
            for (const std::size_t sample : stencil) {
                reliable = reliable && source.confidence[sample] != 0u &&
                    std::isfinite(source.values[sample]);
                minimum = std::min(minimum, source.values[sample]);
                maximum = std::max(maximum, source.values[sample]);
            }
            if (!reliable) {
                ++localReport.fallbackCellCount;
                continue;
            }
            const double corrected = forward.values[index] +
                0.5 * (source.values[index] - backward.values[index]);
            if (!std::isfinite(corrected)) {
                ++localReport.nonFiniteCellCount;
                ++localReport.fallbackCellCount;
                continue;
            }
            const double limited = std::clamp(corrected, minimum, maximum);
            if (limited != corrected) ++localReport.limiterClampCount;
            result.values[index] = limited;
            result.confidence[index] = 1u;
            ++localReport.correctedCellCount;
        }
    }
    if (report != nullptr) *report = localReport;
    return result;
}

LevelSetReinitializationReport reinitializeLevelSet(
    LevelSetField& field,
    double narrowBandWidth) {
    if (!(narrowBandWidth > 0.0)) {
        throw std::invalid_argument("Level-set narrow band must be positive.");
    }
    std::vector<Point> crossings;
    for (std::size_t y = 0; y < field.height; ++y) {
        for (std::size_t x = 0; x + 1 < field.width; ++x) {
            const double a = field.value(x, y);
            const double b = field.value(x + 1, y);
            if (!std::isfinite(a) || !std::isfinite(b) || (a < 0.0) == (b < 0.0)) continue;
            const double theta = std::abs(a) / (std::abs(a) + std::abs(b));
            crossings.push_back({
                (static_cast<double>(x) + 0.5 + theta) * field.cellWidth,
                (static_cast<double>(y) + 0.5) * field.cellHeight});
        }
    }
    for (std::size_t y = 0; y + 1 < field.height; ++y) {
        for (std::size_t x = 0; x < field.width; ++x) {
            const double a = field.value(x, y);
            const double b = field.value(x, y + 1);
            if (!std::isfinite(a) || !std::isfinite(b) || (a < 0.0) == (b < 0.0)) continue;
            const double theta = std::abs(a) / (std::abs(a) + std::abs(b));
            crossings.push_back({
                (static_cast<double>(x) + 0.5) * field.cellWidth,
                (static_cast<double>(y) + 0.5 + theta) * field.cellHeight});
        }
    }

    LevelSetReinitializationReport report;
    report.crossingCount = crossings.size();
    const std::vector<double> original = field.values;
    for (std::size_t y = 0; y < field.height; ++y) {
        for (std::size_t x = 0; x < field.width; ++x) {
            const std::size_t index = indexOf(field.width, x, y);
            if (!std::isfinite(original[index]) || crossings.empty()) {
                field.confidence[index] = 0u;
                ++report.invalidCellCount;
                continue;
            }
            const double px = (static_cast<double>(x) + 0.5) * field.cellWidth;
            const double py = (static_cast<double>(y) + 0.5) * field.cellHeight;
            double distance = std::numeric_limits<double>::infinity();
            for (const Point crossing : crossings) {
                distance = std::min(distance, std::hypot(px - crossing.x, py - crossing.y));
            }
            if (distance <= narrowBandWidth) {
                const double updated = std::copysign(distance, original[index]);
                report.maximumInterfaceDisplacement = std::max(
                    report.maximumInterfaceDisplacement, std::abs(updated - original[index]));
                field.values[index] = updated;
                field.confidence[index] = 1u;
                ++report.validCellCount;
            } else {
                field.values[index] = std::copysign(narrowBandWidth, original[index]);
                field.confidence[index] = 0u;
                ++report.invalidCellCount;
            }
        }
    }
    ++field.sourceVersion;
    return report;
}

LevelSetReinitializationReport redistanceLevelSetNarrowBand(
    LevelSetField& field,
    const LevelSetRedistanceSettings& settings) {
    if (!(settings.narrowBandWidth > 0.0) || settings.areaSamplesPerAxis == 0 ||
        field.width < 2 || field.height < 2 ||
        field.values.size() != field.width * field.height) {
        throw std::invalid_argument("Invalid narrow-band redistance settings or field.");
    }

    LevelSetReinitializationReport report;
    report.areaBefore = estimateLevelSetArea(field, settings.areaSamplesPerAxis).area;
    report.negativeComponentCountBefore = negativeComponentCount(field);
    const std::vector<double> original = field.values;
    const std::vector<unsigned char> originalConfidence = field.confidence;
    const std::vector<Point> crossingsBefore = edgeCrossings(field);
    const std::vector<Segment> segments = extractZeroSegments(field);
    report.crossingCount = crossingsBefore.size();
    report.segmentCount = segments.size();
    if (segments.empty()) {
        std::fill(field.confidence.begin(), field.confidence.end(),
            static_cast<unsigned char>(0));
        report.invalidCellCount = field.values.size();
        return report;
    }

    std::vector<std::vector<std::size_t>> buckets(field.width * field.height);
    for (std::size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex) {
        const Segment& segment = segments[segmentIndex];
        const double minimumX = std::min(segment.a.x, segment.b.x) - settings.narrowBandWidth;
        const double maximumX = std::max(segment.a.x, segment.b.x) + settings.narrowBandWidth;
        const double minimumY = std::min(segment.a.y, segment.b.y) - settings.narrowBandWidth;
        const double maximumY = std::max(segment.a.y, segment.b.y) + settings.narrowBandWidth;
        const std::size_t beginX = static_cast<std::size_t>(std::clamp(
            std::floor(minimumX / field.cellWidth), 0.0, static_cast<double>(field.width - 1)));
        const std::size_t endX = static_cast<std::size_t>(std::clamp(
            std::floor(maximumX / field.cellWidth), 0.0, static_cast<double>(field.width - 1)));
        const std::size_t beginY = static_cast<std::size_t>(std::clamp(
            std::floor(minimumY / field.cellHeight), 0.0, static_cast<double>(field.height - 1)));
        const std::size_t endY = static_cast<std::size_t>(std::clamp(
            std::floor(maximumY / field.cellHeight), 0.0, static_cast<double>(field.height - 1)));
        for (std::size_t y = beginY; y <= endY; ++y) {
            for (std::size_t x = beginX; x <= endX; ++x) {
                buckets[indexOf(field.width, x, y)].push_back(segmentIndex);
            }
        }
    }

    const auto isReliableCrossing = [&](std::size_t a, std::size_t b) {
        return originalConfidence[a] != 0u && originalConfidence[b] != 0u &&
            std::isfinite(original[a]) && std::isfinite(original[b]) &&
            ((original[a] <= 0.0 && original[b] > 0.0) ||
             (original[a] > 0.0 && original[b] <= 0.0));
    };
    const auto touchesReliableCrossing = [&](std::size_t x, std::size_t y) {
        const std::size_t index = indexOf(field.width, x, y);
        return (x > 0 && isReliableCrossing(index, index - 1)) ||
            (x + 1 < field.width && isReliableCrossing(index, index + 1)) ||
            (y > 0 && isReliableCrossing(index, index - field.width)) ||
            (y + 1 < field.height && isReliableCrossing(index, index + field.width));
    };

    for (std::size_t y = 0; y < field.height; ++y) {
        for (std::size_t x = 0; x < field.width; ++x) {
            const std::size_t index = indexOf(field.width, x, y);
            if (!std::isfinite(original[index])) {
                field.confidence[index] = 0u;
                ++report.invalidCellCount;
                continue;
            }
            if (settings.preserveInterfaceCrossings && touchesReliableCrossing(x, y)) {
                field.values[index] = original[index];
                field.confidence[index] = 1u;
                ++report.processedCellCount;
                ++report.validCellCount;
                continue;
            }
            const auto& candidates = buckets[index];
            if (candidates.empty()) {
                field.values[index] = std::copysign(settings.narrowBandWidth, original[index]);
                field.confidence[index] = 0u;
                ++report.invalidCellCount;
                continue;
            }
            const double px = (x + 0.5) * field.cellWidth;
            const double py = (y + 0.5) * field.cellHeight;
            double distance = std::numeric_limits<double>::infinity();
            for (const std::size_t segmentIndex : candidates) {
                distance = std::min(distance,
                    pointSegmentDistance(px, py, segments[segmentIndex]));
            }
            ++report.processedCellCount;
            if (distance <= settings.narrowBandWidth) {
                field.values[index] = std::copysign(distance, original[index]);
                field.confidence[index] = 1u;
                ++report.validCellCount;
            } else {
                field.values[index] = std::copysign(settings.narrowBandWidth, original[index]);
                field.confidence[index] = 0u;
                ++report.invalidCellCount;
            }
        }
    }

    const std::vector<Point> crossingsAfter = edgeCrossings(field);
    const std::size_t crossingCount = std::min(crossingsBefore.size(), crossingsAfter.size());
    double squaredDisplacement = 0.0;
    for (std::size_t i = 0; i < crossingCount; ++i) {
        const double displacement = std::hypot(
            crossingsAfter[i].x - crossingsBefore[i].x,
            crossingsAfter[i].y - crossingsBefore[i].y);
        squaredDisplacement += displacement * displacement;
        report.maximumInterfaceDisplacement = std::max(
            report.maximumInterfaceDisplacement, displacement);
    }
    report.rmsInterfaceDisplacement = std::sqrt(
        squaredDisplacement / std::max<std::size_t>(crossingCount, 1));

    double gradientErrorSum = 0.0;
    std::size_t gradientCount = 0;
    for (std::size_t y = 1; y + 1 < field.height; ++y) {
        for (std::size_t x = 1; x + 1 < field.width; ++x) {
            const std::size_t index = indexOf(field.width, x, y);
            if (field.confidence[index] == 0u) continue;
            const double gx = (field.value(x + 1, y) - field.value(x - 1, y)) /
                (2.0 * field.cellWidth);
            const double gy = (field.value(x, y + 1) - field.value(x, y - 1)) /
                (2.0 * field.cellHeight);
            const double error = std::abs(std::hypot(gx, gy) - 1.0);
            gradientErrorSum += error;
            report.maximumGradientError = std::max(report.maximumGradientError, error);
            ++gradientCount;
        }
    }
    report.meanGradientError = gradientErrorSum / std::max<std::size_t>(gradientCount, 1);
    report.areaAfter = estimateLevelSetArea(field, settings.areaSamplesPerAxis).area;
    report.negativeComponentCountAfter = negativeComponentCount(field);
    ++field.sourceVersion;
    return report;
}

LevelSetAreaEstimate estimateLevelSetArea(
    const LevelSetField& field,
    std::size_t samplesPerAxis) {
    if (samplesPerAxis == 0) {
        throw std::invalid_argument("Level-set area sampling requires at least one sample.");
    }
    LevelSetAreaEstimate result;
    for (std::size_t y = 0; y < field.height; ++y) {
        for (std::size_t x = 0; x < field.width; ++x) {
            std::size_t inside = 0;
            for (std::size_t sy = 0; sy < samplesPerAxis; ++sy) {
                for (std::size_t sx = 0; sx < samplesPerAxis; ++sx) {
                    const double px = (static_cast<double>(x) +
                        (static_cast<double>(sx) + 0.5) / samplesPerAxis) * field.cellWidth;
                    const double py = (static_cast<double>(y) +
                        (static_cast<double>(sy) + 0.5) / samplesPerAxis) * field.cellHeight;
                    const double value = field.sample(px, py);
                    if (std::isfinite(value)) {
                        ++result.validSamples;
                        if (value <= 0.0) ++inside;
                    } else {
                        ++result.invalidSamples;
                    }
                }
            }
            result.area += static_cast<double>(inside) /
                static_cast<double>(samplesPerAxis * samplesPerAxis) *
                field.cellWidth * field.cellHeight;
        }
    }
    return result;
}

} // namespace fluidlab::simulation
