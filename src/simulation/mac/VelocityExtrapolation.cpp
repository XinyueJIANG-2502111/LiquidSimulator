#include "VelocityExtrapolation.h"

#include <algorithm>

namespace fluidlab::simulation {

void VelocityExtrapolator::extrapolate(MacGrid& grid, std::size_t layers) {
    const std::size_t uWidth = grid.width() + 1;
    const std::size_t uHeight = grid.height();
    const std::size_t vWidth = grid.width();
    const std::size_t vHeight = grid.height() + 1;
    m_uValid.assign(uWidth * uHeight, 0);
    m_vValid.assign(vWidth * vHeight, 0);

    const auto uBlocked = [&](std::size_t x, std::size_t y) {
        return x == 0 || x == grid.width() ||
            grid.cellType(x - 1, y) == CellType::solid || grid.cellType(x, y) == CellType::solid;
    };
    const auto vBlocked = [&](std::size_t x, std::size_t y) {
        return y == 0 || y == grid.height() ||
            grid.cellType(x, y - 1) == CellType::solid || grid.cellType(x, y) == CellType::solid;
    };
    for (std::size_t y = 0; y < uHeight; ++y) for (std::size_t x = 1; x < grid.width(); ++x) {
        if (!uBlocked(x, y) &&
            (grid.cellType(x - 1, y) == CellType::fluid || grid.cellType(x, y) == CellType::fluid)) {
            m_uValid[y * uWidth + x] = 1;
        }
    }
    for (std::size_t y = 1; y < grid.height(); ++y) for (std::size_t x = 0; x < vWidth; ++x) {
        if (!vBlocked(x, y) &&
            (grid.cellType(x, y - 1) == CellType::fluid || grid.cellType(x, y) == CellType::fluid)) {
            m_vValid[y * vWidth + x] = 1;
        }
    }

    const auto extend = [&](std::size_t width, std::size_t height, std::vector<unsigned char>& valid,
                            auto blocked, auto read, auto write) {
        for (std::size_t layer = 0; layer < layers; ++layer) {
            m_nextValid = valid;
            m_values.assign(width * height, 0.0f);
            for (std::size_t y = 0; y < height; ++y) for (std::size_t x = 0; x < width; ++x) {
                const std::size_t index = y * width + x;
                if (valid[index] || blocked(x, y)) continue;
                float sum = 0.0f;
                std::size_t count = 0;
                const auto add = [&](std::size_t nx, std::size_t ny) {
                    if (valid[ny * width + nx]) { sum += read(nx, ny); ++count; }
                };
                if (x > 0) add(x - 1, y);
                if (x + 1 < width) add(x + 1, y);
                if (y > 0) add(x, y - 1);
                if (y + 1 < height) add(x, y + 1);
                if (count != 0) {
                    m_values[index] = sum / static_cast<float>(count);
                    m_nextValid[index] = 1;
                }
            }
            for (std::size_t y = 0; y < height; ++y) for (std::size_t x = 0; x < width; ++x) {
                const std::size_t index = y * width + x;
                if (!valid[index] && m_nextValid[index]) write(x, y, m_values[index]);
            }
            valid.swap(m_nextValid);
        }
    };

    extend(uWidth, uHeight, m_uValid, uBlocked,
        [&](std::size_t x, std::size_t y) { return grid.horizontalVelocity(x, y); },
        [&](std::size_t x, std::size_t y, float value) { grid.horizontalVelocity(x, y) = value; });
    extend(vWidth, vHeight, m_vValid, vBlocked,
        [&](std::size_t x, std::size_t y) { return grid.verticalVelocity(x, y); },
        [&](std::size_t x, std::size_t y, float value) { grid.verticalVelocity(x, y) = value; });
    grid.enforceSolidBoundaries();
}

} // namespace fluidlab::simulation
