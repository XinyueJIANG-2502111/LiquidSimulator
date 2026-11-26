#pragma once

#include "MacGrid.h"

#include <cstddef>
#include <vector>

namespace fluidlab::simulation {

class VelocityExtrapolator final {
public:
    void extrapolate(MacGrid& grid, std::size_t layers = 2);

private:
    std::vector<unsigned char> m_uValid;
    std::vector<unsigned char> m_vValid;
    std::vector<unsigned char> m_nextValid;
    std::vector<float> m_values;
};

} // namespace fluidlab::simulation
