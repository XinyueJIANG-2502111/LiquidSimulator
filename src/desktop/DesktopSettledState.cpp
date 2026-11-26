#include "DesktopSettledState.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace fluidlab::desktop {
namespace {

constexpr std::array<char, 8> kMagic{'L', 'S', 'S', 'T', 'A', 'T', 'E', '2'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kMaximumParticleCount = 16u * 1024u * 1024u;

template <typename T>
void appendValue(std::vector<std::byte>& bytes, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* begin = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(T));
}

template <typename T>
bool readValue(const std::vector<std::byte>& bytes, std::size_t& offset, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T)) return false;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

std::uint32_t crc32(const std::byte* data, std::size_t size) noexcept {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= std::to_integer<std::uint8_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1u) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

bool nearlyEqual(float a, float b) noexcept {
    return std::abs(a - b) <= 1.0e-7f * std::max({1.0f, std::abs(a), std::abs(b)});
}

void validateState(const DesktopSettledState& state) {
    const auto& metadata = state.metadata;
    if (metadata.formatVersion != kFormatVersion || metadata.width < 3 ||
        metadata.height < 3 || !(metadata.cellWidth > 0.0f) ||
        !(metadata.cellHeight > 0.0f) || !std::isfinite(metadata.preEvolutionSeconds) ||
        !std::isfinite(metadata.velocityRms) || !std::isfinite(metadata.maximumVelocity) ||
        !std::isfinite(metadata.centroidDriftPerSecond)) {
        throw std::invalid_argument("Settled state metadata is invalid.");
    }
    const std::size_t width = metadata.width;
    const std::size_t height = metadata.height;
    if (state.horizontalVelocity.size() != (width + 1) * height ||
        state.verticalVelocity.size() != width * (height + 1) ||
        state.pressure.size() != width * height || state.cellTypes.size() != width * height ||
        state.particles.size() > kMaximumParticleCount) {
        throw std::invalid_argument("Settled state array lengths are invalid.");
    }
    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(),
            [](float value) { return std::isfinite(value); });
    };
    if (!finite(state.horizontalVelocity) || !finite(state.verticalVelocity) ||
        !finite(state.pressure)) {
        throw std::invalid_argument("Settled state contains a non-finite field value.");
    }
    for (const auto& particle : state.particles) {
        if (!particle.valid || !std::isfinite(particle.position.x) ||
            !std::isfinite(particle.position.y)) {
            throw std::invalid_argument("Settled state contains an invalid marker particle.");
        }
    }
}

} // namespace

std::uint64_t desktopSettledConfigurationId(
    std::size_t width, std::size_t height, float dx, float dy) noexcept {
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    mix(width); mix(height); mix(std::bit_cast<std::uint32_t>(dx));
    mix(std::bit_cast<std::uint32_t>(dy));
    mix(8u); // Marker particles per initial fluid cell.
    mix(0x44414D425245414Bull); // Dam-break source scenario identifier.
    return hash;
}

void saveDesktopSettledState(
    const std::filesystem::path& path, const DesktopSettledState& state) {
    validateState(state);
    std::vector<std::byte> payload;
    payload.reserve(state.horizontalVelocity.size() * sizeof(float) +
        state.verticalVelocity.size() * sizeof(float) + state.pressure.size() * sizeof(float) +
        state.cellTypes.size() + state.particles.size() * (2 * sizeof(float) + 1));
    appendValue(payload, state.metadata.formatVersion);
    appendValue(payload, state.metadata.width);
    appendValue(payload, state.metadata.height);
    appendValue(payload, state.metadata.cellWidth);
    appendValue(payload, state.metadata.cellHeight);
    appendValue(payload, state.metadata.preEvolutionSeconds);
    appendValue(payload, state.metadata.velocityRms);
    appendValue(payload, state.metadata.maximumVelocity);
    appendValue(payload, state.metadata.centroidDriftPerSecond);
    appendValue(payload, state.metadata.stableWindowCount);
    appendValue(payload, state.metadata.configurationId);
    const std::uint64_t uCount = state.horizontalVelocity.size();
    const std::uint64_t vCount = state.verticalVelocity.size();
    const std::uint64_t pCount = state.pressure.size();
    const std::uint64_t typeCount = state.cellTypes.size();
    const std::uint64_t particleCount = state.particles.size();
    appendValue(payload, uCount); appendValue(payload, vCount); appendValue(payload, pCount);
    appendValue(payload, typeCount); appendValue(payload, particleCount);
    for (float value : state.horizontalVelocity) appendValue(payload, value);
    for (float value : state.verticalVelocity) appendValue(payload, value);
    for (float value : state.pressure) appendValue(payload, value);
    for (auto type : state.cellTypes) appendValue(payload, static_cast<std::uint8_t>(type));
    for (const auto& particle : state.particles) {
        appendValue(payload, particle.position.x);
        appendValue(payload, particle.position.y);
        appendValue(payload, static_cast<std::uint8_t>(particle.valid ? 1u : 0u));
    }
    const std::uint32_t checksum = crc32(payload.data(), payload.size());
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Failed to create settled state file.");
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    const std::uint64_t payloadSize = payload.size();
    output.write(reinterpret_cast<const char*>(&payloadSize), sizeof(payloadSize));
    output.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    output.write(reinterpret_cast<const char*>(payload.data()),
        static_cast<std::streamsize>(payload.size()));
    if (!output) throw std::runtime_error("Failed to write settled state file.");
}

DesktopSettledLoadResult loadDesktopSettledState(
    const std::filesystem::path& path, std::size_t expectedWidth,
    std::size_t expectedHeight, float expectedDx, float expectedDy) noexcept {
    DesktopSettledLoadResult result;
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Settled state file was not found.");
        std::array<char, 8> magic{};
        std::uint64_t payloadSize = 0;
        std::uint32_t expectedChecksum = 0;
        input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        input.read(reinterpret_cast<char*>(&payloadSize), sizeof(payloadSize));
        input.read(reinterpret_cast<char*>(&expectedChecksum), sizeof(expectedChecksum));
        if (!input || magic != kMagic || payloadSize > 256ull * 1024ull * 1024ull) {
            throw std::runtime_error("Settled state header is invalid.");
        }
        std::vector<std::byte> payload(static_cast<std::size_t>(payloadSize));
        input.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
        if (!input || input.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("Settled state payload length is invalid.");
        }
        if (crc32(payload.data(), payload.size()) != expectedChecksum) {
            throw std::runtime_error("Settled state checksum does not match.");
        }
        std::size_t offset = 0;
        DesktopSettledState state;
        std::uint64_t uCount = 0, vCount = 0, pCount = 0, typeCount = 0, particleCount = 0;
        if (!readValue(payload, offset, state.metadata.formatVersion) ||
            !readValue(payload, offset, state.metadata.width) ||
            !readValue(payload, offset, state.metadata.height) ||
            !readValue(payload, offset, state.metadata.cellWidth) ||
            !readValue(payload, offset, state.metadata.cellHeight) ||
            !readValue(payload, offset, state.metadata.preEvolutionSeconds) ||
            !readValue(payload, offset, state.metadata.velocityRms) ||
            !readValue(payload, offset, state.metadata.maximumVelocity) ||
            !readValue(payload, offset, state.metadata.centroidDriftPerSecond) ||
            !readValue(payload, offset, state.metadata.stableWindowCount) ||
            !readValue(payload, offset, state.metadata.configurationId) ||
            !readValue(payload, offset, uCount) || !readValue(payload, offset, vCount) ||
            !readValue(payload, offset, pCount) || !readValue(payload, offset, typeCount) ||
            !readValue(payload, offset, particleCount)) {
            throw std::runtime_error("Settled state metadata is truncated.");
        }
        if (uCount > 100000000 || vCount > 100000000 || pCount > 100000000 ||
            typeCount > 100000000 || particleCount > kMaximumParticleCount) {
            throw std::runtime_error("Settled state count is outside the supported range.");
        }
        state.horizontalVelocity.resize(static_cast<std::size_t>(uCount));
        state.verticalVelocity.resize(static_cast<std::size_t>(vCount));
        state.pressure.resize(static_cast<std::size_t>(pCount));
        state.cellTypes.resize(static_cast<std::size_t>(typeCount));
        state.particles.resize(static_cast<std::size_t>(particleCount));
        for (float& value : state.horizontalVelocity) if (!readValue(payload, offset, value)) throw std::runtime_error("Settled U field is truncated.");
        for (float& value : state.verticalVelocity) if (!readValue(payload, offset, value)) throw std::runtime_error("Settled V field is truncated.");
        for (float& value : state.pressure) if (!readValue(payload, offset, value)) throw std::runtime_error("Settled pressure field is truncated.");
        for (auto& type : state.cellTypes) {
            std::uint8_t encoded = 0;
            if (!readValue(payload, offset, encoded) || encoded > static_cast<std::uint8_t>(fluidlab::simulation::CellType::solid)) throw std::runtime_error("Settled classification is invalid.");
            type = static_cast<fluidlab::simulation::CellType>(encoded);
        }
        for (auto& particle : state.particles) {
            std::uint8_t valid = 0;
            if (!readValue(payload, offset, particle.position.x) ||
                !readValue(payload, offset, particle.position.y) ||
                !readValue(payload, offset, valid) || valid != 1u) {
                throw std::runtime_error("Settled marker data is invalid.");
            }
            particle.valid = true;
        }
        if (offset != payload.size()) throw std::runtime_error("Settled state has trailing payload data.");
        validateState(state);
        if (state.metadata.width != expectedWidth || state.metadata.height != expectedHeight ||
            !nearlyEqual(state.metadata.cellWidth, expectedDx) ||
            !nearlyEqual(state.metadata.cellHeight, expectedDy) ||
            state.metadata.configurationId != desktopSettledConfigurationId(
                expectedWidth, expectedHeight, expectedDx, expectedDy)) {
            throw std::runtime_error("Settled state configuration does not match the Desktop scene.");
        }
        result.valid = true;
        result.message = "Settled state loaded.";
        result.state = std::move(state);
    } catch (const std::exception& error) {
        result.message = error.what();
    }
    return result;
}

void applyDesktopSettledState(
    const DesktopSettledState& state, fluidlab::simulation::MacGrid& grid,
    fluidlab::simulation::MarkerParticleSystem& particles) {
    validateState(state);
    if (state.metadata.width != grid.width() || state.metadata.height != grid.height() ||
        !nearlyEqual(state.metadata.cellWidth, grid.cellWidth()) ||
        !nearlyEqual(state.metadata.cellHeight, grid.cellHeight())) {
        throw std::invalid_argument("Settled state cannot be applied to this grid layout.");
    }
    grid.loadState(state.horizontalVelocity, state.verticalVelocity,
        state.pressure, state.cellTypes);
    particles.loadState(grid, state.particles);
}

} // namespace fluidlab::desktop
