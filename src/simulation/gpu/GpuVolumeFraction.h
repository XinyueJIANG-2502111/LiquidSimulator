#pragma once

#include "../../graphics/ShaderProgram.h"
#include "../mac/MacGrid.h"
#include "../mac/VolumeFractionField.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

#include <glad/gl.h>

namespace fluidlab::simulation {

struct GpuVofSnapshot final {
    std::vector<float> fractions;
    std::vector<float> horizontalFlux;
    std::vector<float> verticalFlux;
    std::vector<float> rawHorizontalFlux;
    std::vector<float> rawVerticalFlux;
    std::vector<float> normals;
    std::vector<float> alpha;
    std::vector<unsigned int> plicValid;
    std::vector<unsigned int> compressionCoefficient;
};

enum class GpuVofBoundedMode {
    none,
    localFixed
};

enum class GpuVofUpdatePrecision {
    fp32,
    fp64Local
};

enum class GpuVofRepairMode {
    none,
    localPairwise
};

struct GpuVofCandidateSettings final {
    GpuVofBoundedMode boundedMode = GpuVofBoundedMode::none;
    GpuVofUpdatePrecision updatePrecision = GpuVofUpdatePrecision::fp32;
    GpuVofRepairMode repairMode = GpuVofRepairMode::none;
    std::size_t limiterRounds = 4;
    std::size_t repairRounds = 4;
    float numericalBoundTolerance = 2.0e-6f;
    double conservationTolerance = 2.0e-6;
    bool allowNumericalBoundSlack = false;
};

struct GpuVofStepReport final {
    bool valid = false;
    bool accepted = false;
    double areaBefore = 0.0;
    double areaAfter = 0.0;
    double compressionContribution = 0.0;
    double conservationResidual = 0.0;
    float minimumFraction = 0.0f;
    float maximumFraction = 0.0f;
    std::size_t outOfBoundsCount = 0;
    std::size_t nonFiniteCount = 0;
    std::size_t plicFallbackCount = 0;
    std::size_t invalidVelocityCount = 0;
    std::size_t uniqueLimitedFaceCount = 0;
    std::size_t repeatedLimitedFaceCount = 0;
    std::size_t nearlyBlockedFaceCount = 0;
    std::size_t repairEventCount = 0;
    std::size_t limiterRounds = 0;
    std::size_t repairRounds = 0;
    std::size_t dispatchCount = 0;
    std::size_t memoryBytes = 0;
    double repairAbsoluteVolume = 0.0;
    double cpuSubmitMilliseconds = 0.0;
    double cpuWaitMilliseconds = 0.0;
    std::array<double, 11> gpuPhaseMilliseconds{};
};

class GpuVolumeFraction final {
public:
    explicit GpuVolumeFraction(const std::filesystem::path& shaderDirectory);
    ~GpuVolumeFraction();

    GpuVolumeFraction(const GpuVolumeFraction&) = delete;
    GpuVolumeFraction& operator=(const GpuVolumeFraction&) = delete;

    void upload(const VolumeFractionField& field);
    void reset(const VolumeFractionField& field);
    [[nodiscard]] GpuVofStepReport computeCandidate(
        const MacGrid& velocity,
        double timeStep,
        bool xDirectionFirst = true,
        const GpuVofCandidateSettings& settings = {});
    void acceptCandidate();
    void rejectCandidate() noexcept;

    void computeFluxDiagnostic(const MacGrid& velocity, double timeStep, bool xDirection);
    [[nodiscard]] GpuVofSnapshot readCommitted() const;
    [[nodiscard]] GpuVofSnapshot readCandidate() const;
    [[nodiscard]] GpuVofSnapshot readDiagnosticState() const;

    [[nodiscard]] std::size_t width() const noexcept;
    [[nodiscard]] std::size_t height() const noexcept;
    [[nodiscard]] std::size_t version() const noexcept;
    [[nodiscard]] double physicalTime() const noexcept;
    [[nodiscard]] bool hasCandidate() const noexcept;
    [[nodiscard]] std::size_t memoryBytes() const noexcept;

private:
    void allocate(std::size_t width, std::size_t height);
    void release() noexcept;
    void uploadVelocity(const MacGrid& velocity);
    void dispatchPlic(GLuint fractionBuffer, bool writeCompression, std::size_t phase);
    void dispatchFlux(GLuint fractionBuffer, bool xDirection, double timeStep, std::size_t phase);
    void dispatchUpdate(
        GLuint sourceBuffer,
        GLuint targetBuffer,
        bool xDirection,
        double timeStep,
        GpuVofUpdatePrecision precision,
        std::size_t phase);
    void dispatchReduction(GLuint fractionBuffer, std::size_t phase);
    void dispatchLimiter(
        GLuint sourceBuffer,
        bool xDirection,
        double timeStep,
        std::size_t rounds,
        std::size_t phase);
    void dispatchRepair(GLuint fractionBuffer, std::size_t rounds, std::size_t phase);
    void beginTimedDispatch(std::size_t phase);
    void endTimedDispatch(std::size_t phase);
    [[nodiscard]] GpuVofSnapshot readSnapshot(GLuint fractionBuffer) const;

    graphics::ShaderProgram m_plicProgram;
    graphics::ShaderProgram m_fluxProgram;
    graphics::ShaderProgram m_updateProgram;
    graphics::ShaderProgram m_reduceProgram;
    graphics::ShaderProgram m_limitCellProgram;
    graphics::ShaderProgram m_limitFaceProgram;
    graphics::ShaderProgram m_repairPairProgram;
    std::size_t m_width = 0;
    std::size_t m_height = 0;
    float m_cellWidth = 0.0f;
    float m_cellHeight = 0.0f;
    std::size_t m_version = 0;
    double m_physicalTime = 0.0;
    double m_committedArea = 0.0;
    double m_candidateArea = 0.0;
    double m_candidateTimeStep = 0.0;
    bool m_hasCandidate = false;
    GLuint m_committedFraction = 0;
    GLuint m_workFractionA = 0;
    GLuint m_workFractionB = 0;
    GLuint m_validMask = 0;
    GLuint m_compressionCoefficient = 0;
    GLuint m_normal = 0;
    GLuint m_alpha = 0;
    GLuint m_plicValid = 0;
    GLuint m_horizontalFlux = 0;
    GLuint m_verticalFlux = 0;
    GLuint m_rawHorizontalFlux = 0;
    GLuint m_rawVerticalFlux = 0;
    GLuint m_outgoingScale = 0;
    GLuint m_incomingScale = 0;
    GLuint m_horizontalLimitedMask = 0;
    GLuint m_verticalLimitedMask = 0;
    GLuint m_u = 0;
    GLuint m_v = 0;
    GLuint m_compressionVolume = 0;
    GLuint m_repairVolume = 0;
    GLuint m_status = 0;
    GLuint m_diagnosticFloat = 0;
    GLuint m_diagnosticUint = 0;
    std::array<GLuint, 11> m_queries{};
    std::array<double, 11> m_lastGpuMilliseconds{};
};

} // namespace fluidlab::simulation
