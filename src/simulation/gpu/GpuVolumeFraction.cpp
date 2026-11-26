#include "GpuVolumeFraction.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace fluidlab::simulation {
namespace {

constexpr GLuint kLocalSize = 16;

GLuint groups(std::size_t value) {
    return static_cast<GLuint>((value + kLocalSize - 1) / kLocalSize);
}

void bindCommonUniforms(
    const graphics::ShaderProgram& program,
    std::size_t width,
    std::size_t height,
    float cellWidth,
    float cellHeight) {
    glUniform2ui(program.uniformLocation("gridSize"),
        static_cast<GLuint>(width), static_cast<GLuint>(height));
    glUniform2f(program.uniformLocation("cellSize"), cellWidth, cellHeight);
}

} // namespace

GpuVolumeFraction::GpuVolumeFraction(const std::filesystem::path& shaderDirectory)
    : m_plicProgram(graphics::ShaderProgram::fromComputeFile(shaderDirectory / "vof_plic.comp")),
      m_fluxProgram(graphics::ShaderProgram::fromComputeFile(shaderDirectory / "vof_flux.comp")),
      m_updateProgram(graphics::ShaderProgram::fromComputeFile(shaderDirectory / "vof_update.comp")),
      m_reduceProgram(graphics::ShaderProgram::fromComputeFile(shaderDirectory / "vof_reduce.comp")),
      m_limitCellProgram(graphics::ShaderProgram::fromComputeFile(shaderDirectory / "vof_limit_cell.comp")),
      m_limitFaceProgram(graphics::ShaderProgram::fromComputeFile(shaderDirectory / "vof_limit_face.comp")),
      m_repairPairProgram(graphics::ShaderProgram::fromComputeFile(shaderDirectory / "vof_repair_pair.comp")) {
    glCreateQueries(GL_TIME_ELAPSED, static_cast<GLsizei>(m_queries.size()), m_queries.data());
}

GpuVolumeFraction::~GpuVolumeFraction() {
    release();
    glDeleteQueries(static_cast<GLsizei>(m_queries.size()), m_queries.data());
}

void GpuVolumeFraction::release() noexcept {
    const GLuint buffers[] = {
        m_committedFraction, m_workFractionA, m_workFractionB, m_validMask,
        m_compressionCoefficient, m_normal, m_alpha, m_plicValid,
        m_horizontalFlux, m_verticalFlux, m_u, m_v, m_compressionVolume,
        m_status, m_diagnosticFloat, m_diagnosticUint, m_rawHorizontalFlux,
        m_rawVerticalFlux, m_outgoingScale, m_incomingScale,
        m_horizontalLimitedMask, m_verticalLimitedMask, m_repairVolume,
    };
    glDeleteBuffers(static_cast<GLsizei>(std::size(buffers)), buffers);
    m_committedFraction = m_workFractionA = m_workFractionB = 0;
    m_validMask = m_compressionCoefficient = m_normal = m_alpha = m_plicValid = 0;
    m_horizontalFlux = m_verticalFlux = m_u = m_v = 0;
    m_rawHorizontalFlux = m_rawVerticalFlux = 0;
    m_outgoingScale = m_incomingScale = 0;
    m_horizontalLimitedMask = m_verticalLimitedMask = 0;
    m_compressionVolume = m_repairVolume = m_status = m_diagnosticFloat = m_diagnosticUint = 0;
    m_width = m_height = 0;
}

void GpuVolumeFraction::allocate(std::size_t width, std::size_t height) {
    release();
    m_width = width;
    m_height = height;
    GLuint buffers[23]{};
    glCreateBuffers(23, buffers);
    m_committedFraction = buffers[0];
    m_workFractionA = buffers[1];
    m_workFractionB = buffers[2];
    m_validMask = buffers[3];
    m_compressionCoefficient = buffers[4];
    m_normal = buffers[5];
    m_alpha = buffers[6];
    m_plicValid = buffers[7];
    m_horizontalFlux = buffers[8];
    m_verticalFlux = buffers[9];
    m_u = buffers[10];
    m_v = buffers[11];
    m_compressionVolume = buffers[12];
    m_status = buffers[13];
    m_diagnosticFloat = buffers[14];
    m_diagnosticUint = buffers[15];
    m_rawHorizontalFlux = buffers[16];
    m_rawVerticalFlux = buffers[17];
    m_outgoingScale = buffers[18];
    m_incomingScale = buffers[19];
    m_horizontalLimitedMask = buffers[20];
    m_verticalLimitedMask = buffers[21];
    m_repairVolume = buffers[22];
    const GLsizeiptr cells = static_cast<GLsizeiptr>(width * height);
    const GLsizeiptr xFaces = static_cast<GLsizeiptr>((width + 1) * height);
    const GLsizeiptr yFaces = static_cast<GLsizeiptr>(width * (height + 1));
    glNamedBufferData(m_committedFraction, cells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_workFractionA, cells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_workFractionB, cells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_validMask, cells * sizeof(GLuint), nullptr, GL_STATIC_DRAW);
    glNamedBufferData(m_compressionCoefficient, cells * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_normal, cells * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_alpha, cells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_plicValid, cells * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_horizontalFlux, xFaces * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_verticalFlux, yFaces * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_u, xFaces * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_v, yFaces * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_compressionVolume, cells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_status, 12 * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_diagnosticFloat, 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_diagnosticUint, 4 * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_rawHorizontalFlux, xFaces * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_rawVerticalFlux, yFaces * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_outgoingScale, cells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_incomingScale, cells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_horizontalLimitedMask, xFaces * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_verticalLimitedMask, yFaces * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_repairVolume, cells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
}

void GpuVolumeFraction::upload(const VolumeFractionField& field) {
    if (field.width() != m_width || field.height() != m_height) {
        allocate(field.width(), field.height());
    }
    m_cellWidth = static_cast<float>(field.cellWidth());
    m_cellHeight = static_cast<float>(field.cellHeight());
    std::vector<float> fractions(field.fractions().size());
    std::transform(field.fractions().begin(), field.fractions().end(), fractions.begin(),
        [](double value) { return static_cast<float>(value); });
    std::vector<GLuint> valid(field.validMask().begin(), field.validMask().end());
    glNamedBufferSubData(m_committedFraction, 0,
        static_cast<GLsizeiptr>(fractions.size() * sizeof(float)), fractions.data());
    glNamedBufferSubData(m_validMask, 0,
        static_cast<GLsizeiptr>(valid.size() * sizeof(GLuint)), valid.data());
    m_committedArea = field.totalArea();
    m_physicalTime = field.physicalTime();
    m_version = field.version();
    m_hasCandidate = false;
}

void GpuVolumeFraction::reset(const VolumeFractionField& field) { upload(field); }

void GpuVolumeFraction::uploadVelocity(const MacGrid& velocity) {
    if (velocity.width() != m_width || velocity.height() != m_height) {
        throw std::invalid_argument("GPU VOF velocity layout mismatch.");
    }
    glNamedBufferSubData(m_u, 0,
        static_cast<GLsizeiptr>(velocity.horizontalVelocities().size() * sizeof(float)),
        velocity.horizontalVelocities().data());
    glNamedBufferSubData(m_v, 0,
        static_cast<GLsizeiptr>(velocity.verticalVelocities().size() * sizeof(float)),
        velocity.verticalVelocities().data());
}

void GpuVolumeFraction::beginTimedDispatch(std::size_t phase) {
    glBeginQuery(GL_TIME_ELAPSED, m_queries[phase]);
}

void GpuVolumeFraction::endTimedDispatch(std::size_t phase) {
    glEndQuery(GL_TIME_ELAPSED);
    static_cast<void>(phase);
}

void GpuVolumeFraction::dispatchPlic(GLuint fractionBuffer, bool writeCompression, std::size_t phase) {
    m_plicProgram.use();
    bindCommonUniforms(m_plicProgram, m_width, m_height, m_cellWidth, m_cellHeight);
    glUniform1ui(m_plicProgram.uniformLocation("writeCompression"), writeCompression ? 1u : 0u);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fractionBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_validMask);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_normal);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_alpha);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_plicValid);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_compressionCoefficient);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_status);
    beginTimedDispatch(phase);
    glDispatchCompute(groups(m_width), groups(m_height), 1);
    endTimedDispatch(phase);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GpuVolumeFraction::dispatchFlux(
    GLuint fractionBuffer, bool xDirection, double timeStep, std::size_t phase) {
    m_fluxProgram.use();
    bindCommonUniforms(m_fluxProgram, m_width, m_height, m_cellWidth, m_cellHeight);
    glUniform1f(m_fluxProgram.uniformLocation("timeStep"), static_cast<float>(timeStep));
    glUniform1ui(m_fluxProgram.uniformLocation("xDirection"), xDirection ? 1u : 0u);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fractionBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_validMask);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_normal);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_alpha);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_plicValid);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_u);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_v);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7,
        xDirection ? m_horizontalFlux : m_verticalFlux);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_status);
    beginTimedDispatch(phase);
    glDispatchCompute(
        groups(xDirection ? m_width + 1 : m_width),
        groups(xDirection ? m_height : m_height + 1), 1);
    endTimedDispatch(phase);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GpuVolumeFraction::dispatchUpdate(
    GLuint sourceBuffer,
    GLuint targetBuffer,
    bool xDirection,
    double timeStep,
    GpuVofUpdatePrecision precision,
    std::size_t phase) {
    m_updateProgram.use();
    bindCommonUniforms(m_updateProgram, m_width, m_height, m_cellWidth, m_cellHeight);
    glUniform1f(m_updateProgram.uniformLocation("timeStep"), static_cast<float>(timeStep));
    glUniform1ui(m_updateProgram.uniformLocation("xDirection"), xDirection ? 1u : 0u);
    glUniform1ui(m_updateProgram.uniformLocation("highPrecision"),
        precision == GpuVofUpdatePrecision::fp64Local ? 1u : 0u);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sourceBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, targetBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_validMask);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_compressionCoefficient);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_horizontalFlux);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_verticalFlux);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_u);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_v);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_compressionVolume);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_status);
    beginTimedDispatch(phase);
    glDispatchCompute(groups(m_width), groups(m_height), 1);
    endTimedDispatch(phase);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GpuVolumeFraction::dispatchReduction(GLuint fractionBuffer, std::size_t phase) {
    m_reduceProgram.use();
    glUniform1ui(m_reduceProgram.uniformLocation("cellCount"),
        static_cast<GLuint>(m_width * m_height));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fractionBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_validMask);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_compressionVolume);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_plicValid);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_diagnosticFloat);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_diagnosticUint);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_repairVolume);
    beginTimedDispatch(phase);
    glDispatchCompute(1, 1, 1);
    endTimedDispatch(phase);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

void GpuVolumeFraction::dispatchRepair(
    GLuint fractionBuffer, std::size_t rounds, std::size_t phase) {
    beginTimedDispatch(phase);
    for (std::size_t round = 0; round < rounds; ++round) {
        for (const bool xDirection : {true, false}) {
            for (GLuint parity = 0; parity < 2; ++parity) {
                m_repairPairProgram.use();
                bindCommonUniforms(
                    m_repairPairProgram, m_width, m_height, m_cellWidth, m_cellHeight);
                glUniform1ui(m_repairPairProgram.uniformLocation("xDirection"),
                    xDirection ? 1u : 0u);
                glUniform1ui(m_repairPairProgram.uniformLocation("parity"), parity);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fractionBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_validMask);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_repairVolume);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_status);
                glDispatchCompute(
                    groups(xDirection ? (m_width + 1) / 2 : m_width),
                    groups(xDirection ? m_height : (m_height + 1) / 2), 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
        }
    }
    endTimedDispatch(phase);
}

void GpuVolumeFraction::dispatchLimiter(
    GLuint sourceBuffer,
    bool xDirection,
    double timeStep,
    std::size_t rounds,
    std::size_t phase) {
    const GLuint fluxBuffer = xDirection ? m_horizontalFlux : m_verticalFlux;
    const GLuint rawFluxBuffer = xDirection ? m_rawHorizontalFlux : m_rawVerticalFlux;
    const GLuint limitedMask = xDirection ? m_horizontalLimitedMask : m_verticalLimitedMask;
    const GLsizeiptr faceCount = static_cast<GLsizeiptr>(
        xDirection ? (m_width + 1) * m_height : m_width * (m_height + 1));
    glCopyNamedBufferSubData(
        fluxBuffer, rawFluxBuffer, 0, 0, faceCount * static_cast<GLsizeiptr>(sizeof(float)));
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    beginTimedDispatch(phase);
    for (std::size_t round = 0; round < rounds; ++round) {
        m_limitCellProgram.use();
        bindCommonUniforms(m_limitCellProgram, m_width, m_height, m_cellWidth, m_cellHeight);
        glUniform1f(m_limitCellProgram.uniformLocation("timeStep"), static_cast<float>(timeStep));
        glUniform1ui(m_limitCellProgram.uniformLocation("xDirection"), xDirection ? 1u : 0u);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sourceBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_validMask);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_compressionCoefficient);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fluxBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_u);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_v);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_outgoingScale);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_incomingScale);
        glDispatchCompute(groups(m_width), groups(m_height), 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        m_limitFaceProgram.use();
        glUniform2ui(m_limitFaceProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
        glUniform1ui(m_limitFaceProgram.uniformLocation("xDirection"), xDirection ? 1u : 0u);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_validMask);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_outgoingScale);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_incomingScale);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fluxBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, limitedMask);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_status);
        glDispatchCompute(
            groups(xDirection ? m_width + 1 : m_width),
            groups(xDirection ? m_height : m_height + 1), 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    endTimedDispatch(phase);
}

GpuVofStepReport GpuVolumeFraction::computeCandidate(
    const MacGrid& velocity,
    double timeStep,
    bool xDirectionFirst,
    const GpuVofCandidateSettings& settings) {
    if (!(timeStep > 0.0) || m_width == 0) {
        throw std::invalid_argument("GPU VOF candidate time step is invalid.");
    }
    uploadVelocity(velocity);
    const GLsizeiptr cellBytes = static_cast<GLsizeiptr>(m_width * m_height * sizeof(float));
    glCopyNamedBufferSubData(m_committedFraction, m_workFractionA, 0, 0, cellBytes);
    const GLuint zero = 0;
    const float zeroFloat = 0.0f;
    glClearNamedBufferData(m_status, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearNamedBufferData(m_compressionVolume, GL_R32F, GL_RED, GL_FLOAT, &zeroFloat);
    glClearNamedBufferData(m_repairVolume, GL_R32F, GL_RED, GL_FLOAT, &zeroFloat);
    glClearNamedBufferData(m_horizontalLimitedMask, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearNamedBufferData(m_verticalLimitedMask, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    const auto submitStart = std::chrono::steady_clock::now();
    dispatchPlic(m_workFractionA, true, 0);
    dispatchFlux(m_workFractionA, xDirectionFirst, timeStep, 1);
    if (settings.boundedMode == GpuVofBoundedMode::localFixed) {
        dispatchLimiter(m_workFractionA, xDirectionFirst, timeStep, settings.limiterRounds, 8);
    } else {
        const GLsizeiptr count = static_cast<GLsizeiptr>(
            xDirectionFirst ? (m_width + 1) * m_height : m_width * (m_height + 1));
        glCopyNamedBufferSubData(
            xDirectionFirst ? m_horizontalFlux : m_verticalFlux,
            xDirectionFirst ? m_rawHorizontalFlux : m_rawVerticalFlux,
            0, 0, count * static_cast<GLsizeiptr>(sizeof(float)));
    }
    dispatchUpdate(m_workFractionA, m_workFractionB, xDirectionFirst, timeStep,
        settings.updatePrecision, 2);
    dispatchPlic(m_workFractionB, false, 3);
    dispatchFlux(m_workFractionB, !xDirectionFirst, timeStep, 4);
    if (settings.boundedMode == GpuVofBoundedMode::localFixed) {
        dispatchLimiter(m_workFractionB, !xDirectionFirst, timeStep, settings.limiterRounds, 9);
    } else {
        const bool secondX = !xDirectionFirst;
        const GLsizeiptr count = static_cast<GLsizeiptr>(
            secondX ? (m_width + 1) * m_height : m_width * (m_height + 1));
        glCopyNamedBufferSubData(
            secondX ? m_horizontalFlux : m_verticalFlux,
            secondX ? m_rawHorizontalFlux : m_rawVerticalFlux,
            0, 0, count * static_cast<GLsizeiptr>(sizeof(float)));
    }
    dispatchUpdate(m_workFractionB, m_workFractionA, !xDirectionFirst, timeStep,
        settings.updatePrecision, 5);
    if (settings.repairMode == GpuVofRepairMode::localPairwise) {
        dispatchRepair(m_workFractionA, settings.repairRounds, 10);
    }
    dispatchPlic(m_workFractionA, false, 6);
    dispatchReduction(m_workFractionA, 7);
    GpuVofStepReport report;
    report.cpuSubmitMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - submitStart).count();
    report.dispatchCount = 8 + (settings.boundedMode == GpuVofBoundedMode::localFixed
        ? 4 * settings.limiterRounds : 0) +
        (settings.repairMode == GpuVofRepairMode::localPairwise ? 4 * settings.repairRounds : 0);
    report.memoryBytes = memoryBytes();

    const auto waitStart = std::chrono::steady_clock::now();
    std::array<float, 5> diagnosticFloat{};
    std::array<GLuint, 4> diagnosticUint{};
    std::array<GLuint, 12> status{};
    glGetNamedBufferSubData(m_diagnosticFloat, 0, sizeof(diagnosticFloat), diagnosticFloat.data());
    glGetNamedBufferSubData(m_diagnosticUint, 0, sizeof(diagnosticUint), diagnosticUint.data());
    glGetNamedBufferSubData(m_status, 0, sizeof(status), status.data());
    report.cpuWaitMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - waitStart).count();
    const std::size_t timedPhaseCount = settings.repairMode == GpuVofRepairMode::localPairwise
        ? 11 : (settings.boundedMode == GpuVofBoundedMode::localFixed ? 10 : 8);
    for (std::size_t phase = 0; phase < timedPhaseCount; ++phase) {
        GLuint64 nanoseconds = 0;
        glGetQueryObjectui64v(m_queries[phase], GL_QUERY_RESULT, &nanoseconds);
        report.gpuPhaseMilliseconds[phase] = static_cast<double>(nanoseconds) * 1.0e-6;
    }
    report.areaBefore = m_committedArea;
    report.areaAfter = static_cast<double>(diagnosticFloat[0]) * m_cellWidth * m_cellHeight;
    report.compressionContribution = diagnosticFloat[1];
    report.minimumFraction = diagnosticFloat[2];
    report.maximumFraction = diagnosticFloat[3];
    report.outOfBoundsCount = diagnosticUint[0] + status[2];
    report.nonFiniteCount = diagnosticUint[1] + status[1];
    report.plicFallbackCount = status[0];
    report.invalidVelocityCount = status[3];
    report.uniqueLimitedFaceCount = status[5];
    report.repeatedLimitedFaceCount = status[6];
    report.nearlyBlockedFaceCount = status[7];
    report.repairEventCount = status[8];
    report.repairAbsoluteVolume = diagnosticFloat[4];
    report.limiterRounds = settings.boundedMode == GpuVofBoundedMode::localFixed
        ? settings.limiterRounds : 0;
    report.repairRounds = settings.repairMode == GpuVofRepairMode::localPairwise
        ? settings.repairRounds : 0;
    report.conservationResidual = report.areaAfter - report.areaBefore -
        report.compressionContribution;
    const double tolerance = std::max(
        settings.conservationTolerance, report.areaBefore * 2.0e-5);
    const float boundTolerance = settings.allowNumericalBoundSlack
        ? settings.numericalBoundTolerance : 0.0f;
    const bool withinConfiguredBounds =
        report.minimumFraction >= -boundTolerance &&
        report.maximumFraction <= 1.0f + boundTolerance;
    report.valid = report.outOfBoundsCount == 0 && report.nonFiniteCount == 0 &&
        report.invalidVelocityCount == 0 && std::isfinite(report.conservationResidual) &&
        withinConfiguredBounds && std::abs(report.conservationResidual) <= tolerance;
    m_candidateTimeStep = timeStep;
    m_candidateArea = report.areaAfter;
    m_hasCandidate = report.valid;
    return report;
}

void GpuVolumeFraction::acceptCandidate() {
    if (!m_hasCandidate) {
        throw std::logic_error("No valid GPU VOF candidate is available.");
    }
    const GLsizeiptr bytes = static_cast<GLsizeiptr>(m_width * m_height * sizeof(float));
    glCopyNamedBufferSubData(m_workFractionA, m_committedFraction, 0, 0, bytes);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    m_committedArea = m_candidateArea;
    m_physicalTime += m_candidateTimeStep;
    ++m_version;
    m_hasCandidate = false;
}

void GpuVolumeFraction::rejectCandidate() noexcept { m_hasCandidate = false; }

void GpuVolumeFraction::computeFluxDiagnostic(
    const MacGrid& velocity, double timeStep, bool xDirection) {
    uploadVelocity(velocity);
    const GLuint zero = 0;
    glClearNamedBufferData(m_status, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    dispatchPlic(m_committedFraction, true, 0);
    dispatchFlux(m_committedFraction, xDirection, timeStep, 1);
}

GpuVofSnapshot GpuVolumeFraction::readSnapshot(GLuint fractionBuffer) const {
    GpuVofSnapshot result;
    result.fractions.resize(m_width * m_height);
    result.horizontalFlux.resize((m_width + 1) * m_height);
    result.verticalFlux.resize(m_width * (m_height + 1));
    result.rawHorizontalFlux.resize((m_width + 1) * m_height);
    result.rawVerticalFlux.resize(m_width * (m_height + 1));
    result.normals.resize(2 * m_width * m_height);
    result.alpha.resize(m_width * m_height);
    result.plicValid.resize(m_width * m_height);
    result.compressionCoefficient.resize(m_width * m_height);
    glGetNamedBufferSubData(fractionBuffer, 0,
        static_cast<GLsizeiptr>(result.fractions.size() * sizeof(float)), result.fractions.data());
    glGetNamedBufferSubData(m_horizontalFlux, 0,
        static_cast<GLsizeiptr>(result.horizontalFlux.size() * sizeof(float)), result.horizontalFlux.data());
    glGetNamedBufferSubData(m_verticalFlux, 0,
        static_cast<GLsizeiptr>(result.verticalFlux.size() * sizeof(float)), result.verticalFlux.data());
    glGetNamedBufferSubData(m_rawHorizontalFlux, 0,
        static_cast<GLsizeiptr>(result.rawHorizontalFlux.size() * sizeof(float)),
        result.rawHorizontalFlux.data());
    glGetNamedBufferSubData(m_rawVerticalFlux, 0,
        static_cast<GLsizeiptr>(result.rawVerticalFlux.size() * sizeof(float)),
        result.rawVerticalFlux.data());
    glGetNamedBufferSubData(m_normal, 0,
        static_cast<GLsizeiptr>(result.normals.size() * sizeof(float)), result.normals.data());
    glGetNamedBufferSubData(m_alpha, 0,
        static_cast<GLsizeiptr>(result.alpha.size() * sizeof(float)), result.alpha.data());
    glGetNamedBufferSubData(m_plicValid, 0,
        static_cast<GLsizeiptr>(result.plicValid.size() * sizeof(GLuint)), result.plicValid.data());
    glGetNamedBufferSubData(m_compressionCoefficient, 0,
        static_cast<GLsizeiptr>(result.compressionCoefficient.size() * sizeof(GLuint)),
        result.compressionCoefficient.data());
    return result;
}

GpuVofSnapshot GpuVolumeFraction::readCommitted() const { return readSnapshot(m_committedFraction); }
GpuVofSnapshot GpuVolumeFraction::readCandidate() const { return readSnapshot(m_workFractionA); }
GpuVofSnapshot GpuVolumeFraction::readDiagnosticState() const { return readSnapshot(m_committedFraction); }
std::size_t GpuVolumeFraction::width() const noexcept { return m_width; }
std::size_t GpuVolumeFraction::height() const noexcept { return m_height; }
std::size_t GpuVolumeFraction::version() const noexcept { return m_version; }
double GpuVolumeFraction::physicalTime() const noexcept { return m_physicalTime; }
bool GpuVolumeFraction::hasCandidate() const noexcept { return m_hasCandidate; }

std::size_t GpuVolumeFraction::memoryBytes() const noexcept {
    const std::size_t cells = m_width * m_height;
    const std::size_t xFaces = (m_width + 1) * m_height;
    const std::size_t yFaces = m_width * (m_height + 1);
    return 10 * cells * sizeof(float) +
        (2 * cells + cells) * sizeof(GLuint) +
        3 * (xFaces + yFaces) * sizeof(float) +
        (xFaces + yFaces) * sizeof(GLuint) + 21 * sizeof(GLuint);
}

} // namespace fluidlab::simulation
