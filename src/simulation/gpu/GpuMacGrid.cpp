#include "GpuMacGrid.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace fluidlab::simulation {
namespace {

using Clock = std::chrono::steady_clock;

template <typename Callback>
class ScopeExit final {
public:
    explicit ScopeExit(Callback callback) : m_callback(std::move(callback)) {}
    ~ScopeExit() { m_callback(); }
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
private:
    Callback m_callback;
};

[[nodiscard]] GLsizeiptr checkedBytes(std::size_t count, std::size_t elementSize) {
    if (count == 0 || count > static_cast<std::size_t>(std::numeric_limits<GLsizeiptr>::max()) / elementSize) {
        throw std::invalid_argument("GPU MAC buffer size is invalid or too large.");
    }
    return static_cast<GLsizeiptr>(count * elementSize);
}

void validatePressureSettings(const GpuPressureSettings& settings) {
    if (!std::isfinite(settings.density) || settings.density <= 0.0f ||
        !std::isfinite(settings.timeStep) || settings.timeStep <= 0.0f ||
        !std::isfinite(settings.residualTolerance) || settings.residualTolerance < 0.0f ||
        !std::isfinite(settings.relaxationFactor) || settings.relaxationFactor <= 0.0f ||
        settings.relaxationFactor > 1.0f ||
        settings.maximumIterations == 0 || settings.residualCheckInterval == 0 ||
        settings.gpuResidentBatchCount == 0 ||
        settings.multigridCoarseIterations == 0 ||
        !std::isfinite(settings.multigridRelaxationFactor) ||
        settings.multigridRelaxationFactor <= 0.0f ||
        settings.multigridRelaxationFactor > 1.0f) {
        throw std::invalid_argument("Invalid GPU pressure solver settings.");
    }
}

} // namespace

GpuMacGrid::GpuMacGrid(const std::filesystem::path& divergenceShaderPath)
    : m_divergenceProgram(graphics::ShaderProgram::fromComputeFile(divergenceShaderPath)),
      m_jacobiProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_pressure_jacobi.comp")),
      m_residualProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_pressure_residual.comp")),
      m_reduceResidualProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_reduce_residual.comp")),
      m_reducePartialProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_reduce_partial.comp")),
      m_reduceDotProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_reduce_dot.comp")),
      m_pcgOperatorProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_pcg_operator.comp")),
      m_pcgInitializeProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_pcg_initialize.comp")),
      m_pcgScalarProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_pcg_scalar.comp")),
      m_pcgUpdateProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_pcg_update.comp")),
      m_pcgDirectionProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_pcg_direction.comp")),
      m_mgSmoothProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_smooth.comp")),
      m_mgCoarseSmoothProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_coarse_smooth.comp")),
      m_mgResidualProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_residual.comp")),
      m_mgRestrictProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_restrict.comp")),
      m_mgProlongProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_prolong.comp")),
      m_mgSmoothFloatProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_smooth_fp32.comp")),
      m_mgCoarseSmoothFloatProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_coarse_smooth_fp32.comp")),
      m_mgResidualFloatProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_residual_fp32.comp")),
      m_mgRestrictFloatProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_restrict_fp32.comp")),
      m_mgProlongFloatProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_prolong_fp32.comp")),
      m_mgConvertProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_mg_convert.comp")),
      m_vectorCopyProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_vector_copy.comp")),
      m_projectionProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_pressure_project.comp")),
      m_advectionProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_velocity_advect.comp")),
      m_forceBoundaryProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_force_boundary.comp")),
      m_classificationResetProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_classification_reset.comp")),
      m_classificationMarkProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_classification_mark.comp")),
      m_pressureSanitizeProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_pressure_sanitize.comp")),
      m_particleAdvectionProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_particle_advect.comp")),
      m_extrapolationInitializeProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_extrapolate_init.comp")),
      m_extrapolationLayerProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_extrapolate_layer.comp")),
      m_dynamicFineOperatorProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_dynamic_fine_operator.comp")),
      m_componentInitializeProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_component_initialize.comp")),
      m_componentPropagateProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_component_propagate.comp")),
      m_componentFlagsProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_component_flags.comp")),
      m_componentSummaryProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_component_summary.comp")),
      m_dynamicCoarseOperatorProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_dynamic_coarse_operator.comp")),
      m_classificationDiagnosticsProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_classification_diagnostics.comp")),
      m_particleRedistributionProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_particle_redistribute.comp")),
      m_reduceDivergenceProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_reduce_divergence.comp")),
      m_reduceVelocityProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_reduce_velocity.comp")),
      m_reduceVelocityPartialProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_reduce_velocity_partial.comp")),
      m_visualizationProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_visualize.comp")),
      m_levelSetAdvectionProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_advect.comp")),
      m_levelSetCorrectionProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_maccormack.comp")),
      m_levelSetAreaProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_area.comp")),
      m_levelSetRedistanceProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_redistance.comp")),
      m_levelSetVelocityExtensionInitializeProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_velocity_extend_init.comp")),
      m_levelSetVelocityExtensionLayerProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_velocity_extend_layer.comp")),
      m_levelSetThetaProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_theta.comp")),
      m_levelSetParticleDiagnosticProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_particle_diagnostic.comp")),
      m_levelSetVelocityDiagnosticProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_velocity_diagnostic.comp")),
      m_levelSetComponentInitializeProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_component_initialize.comp")),
      m_levelSetComponentPropagateProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_component_propagate.comp")),
      m_levelSetComponentSummaryProgram(graphics::ShaderProgram::fromComputeFile(
          divergenceShaderPath.parent_path() / "mac_levelset_component_summary.comp")) {
    GLint maximumInvocations = 0;
    GLint maximumWorkgroupX = 0;
    GLint maximumSharedBytes = 0;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maximumInvocations);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maximumWorkgroupX);
    glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &maximumSharedBytes);
    constexpr GLint requiredSharedBytes = static_cast<GLint>(
        2 * 256 * sizeof(double));
    m_fusedCoarseSmootherSupported = maximumInvocations >= 256 &&
        maximumWorkgroupX >= 256 && maximumSharedBytes >= requiredSharedBytes;
    glCreateBuffers(1, &m_uBuffer);
    glCreateBuffers(1, &m_vBuffer);
    glCreateBuffers(1, &m_uScratchBuffer);
    glCreateBuffers(1, &m_vScratchBuffer);
    glCreateBuffers(1, &m_pressureBuffer);
    glCreateBuffers(1, &m_pressureScratchBuffer);
    glCreateBuffers(1, &m_divergenceBuffer);
    glCreateBuffers(1, &m_cellTypeBuffer);
    glCreateBuffers(1, &m_residualBuffer);
    glCreateBuffers(1, &m_reductionBufferA);
    glCreateBuffers(1, &m_reductionBufferB);
    glCreateBuffers(1, &m_pcgResidualBuffer);
    glCreateBuffers(1, &m_pcgPreconditionedBuffer);
    glCreateBuffers(1, &m_pcgDirectionBuffer);
    glCreateBuffers(1, &m_pcgApBuffer);
    glCreateBuffers(1, &m_pcgScalarBuffer);
    glCreateBuffers(1, &m_pressureControlBuffer);
    glCreateBuffers(1, &m_solidMaskBuffer);
    glCreateBuffers(1, &m_previousCellTypeBuffer);
    glCreateBuffers(1, &m_particleBuffer);
    glCreateBuffers(1, &m_classificationCounterBuffer);
    glCreateBuffers(1, &m_particleCellCountBuffer);
    glCreateBuffers(1, &m_classificationDiagnosticBuffer);
    glCreateBuffers(1, &m_classificationFlipBuffer);
    glCreateBuffers(1, &m_redistributionCounterBuffer);
    glCreateBuffers(1, &m_uValidBuffer);
    glCreateBuffers(1, &m_vValidBuffer);
    glCreateBuffers(1, &m_uValidScratchBuffer);
    glCreateBuffers(1, &m_vValidScratchBuffer);
    glCreateBuffers(1, &m_componentLabelBufferA);
    glCreateBuffers(1, &m_componentLabelBufferB);
    glCreateBuffers(1, &m_componentFlagsBuffer);
    glCreateBuffers(1, &m_topologyStatusBuffer);
    glCreateBuffers(1, &m_topologyChangedBuffer);
    glCreateBuffers(1, &m_multigridBuildStatusBuffer);
    glCreateBuffers(1, &m_candidateUBuffer);
    glCreateBuffers(1, &m_candidateVBuffer);
    glCreateBuffers(1, &m_candidatePressureBuffer);
    glCreateBuffers(1, &m_candidateCellTypeBuffer);
    glCreateBuffers(1, &m_candidatePreviousCellTypeBuffer);
    glCreateBuffers(1, &m_candidateParticleBuffer);
    glCreateBuffers(1, &m_candidateParticleCellCountBuffer);
    glCreateBuffers(1, &m_candidateClassificationDiagnosticBuffer);
    glCreateBuffers(1, &m_candidateClassificationFlipBuffer);
    glCreateBuffers(1, &m_committedRenderUBuffer);
    glCreateBuffers(1, &m_committedRenderVBuffer);
    glCreateBuffers(1, &m_committedRenderPressureBuffer);
    glCreateBuffers(1, &m_committedRenderDivergenceBuffer);
    glCreateBuffers(1, &m_committedRenderCellTypeBuffer);
    glCreateBuffers(1, &m_committedRenderParticleBuffer);
    glCreateBuffers(1, &m_committedRenderParticleCellCountBuffer);
    glCreateBuffers(1, &m_committedRenderClassificationDiagnosticBuffer);
    glCreateBuffers(1, &m_levelSetPhiBuffer);
    glCreateBuffers(1, &m_levelSetPhiScratchBuffer);
    glCreateBuffers(1, &m_levelSetPhiAuxBuffer);
    glCreateBuffers(1, &m_levelSetConfidenceBuffer);
    glCreateBuffers(1, &m_levelSetConfidenceScratchBuffer);
    glCreateBuffers(1, &m_levelSetConfidenceAuxBuffer);
    glCreateBuffers(1, &m_levelSetNarrowBandBuffer);
    glCreateBuffers(1, &m_levelSetUBuffer);
    glCreateBuffers(1, &m_levelSetVBuffer);
    glCreateBuffers(1, &m_levelSetUScratchBuffer);
    glCreateBuffers(1, &m_levelSetVScratchBuffer);
    glCreateBuffers(1, &m_levelSetUValidBuffer);
    glCreateBuffers(1, &m_levelSetVValidBuffer);
    glCreateBuffers(1, &m_levelSetUValidScratchBuffer);
    glCreateBuffers(1, &m_levelSetVValidScratchBuffer);
    glCreateBuffers(1, &m_levelSetHorizontalThetaBuffer);
    glCreateBuffers(1, &m_levelSetVerticalThetaBuffer);
    glCreateBuffers(1, &m_levelSetHorizontalThetaStateBuffer);
    glCreateBuffers(1, &m_levelSetVerticalThetaStateBuffer);
    glCreateBuffers(1, &m_levelSetPreviousHorizontalThetaBuffer);
    glCreateBuffers(1, &m_levelSetPreviousVerticalThetaBuffer);
    glCreateBuffers(1, &m_levelSetDiagnosticBuffer);
    glCreateBuffers(1, &m_levelSetAreaBuffer);
    glCreateBuffers(1, &m_levelSetComponentLabelBufferA);
    glCreateBuffers(1, &m_levelSetComponentLabelBufferB);
    glCreateBuffers(1, &m_levelSetComponentChangedBuffer);
    glCreateBuffers(1, &m_candidateLevelSetPhiBuffer);
    glCreateBuffers(1, &m_candidateLevelSetConfidenceBuffer);
    glCreateBuffers(1, &m_candidateLevelSetHorizontalThetaBuffer);
    glCreateBuffers(1, &m_candidateLevelSetVerticalThetaBuffer);
    glCreateBuffers(1, &m_candidateLevelSetHorizontalThetaStateBuffer);
    glCreateBuffers(1, &m_candidateLevelSetVerticalThetaStateBuffer);
    glCreateBuffers(1, &m_committedRenderLevelSetPhiBuffer);
    glCreateBuffers(1, &m_committedRenderLevelSetConfidenceBuffer);
    glCreateBuffers(1, &m_committedRenderLevelSetHorizontalThetaBuffer);
    glCreateBuffers(1, &m_committedRenderLevelSetVerticalThetaBuffer);
    glCreateBuffers(1, &m_committedRenderLevelSetHorizontalThetaStateBuffer);
    glCreateBuffers(1, &m_committedRenderLevelSetVerticalThetaStateBuffer);
    glCreateQueries(GL_TIME_ELAPSED, 1, &m_timerQuery);
    glCreateQueries(GL_TIMESTAMP, static_cast<GLsizei>(m_pressureTimestampQueries.size()),
        m_pressureTimestampQueries.data());
    glCreateQueries(GL_TIME_ELAPSED, static_cast<GLsizei>(asynchronousQueryCount),
        m_asynchronousStepQueries.data());
    if (m_uBuffer == 0 || m_vBuffer == 0 || m_uScratchBuffer == 0 ||
        m_vScratchBuffer == 0 || m_pressureBuffer == 0 ||
        m_pressureScratchBuffer == 0 || m_divergenceBuffer == 0 ||
        m_cellTypeBuffer == 0 || m_residualBuffer == 0 ||
        m_reductionBufferA == 0 || m_reductionBufferB == 0 ||
        m_pcgResidualBuffer == 0 || m_pcgPreconditionedBuffer == 0 ||
        m_pcgDirectionBuffer == 0 || m_pcgApBuffer == 0 ||
        m_pcgScalarBuffer == 0 || m_pressureControlBuffer == 0 ||
        m_solidMaskBuffer == 0 ||
        m_previousCellTypeBuffer == 0 || m_particleBuffer == 0 ||
        m_classificationCounterBuffer == 0 || m_particleCellCountBuffer == 0 ||
        m_classificationDiagnosticBuffer == 0 || m_classificationFlipBuffer == 0 ||
        m_redistributionCounterBuffer == 0 || m_uValidBuffer == 0 ||
        m_vValidBuffer == 0 || m_uValidScratchBuffer == 0 ||
        m_vValidScratchBuffer == 0 || m_componentLabelBufferA == 0 ||
        m_componentLabelBufferB == 0 || m_componentFlagsBuffer == 0 ||
        m_topologyStatusBuffer == 0 || m_topologyChangedBuffer == 0 ||
        m_multigridBuildStatusBuffer == 0 || m_candidateUBuffer == 0 ||
        m_candidateVBuffer == 0 || m_candidatePressureBuffer == 0 ||
        m_candidateCellTypeBuffer == 0 || m_candidatePreviousCellTypeBuffer == 0 ||
        m_candidateParticleBuffer == 0 || m_candidateParticleCellCountBuffer == 0 ||
        m_candidateClassificationDiagnosticBuffer == 0 ||
        m_candidateClassificationFlipBuffer == 0 ||
        m_committedRenderUBuffer == 0 || m_committedRenderVBuffer == 0 ||
        m_committedRenderPressureBuffer == 0 || m_committedRenderDivergenceBuffer == 0 ||
        m_committedRenderCellTypeBuffer == 0 || m_committedRenderParticleBuffer == 0 ||
        m_committedRenderParticleCellCountBuffer == 0 ||
        m_committedRenderClassificationDiagnosticBuffer == 0 ||
        m_levelSetPhiBuffer == 0 || m_levelSetPhiScratchBuffer == 0 ||
        m_levelSetPhiAuxBuffer == 0 || m_levelSetConfidenceBuffer == 0 ||
        m_levelSetConfidenceScratchBuffer == 0 || m_levelSetConfidenceAuxBuffer == 0 ||
        m_levelSetNarrowBandBuffer == 0 || m_levelSetUBuffer == 0 ||
        m_levelSetVBuffer == 0 || m_levelSetUScratchBuffer == 0 ||
        m_levelSetVScratchBuffer == 0 || m_levelSetUValidBuffer == 0 ||
        m_levelSetVValidBuffer == 0 || m_levelSetUValidScratchBuffer == 0 ||
        m_levelSetVValidScratchBuffer == 0 || m_levelSetHorizontalThetaBuffer == 0 ||
        m_levelSetVerticalThetaBuffer == 0 || m_levelSetHorizontalThetaStateBuffer == 0 ||
        m_levelSetVerticalThetaStateBuffer == 0 ||
        m_levelSetPreviousHorizontalThetaBuffer == 0 ||
        m_levelSetPreviousVerticalThetaBuffer == 0 || m_levelSetDiagnosticBuffer == 0 ||
        m_levelSetAreaBuffer == 0 ||
        m_levelSetComponentLabelBufferA == 0 || m_levelSetComponentLabelBufferB == 0 ||
        m_levelSetComponentChangedBuffer == 0 ||
        m_candidateLevelSetPhiBuffer == 0 || m_candidateLevelSetConfidenceBuffer == 0 ||
        m_candidateLevelSetHorizontalThetaBuffer == 0 ||
        m_candidateLevelSetVerticalThetaBuffer == 0 ||
        m_candidateLevelSetHorizontalThetaStateBuffer == 0 ||
        m_candidateLevelSetVerticalThetaStateBuffer == 0 ||
        m_committedRenderLevelSetPhiBuffer == 0 ||
        m_committedRenderLevelSetConfidenceBuffer == 0 ||
        m_committedRenderLevelSetHorizontalThetaBuffer == 0 ||
        m_committedRenderLevelSetVerticalThetaBuffer == 0 ||
        m_committedRenderLevelSetHorizontalThetaStateBuffer == 0 ||
        m_committedRenderLevelSetVerticalThetaStateBuffer == 0 || m_timerQuery == 0 ||
        std::any_of(m_pressureTimestampQueries.begin(), m_pressureTimestampQueries.end(),
            [](GLuint query) { return query == 0; }) ||
        std::any_of(m_asynchronousStepQueries.begin(), m_asynchronousStepQueries.end(),
            [](GLuint query) { return query == 0; })) {
        release();
        throw std::runtime_error("Failed to create GPU MAC resources.");
    }
    checkOpenGlError("GPU MAC resource creation");
}

GpuMacGrid::~GpuMacGrid() { release(); }

void GpuMacGrid::upload(const MacGrid& grid) {
    ensureLayout(grid);
    inspectPressureTopology(grid);
    rebuildMultigridHierarchy(grid);
    const auto start = Clock::now();
    m_cellTypes.resize(grid.cellTypes().size());
    std::vector<GLuint> solidMask(grid.cellTypes().size());
    for (std::size_t i = 0; i < m_cellTypes.size(); ++i) {
        m_cellTypes[i] = static_cast<GLuint>(grid.cellTypes()[i]);
        solidMask[i] = grid.cellTypes()[i] == CellType::solid ? 1u : 0u;
    }
    glNamedBufferSubData(m_uBuffer, 0,
        checkedBytes(grid.horizontalVelocities().size(), sizeof(float)), grid.horizontalVelocities().data());
    glNamedBufferSubData(m_vBuffer, 0,
        checkedBytes(grid.verticalVelocities().size(), sizeof(float)), grid.verticalVelocities().data());
    m_pressureUpload.assign(grid.pressures().begin(), grid.pressures().end());
    glNamedBufferSubData(m_pressureBuffer, 0,
        checkedBytes(m_pressureUpload.size(), sizeof(double)), m_pressureUpload.data());
    glNamedBufferSubData(m_pressureScratchBuffer, 0,
        checkedBytes(m_pressureUpload.size(), sizeof(double)), m_pressureUpload.data());
    glNamedBufferSubData(m_cellTypeBuffer, 0,
        checkedBytes(m_cellTypes.size(), sizeof(GLuint)), m_cellTypes.data());
    glNamedBufferSubData(m_solidMaskBuffer, 0,
        checkedBytes(solidMask.size(), sizeof(GLuint)), solidMask.data());
    glNamedBufferSubData(m_previousCellTypeBuffer, 0,
        checkedBytes(m_cellTypes.size(), sizeof(GLuint)), m_cellTypes.data());
    ++m_classificationVersion;
    m_fineOperatorVersion = m_classificationVersion;
    m_multigridVersion = m_classificationVersion;
    m_activeMultigridLevelCount = m_multigridLevels.size();
    m_dynamicPressureTopologyReady = true;
    m_dynamicPressureTopologyAnalysisFailed = false;
    checkOpenGlError("GPU MAC upload");
    m_lastUploadMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void GpuMacGrid::uploadVelocity(const MacGrid& grid) {
    if (grid.width() != m_width || grid.height() != m_height ||
        grid.cellWidth() != m_dx || grid.cellHeight() != m_dy) {
        throw std::invalid_argument("GPU velocity upload layout does not match.");
    }
    const auto start = Clock::now();
    glNamedBufferSubData(m_uBuffer, 0,
        checkedBytes(grid.horizontalVelocities().size(), sizeof(float)),
        grid.horizontalVelocities().data());
    glNamedBufferSubData(m_vBuffer, 0,
        checkedBytes(grid.verticalVelocities().size(), sizeof(float)),
        grid.verticalVelocities().data());
    checkOpenGlError("GPU MAC velocity upload");
    m_lastUploadMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void GpuMacGrid::uploadMarkerParticles(const MarkerParticleSystem& particles) {
    if (m_width == 0) {
        throw std::logic_error("Upload a GPU MAC layout before marker particles.");
    }
    const auto& source = particles.particles();
    std::vector<std::array<float, 4>> upload(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        upload[i] = {source[i].position.x, source[i].position.y,
            source[i].valid ? 1.0f : 0.0f, 0.0f};
    }
    const std::size_t requestedCapacity = std::max({source.size(),
        m_width * m_height * std::size_t{16}, std::size_t{1}});
    if (requestedCapacity > m_particleCapacity) {
        glNamedBufferData(m_particleBuffer,
            checkedBytes(requestedCapacity, sizeof(std::array<float, 4>)),
            nullptr, GL_DYNAMIC_DRAW);
        glNamedBufferData(m_candidateParticleBuffer,
            checkedBytes(requestedCapacity, sizeof(std::array<float, 4>)),
            nullptr, GL_DYNAMIC_DRAW);
        glNamedBufferData(m_committedRenderParticleBuffer,
            checkedBytes(requestedCapacity, sizeof(std::array<float, 4>)),
            nullptr, GL_DYNAMIC_DRAW);
        if (!source.empty()) {
            glNamedBufferSubData(m_particleBuffer, 0,
                checkedBytes(source.size(), sizeof(std::array<float, 4>)), upload.data());
        }
    } else if (!source.empty()) {
        glNamedBufferSubData(m_particleBuffer, 0,
            checkedBytes(source.size(), sizeof(std::array<float, 4>)), upload.data());
    }
    m_particleCapacity = std::max(m_particleCapacity, requestedCapacity);
    m_particleCount = source.size();
    const GLuint zero = 0u;
    glClearNamedBufferData(m_levelSetDiagnosticBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearNamedBufferData(m_classificationFlipBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    checkOpenGlError("GPU marker particle upload");
}

GpuFluidInjectionReport GpuMacGrid::injectFluidCells(
    Vector2 position,
    float radius,
    std::size_t requestedCellCount,
    std::size_t particlesPerCell) {
    GpuFluidInjectionReport report;
    report.requestedCellCount = requestedCellCount;
    if (requestedCellCount == 0 || particlesPerCell == 0 || radius <= 0.0f) {
        return report;
    }

    const auto cellTypes = readCellTypes();
    struct CandidateCell final {
        std::size_t x = 0;
        std::size_t y = 0;
        float distanceSquared = 0.0f;
    };
    std::vector<CandidateCell> candidates;
    const float radiusSquared = radius * radius;
    for (std::size_t y = 1; y + 1 < m_height; ++y) {
        for (std::size_t x = 1; x + 1 < m_width; ++x) {
            const float centerX = (static_cast<float>(x) + 0.5f) * m_dx;
            const float centerY = (static_cast<float>(y) + 0.5f) * m_dy;
            const float deltaX = centerX - position.x;
            const float deltaY = centerY - position.y;
            const float distanceSquared = deltaX * deltaX + deltaY * deltaY;
            if (distanceSquared > radiusSquared) continue;
            if (cellTypes[y * m_width + x] != CellType::air) {
                ++report.rejectedSolidOrFluidCells;
                continue;
            }
            candidates.push_back({x, y, distanceSquared});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const CandidateCell& a, const CandidateCell& b) {
            if (a.distanceSquared != b.distanceSquared) {
                return a.distanceSquared < b.distanceSquared;
            }
            return a.y != b.y ? a.y < b.y : a.x < b.x;
        });

    const std::size_t availableParticles = m_particleCapacity - m_particleCount;
    const std::size_t capacityCells = availableParticles / particlesPerCell;
    const std::size_t acceptedCells = std::min({
        requestedCellCount, candidates.size(), capacityCells});
    report.capacityReached = acceptedCells < std::min(requestedCellCount, candidates.size());
    if (acceptedCells == 0) return report;

    std::vector<std::array<float, 4>> particles;
    particles.reserve(acceptedCells * particlesPerCell);
    constexpr std::array<std::array<float, 2>, 8> offsets{{
        {0.25f, 0.25f}, {0.75f, 0.25f}, {0.25f, 0.75f}, {0.75f, 0.75f},
        {0.50f, 0.25f}, {0.50f, 0.75f}, {0.25f, 0.50f}, {0.75f, 0.50f}}};
    for (std::size_t cell = 0; cell < acceptedCells; ++cell) {
        const auto& candidate = candidates[cell];
        for (std::size_t particle = 0; particle < particlesPerCell; ++particle) {
            const auto& offset = offsets[particle % offsets.size()];
            particles.push_back({
                (static_cast<float>(candidate.x) + offset[0]) * m_dx,
                (static_cast<float>(candidate.y) + offset[1]) * m_dy,
                1.0f,
                0.0f});
        }
    }
    glNamedBufferSubData(
        m_particleBuffer,
        checkedBytes(m_particleCount, sizeof(std::array<float, 4>)),
        checkedBytes(particles.size(), sizeof(std::array<float, 4>)),
        particles.data());
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    m_particleCount += particles.size();
    report.acceptedCellCount = acceptedCells;
    report.particlesAdded = particles.size();
    checkOpenGlError("GPU fluid injection");
    return report;
}

GpuClassificationReport GpuMacGrid::rebuildClassificationFromParticles(
    bool preserveFluidPressure) {
    if (m_width == 0) {
        throw std::logic_error("Upload a GPU MAC layout before dynamic classification.");
    }
    const std::size_t cellCount = m_width * m_height;
    const GLuint zero = 0;
    glClearNamedBufferData(m_classificationCounterBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    beginTimedQuery();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_solidMaskBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_previousCellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_particleCellCountBuffer);
    m_classificationResetProgram.use();
    glUniform1ui(m_classificationResetProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(cellCount));
    glDispatchCompute(static_cast<GLuint>((cellCount + 255) / 256), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    if (m_particleCount != 0) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_solidMaskBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_particleBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_classificationCounterBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_particleCellCountBuffer);
        m_classificationMarkProgram.use();
        glUniform1ui(m_classificationMarkProgram.uniformLocation("particleCount"),
            static_cast<GLuint>(m_particleCount));
        glUniform2ui(m_classificationMarkProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
        glUniform2f(m_classificationMarkProgram.uniformLocation("cellSize"), m_dx, m_dy);
        glDispatchCompute(static_cast<GLuint>((m_particleCount + 255) / 256), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    }

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_previousCellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_particleCellCountBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_classificationFlipBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_classificationDiagnosticBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_classificationCounterBuffer);
    m_classificationDiagnosticsProgram.use();
    glUniform2ui(m_classificationDiagnosticsProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glDispatchCompute(static_cast<GLuint>((cellCount + 255) / 256), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_previousCellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_pressureBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_pressureScratchBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_classificationCounterBuffer);
    m_pressureSanitizeProgram.use();
    glUniform1ui(m_pressureSanitizeProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(cellCount));
    glUniform1ui(m_pressureSanitizeProgram.uniformLocation("preserveFluidPressure"),
        preserveFluidPressure ? 1u : 0u);
    glDispatchCompute(static_cast<GLuint>((cellCount + 255) / 256), 1, 1);
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    GLuint counters[12]{};
    const auto readStart = Clock::now();
    glGetNamedBufferSubData(m_classificationCounterBuffer, 0, sizeof(counters), counters);
    GpuClassificationReport report;
    report.readbackMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - readStart).count();
    report.gpuMilliseconds = finishTimedQuery();
    report.particleCount = m_particleCount;
    report.validParticleCount = counters[0];
    report.invalidParticleCount = counters[1];
    report.changedCellCount = counters[2];
    report.fluidCellCount = counters[3];
    report.enclosedAirCellCount = counters[4];
    report.surfaceHoleCandidateCount = counters[5];
    report.phaseToggleCount = counters[6];
    report.airCellWithParticleCount = counters[7];
    report.maximumParticlesPerCell = counters[8];
    report.repeatedFlickerCellCount = counters[9];
    report.bytesRead = sizeof(counters);
    const bool canReuseTopology = report.changedCellCount == 0 &&
        m_dynamicPressureTopologyReady &&
        m_fineOperatorVersion == m_classificationVersion &&
        m_multigridVersion == m_classificationVersion;
    ++m_classificationVersion;
    if (canReuseTopology) {
        m_fineOperatorVersion = m_classificationVersion;
        m_multigridVersion = m_classificationVersion;
    } else {
        m_dynamicPressureTopologyReady = false;
        m_dynamicPressureTopologyAnalysisFailed = false;
    }
    checkOpenGlError("GPU dynamic cell classification");
    return report;
}

GpuParticleAdvectionReport GpuMacGrid::advectMarkerParticles(float timeStep) {
    if (!std::isfinite(timeStep) || timeStep < 0.0f) {
        throw std::invalid_argument("GPU particle timestep must be finite and non-negative.");
    }
    GpuParticleAdvectionReport report;
    report.particles.particleCount = m_particleCount;
    if (m_particleCount == 0) return report;
    const GLuint zero = 0;
    glClearNamedBufferData(m_classificationCounterBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_uBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_vBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_particleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_classificationCounterBuffer);
    m_particleAdvectionProgram.use();
    glUniform1ui(m_particleAdvectionProgram.uniformLocation("particleCount"),
        static_cast<GLuint>(m_particleCount));
    glUniform2ui(m_particleAdvectionProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_particleAdvectionProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1f(m_particleAdvectionProgram.uniformLocation("timeStep"), timeStep);
    beginTimedQuery();
    glDispatchCompute(static_cast<GLuint>((m_particleCount + 255) / 256), 1, 1);
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    GLuint counters[2]{};
    const auto readStart = Clock::now();
    glGetNamedBufferSubData(m_classificationCounterBuffer, 0, sizeof(counters), counters);
    report.readbackMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - readStart).count();
    report.gpuMilliseconds = finishTimedQuery();
    report.particles.collisionCount = counters[0];
    report.particles.invalidPositionCount = counters[1];
    report.bytesRead = sizeof(counters);
    checkOpenGlError("GPU RK2 marker particle advection");
    return report;
}

GpuParticleRedistributionReport GpuMacGrid::redistributeMarkerParticles(
    std::size_t targetMinimum) {
    if (targetMinimum == 0 || targetMinimum > 8) {
        throw std::invalid_argument("GPU redistribution target must be between one and eight.");
    }
    if (m_width == 0 || m_particleCapacity == 0) {
        throw std::logic_error("Upload GPU particles before redistribution.");
    }
    GpuParticleRedistributionReport report;
    report.particleCountBefore = m_particleCount;
    const GLuint counters[4]{static_cast<GLuint>(m_particleCount), 0u, 0u, 0u};
    glNamedBufferSubData(m_redistributionCounterBuffer, 0, sizeof(counters), counters);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_particleCellCountBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_particleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_redistributionCounterBuffer);
    m_particleRedistributionProgram.use();
    glUniform2ui(m_particleRedistributionProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_particleRedistributionProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1ui(m_particleRedistributionProgram.uniformLocation("targetMinimum"),
        static_cast<GLuint>(targetMinimum));
    glUniform1ui(m_particleRedistributionProgram.uniformLocation("particleCapacity"),
        static_cast<GLuint>(m_particleCapacity));
    beginTimedQuery();
    glDispatchCompute(static_cast<GLuint>((m_width * m_height + 255) / 256), 1, 1);
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    GLuint result[4]{};
    const auto readStart = Clock::now();
    glGetNamedBufferSubData(m_redistributionCounterBuffer, 0, sizeof(result), result);
    report.readbackMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - readStart).count();
    report.gpuMilliseconds = finishTimedQuery();
    report.capacityOverflowCount = result[1];
    report.addedParticleCount = result[2];
    m_particleCount = std::min<std::size_t>(result[0], m_particleCapacity);
    report.particleCountAfter = m_particleCount;
    checkOpenGlError("GPU marker particle redistribution");
    return report;
}

double GpuMacGrid::extrapolateVelocity(std::size_t layers) {
    if (m_width == 0 || layers == 0) {
        throw std::invalid_argument("GPU velocity extrapolation requires a layout and layers.");
    }
    const GLuint groupsX = static_cast<GLuint>((m_width + 16) / 16);
    const GLuint groupsY = static_cast<GLuint>((m_height + 16) / 16);
    beginTimedQuery();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_uBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_vBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_uValidBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_vValidBuffer);
    m_extrapolationInitializeProgram.use();
    glUniform2ui(m_extrapolationInitializeProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    for (std::size_t layer = 0; layer < layers; ++layer) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_uBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_vBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_cellTypeBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_uValidBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_vValidBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_uScratchBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_vScratchBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_uValidScratchBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_vValidScratchBuffer);
        m_extrapolationLayerProgram.use();
        glUniform2ui(m_extrapolationLayerProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
        glDispatchCompute(groupsX, groupsY, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        std::swap(m_uBuffer, m_uScratchBuffer);
        std::swap(m_vBuffer, m_vScratchBuffer);
        std::swap(m_uValidBuffer, m_uValidScratchBuffer);
        std::swap(m_vValidBuffer, m_vValidScratchBuffer);
    }
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    const double milliseconds = finishTimedQuery();
    checkOpenGlError("GPU layered velocity extrapolation");
    return milliseconds;
}

GpuDynamicTopologyReport GpuMacGrid::rebuildDynamicPressureTopology() {
    if (m_width == 0 || m_height == 0) {
        throw std::logic_error("Upload a GPU MAC layout before dynamic topology construction.");
    }
    const auto cpuStart = Clock::now();
    GpuDynamicTopologyReport report;
    report.classificationVersion = m_classificationVersion;
    allocateDynamicMultigridHierarchy();
    const std::size_t cellCount = m_width * m_height;
    const GLuint zero = 0u;
    glClearNamedBufferData(m_topologyStatusBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearNamedBufferData(m_componentFlagsBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearNamedBufferData(m_multigridBuildStatusBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    GLuint queries[3]{};
    if (m_synchronousTimingEnabled) {
        glCreateQueries(GL_TIME_ELAPSED, 3, queries);
        if (queries[0] == 0 || queries[1] == 0 || queries[2] == 0) {
            glDeleteQueries(3, queries);
            throw std::runtime_error("Failed to create dynamic topology timer queries.");
        }
    }

    MultigridLevel& fine = m_multigridLevels.front();
    if (m_synchronousTimingEnabled) glBeginQuery(GL_TIME_ELAPSED, queries[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, fine.diagonal);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, fine.weights);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fine.active);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_topologyStatusBuffer);
    m_dynamicFineOperatorProgram.use();
    glUniform2ui(m_dynamicFineOperatorProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_dynamicFineOperatorProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glDispatchCompute(static_cast<GLuint>((cellCount + 255) / 256), 1, 1);
    if (m_synchronousTimingEnabled) glEndQuery(GL_TIME_ELAPSED);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    if (m_synchronousTimingEnabled) glBeginQuery(GL_TIME_ELAPSED, queries[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_componentLabelBufferA);
    m_componentInitializeProgram.use();
    glUniform1ui(m_componentInitializeProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(cellCount));
    glDispatchCompute(static_cast<GLuint>((cellCount + 255) / 256), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    GLuint labelInput = m_componentLabelBufferA;
    GLuint labelOutput = m_componentLabelBufferB;
    bool labelsConverged = false;
    GLuint changed = 1u;
    constexpr std::size_t checkInterval = 16;
    for (std::size_t iteration = 0; iteration < cellCount; ++iteration) {
        glClearNamedBufferData(m_topologyChangedBuffer,
            GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, labelInput);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, labelOutput);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_topologyChangedBuffer);
        m_componentPropagateProgram.use();
        glUniform2ui(m_componentPropagateProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
        glDispatchCompute(static_cast<GLuint>((cellCount + 255) / 256), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        std::swap(labelInput, labelOutput);
        report.labelPropagationIterations = iteration + 1;
        if ((iteration + 1) % checkInterval == 0 || iteration + 1 == cellCount) {
            const auto readStart = Clock::now();
            glGetNamedBufferSubData(m_topologyChangedBuffer, 0, sizeof(changed), &changed);
            report.statusReadbackMilliseconds +=
                std::chrono::duration<double, std::milli>(Clock::now() - readStart).count();
            report.statusBytesRead += sizeof(changed);
            if (changed == 0u) {
                labelsConverged = true;
                break;
            }
        }
    }

    glCopyNamedBufferSubData(labelInput, fine.component, 0, 0,
        checkedBytes(cellCount, sizeof(GLuint)));
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, labelInput);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_componentFlagsBuffer);
    m_componentFlagsProgram.use();
    glUniform2ui(m_componentFlagsProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glDispatchCompute(static_cast<GLuint>((cellCount + 255) / 256), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, labelInput);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_componentFlagsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_topologyStatusBuffer);
    m_componentSummaryProgram.use();
    glUniform1ui(m_componentSummaryProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(cellCount));
    glDispatchCompute(static_cast<GLuint>((cellCount + 255) / 256), 1, 1);
    if (m_synchronousTimingEnabled) glEndQuery(GL_TIME_ELAPSED);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    if (m_synchronousTimingEnabled) glBeginQuery(GL_TIME_ELAPSED, queries[2]);
    for (std::size_t levelIndex = 0; levelIndex + 1 < m_multigridLevels.size(); ++levelIndex) {
        MultigridLevel& source = m_multigridLevels[levelIndex];
        MultigridLevel& coarse = m_multigridLevels[levelIndex + 1];
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, source.diagonal);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, source.weights);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, source.active);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, source.component);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coarse.diagonal);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coarse.weights);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coarse.active);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, coarse.component);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_multigridBuildStatusBuffer);
        m_dynamicCoarseOperatorProgram.use();
        glUniform2ui(m_dynamicCoarseOperatorProgram.uniformLocation("fineSize"),
            static_cast<GLuint>(source.width), static_cast<GLuint>(source.height));
        glUniform2ui(m_dynamicCoarseOperatorProgram.uniformLocation("coarseSize"),
            static_cast<GLuint>(coarse.width), static_cast<GLuint>(coarse.height));
        glUniform1ui(m_dynamicCoarseOperatorProgram.uniformLocation("levelIndex"),
            static_cast<GLuint>(levelIndex));
        glDispatchCompute(static_cast<GLuint>((coarse.width * coarse.height + 255) / 256), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    if (m_synchronousTimingEnabled) glEndQuery(GL_TIME_ELAPSED);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    GLuint status[8]{};
    std::vector<GLuint> buildStatus(m_multigridLevels.size(), 0u);
    const auto statusReadStart = Clock::now();
    glGetNamedBufferSubData(m_topologyStatusBuffer, 0, sizeof(status), status);
    glGetNamedBufferSubData(m_multigridBuildStatusBuffer, 0,
        checkedBytes(buildStatus.size(), sizeof(GLuint)), buildStatus.data());
    report.statusReadbackMilliseconds +=
        std::chrono::duration<double, std::milli>(Clock::now() - statusReadStart).count();
    report.statusBytesRead += sizeof(status) + buildStatus.size() * sizeof(GLuint);

    if (m_synchronousTimingEnabled) {
        GLuint64 nanoseconds[3]{};
        for (std::size_t i = 0; i < 3; ++i) {
            glGetQueryObjectui64v(queries[i], GL_QUERY_RESULT, &nanoseconds[i]);
        }
        glDeleteQueries(3, queries);
        report.fineOperatorGpuMilliseconds = static_cast<double>(nanoseconds[0]) / 1.0e6;
        report.topologyGpuMilliseconds = static_cast<double>(nanoseconds[1]) / 1.0e6;
        report.coarseOperatorGpuMilliseconds = static_cast<double>(nanoseconds[2]) / 1.0e6;
    }
    report.fluidCellCount = status[1];
    report.isolatedFluidCellCount = status[2];
    report.connectedComponentCount = status[3];
    report.pureNeumannComponentCount = status[4];
    report.fineOperatorValid = true;
    report.topologyValid = labelsConverged;
    m_activeMultigridLevelCount = m_multigridLevels.size();
    for (std::size_t level = 0; level + 1 < m_multigridLevels.size(); ++level) {
        if (buildStatus[level] != 0u) {
            m_activeMultigridLevelCount = level + 1;
            break;
        }
    }
    report.activeMultigridLevelCount = m_activeMultigridLevelCount;
    report.multigridValid = labelsConverged && m_activeMultigridLevelCount != 0;
    m_pureNeumannComponentCount = report.pureNeumannComponentCount;
    m_connectedComponentCount = report.connectedComponentCount;
    m_fluidCellCount = report.fluidCellCount;
    m_hasIsolatedFluidCell = report.isolatedFluidCellCount != 0;
    m_dynamicPressureTopologyAnalysisFailed = !labelsConverged;
    m_dynamicPressureTopologyReady = labelsConverged;
    m_fineOperatorVersion = labelsConverged ? m_classificationVersion : 0;
    m_multigridVersion = report.multigridValid ? m_classificationVersion : 0;
    report.fineOperatorVersion = m_fineOperatorVersion;
    report.multigridVersion = m_multigridVersion;
    report.totalCpuMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - cpuStart).count();
    checkOpenGlError("GPU dynamic pressure topology construction");
    return report;
}

GpuDynamicTopologyReport GpuMacGrid::reuseDynamicPressureTopology() {
    if (!m_dynamicPressureTopologyReady ||
        m_fineOperatorVersion != m_classificationVersion ||
        m_multigridVersion != m_classificationVersion) {
        throw std::logic_error("GPU dynamic pressure topology is not reusable.");
    }
    GpuDynamicTopologyReport report;
    report.fineOperatorValid = true;
    report.topologyValid = !m_dynamicPressureTopologyAnalysisFailed;
    report.multigridValid = m_activeMultigridLevelCount != 0;
    report.reusedExistingHierarchy = true;
    report.fluidCellCount = m_fluidCellCount;
    report.connectedComponentCount = m_connectedComponentCount;
    report.pureNeumannComponentCount = m_pureNeumannComponentCount;
    report.isolatedFluidCellCount = m_hasIsolatedFluidCell ? 1 : 0;
    report.activeMultigridLevelCount = m_activeMultigridLevelCount;
    report.classificationVersion = m_classificationVersion;
    report.fineOperatorVersion = m_fineOperatorVersion;
    report.multigridVersion = m_multigridVersion;
    return report;
}

void GpuMacGrid::synchronizeDynamicPressureTopology() {
    if (m_dynamicPressureTopologyReady) return;
    const auto types = readCellTypes();
    MacGrid topology(m_width, m_height, m_dx, m_dy);
    for (std::size_t y = 0; y < m_height; ++y) {
        for (std::size_t x = 0; x < m_width; ++x) {
            topology.setCellType(x, y, types[y * m_width + x]);
        }
    }
    inspectPressureTopology(topology);
    rebuildMultigridHierarchy(topology);
    m_cellTypes.resize(types.size());
    std::transform(types.begin(), types.end(), m_cellTypes.begin(),
        [](CellType type) { return static_cast<GLuint>(type); });
    m_dynamicPressureTopologyReady = true;
    m_dynamicPressureTopologyAnalysisFailed = false;
    m_fineOperatorVersion = m_classificationVersion;
    m_multigridVersion = m_classificationVersion;
}

void GpuMacGrid::download(MacGrid& grid) const {
    if (grid.width() != m_width || grid.height() != m_height ||
        grid.cellWidth() != m_dx || grid.cellHeight() != m_dy) {
        throw std::invalid_argument("GPU download target layout does not match.");
    }
    const auto u = readHorizontalVelocity();
    const auto v = readVerticalVelocity();
    const auto p = readPressure();
    const auto types = readCellTypes();
    for (std::size_t y = 0; y < m_height; ++y)
        for (std::size_t x = 0; x <= m_width; ++x)
            grid.horizontalVelocity(x, y) = u[y * (m_width + 1) + x];
    for (std::size_t y = 0; y <= m_height; ++y)
        for (std::size_t x = 0; x < m_width; ++x)
            grid.verticalVelocity(x, y) = v[y * m_width + x];
    for (std::size_t y = 0; y < m_height; ++y)
        for (std::size_t x = 0; x < m_width; ++x)
            grid.pressure(x, y) = p[y * m_width + x];
    for (std::size_t y = 0; y < m_height; ++y)
        for (std::size_t x = 0; x < m_width; ++x)
            grid.setCellType(x, y, types[y * m_width + x]);
    grid.computeDivergence();
}

void GpuMacGrid::dispatchDivergence() {
    if (m_width == 0 || m_height == 0) throw std::logic_error("Upload a GPU MAC layout before dispatch.");
    if (m_queryPending) throw std::logic_error("Read the previous timed result before another dispatch.");
    bindCommonBuffers();
    m_divergenceProgram.use();
    glUniform2ui(m_divergenceProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_divergenceProgram.uniformLocation("cellSize"), m_dx, m_dy);
    beginTimedQuery();
    glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
        static_cast<GLuint>((m_height + 15) / 16), 1);
    ++m_pressureDispatchCount;
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    checkOpenGlError("GPU divergence dispatch");
}

std::vector<float> GpuMacGrid::readDivergence() {
    if (!m_queryPending) throw std::logic_error("No timed GPU divergence dispatch is pending.");
    std::vector<float> values(m_width * m_height);
    glGetNamedBufferSubData(m_divergenceBuffer, 0,
        checkedBytes(values.size(), sizeof(float)), values.data());
    m_lastGpuMilliseconds = finishTimedQuery();
    checkOpenGlError("GPU divergence readback");
    return values;
}

GpuPressureReport GpuMacGrid::solvePressure(const GpuPressureSettings& settings) {
    validatePressureSettings(settings);
    m_pressureDispatchCount = 0;
    if (m_width == 0) throw std::logic_error("Upload a GPU MAC layout before pressure solve.");
    const auto finalize = [this, &settings](GpuPressureReport report) {
        report.precisionMode = settings.precisionMode;
        report.dispatchCount = m_pressureDispatchCount;
        return report;
    };
    if (!m_dynamicPressureTopologyReady) {
        GpuPressureReport report;
        report.solver = settings.solver;
        report.failureReason = m_dynamicPressureTopologyAnalysisFailed
            ? GpuPressureFailureReason::dynamicPressureTopologyAnalysisFailed
            : GpuPressureFailureReason::dynamicPressureTopologyStale;
        return finalize(report);
    }
    if (m_fineOperatorVersion != m_classificationVersion ||
        (settings.solver == GpuPressureSolver::multigridPreconditionedConjugateGradient &&
         m_multigridVersion != m_classificationVersion)) {
        GpuPressureReport report;
        report.solver = settings.solver;
        report.failureReason = GpuPressureFailureReason::dynamicPressureTopologyStale;
        return finalize(report);
    }
    if (settings.solver == GpuPressureSolver::preconditionedConjugateGradient ||
        settings.solver == GpuPressureSolver::multigridPreconditionedConjugateGradient) {
        if (settings.controlMode == GpuPressureControlMode::gpuResidentBatches) {
            return finalize(solvePcgGpuResident(settings));
        }
        return finalize(solvePcg(settings));
    }
    return finalize(solveJacobi(settings));
}

GpuPressureReport GpuMacGrid::solveJacobi(const GpuPressureSettings& settings) {
    GpuPressureReport report;
    report.solver = GpuPressureSolver::weightedJacobi;
    report.warmStartUsed = settings.warmStartEnabled;
    if (!settings.warmStartEnabled) clearPressureBuffers();

    report.initialResidual = computeResidual(
        settings.density, settings.timeStep, report.reductionGpuMilliseconds,
        report.residualReadbackMilliseconds, report.validEquationCount,
        report.residualBytesRead);
    report.finalResidual = report.initialResidual;
    if (report.initialResidual.rootMeanSquare <= settings.residualTolerance) {
        report.converged = true;
        report.termination = PressureSolveTermination::initialResidualWithinTolerance;
        return report;
    }

    m_jacobiProgram.use();
    glUniform2ui(m_jacobiProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_jacobiProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1d(m_jacobiProgram.uniformLocation("rhsScale"),
        static_cast<double>(settings.density) / settings.timeStep);
    glUniform1d(m_jacobiProgram.uniformLocation("relaxationFactor"), settings.relaxationFactor);
    const GLuint groupsX = static_cast<GLuint>((m_width + 15) / 16);
    const GLuint groupsY = static_cast<GLuint>((m_height + 15) / 16);

    while (report.iterations < settings.maximumIterations) {
        const std::size_t batch = std::min(
            settings.residualCheckInterval, settings.maximumIterations - report.iterations);
        m_jacobiProgram.use();
        beginTimedQuery();
        for (std::size_t i = 0; i < batch; ++i) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_pressureBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_divergenceBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_pressureScratchBuffer);
            glDispatchCompute(groupsX, groupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            std::swap(m_pressureBuffer, m_pressureScratchBuffer);
        }
        endTimedQuery();
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        report.iterations += batch;
        report.gpuMilliseconds += finishTimedQuery();
        report.finalResidual = computeResidual(
            settings.density, settings.timeStep, report.reductionGpuMilliseconds,
            report.residualReadbackMilliseconds, report.validEquationCount,
            report.residualBytesRead);
        if (report.finalResidual.rootMeanSquare <= settings.residualTolerance) {
            report.converged = true;
            report.termination = PressureSolveTermination::iterativeResidualWithinTolerance;
            break;
        }
    }
    checkOpenGlError("GPU Jacobi pressure solve");
    return report;
}

GpuPressureReport GpuMacGrid::solvePcg(const GpuPressureSettings& settings) {
    GpuPressureReport report;
    report.solver = settings.solver;
    const bool useMultigrid =
        settings.solver == GpuPressureSolver::multigridPreconditionedConjugateGradient;
    report.multigridLevelCount = useMultigrid ? m_activeMultigridLevelCount : 0;
    report.multigridMemoryBytes = useMultigrid ? m_multigridMemoryBytes : 0;
    report.warmStartUsed = settings.warmStartEnabled;
    if (m_hasIsolatedFluidCell) {
        report.failureReason = GpuPressureFailureReason::isolatedFluidCell;
        return report;
    }
    if (m_pureNeumannComponentCount > 1) {
        report.failureReason =
            GpuPressureFailureReason::unsupportedMultiplePureNeumannComponents;
        return report;
    }
    if (!settings.warmStartEnabled) clearPressureBuffers();
    report.initialResidual = computeResidual(
        settings.density, settings.timeStep, report.reductionGpuMilliseconds,
        report.residualReadbackMilliseconds, report.validEquationCount,
        report.residualBytesRead);
    report.finalResidual = report.initialResidual;
    if (report.initialResidual.rootMeanSquare <= settings.residualTolerance) {
        report.converged = true;
        report.termination = PressureSolveTermination::initialResidualWithinTolerance;
        return report;
    }

    const GLuint groupsX = static_cast<GLuint>((m_width + 15) / 16);
    const GLuint groupsY = static_cast<GLuint>((m_height + 15) / 16);
    const GLuint vectorGroups = static_cast<GLuint>((m_width * m_height + 255) / 256);
    dispatchPcgOperator(m_pressureBuffer, m_pcgApBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_divergenceBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_pcgApBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_pcgResidualBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_pcgPreconditionedBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_pcgDirectionBuffer);
    m_pcgInitializeProgram.use();
    glUniform2ui(m_pcgInitializeProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_pcgInitializeProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1d(m_pcgInitializeProgram.uniformLocation("rhsScale"),
        static_cast<double>(settings.density) / settings.timeStep);
    glDispatchCompute(groupsX, groupsY, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if (useMultigrid) {
        applyMultigridPreconditioner(
            m_pcgResidualBuffer, m_pcgPreconditionedBuffer, settings);
        copyVector(m_pcgPreconditionedBuffer, m_pcgDirectionBuffer, m_width * m_height);
    }
    dispatchPcgScalar(
        reduceDot(m_pcgResidualBuffer, m_pcgPreconditionedBuffer), 0);

    std::vector<GLuint> multigridTimestampQueries;
    if (useMultigrid && m_synchronousTimingEnabled) {
        const std::size_t timestampBatchCapacity = std::min(
            settings.residualCheckInterval, settings.maximumIterations);
        if (timestampBatchCapacity >
            static_cast<std::size_t>(std::numeric_limits<GLsizei>::max()) / 2) {
            throw std::invalid_argument("Multigrid timing interval exceeds the OpenGL query limit.");
        }
        multigridTimestampQueries.resize(timestampBatchCapacity * 2);
        glCreateQueries(GL_TIMESTAMP,
            static_cast<GLsizei>(multigridTimestampQueries.size()),
            multigridTimestampQueries.data());
        if (std::find(multigridTimestampQueries.begin(),
                multigridTimestampQueries.end(), 0u) != multigridTimestampQueries.end()) {
            glDeleteQueries(static_cast<GLsizei>(multigridTimestampQueries.size()),
                multigridTimestampQueries.data());
            throw std::runtime_error("Failed to create multigrid timestamp queries.");
        }
    }

    while (report.iterations < settings.maximumIterations) {
        const std::size_t batch = std::min(
            settings.residualCheckInterval, settings.maximumIterations - report.iterations);
        beginTimedQuery();
        for (std::size_t iteration = 0; iteration < batch; ++iteration) {
            dispatchPcgOperator(m_pcgDirectionBuffer, m_pcgApBuffer);
            dispatchPcgScalar(reduceDot(m_pcgDirectionBuffer, m_pcgApBuffer), 1);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_pressureBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_pcgResidualBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_pcgDirectionBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_pcgApBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_pcgPreconditionedBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_pcgScalarBuffer);
            const int vectorUpdateProfile = beginPressureProfileEvent(
                GpuPressureProfileCategory::vectorUpdate, 0);
            m_pcgUpdateProgram.use();
            glUniform1ui(m_pcgUpdateProgram.uniformLocation("elementCount"),
                static_cast<GLuint>(m_width * m_height));
            glUniform2ui(m_pcgUpdateProgram.uniformLocation("gridSize"),
                static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
            glUniform2f(m_pcgUpdateProgram.uniformLocation("cellSize"), m_dx, m_dy);
            glDispatchCompute(vectorGroups, 1, 1);
            ++m_pressureDispatchCount;
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            endPressureProfileEvent(vectorUpdateProfile, 1);

            if (useMultigrid) {
                if (m_synchronousTimingEnabled) {
                    glQueryCounter(multigridTimestampQueries[iteration * 2], GL_TIMESTAMP);
                }
                applyMultigridPreconditioner(
                    m_pcgResidualBuffer, m_pcgPreconditionedBuffer, settings);
                if (m_synchronousTimingEnabled) {
                    glQueryCounter(multigridTimestampQueries[iteration * 2 + 1], GL_TIMESTAMP);
                }
            }

            dispatchPcgScalar(
                reduceDot(m_pcgResidualBuffer, m_pcgPreconditionedBuffer), 2);
            const int directionUpdateProfile = beginPressureProfileEvent(
                GpuPressureProfileCategory::vectorUpdate, 1);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_pcgPreconditionedBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_pcgDirectionBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_pcgScalarBuffer);
            m_pcgDirectionProgram.use();
            glUniform1ui(m_pcgDirectionProgram.uniformLocation("elementCount"),
                static_cast<GLuint>(m_width * m_height));
            glDispatchCompute(vectorGroups, 1, 1);
            ++m_pressureDispatchCount;
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            endPressureProfileEvent(directionUpdateProfile, 1);
        }
        endTimedQuery();
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        report.gpuMilliseconds += finishTimedQuery();
        if (useMultigrid && m_synchronousTimingEnabled) {
            for (std::size_t iteration = 0; iteration < batch; ++iteration) {
                GLuint64 beginTimestamp = 0;
                GLuint64 endTimestamp = 0;
                glGetQueryObjectui64v(multigridTimestampQueries[iteration * 2],
                    GL_QUERY_RESULT, &beginTimestamp);
                glGetQueryObjectui64v(multigridTimestampQueries[iteration * 2 + 1],
                    GL_QUERY_RESULT, &endTimestamp);
                report.preconditionerGpuMilliseconds +=
                    static_cast<double>(endTimestamp - beginTimestamp) / 1.0e6;
            }
        }
        report.iterations += batch;
        report.finalResidual = computeResidual(
            settings.density, settings.timeStep, report.reductionGpuMilliseconds,
            report.residualReadbackMilliseconds, report.validEquationCount,
            report.residualBytesRead);
        if (report.finalResidual.rootMeanSquare <= settings.residualTolerance) {
            report.converged = true;
            report.termination = PressureSolveTermination::iterativeResidualWithinTolerance;
            break;
        }
        double scalar[4]{};
        const auto readStart = Clock::now();
        glGetNamedBufferSubData(m_pcgScalarBuffer, 0, sizeof(scalar), scalar);
        report.residualReadbackMilliseconds +=
            std::chrono::duration<double, std::milli>(Clock::now() - readStart).count();
        report.residualBytesRead += sizeof(scalar);
        if (scalar[3] != 0.0 || !std::isfinite(scalar[0])) {
            report.failureReason = GpuPressureFailureReason::numericalBreakdown;
            break;
        }
    }
    if (!multigridTimestampQueries.empty()) {
        glDeleteQueries(static_cast<GLsizei>(multigridTimestampQueries.size()),
            multigridTimestampQueries.data());
    }
    checkOpenGlError("GPU PCG pressure solve");
    return report;
}

GpuPressureReport GpuMacGrid::solvePcgGpuResident(const GpuPressureSettings& settings) {
    GpuPressureReport report;
    report.solver = settings.solver;
    report.gpuResidentControlUsed = true;
    const bool useMultigrid =
        settings.solver == GpuPressureSolver::multigridPreconditionedConjugateGradient;
    report.multigridLevelCount = useMultigrid ? m_activeMultigridLevelCount : 0;
    report.multigridMemoryBytes = useMultigrid ? m_multigridMemoryBytes : 0;
    report.warmStartUsed = settings.warmStartEnabled;
    if (m_hasIsolatedFluidCell) {
        report.solveState = GpuPressureSolveState::invalidTopology;
        report.failureReason = GpuPressureFailureReason::isolatedFluidCell;
        return report;
    }
    if (m_pureNeumannComponentCount > 1) {
        report.solveState = GpuPressureSolveState::invalidTopology;
        report.failureReason =
            GpuPressureFailureReason::unsupportedMultiplePureNeumannComponents;
        return report;
    }
    if (!settings.warmStartEnabled) clearPressureBuffers();

    if (m_synchronousTimingEnabled) {
        glQueryCounter(m_pressureTimestampQueries[0], GL_TIMESTAMP);
    }

    initializeGpuPressureControl();
    m_gpuPressureControlActive = true;
    configureGpuPressureControl(true);
    ScopeExit controlGuard([this] {
        m_gpuPressureControlActive = false;
        configureGpuPressureControl(false);
    });

    GLuint residualSummary = dispatchResidualReduction(
        settings.density, settings.timeStep, report.reductionGpuMilliseconds);
    dispatchGpuPressureControl(residualSummary, settings.residualTolerance,
        0, settings.maximumIterations, 0);

    const GLuint groupsX = static_cast<GLuint>((m_width + 15) / 16);
    const GLuint groupsY = static_cast<GLuint>((m_height + 15) / 16);
    const GLuint vectorGroups = static_cast<GLuint>((m_width * m_height + 255) / 256);
    dispatchPcgOperator(m_pressureBuffer, m_pcgApBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_divergenceBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_pcgApBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_pcgResidualBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_pcgPreconditionedBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_pcgDirectionBuffer);
    m_pcgInitializeProgram.use();
    glUniform2ui(m_pcgInitializeProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_pcgInitializeProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1d(m_pcgInitializeProgram.uniformLocation("rhsScale"),
        static_cast<double>(settings.density) / settings.timeStep);
    glDispatchCompute(groupsX, groupsY, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if (useMultigrid) {
        applyMultigridPreconditioner(
            m_pcgResidualBuffer, m_pcgPreconditionedBuffer, settings);
        copyVector(m_pcgPreconditionedBuffer, m_pcgDirectionBuffer, m_width * m_height);
    }
    dispatchPcgScalar(
        reduceDot(m_pcgResidualBuffer, m_pcgPreconditionedBuffer), 0);

    std::size_t submittedIterations = 0;
    while (report.solveState == GpuPressureSolveState::running &&
           submittedIterations < settings.maximumIterations) {
        for (std::size_t windowBatch = 0;
             windowBatch < settings.gpuResidentBatchCount &&
             submittedIterations < settings.maximumIterations;
             ++windowBatch) {
            const std::size_t batch = std::min(settings.residualCheckInterval,
                settings.maximumIterations - submittedIterations);
            for (std::size_t iteration = 0; iteration < batch; ++iteration) {
                dispatchPcgOperator(m_pcgDirectionBuffer, m_pcgApBuffer);
                dispatchPcgScalar(reduceDot(m_pcgDirectionBuffer, m_pcgApBuffer), 1);

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_pressureBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_pcgResidualBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_pcgDirectionBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_pcgApBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_pcgPreconditionedBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_pcgScalarBuffer);
                m_pcgUpdateProgram.use();
                glUniform1ui(m_pcgUpdateProgram.uniformLocation("elementCount"),
                    static_cast<GLuint>(m_width * m_height));
                glUniform2ui(m_pcgUpdateProgram.uniformLocation("gridSize"),
                    static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
                glUniform2f(m_pcgUpdateProgram.uniformLocation("cellSize"), m_dx, m_dy);
                glDispatchCompute(vectorGroups, 1, 1);
                ++m_pressureDispatchCount;
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                if (useMultigrid) {
                    applyMultigridPreconditioner(
                        m_pcgResidualBuffer, m_pcgPreconditionedBuffer, settings);
                }
                dispatchPcgScalar(
                    reduceDot(m_pcgResidualBuffer, m_pcgPreconditionedBuffer), 2);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_pcgPreconditionedBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_pcgDirectionBuffer);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_pcgScalarBuffer);
                m_pcgDirectionProgram.use();
                glUniform1ui(m_pcgDirectionProgram.uniformLocation("elementCount"),
                    static_cast<GLuint>(m_width * m_height));
                glDispatchCompute(vectorGroups, 1, 1);
                ++m_pressureDispatchCount;
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
            submittedIterations += batch;
            residualSummary = dispatchResidualReduction(
                settings.density, settings.timeStep, report.reductionGpuMilliseconds);
            dispatchGpuPressureControl(residualSummary, settings.residualTolerance,
                submittedIterations, settings.maximumIterations, submittedIterations);
        }
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        readGpuPressureControl(report);
    }

    report.submittedIterations = submittedIterations;
    report.skippedIterations = submittedIterations > report.iterations
        ? submittedIterations - report.iterations : 0;
    if (report.solveState == GpuPressureSolveState::running) {
        report.solveState = GpuPressureSolveState::maximumIterationsReached;
        report.termination = PressureSolveTermination::maximumIterationsReached;
    }
    if (m_synchronousTimingEnabled) {
        glQueryCounter(m_pressureTimestampQueries[1], GL_TIMESTAMP);
        GLuint64 beginTimestamp = 0;
        GLuint64 endTimestamp = 0;
        glGetQueryObjectui64v(m_pressureTimestampQueries[0], GL_QUERY_RESULT, &beginTimestamp);
        glGetQueryObjectui64v(m_pressureTimestampQueries[1], GL_QUERY_RESULT, &endTimestamp);
        report.gpuMilliseconds = static_cast<double>(endTimestamp - beginTimestamp) / 1.0e6;
    }
    checkOpenGlError("GPU-resident PCG pressure solve");
    return report;
}

GpuMultigridCycleReport GpuMacGrid::testMultigridCycle(
    const GpuPressureSettings& settings) {
    validatePressureSettings(settings);
    if (m_width == 0 || m_multigridLevels.empty()) {
        throw std::logic_error("Upload a GPU MAC layout before multigrid validation.");
    }
    if (m_hasIsolatedFluidCell || m_pureNeumannComponentCount > 1) {
        throw std::invalid_argument("The current topology is not supported by multigrid validation.");
    }

    dispatchPcgOperator(m_pressureBuffer, m_pcgApBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_divergenceBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_pcgApBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_pcgResidualBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_pcgPreconditionedBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_pcgDirectionBuffer);
    m_pcgInitializeProgram.use();
    glUniform2ui(m_pcgInitializeProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_pcgInitializeProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1d(m_pcgInitializeProgram.uniformLocation("rhsScale"),
        static_cast<double>(settings.density) / settings.timeStep);
    glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
        static_cast<GLuint>((m_height + 15) / 16), 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    beginTimedQuery();
    applyMultigridPreconditioner(
        m_pcgResidualBuffer, m_pcgPreconditionedBuffer, settings);
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    dispatchPcgOperator(m_pcgPreconditionedBuffer, m_pcgApBuffer);

    const std::size_t count = m_width * m_height;
    std::vector<double> residual(count);
    std::vector<double> matrixCorrection(count);
    glGetNamedBufferSubData(m_pcgResidualBuffer, 0,
        checkedBytes(count, sizeof(double)), residual.data());
    glGetNamedBufferSubData(m_pcgApBuffer, 0,
        checkedBytes(count, sizeof(double)), matrixCorrection.data());
    GpuMultigridCycleReport report;
    report.gpuMilliseconds = finishTimedQuery();
    report.levelCount = m_multigridLevels.size();
    double initialSquared = 0.0;
    double correctedSquared = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        if (m_cellTypes[i] != static_cast<GLuint>(CellType::fluid)) continue;
        initialSquared += residual[i] * residual[i];
        const double corrected = residual[i] - matrixCorrection[i];
        correctedSquared += corrected * corrected;
        ++report.validEquationCount;
    }
    if (report.validEquationCount != 0) {
        report.initialRootMeanSquare = static_cast<float>(
            std::sqrt(initialSquared / static_cast<double>(report.validEquationCount)));
        report.correctedRootMeanSquare = static_cast<float>(
            std::sqrt(correctedSquared / static_cast<double>(report.validEquationCount)));
    }
    checkOpenGlError("GPU multigrid V-cycle validation");
    return report;
}

GpuPressureProfileReport GpuMacGrid::profilePressureSolve(
    const GpuPressureSettings& settings) {
    if (m_pressureProfilingActive) {
        throw std::logic_error("A pressure profile is already active.");
    }
    if (std::all_of(m_profileTimestampQueries.begin(), m_profileTimestampQueries.end(),
            [](GLuint query) { return query == 0; })) {
        glCreateQueries(GL_TIMESTAMP, static_cast<GLsizei>(m_profileTimestampQueries.size()),
            m_profileTimestampQueries.data());
        if (std::any_of(m_profileTimestampQueries.begin(), m_profileTimestampQueries.end(),
                [](GLuint query) { return query == 0; })) {
            throw std::runtime_error("Failed to create the offline pressure profile query pool.");
        }
    }
    m_pressureProfileEvents.clear();
    m_pressureProfileCategorySamples.fill(0);
    m_pressureProfilingActive = true;
    const auto start = Clock::now();
    GpuPressureProfileReport report;
    report.pressure = project(settings).pressure;
    report.endToEndMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    m_pressureProfilingActive = false;
    resolvePressureProfile(report);
    report.levels.reserve(m_activeMultigridLevelCount);
    for (std::size_t levelIndex = 0; levelIndex < m_activeMultigridLevelCount; ++levelIndex) {
        const MultigridLevel& level = m_multigridLevels[levelIndex];
        GpuPressureProfileLevelReport levelReport;
        levelReport.width = level.width;
        levelReport.height = level.height;
        levelReport.activeCount = level.activeCount;
        for (const PressureProfileEvent& event : m_pressureProfileEvents) {
            if (event.level != static_cast<int>(levelIndex)) continue;
            if (event.category == GpuPressureProfileCategory::smoother ||
                event.category == GpuPressureProfileCategory::coarseSolve) {
                GLuint64 begin = 0;
                GLuint64 end = 0;
                glGetQueryObjectui64v(event.beginQuery, GL_QUERY_RESULT, &begin);
                glGetQueryObjectui64v(event.endQuery, GL_QUERY_RESULT, &end);
                levelReport.smootherGpuMilliseconds +=
                    static_cast<double>(end - begin) / 1.0e6;
                levelReport.smootherDispatchCount += event.dispatchCount;
            }
        }
        levelReport.smootherIterations = levelIndex + 1 == m_activeMultigridLevelCount
            ? settings.multigridCoarseIterations
            : settings.multigridPreSmoothIterations;
        report.levels.push_back(levelReport);
    }
    return report;
}

std::vector<double> GpuMacGrid::applyPressurePreconditionerForValidation(
    const std::vector<double>& residual, const GpuPressureSettings& settings) {
    validatePressureSettings(settings);
    const std::size_t count = m_width * m_height;
    if (count == 0 || residual.size() != count) {
        throw std::invalid_argument("Pressure preconditioner validation vector size is invalid.");
    }
    if (m_multigridLevels.empty() || m_activeMultigridLevelCount == 0 ||
        m_hasIsolatedFluidCell || m_pureNeumannComponentCount > 1) {
        throw std::logic_error("Pressure preconditioner validation topology is unsupported.");
    }
    glNamedBufferSubData(m_pcgResidualBuffer, 0, checkedBytes(count, sizeof(double)),
        residual.data());
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    applyMultigridPreconditioner(
        m_pcgResidualBuffer, m_pcgPreconditionedBuffer, settings);
    std::vector<double> result(count);
    glGetNamedBufferSubData(m_pcgPreconditionedBuffer, 0,
        checkedBytes(count, sizeof(double)), result.data());
    return result;
}

std::vector<double> GpuMacGrid::readStoredPressureResidualField() const {
    if (m_width == 0) throw std::logic_error("GPU MAC layout is unavailable.");
    std::vector<std::array<double, 2>> packed(m_width * m_height);
    glGetNamedBufferSubData(m_residualBuffer, 0,
        checkedBytes(packed.size(), sizeof(packed.front())), packed.data());
    std::vector<double> result(packed.size());
    for (std::size_t i = 0; i < packed.size(); ++i) result[i] = packed[i][0];
    return result;
}

GpuProjectionReport GpuMacGrid::project(const GpuPressureSettings& settings) {
    GpuProjectionReport report;
    dispatchDivergence();
    report.divergenceGpuMilliseconds = finishTimedQuery();
    double divergenceReadbackMilliseconds = 0.0;
    std::size_t divergenceBytesRead = 0;
    report.divergenceBefore = computeDivergenceStatisticsGpu(
        report.divergenceGpuMilliseconds,
        divergenceReadbackMilliseconds,
        divergenceBytesRead);
    report.pressure = solvePressure(settings);
    if (report.pressure.failureReason != GpuPressureFailureReason::none ||
        !report.pressure.converged) {
        report.divergenceAfter = report.divergenceBefore;
        return report;
    }

    std::vector<float> uBefore;
    std::vector<float> vBefore;
    if (settings.collectVelocityChange) {
        uBefore = readHorizontalVelocity();
        vBefore = readVerticalVelocity();
    }

    bindCommonBuffers();
    m_projectionProgram.use();
    glUniform2ui(m_projectionProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_projectionProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1f(m_projectionProgram.uniformLocation("velocityScale"), settings.timeStep / settings.density);
    beginTimedQuery();
    glDispatchCompute(static_cast<GLuint>((m_width + 16) / 16),
        static_cast<GLuint>((m_height + 16) / 16), 1);
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    report.projectionGpuMilliseconds = finishTimedQuery();
    if (settings.collectVelocityChange) {
        const auto uAfter = readHorizontalVelocity();
        const auto vAfter = readVerticalVelocity();
        for (std::size_t i = 0; i < uBefore.size(); ++i)
            report.maximumVelocityChange = std::max(
                report.maximumVelocityChange, std::abs(uAfter[i] - uBefore[i]));
        for (std::size_t i = 0; i < vBefore.size(); ++i)
            report.maximumVelocityChange = std::max(
                report.maximumVelocityChange, std::abs(vAfter[i] - vBefore[i]));
    }

    dispatchDivergence();
    report.divergenceGpuMilliseconds += finishTimedQuery();
    report.divergenceAfter = computeDivergenceStatisticsGpu(
        report.divergenceGpuMilliseconds,
        divergenceReadbackMilliseconds,
        divergenceBytesRead);
    return report;
}

GpuVelocityExtrema GpuMacGrid::velocityExtrema() {
    if (m_width == 0 || m_height == 0) {
        throw std::logic_error("Upload a GPU MAC layout before velocity reduction.");
    }
    const std::size_t uCount = (m_width + 1) * m_height;
    const std::size_t vCount = m_width * (m_height + 1);
    const std::size_t elementCount = std::max(uCount, vCount);
    const std::size_t partialCount = (elementCount + 255) / 256;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_uBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_vBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_uValidBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_vValidBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_reductionBufferA);
    m_reduceVelocityProgram.use();
    glUniform1ui(m_reduceVelocityProgram.uniformLocation("uCount"), static_cast<GLuint>(uCount));
    glUniform1ui(m_reduceVelocityProgram.uniformLocation("vCount"), static_cast<GLuint>(vCount));
    beginTimedQuery();
    glDispatchCompute(static_cast<GLuint>(partialCount), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    GLuint input = m_reductionBufferA;
    std::size_t count = partialCount;
    while (count > 1) {
        const GLuint output = input == m_reductionBufferA
            ? m_reductionBufferB : m_reductionBufferA;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, input);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, output);
        m_reduceVelocityPartialProgram.use();
        glUniform1ui(m_reduceVelocityPartialProgram.uniformLocation("elementCount"),
            static_cast<GLuint>(count));
        const std::size_t outputCount = (count + 255) / 256;
        glDispatchCompute(static_cast<GLuint>(outputCount), 1, 1);
        ++m_pressureDispatchCount;
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        input = output;
        count = outputCount;
    }
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    double values[4]{};
    const auto readbackStart = Clock::now();
    glGetNamedBufferSubData(input, 0, sizeof(values), values);
    GpuVelocityExtrema result;
    result.readbackMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - readbackStart).count();
    result.gpuMilliseconds = finishTimedQuery();
    result.bytesRead = sizeof(values);
    result.maximumAbsoluteU = static_cast<float>(values[0]);
    result.maximumAbsoluteV = static_cast<float>(values[1]);
    result.maximumUIndex = static_cast<std::size_t>(values[2]);
    result.maximumVIndex = static_cast<std::size_t>(values[3]);
    checkOpenGlError("GPU velocity reduction");
    return result;
}

void GpuMacGrid::dispatchVisualization(GLuint texture, std::size_t mode) {
    if (texture == 0 || mode > 5) {
        throw std::invalid_argument("Invalid GPU visualization target or mode.");
    }
    if (m_committedRenderStateEnabled && m_committedRenderStateValid) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_committedRenderUBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_committedRenderVBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_committedRenderPressureBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_committedRenderDivergenceBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_committedRenderCellTypeBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_committedRenderParticleCellCountBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6,
            m_committedRenderClassificationDiagnosticBuffer);
    } else {
        bindCommonBuffers();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_particleCellCountBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_classificationDiagnosticBuffer);
    }
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    m_visualizationProgram.use();
    glUniform2ui(m_visualizationProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform1ui(m_visualizationProgram.uniformLocation("mode"), static_cast<GLuint>(mode));
    glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
        static_cast<GLuint>((m_height + 15) / 16), 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
        GL_SHADER_STORAGE_BARRIER_BIT);
    checkOpenGlError("GPU visualization dispatch");
}

GLuint GpuMacGrid::particleBufferHandle() const noexcept {
    return m_committedRenderStateEnabled && m_committedRenderStateValid
        ? m_committedRenderParticleBuffer : m_particleBuffer;
}
std::size_t GpuMacGrid::particleCount() const noexcept {
    return m_committedRenderStateEnabled && m_committedRenderStateValid
        ? m_committedRenderParticleCount : m_particleCount;
}

std::size_t GpuMacGrid::particleCapacity() const noexcept { return m_particleCapacity; }
void GpuMacGrid::setSynchronousTimingEnabled(bool enabled) noexcept {
    m_synchronousTimingEnabled = enabled;
}

double GpuMacGrid::advectVelocity(float timeStep) {
    if (!std::isfinite(timeStep) || timeStep < 0.0f) {
        throw std::invalid_argument("GPU advection timestep must be finite and non-negative.");
    }
    bindCommonBuffers();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_uScratchBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_vScratchBuffer);
    m_advectionProgram.use();
    glUniform2ui(m_advectionProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_advectionProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1f(m_advectionProgram.uniformLocation("timeStep"), timeStep);
    beginTimedQuery();
    glDispatchCompute(static_cast<GLuint>((m_width + 16) / 16),
        static_cast<GLuint>((m_height + 16) / 16), 1);
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    const double milliseconds = finishTimedQuery();
    std::swap(m_uBuffer, m_uScratchBuffer);
    std::swap(m_vBuffer, m_vScratchBuffer);
    checkOpenGlError("GPU RK2 velocity advection");
    return milliseconds;
}

double GpuMacGrid::applyAccelerationAndBoundaries(Vector2 acceleration, float timeStep) {
    if (!std::isfinite(acceleration.x) || !std::isfinite(acceleration.y) ||
        !std::isfinite(timeStep) || timeStep < 0.0f) {
        throw std::invalid_argument("GPU force parameters must be finite.");
    }
    bindCommonBuffers();
    m_forceBoundaryProgram.use();
    glUniform2ui(m_forceBoundaryProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_forceBoundaryProgram.uniformLocation("velocityDelta"),
        acceleration.x * timeStep, acceleration.y * timeStep);
    beginTimedQuery();
    glDispatchCompute(static_cast<GLuint>((m_width + 16) / 16),
        static_cast<GLuint>((m_height + 16) / 16), 1);
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    const double milliseconds = finishTimedQuery();
    checkOpenGlError("GPU force and boundary pass");
    return milliseconds;
}

GpuStepReport GpuMacGrid::stepFixedClassification(
    float timeStep,
    Vector2 acceleration,
    const GpuPressureSettings& pressureSettings) {
    const auto start = Clock::now();
    GpuStepReport report;
    report.advectionGpuMilliseconds = advectVelocity(timeStep);
    report.forceBoundaryGpuMilliseconds = applyAccelerationAndBoundaries(acceleration, timeStep);
    GpuPressureSettings adjusted = pressureSettings;
    adjusted.timeStep = timeStep;
    report.projection = project(adjusted);
    report.totalCpuSubmissionMilliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return report;
}

std::vector<float> GpuMacGrid::readHorizontalVelocity() const {
    std::vector<float> values((m_width + 1) * m_height);
    glGetNamedBufferSubData(m_uBuffer, 0, checkedBytes(values.size(), sizeof(float)), values.data());
    return values;
}

std::vector<float> GpuMacGrid::readVerticalVelocity() const {
    std::vector<float> values(m_width * (m_height + 1));
    glGetNamedBufferSubData(m_vBuffer, 0, checkedBytes(values.size(), sizeof(float)), values.data());
    return values;
}

std::vector<float> GpuMacGrid::readPressure() const {
    std::vector<double> preciseValues(m_width * m_height);
    glGetNamedBufferSubData(m_pressureBuffer, 0,
        checkedBytes(preciseValues.size(), sizeof(double)), preciseValues.data());
    std::vector<float> values(preciseValues.size());
    std::transform(preciseValues.begin(), preciseValues.end(), values.begin(),
        [](double value) { return static_cast<float>(value); });
    return values;
}

std::vector<double> GpuMacGrid::readResidualField() const {
    std::vector<double> values(m_width * m_height * 2);
    glGetNamedBufferSubData(m_residualBuffer, 0,
        checkedBytes(values.size(), sizeof(double)), values.data());
    return values;
}

std::vector<CellType> GpuMacGrid::readCellTypes() const {
    std::vector<GLuint> encoded(m_width * m_height);
    glGetNamedBufferSubData(m_cellTypeBuffer, 0,
        checkedBytes(encoded.size(), sizeof(GLuint)), encoded.data());
    std::vector<CellType> values(encoded.size());
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] > static_cast<GLuint>(CellType::solid)) {
            throw std::runtime_error("GPU cell classification contains an invalid value.");
        }
        values[i] = static_cast<CellType>(encoded[i]);
    }
    return values;
}

std::vector<CellType> GpuMacGrid::readPreviousCellTypes() const {
    std::vector<GLuint> encoded(m_width * m_height);
    glGetNamedBufferSubData(m_previousCellTypeBuffer, 0,
        checkedBytes(encoded.size(), sizeof(GLuint)), encoded.data());
    std::vector<CellType> values(encoded.size());
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] > static_cast<GLuint>(CellType::solid)) {
            throw std::runtime_error("GPU previous cell classification contains an invalid value.");
        }
        values[i] = static_cast<CellType>(encoded[i]);
    }
    return values;
}

GpuPressureOperatorSnapshot GpuMacGrid::readPressureOperatorLevel(
    std::size_t levelIndex) const {
    if (levelIndex >= m_activeMultigridLevelCount || levelIndex >= m_multigridLevels.size()) {
        throw std::out_of_range("GPU pressure operator level is not active.");
    }
    const MultigridLevel& level = m_multigridLevels[levelIndex];
    const std::size_t count = level.width * level.height;
    GpuPressureOperatorSnapshot snapshot;
    snapshot.width = level.width;
    snapshot.height = level.height;
    snapshot.diagonal.resize(count);
    snapshot.weights.resize(count);
    snapshot.active.resize(count);
    snapshot.component.resize(count);
    glGetNamedBufferSubData(level.diagonal, 0,
        checkedBytes(count, sizeof(double)), snapshot.diagonal.data());
    glGetNamedBufferSubData(level.weights, 0,
        checkedBytes(count, sizeof(std::array<double, 4>)), snapshot.weights.data());
    glGetNamedBufferSubData(level.active, 0,
        checkedBytes(count, sizeof(GLuint)), snapshot.active.data());
    glGetNamedBufferSubData(level.component, 0,
        checkedBytes(count, sizeof(GLuint)), snapshot.component.data());
    return snapshot;
}

std::vector<MarkerParticle> GpuMacGrid::readMarkerParticles() const {
    std::vector<std::array<float, 4>> encoded(m_particleCount);
    if (!encoded.empty()) {
        glGetNamedBufferSubData(m_particleBuffer, 0,
            checkedBytes(encoded.size(), sizeof(std::array<float, 4>)), encoded.data());
    }
    std::vector<MarkerParticle> particles(encoded.size());
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        particles[i] = {{encoded[i][0], encoded[i][1]}, encoded[i][2] >= 0.5f};
    }
    return particles;
}

std::vector<GLuint> GpuMacGrid::readParticleCellCounts() const {
    std::vector<GLuint> values(m_width * m_height);
    glGetNamedBufferSubData(m_particleCellCountBuffer, 0,
        checkedBytes(values.size(), sizeof(GLuint)), values.data());
    return values;
}

std::vector<GLuint> GpuMacGrid::readClassificationDiagnostics() const {
    std::vector<GLuint> values(m_width * m_height);
    glGetNamedBufferSubData(m_classificationDiagnosticBuffer, 0,
        checkedBytes(values.size(), sizeof(GLuint)), values.data());
    return values;
}

std::vector<GLuint> GpuMacGrid::readHorizontalValidity() const {
    std::vector<GLuint> values((m_width + 1) * m_height);
    glGetNamedBufferSubData(m_uValidBuffer, 0,
        checkedBytes(values.size(), sizeof(GLuint)), values.data());
    return values;
}

std::vector<GLuint> GpuMacGrid::readVerticalValidity() const {
    std::vector<GLuint> values(m_width * (m_height + 1));
    glGetNamedBufferSubData(m_vValidBuffer, 0,
        checkedBytes(values.size(), sizeof(GLuint)), values.data());
    return values;
}

void GpuMacGrid::pollAsynchronousStepTimings() {
    if (m_synchronousTimingEnabled) return;
    for (std::size_t i = 0; i < asynchronousQueryCount; ++i) {
        if (!m_asynchronousStepQueryPending[i]) continue;
        GLint available = GL_FALSE;
        glGetQueryObjectiv(m_asynchronousStepQueries[i], GL_QUERY_RESULT_AVAILABLE, &available);
        if (available != GL_TRUE) continue;
        GLuint64 nanoseconds = 0;
        glGetQueryObjectui64v(m_asynchronousStepQueries[i], GL_QUERY_RESULT, &nanoseconds);
        m_asynchronousStepQueryPending[i] = false;
        m_lastAsynchronousStepTiming.milliseconds =
            static_cast<double>(nanoseconds) / 1.0e6;
        m_lastAsynchronousStepTiming.latencySteps = static_cast<std::size_t>(
            m_asynchronousStepSerial - m_asynchronousStepQuerySerial[i] + 1);
        m_lastAsynchronousStepTiming.available = true;
    }
}

void GpuMacGrid::beginAsynchronousStepTiming() {
    if (m_synchronousTimingEnabled) return;
    pollAsynchronousStepTimings();
    const std::size_t slot = m_asynchronousStepQueryCursor;
    if (m_asynchronousStepQueryPending[slot]) {
        m_asynchronousStepQueryActive = false;
        return;
    }
    glBeginQuery(GL_TIME_ELAPSED, m_asynchronousStepQueries[slot]);
    m_asynchronousStepQueryActive = true;
}

void GpuMacGrid::endAsynchronousStepTiming() {
    if (m_synchronousTimingEnabled || !m_asynchronousStepQueryActive) return;
    glEndQuery(GL_TIME_ELAPSED);
    const std::size_t slot = m_asynchronousStepQueryCursor;
    m_asynchronousStepQueryPending[slot] = true;
    m_asynchronousStepQuerySerial[slot] = ++m_asynchronousStepSerial;
    m_asynchronousStepQueryCursor = (slot + 1) % asynchronousQueryCount;
    m_asynchronousStepQueryActive = false;
}

GpuAsyncTimingSample GpuMacGrid::asynchronousStepTiming() const noexcept {
    return m_lastAsynchronousStepTiming;
}

double GpuMacGrid::measureExperimentalLevelSetArea(GLuint phiBuffer) {
    const GLuint zero = 0u;
    glClearNamedBufferData(m_levelSetAreaBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, phiBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_solidMaskBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_levelSetAreaBuffer);
    m_levelSetAreaProgram.use();
    glUniform2ui(m_levelSetAreaProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_levelSetAreaProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
        static_cast<GLuint>((m_height + 15) / 16), 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    std::array<GLuint, 4> area{};
    glGetNamedBufferSubData(m_levelSetAreaBuffer, 0,
        static_cast<GLsizeiptr>(sizeof(area)), area.data());
    return static_cast<double>(area[0]) / 65535.0 *
        static_cast<double>(m_dx) * static_cast<double>(m_dy);
}

void GpuMacGrid::initializeExperimentalLevelSet(
    const LevelSetField& field,
    const GpuLevelSetSettings& settings) {
    if (field.width != m_width || field.height != m_height ||
        field.values.size() != m_width * m_height ||
        field.confidence.size() != m_width * m_height ||
        !std::isfinite(settings.thetaMinimum) || settings.thetaMinimum <= 0.0f ||
        settings.thetaMinimum > 1.0f || !std::isfinite(settings.minimumConfidence) ||
        settings.minimumConfidence < 0.0f || settings.minimumConfidence > 1.0f ||
        settings.redistanceInterval == 0 || settings.diagnosticReadbackInterval == 0 ||
        settings.velocityExtensionLayers == 0 ||
        !std::isfinite(settings.redistancePhysicalInterval) ||
        settings.redistancePhysicalInterval < 0.0) {
        throw std::invalid_argument("Invalid experimental GPU level-set initialization.");
    }
    std::vector<float> phi(field.values.size());
    std::vector<GLuint> confidence(field.confidence.size());
    for (std::size_t i = 0; i < phi.size(); ++i) {
        phi[i] = static_cast<float>(field.values[i]);
        confidence[i] = field.confidence[i] == 0 ? 0u : 1u;
        if (!std::isfinite(phi[i])) {
            throw std::invalid_argument("Experimental GPU level-set contains a non-finite value.");
        }
    }
    m_levelSetSettings = settings;
    if (!(m_levelSetSettings.narrowBandWidth > 0.0f)) {
        m_levelSetSettings.narrowBandWidth = 6.0f * std::max(m_dx, m_dy);
    }
    glNamedBufferSubData(m_levelSetPhiBuffer, 0,
        checkedBytes(phi.size(), sizeof(float)), phi.data());
    glNamedBufferSubData(m_levelSetConfidenceBuffer, 0,
        checkedBytes(confidence.size(), sizeof(GLuint)), confidence.data());
    glCopyNamedBufferSubData(m_levelSetPhiBuffer, m_levelSetPhiScratchBuffer, 0, 0,
        checkedBytes(phi.size(), sizeof(float)));
    glCopyNamedBufferSubData(m_levelSetPhiBuffer, m_levelSetPhiAuxBuffer, 0, 0,
        checkedBytes(phi.size(), sizeof(float)));
    glCopyNamedBufferSubData(m_levelSetConfidenceBuffer,
        m_levelSetConfidenceScratchBuffer, 0, 0,
        checkedBytes(confidence.size(), sizeof(GLuint)));
    glCopyNamedBufferSubData(m_levelSetConfidenceBuffer,
        m_levelSetConfidenceAuxBuffer, 0, 0,
        checkedBytes(confidence.size(), sizeof(GLuint)));
    const float one = 1.0f;
    const GLuint unused = static_cast<GLuint>(GpuLevelSetThetaState::unused);
    glClearNamedBufferData(m_levelSetHorizontalThetaBuffer, GL_R32F, GL_RED, GL_FLOAT, &one);
    glClearNamedBufferData(m_levelSetVerticalThetaBuffer, GL_R32F, GL_RED, GL_FLOAT, &one);
    glClearNamedBufferData(m_levelSetPreviousHorizontalThetaBuffer,
        GL_R32F, GL_RED, GL_FLOAT, &one);
    glClearNamedBufferData(m_levelSetPreviousVerticalThetaBuffer,
        GL_R32F, GL_RED, GL_FLOAT, &one);
    glClearNamedBufferData(m_levelSetHorizontalThetaStateBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &unused);
    glClearNamedBufferData(m_levelSetVerticalThetaStateBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &unused);
    const GLuint zero = 0u;
    glClearNamedBufferData(m_levelSetDiagnosticBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    m_levelSetReport = {};
    m_levelSetReport.initialized = true;
    m_levelSetReport.candidateValid = true;
    m_levelSetReport.geometryVersion = std::max<std::uint64_t>(field.sourceVersion, 1u);
    m_levelSetReport.thetaVersion = 1;
    m_levelSetAcceptedSubsteps = 0;
    if (m_levelSetSettings.stageAreaAuditEnabled) {
        m_levelSetReport.initialArea = measureExperimentalLevelSetArea(m_levelSetPhiBuffer);
        m_levelSetReport.areaBeforeAdvection = m_levelSetReport.initialArea;
        m_levelSetReport.areaAfterAdvection = m_levelSetReport.initialArea;
        m_levelSetReport.areaAfterRedistance = m_levelSetReport.initialArea;
    }
    static_cast<void>(updateExperimentalLevelSetDiagnostics());
    checkOpenGlError("Experimental GPU level-set initialization");
}

GpuLevelSetReport GpuMacGrid::advectExperimentalLevelSet(float timeStep) {
    if (!m_levelSetReport.initialized || !m_candidateStateSaved ||
        !std::isfinite(timeStep) || timeStep <= 0.0f) {
        throw std::logic_error("Experimental GPU level-set advection requires an active candidate.");
    }
    const auto start = Clock::now();
    const GLuint zero = 0u;
    glClearNamedBufferData(m_levelSetDiagnosticBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    GLuint advectionUBuffer = m_uBuffer;
    GLuint advectionVBuffer = m_vBuffer;
    if (m_levelSetSettings.velocityMode == GpuLevelSetVelocityMode::normalExtended) {
        const auto extensionStart = Clock::now();
        const GLuint groupsX = static_cast<GLuint>((m_width + 16) / 16);
        const GLuint groupsY = static_cast<GLuint>((m_height + 16) / 16);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_uBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_vBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_cellTypeBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_levelSetPhiBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_levelSetConfidenceBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_levelSetUBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_levelSetVBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_levelSetUValidBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_levelSetVValidBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_levelSetDiagnosticBuffer);
        m_levelSetVelocityExtensionInitializeProgram.use();
        glUniform2ui(m_levelSetVelocityExtensionInitializeProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
        glDispatchCompute(groupsX, groupsY, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        GLuint inputU = m_levelSetUBuffer;
        GLuint inputV = m_levelSetVBuffer;
        GLuint inputUValid = m_levelSetUValidBuffer;
        GLuint inputVValid = m_levelSetVValidBuffer;
        GLuint outputU = m_levelSetUScratchBuffer;
        GLuint outputV = m_levelSetVScratchBuffer;
        GLuint outputUValid = m_levelSetUValidScratchBuffer;
        GLuint outputVValid = m_levelSetVValidScratchBuffer;
        for (std::size_t layer = 0; layer < m_levelSetSettings.velocityExtensionLayers;
             ++layer) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputU);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, inputV);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, inputUValid);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, inputVValid);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_levelSetPhiBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_levelSetConfidenceBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, outputU);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, outputV);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, outputUValid);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, outputVValid);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, m_levelSetDiagnosticBuffer);
            m_levelSetVelocityExtensionLayerProgram.use();
            glUniform2ui(m_levelSetVelocityExtensionLayerProgram.uniformLocation("gridSize"),
                static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
            glUniform2f(m_levelSetVelocityExtensionLayerProgram.uniformLocation("cellSize"),
                m_dx, m_dy);
            glUniform1f(m_levelSetVelocityExtensionLayerProgram.uniformLocation("bandWidth"),
                m_levelSetSettings.narrowBandWidth);
            glUniform1ui(m_levelSetVelocityExtensionLayerProgram.uniformLocation("finalLayer"),
                layer + 1 == m_levelSetSettings.velocityExtensionLayers ? 1u : 0u);
            glDispatchCompute(groupsX, groupsY, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            std::swap(inputU, outputU);
            std::swap(inputV, outputV);
            std::swap(inputUValid, outputUValid);
            std::swap(inputVValid, outputVValid);
        }
        advectionUBuffer = inputU;
        advectionVBuffer = inputV;
        m_levelSetReport.velocityExtensionCpuMilliseconds +=
            std::chrono::duration<double, std::milli>(Clock::now() - extensionStart).count();
    }
    if (m_levelSetSettings.stageAreaAuditEnabled) {
        m_levelSetReport.areaBeforeAdvection =
            measureExperimentalLevelSetArea(m_levelSetPhiBuffer);
    }
    const auto dispatchAdvection = [&](GLuint sourcePhi, GLuint sourceConfidence,
                                        GLuint targetPhi, GLuint targetConfidence,
                                        float step, bool diagnostics) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, advectionUBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, advectionVBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_cellTypeBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, sourcePhi);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, sourceConfidence);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, targetPhi);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, targetConfidence);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_levelSetDiagnosticBuffer);
        m_levelSetAdvectionProgram.use();
        glUniform2ui(m_levelSetAdvectionProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
        glUniform2f(m_levelSetAdvectionProgram.uniformLocation("cellSize"), m_dx, m_dy);
        glUniform1f(m_levelSetAdvectionProgram.uniformLocation("timeStep"), step);
        glUniform1ui(m_levelSetAdvectionProgram.uniformLocation("writeDiagnostics"),
            diagnostics ? 1u : 0u);
        glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
            static_cast<GLuint>((m_height + 15) / 16), 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    dispatchAdvection(m_levelSetPhiBuffer, m_levelSetConfidenceBuffer,
        m_levelSetPhiScratchBuffer, m_levelSetConfidenceScratchBuffer, timeStep, true);
    if (m_levelSetSettings.advectionMethod ==
        GpuLevelSetAdvectionMethod::limitedMacCormack) {
        dispatchAdvection(m_levelSetPhiScratchBuffer, m_levelSetConfidenceScratchBuffer,
            m_levelSetPhiAuxBuffer, m_levelSetConfidenceAuxBuffer, -timeStep, false);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, advectionUBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, advectionVBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_cellTypeBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_candidateLevelSetPhiBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_candidateLevelSetConfidenceBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_levelSetPhiScratchBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_levelSetConfidenceScratchBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_levelSetPhiAuxBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_levelSetConfidenceAuxBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_levelSetPhiBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, m_levelSetConfidenceBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, m_levelSetDiagnosticBuffer);
        m_levelSetCorrectionProgram.use();
        glUniform2ui(m_levelSetCorrectionProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
        glUniform2f(m_levelSetCorrectionProgram.uniformLocation("cellSize"), m_dx, m_dy);
        glUniform1f(m_levelSetCorrectionProgram.uniformLocation("timeStep"), timeStep);
        glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
            static_cast<GLuint>((m_height + 15) / 16), 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    } else {
        glCopyNamedBufferSubData(m_levelSetPhiScratchBuffer, m_levelSetPhiBuffer, 0, 0,
            checkedBytes(m_width * m_height, sizeof(float)));
        glCopyNamedBufferSubData(m_levelSetConfidenceScratchBuffer,
            m_levelSetConfidenceBuffer, 0, 0,
            checkedBytes(m_width * m_height, sizeof(GLuint)));
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    }
    if (m_levelSetSettings.stageAreaAuditEnabled) {
        m_levelSetReport.areaAfterAdvection =
            measureExperimentalLevelSetArea(m_levelSetPhiBuffer);
        m_levelSetReport.lastAdvectionAreaChange =
            m_levelSetReport.areaAfterAdvection - m_levelSetReport.areaBeforeAdvection;
        m_levelSetReport.cumulativeAdvectionAreaChange +=
            m_levelSetReport.lastAdvectionAreaChange;
    }
    m_levelSetReport.advectionCpuMilliseconds +=
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    ++m_levelSetReport.advectionCount;
    ++m_levelSetReport.geometryVersion;
    m_levelSetReport.physicalTime += timeStep;

    const bool redistanceByPhysicalTime =
        m_levelSetSettings.redistancePhysicalInterval > 0.0 &&
        m_levelSetReport.physicalTime - m_levelSetReport.lastRedistancePhysicalTime + 1.0e-12 >=
            m_levelSetSettings.redistancePhysicalInterval;
    const bool redistanceByAcceptedCount =
        m_levelSetSettings.redistancePhysicalInterval <= 0.0 &&
        (m_levelSetAcceptedSubsteps + 1) % m_levelSetSettings.redistanceInterval == 0;
    if (redistanceByPhysicalTime || redistanceByAcceptedCount) {
        const auto redistanceStart = Clock::now();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_levelSetPhiBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_levelSetConfidenceBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_levelSetPhiScratchBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_levelSetConfidenceScratchBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_levelSetNarrowBandBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_solidMaskBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_levelSetDiagnosticBuffer);
        m_levelSetRedistanceProgram.use();
        glUniform2ui(m_levelSetRedistanceProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
        glUniform2f(m_levelSetRedistanceProgram.uniformLocation("cellSize"), m_dx, m_dy);
        glUniform1f(m_levelSetRedistanceProgram.uniformLocation("bandWidth"),
            m_levelSetSettings.narrowBandWidth);
        glUniform1ui(m_levelSetRedistanceProgram.uniformLocation("preserveInterface"),
            m_levelSetSettings.redistanceMethod ==
                GpuLevelSetRedistanceMethod::interfacePreserving ? 1u : 0u);
        glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
            static_cast<GLuint>((m_height + 15) / 16), 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glCopyNamedBufferSubData(m_levelSetPhiScratchBuffer, m_levelSetPhiBuffer, 0, 0,
            checkedBytes(m_width * m_height, sizeof(float)));
        glCopyNamedBufferSubData(m_levelSetConfidenceScratchBuffer,
            m_levelSetConfidenceBuffer, 0, 0,
            checkedBytes(m_width * m_height, sizeof(GLuint)));
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
        m_levelSetReport.redistanceCpuMilliseconds +=
            std::chrono::duration<double, std::milli>(Clock::now() - redistanceStart).count();
        m_levelSetReport.lastRedistancePhysicalTime = m_levelSetReport.physicalTime;
        ++m_levelSetReport.redistanceCount;
    }
    if (m_levelSetSettings.stageAreaAuditEnabled) {
        m_levelSetReport.areaAfterRedistance =
            measureExperimentalLevelSetArea(m_levelSetPhiBuffer);
        m_levelSetReport.lastRedistanceAreaChange =
            m_levelSetReport.areaAfterRedistance - m_levelSetReport.areaAfterAdvection;
        m_levelSetReport.cumulativeRedistanceAreaChange +=
            m_levelSetReport.lastRedistanceAreaChange;
        ++m_levelSetReport.stageAreaAuditCount;
    }
    m_levelSetReport.candidateValid = true;
    checkOpenGlError("Experimental GPU level-set advance");
    return m_levelSetReport;
}

GpuLevelSetReport GpuMacGrid::updateExperimentalLevelSetDiagnostics() {
    if (!m_levelSetReport.initialized) return m_levelSetReport;
    const auto start = Clock::now();
    glCopyNamedBufferSubData(m_levelSetHorizontalThetaBuffer,
        m_levelSetPreviousHorizontalThetaBuffer, 0, 0,
        checkedBytes((m_width + 1) * m_height, sizeof(float)));
    glCopyNamedBufferSubData(m_levelSetVerticalThetaBuffer,
        m_levelSetPreviousVerticalThetaBuffer, 0, 0,
        checkedBytes(m_width * (m_height + 1), sizeof(float)));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_levelSetPhiBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_levelSetConfidenceBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_previousCellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_particleCellCountBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_levelSetHorizontalThetaBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_levelSetVerticalThetaBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_levelSetHorizontalThetaStateBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_levelSetVerticalThetaStateBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_levelSetPreviousHorizontalThetaBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, m_levelSetPreviousVerticalThetaBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, m_levelSetDiagnosticBuffer);
    m_levelSetThetaProgram.use();
    glUniform2ui(m_levelSetThetaProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform1f(m_levelSetThetaProgram.uniformLocation("thetaMinimum"),
        m_levelSetSettings.thetaMinimum);
    glUniform2f(m_levelSetThetaProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glDispatchCompute(static_cast<GLuint>((m_width + 16) / 16),
        static_cast<GLuint>((m_height + 16) / 16), 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_particleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_levelSetPhiBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_levelSetConfidenceBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_levelSetDiagnosticBuffer);
    m_levelSetParticleDiagnosticProgram.use();
    glUniform1ui(m_levelSetParticleDiagnosticProgram.uniformLocation("particleCount"),
        static_cast<GLuint>(m_particleCount));
    glUniform2ui(m_levelSetParticleDiagnosticProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_levelSetParticleDiagnosticProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1f(m_levelSetParticleDiagnosticProgram.uniformLocation("distanceTolerance"),
        std::max(m_dx, m_dy));
    if (m_particleCount != 0) {
        glDispatchCompute(static_cast<GLuint>((m_particleCount + 255) / 256), 1, 1);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    const bool useExtendedVelocity = m_levelSetSettings.velocityMode ==
        GpuLevelSetVelocityMode::normalExtended;
    const bool extensionEndsInScratch =
        (m_levelSetSettings.velocityExtensionLayers % 2) != 0;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, useExtendedVelocity
        ? (extensionEndsInScratch ? m_levelSetUScratchBuffer : m_levelSetUBuffer)
        : m_uBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, useExtendedVelocity
        ? (extensionEndsInScratch ? m_levelSetVScratchBuffer : m_levelSetVBuffer)
        : m_vBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_levelSetPhiBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_levelSetConfidenceBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_levelSetDiagnosticBuffer);
    m_levelSetVelocityDiagnosticProgram.use();
    glUniform2ui(m_levelSetVelocityDiagnosticProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_levelSetVelocityDiagnosticProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1f(m_levelSetVelocityDiagnosticProgram.uniformLocation("interfaceBandWidth"),
        2.0f * std::max(m_dx, m_dy));
    glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
        static_cast<GLuint>((m_height + 15) / 16), 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    ++m_levelSetReport.thetaVersion;
    const bool readback = m_levelSetReport.advectionCount == 0 ||
        (m_levelSetAcceptedSubsteps + 1) % m_levelSetSettings.diagnosticReadbackInterval == 0;
    if (readback) {
        const std::size_t elementCount = m_width * m_height;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_levelSetPhiBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_levelSetConfidenceBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_levelSetComponentLabelBufferA);
        m_levelSetComponentInitializeProgram.use();
        glUniform1ui(m_levelSetComponentInitializeProgram.uniformLocation("elementCount"),
            static_cast<GLuint>(elementCount));
        glDispatchCompute(static_cast<GLuint>((elementCount + 255) / 256), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        GLuint inputLabels = m_levelSetComponentLabelBufferA;
        GLuint outputLabels = m_levelSetComponentLabelBufferB;
        std::size_t propagationIterations = 0;
        for (std::size_t iteration = 0; iteration < m_width + m_height; ++iteration) {
            const GLuint zero = 0u;
            glClearNamedBufferData(m_levelSetComponentChangedBuffer,
                GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputLabels);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputLabels);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_levelSetComponentChangedBuffer);
            m_levelSetComponentPropagateProgram.use();
            glUniform2ui(m_levelSetComponentPropagateProgram.uniformLocation("gridSize"),
                static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
            glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
                static_cast<GLuint>((m_height + 15) / 16), 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
            std::swap(inputLabels, outputLabels);
            propagationIterations = iteration + 1;
            if ((iteration + 1) % 16 == 0 || iteration + 1 == m_width + m_height) {
                GLuint changed = 0u;
                glGetNamedBufferSubData(m_levelSetComponentChangedBuffer, 0,
                    static_cast<GLsizeiptr>(sizeof(changed)), &changed);
                if (changed == 0u) break;
            }
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputLabels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_levelSetDiagnosticBuffer);
        m_levelSetComponentSummaryProgram.use();
        glUniform1ui(m_levelSetComponentSummaryProgram.uniformLocation("elementCount"),
            static_cast<GLuint>(elementCount));
        glDispatchCompute(static_cast<GLuint>((elementCount + 255) / 256), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<GLuint, 40> diagnostics{};
        glGetNamedBufferSubData(m_levelSetDiagnosticBuffer, 0,
            static_cast<GLsizeiptr>(sizeof(diagnostics)), diagnostics.data());
        m_levelSetReport.invalidSampleCount = diagnostics[0];
        m_levelSetReport.invalidPhiCount = diagnostics[1];
        m_levelSetReport.lowConfidenceCellCount = diagnostics[2];
        m_levelSetReport.signConflictCount = diagnostics[3];
        m_levelSetReport.positivePhiParticleCount = diagnostics[4];
        m_levelSetReport.validThetaCount = diagnostics[5];
        m_levelSetReport.clampedThetaCount = diagnostics[6];
        m_levelSetReport.fallbackThetaCount = diagnostics[7];
        m_levelSetReport.classificationStableThetaChangeCount = diagnostics[8];
        m_levelSetReport.maximumThetaChange = std::bit_cast<float>(diagnostics[9]);
        m_levelSetReport.geometricArea = static_cast<double>(diagnostics[11]) /
            65535.0 * static_cast<double>(m_dx) * static_cast<double>(m_dy);
        m_levelSetReport.negativeCellCount = diagnostics[12];
        m_levelSetReport.negativeComponentCount = diagnostics[13];
        m_levelSetReport.topologyPropagationIterations = propagationIterations;
        m_levelSetReport.highOrderFallbackCount += diagnostics[14];
        m_levelSetReport.domainClampedBacktraceCount += diagnostics[15];
        m_levelSetReport.solidBacktraceCount += diagnostics[16];
        m_levelSetReport.lowConfidenceBacktraceCount += diagnostics[17];
        m_levelSetReport.highOrderNonFiniteCount += diagnostics[19];
        m_levelSetReport.limiterClampCount += diagnostics[20];
        m_levelSetReport.highOrderCorrectionCount += diagnostics[21];
        const auto meanDivergence = [&](std::size_t sum, std::size_t count) {
            return diagnostics[count] == 0u ? 0.0 :
                static_cast<double>(diagnostics[sum]) /
                (1000.0 * static_cast<double>(diagnostics[count]));
        };
        m_levelSetReport.fluidMeanAbsoluteDivergence = meanDivergence(22, 23);
        m_levelSetReport.interfaceMeanAbsoluteDivergence = meanDivergence(24, 25);
        m_levelSetReport.airBandMeanAbsoluteDivergence = meanDivergence(26, 27);
        m_levelSetReport.fluidMaximumAbsoluteDivergence =
            std::bit_cast<float>(diagnostics[28]);
        m_levelSetReport.interfaceMaximumAbsoluteDivergence =
            std::bit_cast<float>(diagnostics[29]);
        m_levelSetReport.airBandMaximumAbsoluteDivergence =
            std::bit_cast<float>(diagnostics[30]);
        m_levelSetReport.interfacePreservedCellCount += diagnostics[31];
        m_levelSetReport.velocityExtensionSeedCount += diagnostics[32];
        m_levelSetReport.velocityExtensionFilledCount += diagnostics[33];
        m_levelSetReport.velocityExtensionUnresolvedCount += diagnostics[34];
    }
    m_levelSetReport.thetaCpuMilliseconds +=
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    checkOpenGlError("Experimental GPU level-set diagnostics");
    return m_levelSetReport;
}

GpuLevelSetSnapshot GpuMacGrid::readExperimentalLevelSet() const {
    GpuLevelSetSnapshot result;
    if (!m_levelSetReport.initialized) return result;
    result.phi.resize(m_width * m_height);
    result.confidence.resize(m_width * m_height);
    result.horizontalTheta.resize((m_width + 1) * m_height);
    result.verticalTheta.resize(m_width * (m_height + 1));
    result.horizontalThetaState.resize((m_width + 1) * m_height);
    result.verticalThetaState.resize(m_width * (m_height + 1));
    const bool committed = committedRenderStateEnabled() &&
        m_committedRenderLevelSetReport.initialized;
    const GLuint phiBuffer = committed ? m_committedRenderLevelSetPhiBuffer : m_levelSetPhiBuffer;
    const GLuint confidenceBuffer = committed
        ? m_committedRenderLevelSetConfidenceBuffer : m_levelSetConfidenceBuffer;
    const GLuint horizontalThetaBuffer = committed
        ? m_committedRenderLevelSetHorizontalThetaBuffer : m_levelSetHorizontalThetaBuffer;
    const GLuint verticalThetaBuffer = committed
        ? m_committedRenderLevelSetVerticalThetaBuffer : m_levelSetVerticalThetaBuffer;
    const GLuint horizontalStateBuffer = committed
        ? m_committedRenderLevelSetHorizontalThetaStateBuffer : m_levelSetHorizontalThetaStateBuffer;
    const GLuint verticalStateBuffer = committed
        ? m_committedRenderLevelSetVerticalThetaStateBuffer : m_levelSetVerticalThetaStateBuffer;
    glGetNamedBufferSubData(phiBuffer, 0,
        checkedBytes(result.phi.size(), sizeof(float)), result.phi.data());
    glGetNamedBufferSubData(confidenceBuffer, 0,
        checkedBytes(result.confidence.size(), sizeof(GLuint)), result.confidence.data());
    glGetNamedBufferSubData(horizontalThetaBuffer, 0,
        checkedBytes(result.horizontalTheta.size(), sizeof(float)), result.horizontalTheta.data());
    glGetNamedBufferSubData(verticalThetaBuffer, 0,
        checkedBytes(result.verticalTheta.size(), sizeof(float)), result.verticalTheta.data());
    glGetNamedBufferSubData(horizontalStateBuffer, 0,
        checkedBytes(result.horizontalThetaState.size(), sizeof(GLuint)),
        result.horizontalThetaState.data());
    glGetNamedBufferSubData(verticalStateBuffer, 0,
        checkedBytes(result.verticalThetaState.size(), sizeof(GLuint)),
        result.verticalThetaState.data());
    result.report = committed ? m_committedRenderLevelSetReport : m_levelSetReport;
    return result;
}

bool GpuMacGrid::experimentalLevelSetEnabled() const noexcept {
    return m_levelSetReport.initialized;
}

const GpuLevelSetReport& GpuMacGrid::experimentalLevelSetReport() const noexcept {
    return m_levelSetReport;
}

void GpuMacGrid::saveCandidateState() {
    if (m_width == 0 || m_particleCapacity == 0) {
        throw std::logic_error("GPU candidate state requires initialized grid and particle buffers.");
    }
    if (m_candidateStateSaved) {
        throw std::logic_error("A GPU candidate state is already active.");
    }
    const GLsizeiptr uBytes = checkedBytes((m_width + 1) * m_height, sizeof(float));
    const GLsizeiptr vBytes = checkedBytes(m_width * (m_height + 1), sizeof(float));
    const GLsizeiptr pressureBytes = checkedBytes(m_width * m_height, sizeof(double));
    const GLsizeiptr typeBytes = checkedBytes(m_width * m_height, sizeof(GLuint));
    const GLsizeiptr particleBytes = m_particleCount == 0 ? 0 :
        checkedBytes(m_particleCount, sizeof(std::array<float, 4>));
    glCopyNamedBufferSubData(m_uBuffer, m_candidateUBuffer, 0, 0, uBytes);
    glCopyNamedBufferSubData(m_vBuffer, m_candidateVBuffer, 0, 0, vBytes);
    glCopyNamedBufferSubData(m_pressureBuffer, m_candidatePressureBuffer, 0, 0, pressureBytes);
    glCopyNamedBufferSubData(m_cellTypeBuffer, m_candidateCellTypeBuffer, 0, 0, typeBytes);
    glCopyNamedBufferSubData(
        m_previousCellTypeBuffer, m_candidatePreviousCellTypeBuffer, 0, 0, typeBytes);
    if (particleBytes != 0) {
        glCopyNamedBufferSubData(m_particleBuffer, m_candidateParticleBuffer, 0, 0, particleBytes);
    }
    glCopyNamedBufferSubData(
        m_particleCellCountBuffer, m_candidateParticleCellCountBuffer, 0, 0, typeBytes);
    glCopyNamedBufferSubData(m_classificationDiagnosticBuffer,
        m_candidateClassificationDiagnosticBuffer, 0, 0, typeBytes);
    glCopyNamedBufferSubData(
        m_classificationFlipBuffer, m_candidateClassificationFlipBuffer, 0, 0, typeBytes);
    if (m_levelSetReport.initialized) {
        const GLsizeiptr phiBytes = checkedBytes(m_width * m_height, sizeof(float));
        const GLsizeiptr uThetaBytes = checkedBytes((m_width + 1) * m_height, sizeof(float));
        const GLsizeiptr vThetaBytes = checkedBytes(m_width * (m_height + 1), sizeof(float));
        glCopyNamedBufferSubData(m_levelSetPhiBuffer, m_candidateLevelSetPhiBuffer,
            0, 0, phiBytes);
        glCopyNamedBufferSubData(m_levelSetConfidenceBuffer,
            m_candidateLevelSetConfidenceBuffer, 0, 0, typeBytes);
        glCopyNamedBufferSubData(m_levelSetHorizontalThetaBuffer,
            m_candidateLevelSetHorizontalThetaBuffer, 0, 0, uThetaBytes);
        glCopyNamedBufferSubData(m_levelSetVerticalThetaBuffer,
            m_candidateLevelSetVerticalThetaBuffer, 0, 0, vThetaBytes);
        glCopyNamedBufferSubData(m_levelSetHorizontalThetaStateBuffer,
            m_candidateLevelSetHorizontalThetaStateBuffer, 0, 0,
            checkedBytes((m_width + 1) * m_height, sizeof(GLuint)));
        glCopyNamedBufferSubData(m_levelSetVerticalThetaStateBuffer,
            m_candidateLevelSetVerticalThetaStateBuffer, 0, 0,
            checkedBytes(m_width * (m_height + 1), sizeof(GLuint)));
        m_candidateLevelSetReport = m_levelSetReport;
    }
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    m_candidateParticleCount = m_particleCount;
    m_candidateFluidCellCount = m_fluidCellCount;
    m_candidateConnectedComponentCount = m_connectedComponentCount;
    m_candidateActiveMultigridLevelCount = m_activeMultigridLevelCount;
    m_candidatePureNeumannComponentCount = m_pureNeumannComponentCount;
    m_candidateHasIsolatedFluidCell = m_hasIsolatedFluidCell;
    m_candidateClassificationVersion = m_classificationVersion;
    m_candidateStateSaved = true;
    checkOpenGlError("GPU candidate state save");
}

bool GpuMacGrid::restoreCandidateState() {
    if (!m_candidateStateSaved) {
        throw std::logic_error("No GPU candidate state is available to restore.");
    }
    const bool topologyChanged = m_classificationVersion != m_candidateClassificationVersion;
    const GLsizeiptr uBytes = checkedBytes((m_width + 1) * m_height, sizeof(float));
    const GLsizeiptr vBytes = checkedBytes(m_width * (m_height + 1), sizeof(float));
    const GLsizeiptr pressureBytes = checkedBytes(m_width * m_height, sizeof(double));
    const GLsizeiptr typeBytes = checkedBytes(m_width * m_height, sizeof(GLuint));
    const GLsizeiptr particleBytes = m_candidateParticleCount == 0 ? 0 :
        checkedBytes(m_candidateParticleCount, sizeof(std::array<float, 4>));
    glCopyNamedBufferSubData(m_candidateUBuffer, m_uBuffer, 0, 0, uBytes);
    glCopyNamedBufferSubData(m_candidateVBuffer, m_vBuffer, 0, 0, vBytes);
    glCopyNamedBufferSubData(m_candidatePressureBuffer, m_pressureBuffer, 0, 0, pressureBytes);
    glCopyNamedBufferSubData(m_candidateCellTypeBuffer, m_cellTypeBuffer, 0, 0, typeBytes);
    glCopyNamedBufferSubData(
        m_candidatePreviousCellTypeBuffer, m_previousCellTypeBuffer, 0, 0, typeBytes);
    if (particleBytes != 0) {
        glCopyNamedBufferSubData(m_candidateParticleBuffer, m_particleBuffer, 0, 0, particleBytes);
    }
    glCopyNamedBufferSubData(
        m_candidateParticleCellCountBuffer, m_particleCellCountBuffer, 0, 0, typeBytes);
    glCopyNamedBufferSubData(m_candidateClassificationDiagnosticBuffer,
        m_classificationDiagnosticBuffer, 0, 0, typeBytes);
    glCopyNamedBufferSubData(
        m_candidateClassificationFlipBuffer, m_classificationFlipBuffer, 0, 0, typeBytes);
    if (m_levelSetReport.initialized) {
        glCopyNamedBufferSubData(m_candidateLevelSetPhiBuffer, m_levelSetPhiBuffer,
            0, 0, checkedBytes(m_width * m_height, sizeof(float)));
        glCopyNamedBufferSubData(m_candidateLevelSetConfidenceBuffer,
            m_levelSetConfidenceBuffer, 0, 0, typeBytes);
        glCopyNamedBufferSubData(m_candidateLevelSetHorizontalThetaBuffer,
            m_levelSetHorizontalThetaBuffer, 0, 0,
            checkedBytes((m_width + 1) * m_height, sizeof(float)));
        glCopyNamedBufferSubData(m_candidateLevelSetVerticalThetaBuffer,
            m_levelSetVerticalThetaBuffer, 0, 0,
            checkedBytes(m_width * (m_height + 1), sizeof(float)));
        glCopyNamedBufferSubData(m_candidateLevelSetHorizontalThetaStateBuffer,
            m_levelSetHorizontalThetaStateBuffer, 0, 0,
            checkedBytes((m_width + 1) * m_height, sizeof(GLuint)));
        glCopyNamedBufferSubData(m_candidateLevelSetVerticalThetaStateBuffer,
            m_levelSetVerticalThetaStateBuffer, 0, 0,
            checkedBytes(m_width * (m_height + 1), sizeof(GLuint)));
        m_levelSetReport = m_candidateLevelSetReport;
    }
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    m_particleCount = m_candidateParticleCount;
    m_fluidCellCount = m_candidateFluidCellCount;
    m_connectedComponentCount = m_candidateConnectedComponentCount;
    m_activeMultigridLevelCount = m_candidateActiveMultigridLevelCount;
    m_pureNeumannComponentCount = m_candidatePureNeumannComponentCount;
    m_hasIsolatedFluidCell = m_candidateHasIsolatedFluidCell;
    if (topologyChanged) {
        ++m_classificationVersion;
        m_fineOperatorVersion = 0;
        m_multigridVersion = 0;
        m_dynamicPressureTopologyReady = false;
        m_dynamicPressureTopologyAnalysisFailed = false;
    }
    m_candidateStateSaved = false;
    checkOpenGlError("GPU candidate state restore");
    return topologyChanged;
}

void GpuMacGrid::commitCandidateState() noexcept {
    if (m_levelSetReport.initialized) ++m_levelSetAcceptedSubsteps;
    m_candidateStateSaved = false;
}

bool GpuMacGrid::hasCandidateState() const noexcept { return m_candidateStateSaved; }

void GpuMacGrid::captureCommittedRenderState() {
    if (m_width == 0 || m_particleCapacity == 0) {
        throw std::logic_error("GPU committed render state requires initialized resources.");
    }
    const GLsizeiptr uBytes = checkedBytes((m_width + 1) * m_height, sizeof(float));
    const GLsizeiptr vBytes = checkedBytes(m_width * (m_height + 1), sizeof(float));
    const GLsizeiptr pressureBytes = checkedBytes(m_width * m_height, sizeof(double));
    const GLsizeiptr cellFloatBytes = checkedBytes(m_width * m_height, sizeof(float));
    const GLsizeiptr cellUintBytes = checkedBytes(m_width * m_height, sizeof(GLuint));
    glCopyNamedBufferSubData(m_uBuffer, m_committedRenderUBuffer, 0, 0, uBytes);
    glCopyNamedBufferSubData(m_vBuffer, m_committedRenderVBuffer, 0, 0, vBytes);
    glCopyNamedBufferSubData(
        m_pressureBuffer, m_committedRenderPressureBuffer, 0, 0, pressureBytes);
    glCopyNamedBufferSubData(
        m_divergenceBuffer, m_committedRenderDivergenceBuffer, 0, 0, cellFloatBytes);
    glCopyNamedBufferSubData(
        m_cellTypeBuffer, m_committedRenderCellTypeBuffer, 0, 0, cellUintBytes);
    if (m_particleCount != 0) {
        glCopyNamedBufferSubData(m_particleBuffer, m_committedRenderParticleBuffer, 0, 0,
            checkedBytes(m_particleCount, sizeof(std::array<float, 4>)));
    }
    glCopyNamedBufferSubData(m_particleCellCountBuffer,
        m_committedRenderParticleCellCountBuffer, 0, 0, cellUintBytes);
    glCopyNamedBufferSubData(m_classificationDiagnosticBuffer,
        m_committedRenderClassificationDiagnosticBuffer, 0, 0, cellUintBytes);
    if (m_levelSetReport.initialized) {
        glCopyNamedBufferSubData(m_levelSetPhiBuffer,
            m_committedRenderLevelSetPhiBuffer, 0, 0, cellFloatBytes);
        glCopyNamedBufferSubData(m_levelSetConfidenceBuffer,
            m_committedRenderLevelSetConfidenceBuffer, 0, 0, cellUintBytes);
        glCopyNamedBufferSubData(m_levelSetHorizontalThetaBuffer,
            m_committedRenderLevelSetHorizontalThetaBuffer, 0, 0, uBytes);
        glCopyNamedBufferSubData(m_levelSetVerticalThetaBuffer,
            m_committedRenderLevelSetVerticalThetaBuffer, 0, 0, vBytes);
        glCopyNamedBufferSubData(m_levelSetHorizontalThetaStateBuffer,
            m_committedRenderLevelSetHorizontalThetaStateBuffer, 0, 0,
            checkedBytes((m_width + 1) * m_height, sizeof(GLuint)));
        glCopyNamedBufferSubData(m_levelSetVerticalThetaStateBuffer,
            m_committedRenderLevelSetVerticalThetaStateBuffer, 0, 0,
            checkedBytes(m_width * (m_height + 1), sizeof(GLuint)));
        m_committedRenderLevelSetReport = m_levelSetReport;
    }
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT |
        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
    m_committedRenderParticleCount = m_particleCount;
    m_committedRenderStateValid = true;
    checkOpenGlError("GPU committed render state capture");
}

void GpuMacGrid::setCommittedRenderStateEnabled(bool enabled) noexcept {
    m_committedRenderStateEnabled = enabled;
}

bool GpuMacGrid::committedRenderStateEnabled() const noexcept {
    return m_committedRenderStateEnabled && m_committedRenderStateValid;
}

double GpuMacGrid::lastUploadMilliseconds() const noexcept { return m_lastUploadMilliseconds; }
double GpuMacGrid::lastGpuMilliseconds() const noexcept { return m_lastGpuMilliseconds; }
std::size_t GpuMacGrid::width() const noexcept { return m_width; }
std::size_t GpuMacGrid::height() const noexcept { return m_height; }
float GpuMacGrid::cellWidth() const noexcept { return m_dx; }
float GpuMacGrid::cellHeight() const noexcept { return m_dy; }

void GpuMacGrid::ensureLayout(const MacGrid& grid) {
    if (grid.width() > static_cast<std::size_t>(std::numeric_limits<GLuint>::max()) ||
        grid.height() > static_cast<std::size_t>(std::numeric_limits<GLuint>::max())) {
        throw std::invalid_argument("GPU MAC grid dimensions exceed GLSL uint range.");
    }
    if (m_width == grid.width() && m_height == grid.height() &&
        m_dx == grid.cellWidth() && m_dy == grid.cellHeight()) return;
    m_width = grid.width();
    m_height = grid.height();
    m_dx = grid.cellWidth();
    m_dy = grid.cellHeight();
    const GLsizeiptr uBytes = checkedBytes((m_width + 1) * m_height, sizeof(float));
    const GLsizeiptr vBytes = checkedBytes(m_width * (m_height + 1), sizeof(float));
    const GLsizeiptr cellBytes = checkedBytes(m_width * m_height, sizeof(float));
    const GLsizeiptr pressureBytes = checkedBytes(m_width * m_height, sizeof(double));
    const std::size_t maximumElementCount = std::max({m_width * m_height,
        (m_width + 1) * m_height, m_width * (m_height + 1)});
    const std::size_t partialCount = (maximumElementCount + 255) / 256;
    const GLsizeiptr partialBytes = checkedBytes(partialCount * 4, sizeof(double));
    glNamedBufferData(m_uBuffer, uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_vBuffer, vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_uScratchBuffer, uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_vScratchBuffer, vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_pressureBuffer, pressureBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_pressureScratchBuffer, pressureBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateUBuffer, uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateVBuffer, vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidatePressureBuffer, pressureBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderUBuffer, uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderVBuffer, vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderPressureBuffer, pressureBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderDivergenceBuffer, cellBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_divergenceBuffer, cellBytes, nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_cellTypeBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_residualBuffer,
        checkedBytes(m_width * m_height * 2, sizeof(double)), nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_reductionBufferA, partialBytes, nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_reductionBufferB, partialBytes, nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_pcgResidualBuffer, pressureBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_pcgPreconditionedBuffer, pressureBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_pcgDirectionBuffer, pressureBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_pcgApBuffer, pressureBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_pcgScalarBuffer, checkedBytes(4, sizeof(double)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(
        m_pressureControlBuffer, checkedBytes(10, sizeof(double)), nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_solidMaskBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_previousCellTypeBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateCellTypeBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidatePreviousCellTypeBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderCellTypeBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_classificationCounterBuffer,
        checkedBytes(12, sizeof(GLuint)), nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_particleCellCountBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_classificationDiagnosticBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_classificationFlipBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateParticleCellCountBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateClassificationDiagnosticBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateClassificationFlipBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderParticleCellCountBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderClassificationDiagnosticBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetPhiBuffer, cellBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetPhiScratchBuffer, cellBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetPhiAuxBuffer, cellBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetConfidenceBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetConfidenceScratchBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetConfidenceAuxBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetNarrowBandBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetUBuffer, uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetVBuffer, vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetUScratchBuffer, uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetVScratchBuffer, vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetUValidBuffer,
        checkedBytes((m_width + 1) * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetVValidBuffer,
        checkedBytes(m_width * (m_height + 1), sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetUValidScratchBuffer,
        checkedBytes((m_width + 1) * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetVValidScratchBuffer,
        checkedBytes(m_width * (m_height + 1), sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetHorizontalThetaBuffer, uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetVerticalThetaBuffer, vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetHorizontalThetaStateBuffer,
        checkedBytes((m_width + 1) * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetVerticalThetaStateBuffer,
        checkedBytes(m_width * (m_height + 1), sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetPreviousHorizontalThetaBuffer, uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetPreviousVerticalThetaBuffer, vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetDiagnosticBuffer,
        checkedBytes(40, sizeof(GLuint)), nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_levelSetAreaBuffer,
        checkedBytes(4, sizeof(GLuint)), nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_levelSetComponentLabelBufferA,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetComponentLabelBufferB,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_levelSetComponentChangedBuffer,
        checkedBytes(1, sizeof(GLuint)), nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_candidateLevelSetPhiBuffer, cellBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateLevelSetConfidenceBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateLevelSetHorizontalThetaBuffer, uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateLevelSetVerticalThetaBuffer, vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateLevelSetHorizontalThetaStateBuffer,
        checkedBytes((m_width + 1) * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_candidateLevelSetVerticalThetaStateBuffer,
        checkedBytes(m_width * (m_height + 1), sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderLevelSetPhiBuffer, cellBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderLevelSetConfidenceBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderLevelSetHorizontalThetaBuffer,
        uBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderLevelSetVerticalThetaBuffer,
        vBytes, nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderLevelSetHorizontalThetaStateBuffer,
        checkedBytes((m_width + 1) * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_committedRenderLevelSetVerticalThetaStateBuffer,
        checkedBytes(m_width * (m_height + 1), sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_redistributionCounterBuffer,
        checkedBytes(4, sizeof(GLuint)), nullptr, GL_DYNAMIC_READ);
    const GLuint zero = 0u;
    glClearNamedBufferData(m_classificationFlipBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glNamedBufferData(m_uValidBuffer,
        checkedBytes((m_width + 1) * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_vValidBuffer,
        checkedBytes(m_width * (m_height + 1), sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_uValidScratchBuffer,
        checkedBytes((m_width + 1) * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_vValidScratchBuffer,
        checkedBytes(m_width * (m_height + 1), sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glClearNamedBufferData(m_uValidBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearNamedBufferData(m_vValidBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearNamedBufferData(m_uValidScratchBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glClearNamedBufferData(m_vValidScratchBuffer,
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glNamedBufferData(m_componentLabelBufferA,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_componentLabelBufferB,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_componentFlagsBuffer,
        checkedBytes(m_width * m_height, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_topologyStatusBuffer,
        checkedBytes(8, sizeof(GLuint)), nullptr, GL_DYNAMIC_READ);
    glNamedBufferData(m_topologyChangedBuffer,
        checkedBytes(1, sizeof(GLuint)), nullptr, GL_DYNAMIC_READ);
    const std::size_t maximumLevels = 1 + static_cast<std::size_t>(
        std::ceil(std::log2(static_cast<double>(std::max(m_width, m_height)))));
    glNamedBufferData(m_multigridBuildStatusBuffer,
        checkedBytes(maximumLevels, sizeof(GLuint)), nullptr, GL_DYNAMIC_READ);
    checkOpenGlError("GPU MAC buffer allocation");
}

void GpuMacGrid::bindCommonBuffers() const noexcept {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_uBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_vBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_pressureBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_divergenceBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
}

void GpuMacGrid::clearPressureBuffers() {
    const float zero = 0.0f;
    glClearNamedBufferData(m_pressureBuffer, GL_R32F, GL_RED, GL_FLOAT, &zero);
    glClearNamedBufferData(m_pressureScratchBuffer, GL_R32F, GL_RED, GL_FLOAT, &zero);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

DivergenceStatistics GpuMacGrid::computeDivergenceStatisticsGpu(
    double& gpuMilliseconds,
    double& readbackMilliseconds,
    std::size_t& bytesRead) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_divergenceBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_reductionBufferA);
    m_reduceDivergenceProgram.use();
    const std::size_t elementCount = m_width * m_height;
    const std::size_t partialCount = (elementCount + 255) / 256;
    glUniform1ui(m_reduceDivergenceProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(elementCount));
    beginTimedQuery();
    glDispatchCompute(static_cast<GLuint>(partialCount), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    const GLuint finalBuffer = reducePartials(m_reductionBufferA, partialCount);
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    double values[4]{};
    const auto readbackStart = Clock::now();
    glGetNamedBufferSubData(finalBuffer, 0, sizeof(values), values);
    readbackMilliseconds += std::chrono::duration<double, std::milli>(
        Clock::now() - readbackStart).count();
    gpuMilliseconds += finishTimedQuery();
    bytesRead += sizeof(values);
    const std::size_t count = static_cast<std::size_t>(std::llround(values[2]));
    DivergenceStatistics result;
    result.maximumAbsolute = static_cast<float>(values[1]);
    result.meanAbsolute = count != 0
        ? static_cast<float>(values[0] / static_cast<double>(count)) : 0.0f;
    return result;
}

PressureResidual GpuMacGrid::computeResidual(
    float density,
    float timeStep,
    double& reductionGpuMs,
    double& readbackMs,
    std::size_t& validEquationCount,
    std::size_t& bytesRead) {
    const GLuint finalBuffer = dispatchResidualReduction(density, timeStep, reductionGpuMs);

    double values[4]{};
    const auto start = Clock::now();
    glGetNamedBufferSubData(finalBuffer, 0, sizeof(values), values);
    readbackMs += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    bytesRead += sizeof(values);
    PressureResidual residual;
    validEquationCount = static_cast<std::size_t>(std::llround(values[2]));
    residual.maximumAbsolute = static_cast<float>(values[1]);
    residual.rootMeanSquare = validEquationCount != 0
        ? static_cast<float>(std::sqrt(values[0] / static_cast<double>(validEquationCount))) : 0.0f;
    return residual;
}

GLuint GpuMacGrid::dispatchResidualReduction(
    float density, float timeStep, double& reductionGpuMs) {
    const int profileEvent = beginPressureProfileEvent(
        GpuPressureProfileCategory::trueResidual);
    const std::size_t dispatchBefore = m_pressureDispatchCount;
    bindCommonBuffers();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_residualBuffer);
    m_residualProgram.use();
    glUniform2ui(m_residualProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_residualProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glUniform1d(m_residualProgram.uniformLocation("rhsScale"),
        static_cast<double>(density) / timeStep);
    beginTimedQuery();
    glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
        static_cast<GLuint>((m_height + 15) / 16), 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_residualBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_reductionBufferA);
    m_reduceResidualProgram.use();
    glUniform1ui(m_reduceResidualProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(m_width * m_height));
    const std::size_t partialCount = (m_width * m_height + 255) / 256;
    glDispatchCompute(static_cast<GLuint>(partialCount), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    const GLuint finalBuffer = reducePartials(m_reductionBufferA, partialCount);
    endTimedQuery();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    reductionGpuMs += finishTimedQuery();
    endPressureProfileEvent(profileEvent, m_pressureDispatchCount - dispatchBefore);
    return finalBuffer;
}

GLuint GpuMacGrid::reducePartials(GLuint inputBuffer, std::size_t inputCount) {
    GLuint input = inputBuffer;
    std::size_t count = inputCount;
    while (count > 1) {
        const GLuint output = input == m_reductionBufferA
            ? m_reductionBufferB : m_reductionBufferA;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, input);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, output);
        m_reducePartialProgram.use();
        glUniform1ui(m_reducePartialProgram.uniformLocation("elementCount"),
            static_cast<GLuint>(count));
        const std::size_t outputCount = (count + 255) / 256;
        glDispatchCompute(static_cast<GLuint>(outputCount), 1, 1);
        ++m_pressureDispatchCount;
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        input = output;
        count = outputCount;
    }
    return input;
}

GLuint GpuMacGrid::reduceDot(GLuint leftBuffer, GLuint rightBuffer) {
    const int profileEvent = beginPressureProfileEvent(
        GpuPressureProfileCategory::dotReduction);
    const std::size_t dispatchBefore = m_pressureDispatchCount;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, leftBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, rightBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_reductionBufferA);
    m_reduceDotProgram.use();
    glUniform1ui(m_reduceDotProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(m_width * m_height));
    const std::size_t partialCount = (m_width * m_height + 255) / 256;
    glDispatchCompute(static_cast<GLuint>(partialCount), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    const GLuint result = reducePartials(m_reductionBufferA, partialCount);
    endPressureProfileEvent(profileEvent, m_pressureDispatchCount - dispatchBefore);
    return result;
}

void GpuMacGrid::dispatchPcgOperator(GLuint inputBuffer, GLuint outputBuffer) {
    const int profileEvent = beginPressureProfileEvent(
        GpuPressureProfileCategory::matrixVector);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_cellTypeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, inputBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, outputBuffer);
    m_pcgOperatorProgram.use();
    glUniform2ui(m_pcgOperatorProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(m_width), static_cast<GLuint>(m_height));
    glUniform2f(m_pcgOperatorProgram.uniformLocation("cellSize"), m_dx, m_dy);
    glDispatchCompute(static_cast<GLuint>((m_width + 15) / 16),
        static_cast<GLuint>((m_height + 15) / 16), 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(profileEvent, 1);
}

void GpuMacGrid::dispatchPcgScalar(GLuint reductionBuffer, GLuint operation) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, reductionBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_pcgScalarBuffer);
    m_pcgScalarProgram.use();
    glUniform1ui(m_pcgScalarProgram.uniformLocation("operation"), operation);
    glDispatchCompute(1, 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GpuMacGrid::dispatchGpuPressureControl(
    GLuint residualSummaryBuffer,
    float tolerance,
    std::size_t completedIterations,
    std::size_t maximumIterations,
    std::size_t submittedIterations) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, residualSummaryBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m_pcgScalarBuffer);
    m_pcgScalarProgram.use();
    glUniform1ui(m_pcgScalarProgram.uniformLocation("operation"), 3u);
    glUniform1ui(m_pcgScalarProgram.uniformLocation("pressureControlEnabled"), 1u);
    glUniform1d(m_pcgScalarProgram.uniformLocation("residualTolerance"),
        static_cast<double>(tolerance));
    glUniform1ui(m_pcgScalarProgram.uniformLocation("completedIterations"),
        static_cast<GLuint>(completedIterations));
    glUniform1ui(m_pcgScalarProgram.uniformLocation("maximumIterations"),
        static_cast<GLuint>(maximumIterations));
    glUniform1ui(m_pcgScalarProgram.uniformLocation("submittedIterations"),
        static_cast<GLuint>(submittedIterations));
    glDispatchCompute(1, 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

void GpuMacGrid::initializeGpuPressureControl() {
    const std::array<double, 10> initial{};
    glNamedBufferSubData(m_pressureControlBuffer, 0,
        static_cast<GLsizeiptr>(sizeof(initial)), initial.data());
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

void GpuMacGrid::readGpuPressureControl(GpuPressureReport& report) {
    std::array<double, 10> values{};
    const auto readStart = Clock::now();
    glGetNamedBufferSubData(m_pressureControlBuffer, 0,
        static_cast<GLsizeiptr>(sizeof(values)), values.data());
    report.residualReadbackMilliseconds +=
        std::chrono::duration<double, std::milli>(Clock::now() - readStart).count();
    report.residualBytesRead += sizeof(values);
    ++report.stateReadbackCount;

    const auto stateCode = static_cast<std::uint32_t>(std::llround(values[0]));
    report.solveState = stateCode <=
        static_cast<std::uint32_t>(GpuPressureSolveState::invalidResidual)
        ? static_cast<GpuPressureSolveState>(stateCode)
        : GpuPressureSolveState::invalidResidual;
    report.iterations = static_cast<std::size_t>(std::max(0LL, std::llround(values[1])));
    report.finalResidual.rootMeanSquare = static_cast<float>(values[2]);
    report.finalResidual.maximumAbsolute = static_cast<float>(values[3]);
    report.initialResidual.rootMeanSquare = static_cast<float>(values[8]);
    report.initialResidual.maximumAbsolute = static_cast<float>(values[9]);
    report.validEquationCount =
        static_cast<std::size_t>(std::max(0LL, std::llround(values[4])));
    report.submittedIterations =
        static_cast<std::size_t>(std::max(0LL, std::llround(values[7])));
    report.skippedIterations = report.submittedIterations > report.iterations
        ? report.submittedIterations - report.iterations : 0;

    switch (report.solveState) {
    case GpuPressureSolveState::converged:
        report.converged = true;
        report.termination = report.iterations == 0
            ? PressureSolveTermination::initialResidualWithinTolerance
            : PressureSolveTermination::iterativeResidualWithinTolerance;
        break;
    case GpuPressureSolveState::maximumIterationsReached:
        report.termination = PressureSolveTermination::maximumIterationsReached;
        break;
    case GpuPressureSolveState::numericalBreakdown:
    case GpuPressureSolveState::invalidResidual:
        report.failureReason = GpuPressureFailureReason::numericalBreakdown;
        break;
    case GpuPressureSolveState::invalidTopology:
        report.failureReason = GpuPressureFailureReason::dynamicPressureTopologyAnalysisFailed;
        break;
    case GpuPressureSolveState::running:
        break;
    }
}

void GpuMacGrid::configureGpuPressureControl(bool enabled) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, m_pressureControlBuffer);
    const GLuint value = enabled ? 1u : 0u;
    const auto configure = [value](graphics::ShaderProgram& program) {
        program.use();
        glUniform1ui(program.uniformLocation("pressureControlEnabled"), value);
    };
    configure(m_residualProgram);
    configure(m_reduceResidualProgram);
    configure(m_reducePartialProgram);
    configure(m_reduceDotProgram);
    configure(m_pcgOperatorProgram);
    configure(m_pcgScalarProgram);
    configure(m_pcgUpdateProgram);
    configure(m_pcgDirectionProgram);
    configure(m_vectorCopyProgram);
    configure(m_mgSmoothProgram);
    configure(m_mgCoarseSmoothProgram);
    configure(m_mgResidualProgram);
    configure(m_mgRestrictProgram);
    configure(m_mgProlongProgram);
    configure(m_mgSmoothFloatProgram);
    configure(m_mgCoarseSmoothFloatProgram);
    configure(m_mgResidualFloatProgram);
    configure(m_mgRestrictFloatProgram);
    configure(m_mgProlongFloatProgram);
    configure(m_mgConvertProgram);
}

void GpuMacGrid::allocateDynamicMultigridHierarchy() {
    std::vector<std::pair<std::size_t, std::size_t>> dimensions;
    std::size_t width = m_width;
    std::size_t height = m_height;
    dimensions.emplace_back(width, height);
    while (std::max(width, height) > 16) {
        width = (width + 1) / 2;
        height = (height + 1) / 2;
        dimensions.emplace_back(width, height);
    }
    bool layoutMatches = m_multigridLevels.size() == dimensions.size();
    if (layoutMatches) {
        for (std::size_t i = 0; i < dimensions.size(); ++i) {
            layoutMatches = layoutMatches &&
                m_multigridLevels[i].width == dimensions[i].first &&
                m_multigridLevels[i].height == dimensions[i].second;
        }
    }
    if (layoutMatches) return;

    releaseMultigrid();
    try {
        for (const auto [levelWidth, levelHeight] : dimensions) {
            MultigridLevel level;
            level.width = levelWidth;
            level.height = levelHeight;
            GLuint buffers[12]{};
            glCreateBuffers(12, buffers);
            level.correction = buffers[0];
            level.scratch = buffers[1];
            level.rightHandSide = buffers[2];
            level.residual = buffers[3];
            level.diagonal = buffers[4];
            level.weights = buffers[5];
            level.active = buffers[6];
            level.component = buffers[7];
            level.correctionFloat = buffers[8];
            level.scratchFloat = buffers[9];
            level.rightHandSideFloat = buffers[10];
            level.residualFloat = buffers[11];
            if (std::find(std::begin(buffers), std::end(buffers), 0u) != std::end(buffers)) {
                glDeleteBuffers(12, buffers);
                throw std::runtime_error("Failed to create dynamic multigrid buffers.");
            }
            const std::size_t count = levelWidth * levelHeight;
            const GLsizeiptr vectorBytes = checkedBytes(count, sizeof(double));
            glNamedBufferData(level.correction, vectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.scratch, vectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.rightHandSide, vectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.residual, vectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.diagonal, vectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.weights,
                checkedBytes(count, sizeof(std::array<double, 4>)), nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.active,
                checkedBytes(count, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.component,
                checkedBytes(count, sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
            const GLsizeiptr floatVectorBytes = checkedBytes(count, sizeof(float));
            glNamedBufferData(level.correctionFloat, floatVectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.scratchFloat, floatVectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.rightHandSideFloat, floatVectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.residualFloat, floatVectorBytes, nullptr, GL_DYNAMIC_DRAW);
            m_multigridMemoryBytes += count *
                (5 * sizeof(double) + 4 * sizeof(float) +
                 sizeof(std::array<double, 4>) + 2 * sizeof(GLuint));
            m_multigridLevels.push_back(level);
        }
    } catch (...) {
        releaseMultigrid();
        throw;
    }
    m_activeMultigridLevelCount = m_multigridLevels.size();
}

void GpuMacGrid::rebuildMultigridHierarchy(const MacGrid& grid) {
    std::size_t topologyHash = grid.width();
    topologyHash ^= grid.height() + 0x9e3779b9u + (topologyHash << 6) + (topologyHash >> 2);
    topologyHash ^= std::hash<float>{}(grid.cellWidth()) + 0x9e3779b9u +
        (topologyHash << 6) + (topologyHash >> 2);
    topologyHash ^= std::hash<float>{}(grid.cellHeight()) + 0x9e3779b9u +
        (topologyHash << 6) + (topologyHash >> 2);
    for (const CellType type : grid.cellTypes()) {
        topologyHash ^= static_cast<std::size_t>(type) + 0x9e3779b9u +
            (topologyHash << 6) + (topologyHash >> 2);
    }
    if (!m_multigridLevels.empty() && topologyHash == m_multigridTopologyHash) return;

    struct CpuLevel final {
        std::size_t width = 0;
        std::size_t height = 0;
        std::vector<double> diagonal;
        std::vector<std::array<double, 4>> weights;
        std::vector<GLuint> active;
        std::vector<GLuint> component;
    };

    releaseMultigrid();
    std::vector<CpuLevel> cpuLevels;
    CpuLevel fine;
    fine.width = grid.width();
    fine.height = grid.height();
    const std::size_t fineCount = fine.width * fine.height;
    fine.diagonal.assign(fineCount, 0.0);
    fine.weights.assign(fineCount, {});
    fine.active.assign(fineCount, 0u);
    fine.component = m_componentIds;
    const double inverseDx2 = 1.0 /
        (static_cast<double>(grid.cellWidth()) * static_cast<double>(grid.cellWidth()));
    const double inverseDy2 = 1.0 /
        (static_cast<double>(grid.cellHeight()) * static_cast<double>(grid.cellHeight()));
    for (std::size_t y = 0; y < fine.height; ++y) {
        for (std::size_t x = 0; x < fine.width; ++x) {
            const std::size_t index = y * fine.width + x;
            if (grid.cellType(x, y) != CellType::fluid) continue;
            fine.active[index] = 1u;
            const auto addNeighbor = [&](std::size_t nx, std::size_t ny,
                                         double coefficient, std::size_t direction) {
                const CellType neighbor = grid.cellType(nx, ny);
                if (neighbor == CellType::fluid || neighbor == CellType::air) {
                    fine.diagonal[index] += coefficient;
                }
                if (neighbor == CellType::fluid) fine.weights[index][direction] = coefficient;
            };
            if (x > 0) addNeighbor(x - 1, y, inverseDx2, 0);
            if (x + 1 < fine.width) addNeighbor(x + 1, y, inverseDx2, 1);
            if (y > 0) addNeighbor(x, y - 1, inverseDy2, 2);
            if (y + 1 < fine.height) addNeighbor(x, y + 1, inverseDy2, 3);
        }
    }
    cpuLevels.push_back(std::move(fine));

    while (std::max(cpuLevels.back().width, cpuLevels.back().height) > 16) {
        const CpuLevel& source = cpuLevels.back();
        CpuLevel coarse;
        coarse.width = (source.width + 1) / 2;
        coarse.height = (source.height + 1) / 2;
        const std::size_t coarseCount = coarse.width * coarse.height;
        coarse.diagonal.assign(coarseCount, 0.0);
        coarse.weights.assign(coarseCount, {});
        coarse.active.assign(coarseCount, 0u);
        coarse.component.assign(coarseCount, std::numeric_limits<GLuint>::max());
        bool componentCollision = false;

        for (std::size_t y = 0; y < source.height; ++y) {
            for (std::size_t x = 0; x < source.width; ++x) {
                const std::size_t fineIndex = y * source.width + x;
                if (source.active[fineIndex] == 0u) continue;
                const std::size_t coarseIndex = (y / 2) * coarse.width + x / 2;
                coarse.active[coarseIndex] = 1u;
                const GLuint component = source.component[fineIndex];
                if (coarse.component[coarseIndex] == std::numeric_limits<GLuint>::max()) {
                    coarse.component[coarseIndex] = component;
                } else if (coarse.component[coarseIndex] != component) {
                    componentCollision = true;
                }
                coarse.diagonal[coarseIndex] += source.diagonal[fineIndex];
            }
        }
        if (componentCollision) break;

        const auto accumulateEdge = [&](std::size_t x, std::size_t y,
                                        std::size_t nx, std::size_t ny,
                                        double weight) {
            if (weight == 0.0) return;
            const std::size_t coarseIndex = (y / 2) * coarse.width + x / 2;
            const std::size_t neighborIndex = (ny / 2) * coarse.width + nx / 2;
            if (coarseIndex == neighborIndex) {
                coarse.diagonal[coarseIndex] -= weight;
                return;
            }
            if (nx / 2 < x / 2) coarse.weights[coarseIndex][0] += weight;
            else if (nx / 2 > x / 2) coarse.weights[coarseIndex][1] += weight;
            else if (ny / 2 < y / 2) coarse.weights[coarseIndex][2] += weight;
            else coarse.weights[coarseIndex][3] += weight;
        };
        for (std::size_t y = 0; y < source.height; ++y) {
            for (std::size_t x = 0; x < source.width; ++x) {
                const std::size_t index = y * source.width + x;
                if (source.active[index] == 0u) continue;
                if (x > 0) accumulateEdge(x, y, x - 1, y, source.weights[index][0]);
                if (x + 1 < source.width)
                    accumulateEdge(x, y, x + 1, y, source.weights[index][1]);
                if (y > 0) accumulateEdge(x, y, x, y - 1, source.weights[index][2]);
                if (y + 1 < source.height)
                    accumulateEdge(x, y, x, y + 1, source.weights[index][3]);
            }
        }
        cpuLevels.push_back(std::move(coarse));
    }

    try {
        m_multigridLevels.reserve(cpuLevels.size());
        for (const CpuLevel& cpu : cpuLevels) {
            MultigridLevel level;
            level.width = cpu.width;
            level.height = cpu.height;
            level.activeCount = static_cast<std::size_t>(
                std::count(cpu.active.begin(), cpu.active.end(), 1u));
            GLuint buffers[12]{};
            glCreateBuffers(12, buffers);
            level.correction = buffers[0];
            level.scratch = buffers[1];
            level.rightHandSide = buffers[2];
            level.residual = buffers[3];
            level.diagonal = buffers[4];
            level.weights = buffers[5];
            level.active = buffers[6];
            level.component = buffers[7];
            level.correctionFloat = buffers[8];
            level.scratchFloat = buffers[9];
            level.rightHandSideFloat = buffers[10];
            level.residualFloat = buffers[11];
            if (level.correction == 0 || level.scratch == 0 || level.rightHandSide == 0 ||
                level.residual == 0 || level.diagonal == 0 || level.weights == 0 ||
                level.active == 0 || level.component == 0 ||
                level.correctionFloat == 0 || level.scratchFloat == 0 ||
                level.rightHandSideFloat == 0 || level.residualFloat == 0) {
                glDeleteBuffers(12, buffers);
                throw std::runtime_error("Failed to create multigrid buffers.");
            }
            const std::size_t count = cpu.width * cpu.height;
            const GLsizeiptr vectorBytes = checkedBytes(count, sizeof(double));
            glNamedBufferData(level.correction, vectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.scratch, vectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.rightHandSide, vectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.residual, vectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.diagonal, vectorBytes, cpu.diagonal.data(), GL_STATIC_DRAW);
            glNamedBufferData(level.weights,
                checkedBytes(count, sizeof(std::array<double, 4>)), cpu.weights.data(), GL_STATIC_DRAW);
            glNamedBufferData(level.active,
                checkedBytes(count, sizeof(GLuint)), cpu.active.data(), GL_STATIC_DRAW);
            glNamedBufferData(level.component,
                checkedBytes(count, sizeof(GLuint)), cpu.component.data(), GL_STATIC_DRAW);
            const GLsizeiptr floatVectorBytes = checkedBytes(count, sizeof(float));
            glNamedBufferData(level.correctionFloat, floatVectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.scratchFloat, floatVectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.rightHandSideFloat, floatVectorBytes, nullptr, GL_DYNAMIC_DRAW);
            glNamedBufferData(level.residualFloat, floatVectorBytes, nullptr, GL_DYNAMIC_DRAW);
            m_multigridMemoryBytes += count *
                (5 * sizeof(double) + 4 * sizeof(float) +
                 sizeof(std::array<double, 4>) + 2 * sizeof(GLuint));
            m_multigridLevels.push_back(level);
        }
    } catch (...) {
        releaseMultigrid();
        throw;
    }
    m_multigridTopologyHash = topologyHash;
    m_activeMultigridLevelCount = m_multigridLevels.size();
    checkOpenGlError("GPU multigrid hierarchy creation");
}

void GpuMacGrid::releaseMultigrid() noexcept {
    for (const MultigridLevel& level : m_multigridLevels) {
        const GLuint buffers[]{level.correction, level.scratch, level.rightHandSide,
            level.residual, level.diagonal, level.weights, level.active, level.component,
            level.correctionFloat, level.scratchFloat, level.rightHandSideFloat,
            level.residualFloat};
        glDeleteBuffers(static_cast<GLsizei>(std::size(buffers)), buffers);
    }
    m_multigridLevels.clear();
    m_multigridMemoryBytes = 0;
    m_multigridTopologyHash = 0;
    m_activeMultigridLevelCount = 0;
}

void GpuMacGrid::copyVector(
    GLuint inputBuffer, GLuint outputBuffer, std::size_t elementCount) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputBuffer);
    m_vectorCopyProgram.use();
    glUniform1ui(m_vectorCopyProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(elementCount));
    glUniform1ui(m_vectorCopyProgram.uniformLocation("clearOutput"), 0u);
    glDispatchCompute(static_cast<GLuint>((elementCount + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GpuMacGrid::clearVector(GLuint outputBuffer, std::size_t elementCount) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputBuffer);
    m_vectorCopyProgram.use();
    glUniform1ui(m_vectorCopyProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(elementCount));
    glUniform1ui(m_vectorCopyProgram.uniformLocation("clearOutput"), 1u);
    glDispatchCompute(static_cast<GLuint>((elementCount + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GpuMacGrid::smoothMultigridLevel(
    MultigridLevel& level, std::size_t iterations, float relaxationFactor) {
    const int levelIndex = static_cast<int>(&level - m_multigridLevels.data());
    const auto category = static_cast<std::size_t>(levelIndex + 1) ==
            m_activeMultigridLevelCount
        ? GpuPressureProfileCategory::coarseSolve
        : GpuPressureProfileCategory::smoother;
    const int profileEvent = beginPressureProfileEvent(category, levelIndex);
    const std::size_t dispatchBefore = m_pressureDispatchCount;
    const GLuint groups = static_cast<GLuint>((level.width * level.height + 255) / 256);
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.correction);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.rightHandSide);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, level.diagonal);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, level.weights);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, level.active);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, level.scratch);
        m_mgSmoothProgram.use();
        glUniform2ui(m_mgSmoothProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
        glUniform1d(m_mgSmoothProgram.uniformLocation("relaxationFactor"), relaxationFactor);
        glDispatchCompute(groups, 1, 1);
        ++m_pressureDispatchCount;
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        std::swap(level.correction, level.scratch);
    }
    endPressureProfileEvent(profileEvent, m_pressureDispatchCount - dispatchBefore);
}

void GpuMacGrid::smoothMultigridLevelFloat(
    MultigridLevel& level, std::size_t iterations, float relaxationFactor) {
    const int levelIndex = static_cast<int>(&level - m_multigridLevels.data());
    const auto category = static_cast<std::size_t>(levelIndex + 1) ==
            m_activeMultigridLevelCount
        ? GpuPressureProfileCategory::coarseSolve
        : GpuPressureProfileCategory::smoother;
    const int profileEvent = beginPressureProfileEvent(category, levelIndex);
    const std::size_t dispatchBefore = m_pressureDispatchCount;
    const GLuint groups = static_cast<GLuint>((level.width * level.height + 255) / 256);
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.correctionFloat);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.rightHandSideFloat);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, level.diagonal);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, level.weights);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, level.active);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, level.scratchFloat);
        m_mgSmoothFloatProgram.use();
        glUniform2ui(m_mgSmoothFloatProgram.uniformLocation("gridSize"),
            static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
        glUniform1f(m_mgSmoothFloatProgram.uniformLocation("relaxationFactor"),
            relaxationFactor);
        glDispatchCompute(groups, 1, 1);
        ++m_pressureDispatchCount;
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        std::swap(level.correctionFloat, level.scratchFloat);
    }
    endPressureProfileEvent(profileEvent, m_pressureDispatchCount - dispatchBefore);
}

void GpuMacGrid::smoothCoarseLevelFused(
    MultigridLevel& level, std::size_t iterations, float relaxationFactor) {
    if (level.width * level.height > 256) {
        throw std::logic_error("Fused coarse smoother exceeds one workgroup.");
    }
    const int levelIndex = static_cast<int>(&level - m_multigridLevels.data());
    const int profileEvent = beginPressureProfileEvent(
        GpuPressureProfileCategory::coarseSolve, levelIndex);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.correction);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.rightHandSide);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, level.diagonal);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, level.weights);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, level.active);
    m_mgCoarseSmoothProgram.use();
    glUniform2ui(m_mgCoarseSmoothProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
    glUniform1d(m_mgCoarseSmoothProgram.uniformLocation("relaxationFactor"), relaxationFactor);
    glUniform1ui(m_mgCoarseSmoothProgram.uniformLocation("iterationCount"),
        static_cast<GLuint>(iterations));
    glDispatchCompute(1, 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(profileEvent, 1);
}

void GpuMacGrid::smoothCoarseLevelFusedFloat(
    MultigridLevel& level, std::size_t iterations, float relaxationFactor) {
    if (level.width * level.height > 256) {
        throw std::logic_error("Fused FP32 coarse smoother exceeds one workgroup.");
    }
    const int levelIndex = static_cast<int>(&level - m_multigridLevels.data());
    const int profileEvent = beginPressureProfileEvent(
        GpuPressureProfileCategory::coarseSolve, levelIndex);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.correctionFloat);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.rightHandSideFloat);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, level.diagonal);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, level.weights);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, level.active);
    m_mgCoarseSmoothFloatProgram.use();
    glUniform2ui(m_mgCoarseSmoothFloatProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
    glUniform1f(m_mgCoarseSmoothFloatProgram.uniformLocation("relaxationFactor"),
        relaxationFactor);
    glUniform1ui(m_mgCoarseSmoothFloatProgram.uniformLocation("iterationCount"),
        static_cast<GLuint>(iterations));
    glDispatchCompute(1, 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(profileEvent, 1);
}

bool GpuMacGrid::canUseFusedCoarseSmoother(
    const MultigridLevel& level) const noexcept {
    return m_fusedCoarseSmootherSupported && level.width * level.height <= 256;
}

void GpuMacGrid::convertDoubleToFloat(
    GLuint inputBuffer, GLuint outputBuffer, std::size_t elementCount) {
    const int profileEvent = beginPressureProfileEvent(
        GpuPressureProfileCategory::typeConversion, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputBuffer);
    m_mgConvertProgram.use();
    glUniform1ui(m_mgConvertProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(elementCount));
    glUniform1ui(m_mgConvertProgram.uniformLocation("direction"), 0u);
    glDispatchCompute(static_cast<GLuint>((elementCount + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(profileEvent, 1);
}

void GpuMacGrid::convertFloatToDouble(
    GLuint inputBuffer, GLuint outputBuffer, std::size_t elementCount) {
    const int profileEvent = beginPressureProfileEvent(
        GpuPressureProfileCategory::typeConversion, 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, inputBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, outputBuffer);
    m_mgConvertProgram.use();
    glUniform1ui(m_mgConvertProgram.uniformLocation("elementCount"),
        static_cast<GLuint>(elementCount));
    glUniform1ui(m_mgConvertProgram.uniformLocation("direction"), 1u);
    glDispatchCompute(static_cast<GLuint>((elementCount + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(profileEvent, 1);
}

void GpuMacGrid::runMultigridCycle(
    std::size_t levelIndex, const GpuPressureSettings& settings) {
    MultigridLevel& level = m_multigridLevels[levelIndex];
    if (levelIndex + 1 == m_activeMultigridLevelCount) {
        if (settings.fusedCoarseSmootherEnabled && canUseFusedCoarseSmoother(level)) {
            smoothCoarseLevelFused(
                level, settings.multigridCoarseIterations, settings.multigridRelaxationFactor);
        } else {
            smoothMultigridLevel(
                level, settings.multigridCoarseIterations, settings.multigridRelaxationFactor);
        }
        return;
    }

    smoothMultigridLevel(
        level, settings.multigridPreSmoothIterations, settings.multigridRelaxationFactor);
    const int residualProfile = beginPressureProfileEvent(
        GpuPressureProfileCategory::multigridResidual, static_cast<int>(levelIndex));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.correction);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.rightHandSide);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, level.diagonal);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, level.weights);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, level.active);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, level.residual);
    m_mgResidualProgram.use();
    glUniform2ui(m_mgResidualProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
    glDispatchCompute(static_cast<GLuint>((level.width * level.height + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(residualProfile, 1);

    MultigridLevel& coarse = m_multigridLevels[levelIndex + 1];
    if (m_gpuPressureControlActive) {
        clearVector(coarse.correction, coarse.width * coarse.height);
        clearVector(coarse.scratch, coarse.width * coarse.height);
    } else {
        const float zero = 0.0f;
        glClearNamedBufferData(coarse.correction, GL_R32F, GL_RED, GL_FLOAT, &zero);
        glClearNamedBufferData(coarse.scratch, GL_R32F, GL_RED, GL_FLOAT, &zero);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    }
    const int restrictionProfile = beginPressureProfileEvent(
        GpuPressureProfileCategory::restriction, static_cast<int>(levelIndex));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.residual);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.active);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coarse.active);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coarse.rightHandSide);
    m_mgRestrictProgram.use();
    glUniform2ui(m_mgRestrictProgram.uniformLocation("fineSize"),
        static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
    glUniform2ui(m_mgRestrictProgram.uniformLocation("coarseSize"),
        static_cast<GLuint>(coarse.width), static_cast<GLuint>(coarse.height));
    glDispatchCompute(static_cast<GLuint>((coarse.width * coarse.height + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(restrictionProfile, 1);

    runMultigridCycle(levelIndex + 1, settings);

    const int prolongationProfile = beginPressureProfileEvent(
        GpuPressureProfileCategory::prolongation, static_cast<int>(levelIndex));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.correction);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.active);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coarse.correction);
    m_mgProlongProgram.use();
    glUniform2ui(m_mgProlongProgram.uniformLocation("fineSize"),
        static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
    glUniform2ui(m_mgProlongProgram.uniformLocation("coarseSize"),
        static_cast<GLuint>(coarse.width), static_cast<GLuint>(coarse.height));
    glDispatchCompute(static_cast<GLuint>((level.width * level.height + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(prolongationProfile, 1);
    smoothMultigridLevel(
        level, settings.multigridPostSmoothIterations, settings.multigridRelaxationFactor);
}

void GpuMacGrid::runMultigridCycleFloat(
    std::size_t levelIndex, const GpuPressureSettings& settings) {
    MultigridLevel& level = m_multigridLevels[levelIndex];
    if (levelIndex + 1 == m_activeMultigridLevelCount) {
        if (settings.fusedCoarseSmootherEnabled && canUseFusedCoarseSmoother(level)) {
            smoothCoarseLevelFusedFloat(
                level, settings.multigridCoarseIterations, settings.multigridRelaxationFactor);
        } else {
            smoothMultigridLevelFloat(
                level, settings.multigridCoarseIterations, settings.multigridRelaxationFactor);
        }
        return;
    }

    smoothMultigridLevelFloat(
        level, settings.multigridPreSmoothIterations, settings.multigridRelaxationFactor);
    const int residualProfile = beginPressureProfileEvent(
        GpuPressureProfileCategory::multigridResidual, static_cast<int>(levelIndex));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.correctionFloat);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.rightHandSideFloat);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, level.diagonal);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, level.weights);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, level.active);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, level.residualFloat);
    m_mgResidualFloatProgram.use();
    glUniform2ui(m_mgResidualFloatProgram.uniformLocation("gridSize"),
        static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
    glDispatchCompute(static_cast<GLuint>((level.width * level.height + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(residualProfile, 1);

    MultigridLevel& coarse = m_multigridLevels[levelIndex + 1];
    const float zero = 0.0f;
    glClearNamedBufferData(coarse.correctionFloat, GL_R32F, GL_RED, GL_FLOAT, &zero);
    glClearNamedBufferData(coarse.scratchFloat, GL_R32F, GL_RED, GL_FLOAT, &zero);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    const int restrictionProfile = beginPressureProfileEvent(
        GpuPressureProfileCategory::restriction, static_cast<int>(levelIndex));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.residualFloat);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.active);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coarse.active);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coarse.rightHandSideFloat);
    m_mgRestrictFloatProgram.use();
    glUniform2ui(m_mgRestrictFloatProgram.uniformLocation("fineSize"),
        static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
    glUniform2ui(m_mgRestrictFloatProgram.uniformLocation("coarseSize"),
        static_cast<GLuint>(coarse.width), static_cast<GLuint>(coarse.height));
    glDispatchCompute(static_cast<GLuint>((coarse.width * coarse.height + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(restrictionProfile, 1);

    runMultigridCycleFloat(levelIndex + 1, settings);

    const int prolongationProfile = beginPressureProfileEvent(
        GpuPressureProfileCategory::prolongation, static_cast<int>(levelIndex));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, level.correctionFloat);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, level.active);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coarse.correctionFloat);
    m_mgProlongFloatProgram.use();
    glUniform2ui(m_mgProlongFloatProgram.uniformLocation("fineSize"),
        static_cast<GLuint>(level.width), static_cast<GLuint>(level.height));
    glUniform2ui(m_mgProlongFloatProgram.uniformLocation("coarseSize"),
        static_cast<GLuint>(coarse.width), static_cast<GLuint>(coarse.height));
    glDispatchCompute(static_cast<GLuint>((level.width * level.height + 255) / 256), 1, 1);
    ++m_pressureDispatchCount;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    endPressureProfileEvent(prolongationProfile, 1);
    smoothMultigridLevelFloat(
        level, settings.multigridPostSmoothIterations, settings.multigridRelaxationFactor);
}

void GpuMacGrid::applyMultigridPreconditioner(
    GLuint inputResidual, GLuint outputPreconditioned,
    const GpuPressureSettings& settings) {
    const int profileEvent = beginPressureProfileEvent(GpuPressureProfileCategory::vCycle);
    const std::size_t dispatchBefore = m_pressureDispatchCount;
    if (m_multigridLevels.empty() || m_activeMultigridLevelCount == 0) {
        throw std::logic_error("Multigrid hierarchy is unavailable.");
    }
    MultigridLevel& fine = m_multigridLevels.front();
    if (settings.precisionMode == GpuPressurePrecisionMode::fp64OuterFp32Multigrid) {
        const float zero = 0.0f;
        glClearNamedBufferData(fine.correctionFloat, GL_R32F, GL_RED, GL_FLOAT, &zero);
        glClearNamedBufferData(fine.scratchFloat, GL_R32F, GL_RED, GL_FLOAT, &zero);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
        convertDoubleToFloat(inputResidual, fine.rightHandSideFloat, m_width * m_height);
        runMultigridCycleFloat(0, settings);
        convertFloatToDouble(fine.correctionFloat, outputPreconditioned, m_width * m_height);
        endPressureProfileEvent(profileEvent, m_pressureDispatchCount - dispatchBefore);
        return;
    }
    if (m_gpuPressureControlActive) {
        clearVector(fine.correction, m_width * m_height);
        clearVector(fine.scratch, m_width * m_height);
    } else {
        const float zero = 0.0f;
        glClearNamedBufferData(fine.correction, GL_R32F, GL_RED, GL_FLOAT, &zero);
        glClearNamedBufferData(fine.scratch, GL_R32F, GL_RED, GL_FLOAT, &zero);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    }
    copyVector(inputResidual, fine.rightHandSide, m_width * m_height);
    runMultigridCycle(0, settings);
    copyVector(fine.correction, outputPreconditioned, m_width * m_height);
    endPressureProfileEvent(profileEvent, m_pressureDispatchCount - dispatchBefore);
}

void GpuMacGrid::inspectPressureTopology(const MacGrid& grid) {
    const std::size_t count = grid.width() * grid.height();
    std::vector<unsigned char> visited(count, 0);
    m_componentIds.assign(count, std::numeric_limits<GLuint>::max());
    m_pureNeumannComponentCount = 0;
    m_hasIsolatedFluidCell = false;
    m_fluidCellCount = static_cast<std::size_t>(std::count(
        grid.cellTypes().begin(), grid.cellTypes().end(), CellType::fluid));
    const auto indexOf = [&](std::size_t x, std::size_t y) { return y * grid.width() + x; };
    GLuint componentId = 0;
    for (std::size_t start = 0; start < count; ++start) {
        if (visited[start] != 0 || grid.cellTypes()[start] != CellType::fluid) continue;
        bool touchesAir = false;
        std::deque<std::size_t> queue{start};
        visited[start] = 1;
        while (!queue.empty()) {
            const std::size_t index = queue.front();
            queue.pop_front();
            m_componentIds[index] = componentId;
            const std::size_t x = index % grid.width();
            const std::size_t y = index / grid.width();
            std::size_t activeNeighbors = 0;
            const auto inspect = [&](std::size_t nx, std::size_t ny) {
                const CellType type = grid.cellType(nx, ny);
                if (type == CellType::air) {
                    touchesAir = true;
                    ++activeNeighbors;
                } else if (type == CellType::fluid) {
                    ++activeNeighbors;
                    const std::size_t neighbor = indexOf(nx, ny);
                    if (visited[neighbor] == 0) {
                        visited[neighbor] = 1;
                        queue.push_back(neighbor);
                    }
                }
            };
            if (x > 0) inspect(x - 1, y);
            if (x + 1 < grid.width()) inspect(x + 1, y);
            if (y > 0) inspect(x, y - 1);
            if (y + 1 < grid.height()) inspect(x, y + 1);
            if (activeNeighbors == 0) m_hasIsolatedFluidCell = true;
        }
        if (!touchesAir) ++m_pureNeumannComponentCount;
        ++componentId;
    }
}

DivergenceStatistics GpuMacGrid::divergenceStatistics(const std::vector<float>& values) const noexcept {
    DivergenceStatistics result;
    double sum = 0.0;
    for (const float value : values) {
        const float absolute = std::abs(value);
        result.maximumAbsolute = std::max(result.maximumAbsolute, absolute);
        sum += absolute;
    }
    result.meanAbsolute = m_fluidCellCount != 0
        ? static_cast<float>(sum / m_fluidCellCount) : 0.0f;
    return result;
}

void GpuMacGrid::beginTimedQuery() {
    if (!m_synchronousTimingEnabled) return;
    if (m_queryPending) throw std::logic_error("A GPU timer query is already pending.");
    glBeginQuery(GL_TIME_ELAPSED, m_timerQuery);
}

void GpuMacGrid::endTimedQuery() {
    if (!m_synchronousTimingEnabled) return;
    glEndQuery(GL_TIME_ELAPSED);
    m_queryPending = true;
}

double GpuMacGrid::finishTimedQuery() {
    if (!m_synchronousTimingEnabled) return 0.0;
    if (!m_queryPending) return 0.0;
    GLuint64 nanoseconds = 0;
    glGetQueryObjectui64v(m_timerQuery, GL_QUERY_RESULT, &nanoseconds);
    m_queryPending = false;
    return static_cast<double>(nanoseconds) / 1.0e6;
}

int GpuMacGrid::beginPressureProfileEvent(
    GpuPressureProfileCategory category, int level) {
    if (!m_pressureProfilingActive) return -1;
    const auto duplicate = std::find_if(m_pressureProfileEvents.begin(),
        m_pressureProfileEvents.end(), [category, level](const PressureProfileEvent& event) {
            return event.category == category && event.level == level;
        });
    if (duplicate != m_pressureProfileEvents.end() ||
        (m_pressureProfileEvents.size() + 1) * 2 > m_profileTimestampQueries.size()) {
        return -1;
    }
    PressureProfileEvent event;
    event.category = category;
    event.level = level;
    const std::size_t queryIndex = m_pressureProfileEvents.size() * 2;
    event.beginQuery = m_profileTimestampQueries[queryIndex];
    event.endQuery = m_profileTimestampQueries[queryIndex + 1];
    event.cpuBegin = Clock::now();
    glQueryCounter(event.beginQuery, GL_TIMESTAMP);
    m_pressureProfileEvents.push_back(event);
    ++m_pressureProfileCategorySamples[static_cast<std::size_t>(category)];
    return static_cast<int>(m_pressureProfileEvents.size() - 1);
}

void GpuMacGrid::endPressureProfileEvent(int eventIndex, std::size_t dispatchCount) {
    if (eventIndex < 0) return;
    PressureProfileEvent& event = m_pressureProfileEvents[static_cast<std::size_t>(eventIndex)];
    glQueryCounter(event.endQuery, GL_TIMESTAMP);
    event.cpuSubmissionMilliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - event.cpuBegin).count();
    event.dispatchCount = dispatchCount;
}

void GpuMacGrid::resolvePressureProfile(GpuPressureProfileReport& report) {
    for (const PressureProfileEvent& event : m_pressureProfileEvents) {
        GLuint64 begin = 0;
        GLuint64 end = 0;
        glGetQueryObjectui64v(event.beginQuery, GL_QUERY_RESULT, &begin);
        glGetQueryObjectui64v(event.endQuery, GL_QUERY_RESULT, &end);
        auto& category = report.categories[static_cast<std::size_t>(event.category)];
        category.gpuMilliseconds += static_cast<double>(end - begin) / 1.0e6;
        category.cpuSubmissionMilliseconds += event.cpuSubmissionMilliseconds;
        category.dispatchCount += event.dispatchCount;
        ++category.sampleCount;
    }
}

void GpuMacGrid::release() noexcept {
    releaseMultigrid();
    if (m_asynchronousStepQueryActive) {
        glEndQuery(GL_TIME_ELAPSED);
        m_asynchronousStepQueryActive = false;
    }
    if (m_timerQuery != 0) glDeleteQueries(1, &m_timerQuery);
    glDeleteQueries(static_cast<GLsizei>(m_pressureTimestampQueries.size()),
        m_pressureTimestampQueries.data());
    if (std::any_of(m_profileTimestampQueries.begin(), m_profileTimestampQueries.end(),
            [](GLuint query) { return query != 0; })) {
        glDeleteQueries(static_cast<GLsizei>(m_profileTimestampQueries.size()),
            m_profileTimestampQueries.data());
    }
    glDeleteQueries(static_cast<GLsizei>(asynchronousQueryCount),
        m_asynchronousStepQueries.data());
    const GLuint buffers[]{m_uBuffer, m_vBuffer, m_uScratchBuffer, m_vScratchBuffer,
        m_pressureBuffer, m_pressureScratchBuffer, m_divergenceBuffer,
        m_cellTypeBuffer, m_residualBuffer, m_reductionBufferA, m_reductionBufferB,
        m_pcgResidualBuffer, m_pcgPreconditionedBuffer, m_pcgDirectionBuffer,
        m_pcgApBuffer, m_pcgScalarBuffer, m_pressureControlBuffer, m_solidMaskBuffer,
        m_previousCellTypeBuffer, m_particleBuffer, m_classificationCounterBuffer,
        m_particleCellCountBuffer, m_classificationDiagnosticBuffer,
        m_classificationFlipBuffer, m_redistributionCounterBuffer,
        m_uValidBuffer, m_vValidBuffer, m_uValidScratchBuffer, m_vValidScratchBuffer,
        m_componentLabelBufferA, m_componentLabelBufferB, m_componentFlagsBuffer,
        m_topologyStatusBuffer, m_topologyChangedBuffer, m_multigridBuildStatusBuffer,
        m_candidateUBuffer, m_candidateVBuffer, m_candidatePressureBuffer,
        m_candidateCellTypeBuffer, m_candidatePreviousCellTypeBuffer,
        m_candidateParticleBuffer, m_candidateParticleCellCountBuffer,
        m_candidateClassificationDiagnosticBuffer, m_candidateClassificationFlipBuffer,
        m_committedRenderUBuffer, m_committedRenderVBuffer,
        m_committedRenderPressureBuffer, m_committedRenderDivergenceBuffer,
        m_committedRenderCellTypeBuffer, m_committedRenderParticleBuffer,
        m_committedRenderParticleCellCountBuffer,
        m_committedRenderClassificationDiagnosticBuffer,
        m_levelSetPhiBuffer, m_levelSetPhiScratchBuffer, m_levelSetPhiAuxBuffer,
        m_levelSetConfidenceBuffer, m_levelSetConfidenceScratchBuffer,
        m_levelSetConfidenceAuxBuffer,
        m_levelSetNarrowBandBuffer, m_levelSetUBuffer, m_levelSetVBuffer,
        m_levelSetUScratchBuffer, m_levelSetVScratchBuffer,
        m_levelSetUValidBuffer, m_levelSetVValidBuffer,
        m_levelSetUValidScratchBuffer, m_levelSetVValidScratchBuffer,
        m_levelSetHorizontalThetaBuffer,
        m_levelSetVerticalThetaBuffer, m_levelSetHorizontalThetaStateBuffer,
        m_levelSetVerticalThetaStateBuffer, m_levelSetPreviousHorizontalThetaBuffer,
        m_levelSetPreviousVerticalThetaBuffer, m_levelSetDiagnosticBuffer,
        m_levelSetAreaBuffer,
        m_levelSetComponentLabelBufferA, m_levelSetComponentLabelBufferB,
        m_levelSetComponentChangedBuffer,
        m_candidateLevelSetPhiBuffer, m_candidateLevelSetConfidenceBuffer,
        m_candidateLevelSetHorizontalThetaBuffer, m_candidateLevelSetVerticalThetaBuffer,
        m_candidateLevelSetHorizontalThetaStateBuffer,
        m_candidateLevelSetVerticalThetaStateBuffer,
        m_committedRenderLevelSetPhiBuffer, m_committedRenderLevelSetConfidenceBuffer,
        m_committedRenderLevelSetHorizontalThetaBuffer,
        m_committedRenderLevelSetVerticalThetaBuffer,
        m_committedRenderLevelSetHorizontalThetaStateBuffer,
        m_committedRenderLevelSetVerticalThetaStateBuffer};
    glDeleteBuffers(static_cast<GLsizei>(std::size(buffers)), buffers);
    m_timerQuery = 0;
    m_pressureTimestampQueries.fill(0);
    m_profileTimestampQueries.fill(0);
    m_asynchronousStepQueries.fill(0);
    m_asynchronousStepQueryPending.fill(false);
    m_uBuffer = m_vBuffer = m_uScratchBuffer = m_vScratchBuffer = 0;
    m_pressureBuffer = m_pressureScratchBuffer = m_divergenceBuffer = 0;
    m_cellTypeBuffer = m_residualBuffer = 0;
    m_reductionBufferA = m_reductionBufferB = 0;
    m_pcgResidualBuffer = m_pcgPreconditionedBuffer = 0;
    m_pcgDirectionBuffer = m_pcgApBuffer = m_pcgScalarBuffer = 0;
    m_pressureControlBuffer = 0;
    m_solidMaskBuffer = m_previousCellTypeBuffer = 0;
    m_particleBuffer = m_classificationCounterBuffer = 0;
    m_particleCellCountBuffer = m_classificationDiagnosticBuffer = 0;
    m_classificationFlipBuffer = m_redistributionCounterBuffer = 0;
    m_uValidBuffer = m_vValidBuffer = 0;
    m_uValidScratchBuffer = m_vValidScratchBuffer = 0;
    m_componentLabelBufferA = m_componentLabelBufferB = 0;
    m_componentFlagsBuffer = m_topologyStatusBuffer = m_topologyChangedBuffer = 0;
    m_multigridBuildStatusBuffer = 0;
    m_candidateUBuffer = m_candidateVBuffer = m_candidatePressureBuffer = 0;
    m_candidateCellTypeBuffer = m_candidatePreviousCellTypeBuffer = 0;
    m_candidateParticleBuffer = m_candidateParticleCellCountBuffer = 0;
    m_candidateClassificationDiagnosticBuffer = m_candidateClassificationFlipBuffer = 0;
    m_committedRenderUBuffer = m_committedRenderVBuffer = 0;
    m_committedRenderPressureBuffer = m_committedRenderDivergenceBuffer = 0;
    m_committedRenderCellTypeBuffer = m_committedRenderParticleBuffer = 0;
    m_committedRenderParticleCellCountBuffer = 0;
    m_committedRenderClassificationDiagnosticBuffer = 0;
    m_levelSetPhiBuffer = m_levelSetPhiScratchBuffer = m_levelSetPhiAuxBuffer = 0;
    m_levelSetConfidenceBuffer = m_levelSetConfidenceScratchBuffer = 0;
    m_levelSetConfidenceAuxBuffer = 0;
    m_levelSetNarrowBandBuffer = 0;
    m_levelSetUBuffer = m_levelSetVBuffer = 0;
    m_levelSetUScratchBuffer = m_levelSetVScratchBuffer = 0;
    m_levelSetUValidBuffer = m_levelSetVValidBuffer = 0;
    m_levelSetUValidScratchBuffer = m_levelSetVValidScratchBuffer = 0;
    m_levelSetHorizontalThetaBuffer = m_levelSetVerticalThetaBuffer = 0;
    m_levelSetHorizontalThetaStateBuffer = m_levelSetVerticalThetaStateBuffer = 0;
    m_levelSetPreviousHorizontalThetaBuffer = m_levelSetPreviousVerticalThetaBuffer = 0;
    m_levelSetDiagnosticBuffer = m_levelSetAreaBuffer = 0;
    m_levelSetComponentLabelBufferA = m_levelSetComponentLabelBufferB = 0;
    m_levelSetComponentChangedBuffer = 0;
    m_candidateLevelSetPhiBuffer = m_candidateLevelSetConfidenceBuffer = 0;
    m_candidateLevelSetHorizontalThetaBuffer = m_candidateLevelSetVerticalThetaBuffer = 0;
    m_candidateLevelSetHorizontalThetaStateBuffer = 0;
    m_candidateLevelSetVerticalThetaStateBuffer = 0;
    m_committedRenderLevelSetPhiBuffer = m_committedRenderLevelSetConfidenceBuffer = 0;
    m_committedRenderLevelSetHorizontalThetaBuffer = 0;
    m_committedRenderLevelSetVerticalThetaBuffer = 0;
    m_committedRenderLevelSetHorizontalThetaStateBuffer = 0;
    m_committedRenderLevelSetVerticalThetaStateBuffer = 0;
    m_levelSetReport = {};
    m_candidateLevelSetReport = {};
    m_committedRenderLevelSetReport = {};
    m_levelSetAcceptedSubsteps = 0;
    m_particleCapacity = 0;
    m_candidateParticleCount = 0;
    m_candidateStateSaved = false;
    m_committedRenderParticleCount = 0;
    m_committedRenderStateEnabled = false;
    m_committedRenderStateValid = false;
}

void GpuMacGrid::checkOpenGlError(const char* operation) const {
    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        throw std::runtime_error(std::string(operation) + " failed with OpenGL error " +
            std::to_string(static_cast<unsigned int>(error)) + '.');
    }
}

} // namespace fluidlab::simulation
