#include "WindowsMoveLoopAdapter.h"

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#pragma comment(lib, "Comctl32.lib")
#endif

namespace fluidlab::desktop {
namespace {
constexpr std::uintptr_t kSubclassId = 0x4C53574Du;
constexpr std::uintptr_t kTimerId = 0x4C530201u;
}

WindowsMoveLoopAdapter::~WindowsMoveLoopAdapter() { uninstall(); }

bool WindowsMoveLoopAdapter::install(
    GLFWwindow* window, Tick tick, unsigned timerMilliseconds) {
    uninstall();
    if (window == nullptr || !tick || timerMilliseconds == 0) return false;
#if defined(_WIN32)
    HWND native = glfwGetWin32Window(window);
    if (native == nullptr) return false;
    m_nativeWindow = native;
    m_tick = std::move(tick);
    m_timerMilliseconds = timerMilliseconds;
    if (SetWindowSubclass(native, &WindowsMoveLoopAdapter::windowProcedure,
            kSubclassId, reinterpret_cast<DWORD_PTR>(this)) == FALSE) {
        m_nativeWindow = nullptr;
        m_tick = {};
        return false;
    }
    return true;
#else
    static_cast<void>(window);
    static_cast<void>(timerMilliseconds);
    return false;
#endif
}

void WindowsMoveLoopAdapter::uninstall() noexcept {
#if defined(_WIN32)
    if (m_nativeWindow != nullptr) {
        const HWND native = static_cast<HWND>(m_nativeWindow);
        KillTimer(native, kTimerId);
        RemoveWindowSubclass(native, &WindowsMoveLoopAdapter::windowProcedure, kSubclassId);
    }
#endif
    m_nativeWindow = nullptr;
    m_tick = {};
    m_scheduler.exit();
}

bool WindowsMoveLoopAdapter::installed() const noexcept { return m_nativeWindow != nullptr; }
bool WindowsMoveLoopAdapter::inMoveSize() const noexcept { return m_scheduler.inMoveSize(); }
std::size_t WindowsMoveLoopAdapter::timerFrameCount() const noexcept { return m_timerFrameCount; }
std::size_t WindowsMoveLoopAdapter::skippedReentrantFrames() const noexcept {
    return m_scheduler.skippedReentrantFrames();
}

void WindowsMoveLoopAdapter::enterMoveSize() noexcept {
    m_scheduler.enter();
#if defined(_WIN32)
    if (m_nativeWindow != nullptr) {
        SetTimer(static_cast<HWND>(m_nativeWindow), kTimerId, m_timerMilliseconds, nullptr);
    }
#endif
}

void WindowsMoveLoopAdapter::exitMoveSize() noexcept {
#if defined(_WIN32)
    if (m_nativeWindow != nullptr) KillTimer(static_cast<HWND>(m_nativeWindow), kTimerId);
#endif
    m_scheduler.exit();
}

void WindowsMoveLoopAdapter::dispatchTimer() {
    if (!m_scheduler.inMoveSize() || !m_tick || !m_scheduler.beginFrame()) return;
    struct Guard final {
        ModalFrameScheduler& scheduler;
        ~Guard() { scheduler.endFrame(); }
    } guard{m_scheduler};
    ++m_timerFrameCount;
    m_tick();
}

void WindowsMoveLoopAdapter::syntheticEnterForValidation() noexcept { enterMoveSize(); }
void WindowsMoveLoopAdapter::syntheticTimerForValidation() { dispatchTimer(); }
void WindowsMoveLoopAdapter::syntheticExitForValidation() noexcept { exitMoveSize(); }

#if defined(_WIN32)
LRESULT CALLBACK WindowsMoveLoopAdapter::windowProcedure(
    HWND window, UINT message, WPARAM wParam,
    LPARAM lParam, UINT_PTR subclassId, DWORD_PTR referenceData) {
    static_cast<void>(subclassId);
    auto* adapter = reinterpret_cast<WindowsMoveLoopAdapter*>(referenceData);
    if (adapter != nullptr) {
        if (message == WM_ENTERSIZEMOVE) adapter->enterMoveSize();
        else if (message == WM_EXITSIZEMOVE) adapter->exitMoveSize();
        else if (message == WM_TIMER && wParam == kTimerId) adapter->dispatchTimer();
        else if (message == WM_NCDESTROY) adapter->uninstall();
    }
    return DefSubclassProc(window, message, wParam, lParam);
}
#endif

} // namespace fluidlab::desktop
