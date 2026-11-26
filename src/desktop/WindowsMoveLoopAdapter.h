#pragma once

#include "DesktopPerformanceMetrics.h"

#include <cstdint>
#include <functional>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <CommCtrl.h>
#endif

struct GLFWwindow;

namespace fluidlab::desktop {

class WindowsMoveLoopAdapter final {
public:
    using Tick = std::function<void()>;

    WindowsMoveLoopAdapter() = default;
    ~WindowsMoveLoopAdapter();
    WindowsMoveLoopAdapter(const WindowsMoveLoopAdapter&) = delete;
    WindowsMoveLoopAdapter& operator=(const WindowsMoveLoopAdapter&) = delete;

    [[nodiscard]] bool install(GLFWwindow* window, Tick tick, unsigned timerMilliseconds = 16);
    void uninstall() noexcept;
    [[nodiscard]] bool installed() const noexcept;
    [[nodiscard]] bool inMoveSize() const noexcept;
    [[nodiscard]] std::size_t timerFrameCount() const noexcept;
    [[nodiscard]] std::size_t skippedReentrantFrames() const noexcept;

    void syntheticEnterForValidation() noexcept;
    void syntheticTimerForValidation();
    void syntheticExitForValidation() noexcept;

private:
#if defined(_WIN32)
    static LRESULT CALLBACK windowProcedure(
        HWND window, UINT message, WPARAM wParam,
        LPARAM lParam, UINT_PTR subclassId, DWORD_PTR referenceData);
#endif
    void enterMoveSize() noexcept;
    void exitMoveSize() noexcept;
    void dispatchTimer();

    void* m_nativeWindow = nullptr;
    Tick m_tick;
    ModalFrameScheduler m_scheduler;
    unsigned m_timerMilliseconds = 16;
    std::size_t m_timerFrameCount = 0;
};

} // namespace fluidlab::desktop
