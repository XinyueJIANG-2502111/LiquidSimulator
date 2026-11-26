# LiquidSimulator Desktop 2.0 Portfolio Edition

## 日本語

LiquidSimulator は、C++20、OpenGL 4.6 Compute Shader、および二次元 MAC グリッドで実装された Windows 向けインタラクティブ流体シミュレーターです。

### 動作環境

- Windows 10/11 x64。
- OpenGL 4.6 Compute Shader に対応した GPU とグラフィックスドライバー。
- NVIDIA RTX シリーズを推奨。現在の検証環境は NVIDIA GeForce RTX 3070 です。
- Release ZIP には必要な再頒布可能 MSVC ランタイムが含まれているため、Visual Studio のインストールは不要です。

### 起動方法

`LiquidSimulator.exe` をダブルクリックすると、安定した水面を初期状態とする標準 Desktop モードが起動します。

ターミナルから起動する場合：

```text
LiquidSimulator.exe --desktop-interactive
```

ダム崩壊のデモを起動する場合：

```text
LiquidSimulator.exe --desktop-interactive-dam-break
```

### 基本操作

- シミュレーション領域の AIR 部分でマウス左ボタンを押し続ける：カーソル位置から水を継続的に注入します。ボタンを離すと停止します。
- ウィンドウのタイトルバーを任意の方向へドラッグ：容器の二次元並進加速度に応じて水が慣性運動します。
- `I`：ウィンドウ慣性入力の有効・無効を切り替えます。
- `O`：性能および物理時間の診断表示を切り替えます。
- `V`：VSync の有効・無効を切り替えます。
- `P`：Marker Particle の表示・非表示を切り替えます。起動時は非表示です。
- 一時停止、1 ステップ実行、Reset、CPU/GPU、同期/非同期の切り替えには、プログラム内の既存キー操作を使用します。

ウィンドウが静止している場合や等速で移動している場合、継続的な慣性力は発生しません。急な移動開始、急停止、方向反転では、それぞれに対応する過渡的な水面応答が発生します。

表示領域はシミュレーション領域と同じ 4:3 のアスペクト比に固定されており、ウィンドウの端をドラッグした場合も縦横比を維持したまま拡大・縮小されます。

### Portfolio Edition の標準設定

- 64x48 の安定した水面を初期状態として使用。
- 正式な GPU MAC ソルバーと FP64 MGPCG 圧力ソルバー。
- 適応 CFL、Candidate ロールバック、安全な Committed スナップショット。
- 濃紺の背景、低コントラストのグリッド、半透明の青い水面。
- 独立した実時間ベースの物理予算と、上限付き非同期 Batch スケジューリング。

### 既知の制限

- 現在は二次元流体と容器の並進慣性のみをサポートしています。
- 容器の回転、傾斜、三次元流体、容器外への流出には対応していません。
- 128x128 などの高解像度ではリアルタイム進行を保証しません。
- GPU VOF、CLSVOF、PLIC、Ghost Fluid は保存された研究モジュールであり、正式な Desktop 物理経路には接続されていません。
- Windows のタイトルバーをドラッグしている間の描画間隔は、システムのメッセージループとグラフィックスドライバーの影響を受ける場合があります。

技術詳細、ビルド方法、検証用コマンドについては `README.md` を参照してください。

---

## English

LiquidSimulator is an interactive fluid simulator for Windows implemented with C++20, OpenGL 4.6 Compute Shaders, and a two-dimensional MAC grid.

### System requirements

- Windows 10/11 x64.
- A GPU and graphics driver supporting OpenGL 4.6 Compute Shaders.
- An NVIDIA RTX-series GPU is recommended. The currently validated GPU is the NVIDIA GeForce RTX 3070.
- The Release ZIP includes the required redistributable MSVC runtime libraries, so Visual Studio is not required to run the application.

### Launching the application

Double-click `LiquidSimulator.exe` to start the default Desktop mode with a settled water state.

To start it from a terminal:

```text
LiquidSimulator.exe --desktop-interactive
```

To run the original dam-break demonstration:

```text
LiquidSimulator.exe --desktop-interactive-dam-break
```

### Basic controls

- Hold the left mouse button over an AIR region: continuously inject water at the cursor; release the button to stop.
- Drag the window title bar in any direction: the water responds to the two-dimensional translational acceleration of the container.
- `I`: enable or disable window-inertia input.
- `O`: show or hide performance and physical-time diagnostics.
- `V`: enable or disable VSync.
- `P`: show or hide Marker Particles. They are hidden at startup.
- Use the existing in-application controls for pause, single-step, Reset, CPU/GPU switching, and synchronous/asynchronous switching.

A stationary window or a window moving at constant velocity does not generate a continuous inertial force. Rapid starts, sudden stops, and direction reversals produce the corresponding transient fluid response.

The client area is constrained to the same 4:3 aspect ratio as the simulation domain. Resizing the window preserves that ratio and does not stretch the rendered fluid.

### Portfolio Edition defaults

- A settled 64x48 water state.
- The production GPU MAC solver and FP64 MGPCG pressure solver.
- Adaptive CFL control, Candidate rollback, and safe Committed snapshots.
- A dark blue-black background, low-contrast grid, and translucent blue water.
- An independent wall-clock physics budget with bounded asynchronous Batch scheduling.

### Known limitations

- The current application supports only two-dimensional fluid and translational container inertia.
- Container rotation, tilting, three-dimensional fluid, and liquid leaving the container are not supported.
- Real-time progression is not guaranteed at higher resolutions such as 128x128.
- GPU VOF, CLSVOF, PLIC, and Ghost Fluid remain preserved research modules and are not connected to the production Desktop physics path.
- Rendering cadence while dragging the Windows title bar may be affected by the system message loop and graphics driver.

See `README.md` for technical details, build instructions, and validation commands.

---

## 中文

LiquidSimulator 是一个使用 C++20、OpenGL 4.6 Compute Shader 和二维 MAC 网格实现的 Windows 互动流体模拟器。

### 运行要求

- Windows 10/11 x64。
- 支持 OpenGL 4.6 Compute Shader 的显卡与驱动。
- 推荐 NVIDIA RTX 系列显卡；当前验证硬件为 NVIDIA GeForce RTX 3070。
- Release ZIP 已包含程序所需的可再发行 MSVC 运行库，不要求安装 Visual Studio。

### 启动

双击 `LiquidSimulator.exe` 即可进入默认稳定水体 Desktop 模式。

也可以从终端运行：

```text
LiquidSimulator.exe --desktop-interactive
```

原始溃坝演示：

```text
LiquidSimulator.exe --desktop-interactive-dam-break
```

### 基本操作

- 在模拟区域的 AIR 区域按住鼠标左键：持续从光标位置注水；松开后停止。
- 向任意方向拖动窗口标题栏：通过容器二维平移加速度驱动水体惯性晃动。
- `I`：开关窗口惯性输入。
- `O`：开关性能与物理时间诊断。
- `V`：开关 VSync。
- `P`：显示或隐藏 Marker Particle；启动时默认隐藏。
- 其他暂停、单步、Reset、CPU/GPU 及同步/异步按键沿用程序内现有操作。

窗口静止或匀速移动不会产生持续惯性力；快速启动、急停和反向拖动会产生相应的瞬态响应。

窗口客户区固定为与模拟区域一致的 4:3 宽高比；拖动窗口边缘时会保持等比例缩放，不会压扁或拉伸液体画面。

### 作品版默认配置

- 64x48 稳定水体初态。
- 正式 GPU MAC 与 FP64 MGPCG 压力求解。
- 自适应 CFL、Candidate 回滚和安全 Committed 快照。
- 深蓝黑背景、低对比度网格和半透明蓝色水体。
- 独立墙钟物理预算与有界异步 Batch 调度。

### 已知限制

- 当前仅支持二维流体和容器平移惯性。
- 不支持容器旋转、倾倒、三维流体或液体离开容器。
- 128x128 等较高分辨率不保证实时推进。
- GPU VOF、CLSVOF、PLIC 和 Ghost Fluid 为保留的研究模块，未接入正式 Desktop 物理路径。
- Windows 标题栏拖动期间的渲染节奏会受到系统消息循环和显卡驱动影响。

完整技术说明、构建方式和验证入口见 `README.md`。
