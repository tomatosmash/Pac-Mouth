# 🎮 Face Tracking Pac-Man Game (吃豆人面部体感游戏)

这是一个基于 AI 面部识别的 Web 互动游戏。利用 **MediaPipe** 和 **OpenCV** 进行实时的面部关键点检测，玩家可以通过张嘴和头部移动来控制游戏中的角色吃掉金豆、躲避炸弹并获取强力道具。

---

## ✨ 功能特色

*   **面部控制**：张嘴吃豆，摇头控制角色旋转方向。
*   **实时互动**：基于 Flask + Socket.IO 实现低延迟的画面传输。
*   **丰富道具**：
    *   🛡️ **护盾 (Shield)**：抵挡炸弹伤害。
    *   🧲 **磁铁 (Magnet)**：自动吸附周围的金豆。
    *   ❄️ **减速 (Slow)**：减缓炸弹和金豆的下落速度。
    *   ✨ **双倍 (Double)**：短时间内得分翻倍。
*   **视觉特效**：全屏冰冻效果、动态能量罩、磁力波纹和金色光晕等炫酷反馈。
*   **多重难度**：包含简单、普通、困难、地狱四种模式。
*   **成就系统**：记录并在达成特定条件时解锁成就徽章。

---

## 🛠️ 环境要求

*   Python 3.8+
*   摄像头 (用于面部捕捉)

---

## 📦 安装与运行

1.  **安装依赖库**
    在终端中运行以下命令安装所需 Python 库：
    ```bash
    pip install flask flask-socketio opencv-python mediapipe numpy
    ```

2.  **准备模型文件**
    确保目录下包含 MediaPipe 的面部标记模型文件：`face_landmarker.task`。

3.  **启动游戏服务器**
    ```bash
    python game_server.py
    ```

4.  **开始游戏**
    打开浏览器访问：[http://localhost:5000](http://localhost:5000)

---

## 🕹️ 游戏玩法

1.  **准备阶段**：点击“开始游戏”，会有 3 秒倒计时，请调整好坐姿，确保摄像头能清晰拍到你的脸。
2.  **吃豆得分**：当金豆 🟡 落下经过嘴巴时，**张大嘴巴**即可吃掉得分。
3.  **躲避炸弹**：炸弹 💣 会扣分并中断连击，请在炸弹落下时**闭上嘴巴**或移动头部躲避。
4.  **吃道具**：特殊的彩色道具会随机掉落，张嘴吃掉它们可以获得超能力！

---

## 📂 文件结构

*   `game_server.py`: 游戏后端核心逻辑 (Flask 服务 + OpenCV 处理)。
*   `templates/index.html`: 前端界面 (Vue.js + TailwindCSS)。
*   `face_landmarker.task`: Google MediaPipe 模型文件。
*   `*.png`: 游戏素材图片 (open, close, bean, bomb, shield, magnet, slow, double)。

---

## 📝 贡献
欢迎提交 Issue 或 Pull Request 来改进游戏体验！

---
Enjoy the game! 👻
