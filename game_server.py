import cv2
import mediapipe as mp
import numpy as np
import math
import random
from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit
import base64
import time
from threading import Thread
import json
from datetime import datetime

# 配置区域
CONFIG = {
    "MODEL_PATH": 'face_landmarker.task',
    "PLAYER_SIZE": 80,
    "ITEM_SIZE": 40,
    "MOUTH_OPEN_THRESHOLD": 0.01,
    "EAT_DISTANCE": 50,
    "VIDEO_WIDTH": 640,
    "VIDEO_HEIGHT": 480
}

# 难度配置
DIFFICULTY_CONFIGS = {
    "easy": {
        "name": "简单",
        "bomb_chance": 0.1,
        "item_fall_speed": 3,
        "spawn_interval": 40,
        "game_duration": 45,
        "lose_score": -5
    },
    "normal": {
        "name": "普通",
        "bomb_chance": 0.2,
        "item_fall_speed": 5,
        "spawn_interval": 30,
        "game_duration": 30,
        "lose_score": -3
    },
    "hard": {
        "name": "困难",
        "bomb_chance": 0.3,
        "item_fall_speed": 7,
        "spawn_interval": 20,
        "game_duration": 25,
        "lose_score": -1
    },
    "hell": {
        "name": "地狱",
        "bomb_chance": 0.4,
        "item_fall_speed": 10,
        "spawn_interval": 15,
        "game_duration": 20,
        "lose_score": -1
    }
}

# 道具配置
POWERUP_TYPES = {
    "shield": {"name": "护盾", "label": "SHIELD", "duration": 5, "color": (0, 255, 255)},
    "magnet": {"name": "磁铁", "label": "MAGNET", "duration": 5, "color": (255, 0, 255)},
    "slow": {"name": "减速", "label": "SLOW", "duration": 5, "color": (0, 255, 0)},
    "double": {"name": "双倍", "label": "DOUBLE", "duration": 5, "color": (255, 255, 0)}
}

# 成就配置
ACHIEVEMENTS = {
    "first_blood": {"name": "首次得分", "desc": "获得第一个金币", "icon": "🌟"},
    "combo_master": {"name": "连击大师", "desc": "达成10连击", "icon": "🔥"},
    "survivor": {"name": "幸存者", "desc": "完成一局游戏且分数>0", "icon": "🛡️"},
    "speed_demon": {"name": "速度恶魔", "desc": "5秒内吃10个金币", "icon": "⚡"},
    "bomb_dodger": {"name": "炸弹闪避", "desc": "完成游戏且未碰炸弹", "icon": "💎"},
    "high_scorer": {"name": "高分玩家", "desc": "单局得分超过50", "icon": "👑"}
}

# 图片素材路径
IMG_OPEN_PATH = "open.png"
IMG_CLOSE_PATH = "close.png"
IMG_BEAN_PATH = "bean.png"
IMG_BOMB_PATH = "bomb.png"
IMG_SHIELD_PATH = "shield.png"
IMG_MAGNET_PATH = "magnet.png"
IMG_SLOW_PATH = "slow.png"
IMG_DOUBLE_PATH = "double.png"

# Flask + SocketIO 初始化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pacman_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# 游戏状态
game_state = {
    "running": False,
    "paused": False,
    "score": 0,
    "frame_count": 0,
    "damage_timer": 0,
    "game_objects": [],
    "start_time": 0,
    "time_left": 30,
    "last_mouth_x": 0,
    "last_mouth_y": 0,
    "tilt_angle": 0,
    "cap": None,
    "latest_result": None,
    "difficulty": "normal",
    "combo": 0,
    "max_combo": 0,
    "powerups": {},
    "achievements_unlocked": set(),
    "bombs_hit": 0,
    "beans_collected": 0,
    "settings": {
        "sensitivity": 0.01,
        "music_volume": 0.5,
        "sfx_volume": 0.7,
        "quality": "standard"
    }
}

# 排行榜存储
leaderboard = []

# MediaPipe 初始化
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result, output_image, timestamp_ms):
    """MediaPipe 异步回调"""
    game_state["latest_result"] = result


# 工具函数
def rotate_image(image, angle):
    """旋转图像"""
    if image is None:
        return None
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated


def load_image_with_alpha(path, size):
    """加载带透明通道的图片"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"警告：未找到图片 {path}，使用占位图替代")
        img = np.zeros((size, size, 4), dtype=np.uint8)
        color = (0, 255, 0) if "open" in path else (255, 255, 0) if "bean" in path else (255, 0, 0)
        cv2.circle(img, (size // 2, size // 2), size // 2, (*color, 255), -1)
    else:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        if len(img.shape) == 3 and img.shape[2] == 3:
            alpha_channel = np.ones((size, size), dtype=np.uint8) * 255
            img = np.dstack((img, alpha_channel))
    return img


def overlay_image(background, overlay, x, y):
    """叠加透明图片"""
    if overlay is None:
        return background

    h_fg, w_fg = overlay.shape[:2]
    h_bg, w_bg = background.shape[:2]

    x1, y1 = x - w_fg // 2, y - h_fg // 2
    x2, y2 = x1 + w_fg, y1 + h_fg

    if x1 >= w_bg or y1 >= h_bg or x2 <= 0 or y2 <= 0:
        return background

    bg_x1 = max(0, x1)
    bg_y1 = max(0, y1)
    bg_x2 = min(w_bg, x2)
    bg_y2 = min(h_bg, y2)
    fg_x1 = max(0, -x1)
    fg_y1 = max(0, -y1)
    fg_x2 = fg_x1 + (bg_x2 - bg_x1)
    fg_y2 = fg_y1 + (bg_y2 - bg_y1)

    foreground = overlay[fg_y1:fg_y2, fg_x1:fg_x2, :3]
    alpha = overlay[fg_y1:fg_y2, fg_x1:fg_x2, 3] / 255.0

    background_roi = background[bg_y1:bg_y2, bg_x1:bg_x2]
    for c in range(3):
        background_roi[:, :, c] = (alpha * foreground[:, :, c] +
                                   (1 - alpha) * background_roi[:, :, c])
    return background


def check_achievement(achievement_id):
    """检查并解锁成就"""
    if achievement_id not in game_state["achievements_unlocked"]:
        game_state["achievements_unlocked"].add(achievement_id)
        socketio.emit("achievement_unlocked", {
            "id": achievement_id,
            "name": ACHIEVEMENTS[achievement_id]["name"],
            "desc": ACHIEVEMENTS[achievement_id]["desc"],
            "icon": ACHIEVEMENTS[achievement_id]["icon"]
        })


def update_leaderboard(score, difficulty):
    """更新排行榜"""
    global leaderboard
    entry = {
        "score": score,
        "difficulty": difficulty,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "combo": game_state["max_combo"],
        "beans": game_state["beans_collected"]
    }
    leaderboard.append(entry)
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    leaderboard = leaderboard[:10]  # 只保留前10名


# 加载素材图片
IMAGES = {
    "open": load_image_with_alpha(IMG_OPEN_PATH, CONFIG["PLAYER_SIZE"]),
    "close": load_image_with_alpha(IMG_CLOSE_PATH, CONFIG["PLAYER_SIZE"]),
    "bean": load_image_with_alpha(IMG_BEAN_PATH, CONFIG["ITEM_SIZE"]),
    "bomb": load_image_with_alpha(IMG_BOMB_PATH, CONFIG["ITEM_SIZE"]),
    "shield": load_image_with_alpha(IMG_SHIELD_PATH, CONFIG["ITEM_SIZE"]),
    "magnet": load_image_with_alpha(IMG_MAGNET_PATH, CONFIG["ITEM_SIZE"]),
    "slow": load_image_with_alpha(IMG_SLOW_PATH, CONFIG["ITEM_SIZE"]),
    "double": load_image_with_alpha(IMG_DOUBLE_PATH, CONFIG["ITEM_SIZE"])
}


# 游戏主循环
def game_loop():
    """游戏主循环"""
    # 检查模型文件
    try:
        with open(CONFIG["MODEL_PATH"], 'r') as f:
            pass
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {CONFIG['MODEL_PATH']}")
        socketio.emit("error", {"message": f"找不到模型文件 {CONFIG['MODEL_PATH']}"})
        game_state["running"] = False
        return

    # 获取难度配置
    diff_config = DIFFICULTY_CONFIGS[game_state["difficulty"]]

    # 初始化 MediaPipe
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=CONFIG["MODEL_PATH"]),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        # 初始化摄像头
        game_state["cap"] = cv2.VideoCapture(0)
        game_state["cap"].set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["VIDEO_WIDTH"])
        game_state["cap"].set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["VIDEO_HEIGHT"])

        if not game_state["cap"].isOpened():
            print("错误: 无法打开摄像头")
            socketio.emit("error", {"message": "无法打开摄像头"})
            game_state["running"] = False
            return

        # 3秒倒计时
        countdown_start = time.time()
        while game_state["running"]:
            elapsed = time.time() - countdown_start
            if elapsed >= 3:
                break

            success, frame = game_state["cap"].read()
            if not success: break

            frame = cv2.flip(frame, 1)

            # 计算倒计时数字 (3 -> 2 -> 1)
            display_num = str(3 - int(elapsed))

            font_scale = 4 + (elapsed % 1) * 0.5

            cv2.putText(frame, display_num, (CONFIG["VIDEO_WIDTH"] // 2 - 30, CONFIG["VIDEO_HEIGHT"] // 2 + 30),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 255), 4)

            _, buffer = cv2.imencode('.jpg', frame)

            socketio.emit("game_update", {
                "frame": base64.b64encode(buffer).decode('utf-8'),
                "score": 0,
                "time_left": diff_config["game_duration"],
                "combo": 0,
                "powerups": {}
            })
            time.sleep(0.03)  # 保持约30FPS的流畅度

        # 倒计时结束后，正式开始计时
        game_state["start_time"] = time.time()
        last_bean_time = time.time()
        combo_beans_count = 0

        while game_state["running"] and game_state["cap"].isOpened():
            # 暂停逻辑
            if game_state["paused"]:
                time.sleep(0.1)
                continue

            success, frame = game_state["cap"].read()
            if not success:
                break

            height, width = CONFIG["VIDEO_HEIGHT"], CONFIG["VIDEO_WIDTH"]

            # 1. 时间计算
            elapsed_time = int(time.time() - game_state["start_time"])
            game_state["time_left"] = max(0, diff_config["game_duration"] - elapsed_time)

            # 2. 游戏结束判定
            if game_state["time_left"] <= 0 or game_state["score"] <= diff_config["lose_score"]:
                game_state["running"] = False

                # 检查成就
                if game_state["score"] > 0:
                    check_achievement("survivor")
                if game_state["score"] > 50:
                    check_achievement("high_scorer")
                if game_state["max_combo"] >= 10:
                    check_achievement("combo_master")
                if game_state["bombs_hit"] == 0 and game_state["beans_collected"] > 0:
                    check_achievement("bomb_dodger")

                # 更新排行榜
                update_leaderboard(game_state["score"], game_state["difficulty"])

                socketio.emit("game_over", {
                    "score": game_state["score"],
                    "combo": game_state["max_combo"],
                    "beans": game_state["beans_collected"],
                    "achievements": list(game_state["achievements_unlocked"])
                })
                break

            # 镜像翻转
            frame = cv2.flip(frame, 1)

            # 3. 面部检测
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp)

            mouth_center = {"x": width // 2, "y": height // 2}
            is_mouth_open = False
            tilt_angle = game_state["tilt_angle"]

            if game_state["latest_result"] and game_state["latest_result"].face_landmarks:
                face = game_state["latest_result"].face_landmarks[0]
                upper = face[13]
                lower = face[14]

                ux = int(upper.x * width)
                uy = int(upper.y * height)
                lx = int(lower.x * width)
                ly = int(lower.y * height)

                mouth_center = {"x": (ux + lx) // 2, "y": (uy + ly) // 2}

                # 判断嘴巴是否张开（使用设置中的灵敏度）
                open_val = abs(upper.y - lower.y)
                is_mouth_open = open_val > game_state["settings"]["sensitivity"]

                # 计算朝向（8方向旋转）
                dx = mouth_center["x"] - game_state["last_mouth_x"]
                dy = mouth_center["y"] - game_state["last_mouth_y"]

                if abs(dx) > 8 or abs(dy) > 8:
                    angle_deg = math.degrees(math.atan2(dy, dx))
                    target_angle = round(angle_deg / 45) * 45
                    tilt_angle = -target_angle
                    game_state["last_mouth_x"] = mouth_center["x"]
                    game_state["last_mouth_y"] = mouth_center["y"]
                    game_state["tilt_angle"] = tilt_angle

            # 4. 绘制玩家（吃豆人）
            # 状态特效渲染

            # 1. 减速 (Slow) - 全屏冰冻滤镜
            if "slow" in game_state["powerups"] and game_state["powerups"]["slow"] > 0:
                ice_overlay = np.zeros_like(frame)
                ice_overlay[:] = (255, 255, 0)  # 青蓝色 (BGR)
                cv2.addWeighted(frame, 0.9, ice_overlay, 0.1, 0, frame)
                # 边缘冰霜效果
                cv2.rectangle(frame, (0, 0), (width, height), (255, 200, 0), 10)

            # 2. 磁铁 (Magnet) - 磁力波纹
            if "magnet" in game_state["powerups"] and game_state["powerups"]["magnet"] > 0:
                # 动态半径，造成一种扩散的波纹效果
                wave_radius = 60 + int(math.sin(time.time() * 10) * 10)
                cv2.circle(frame, (mouth_center["x"], mouth_center["y"]), wave_radius, (255, 0, 255), 2)
                cv2.circle(frame, (mouth_center["x"], mouth_center["y"]), wave_radius - 20, (200, 0, 200), 1)

            # 3. 双倍 (Double) - 金色光晕
            if "double" in game_state["powerups"] and game_state["powerups"]["double"] > 0:
                # 在玩家位置画多层半透明黄色圆，模拟发光
                glow_overlay = frame.copy()
                cv2.circle(glow_overlay, (mouth_center["x"], mouth_center["y"]), 70, (0, 215, 255), -1)  # 金色
                cv2.addWeighted(frame, 0.7, glow_overlay, 0.3, 0, frame)
                # 头顶显示 x2
                cv2.putText(frame, "x2", (mouth_center["x"] + 40, mouth_center["y"] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 255), 3)  # 加粗
                cv2.putText(frame, "x2", (mouth_center["x"] + 40, mouth_center["y"] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            # 4. 护盾 (Shield) - 能量罩
            if "shield" in game_state["powerups"] and game_state["powerups"]["shield"] > 0:
                shield_overlay = frame.copy()
                cv2.circle(shield_overlay, (mouth_center["x"], mouth_center["y"]), 60, (255, 255, 0), -1)  # 青色填充
                cv2.circle(shield_overlay, (mouth_center["x"], mouth_center["y"]), 60, (0, 255, 255), 3)  # 亮边框
                cv2.addWeighted(frame, 0.6, shield_overlay, 0.4, 0, frame)

            # 绘制玩家主体
            player_img = IMAGES["open"] if is_mouth_open else IMAGES["close"]
            rotated_player = rotate_image(player_img, tilt_angle)
            overlay_image(frame, rotated_player, mouth_center["x"], mouth_center["y"])

            # 5. 生成/处理物品
            current_fall_speed = diff_config["item_fall_speed"]
            if "slow" in game_state["powerups"] and game_state["powerups"]["slow"] > 0:
                current_fall_speed = max(2, current_fall_speed // 2)

            if game_state["frame_count"] % (diff_config["spawn_interval"] // 2) == 0:  # 增加生成频率
                rand_val = random.random()
                # 20% 概率生成道具
                if rand_val < 0.2:
                    powerup_type = random.choice(list(POWERUP_TYPES.keys()))
                    game_state["game_objects"].append({
                        "x": random.randint(50, width - 50),
                        "y": 0,
                        "type": "powerup",
                        "powerup_type": powerup_type
                    })
                # 80% 概率生成普通物品 (炸弹或金豆)
                else:
                    obj_type = "bomb" if random.random() < diff_config["bomb_chance"] else "bean"
                    game_state["game_objects"].append({
                        "x": random.randint(50, width - 50),
                        "y": 0,
                        "type": obj_type
                    })

            new_objects = []
            for obj in game_state["game_objects"]:
                obj["y"] += current_fall_speed

                # 磁铁效果
                if "magnet" in game_state["powerups"] and game_state["powerups"]["magnet"] > 0:
                    if obj["type"] == "bean":
                        dx = mouth_center["x"] - obj["x"]
                        dy = mouth_center["y"] - obj["y"]
                        dist = math.hypot(dx, dy)
                        if dist > 0 and dist < 200:
                            obj["x"] += int(dx / dist * 3)
                            obj["y"] += int(dy / dist * 3)

                # 碰撞检测
                dist = math.hypot(mouth_center["x"] - obj["x"], mouth_center["y"] - obj["y"])
                if dist < CONFIG["EAT_DISTANCE"] and is_mouth_open:
                    if obj["type"] == "bean":
                        score_add = 1
                        if "double" in game_state["powerups"] and game_state["powerups"]["double"] > 0:
                            score_add = 2

                        game_state["score"] += score_add
                        game_state["combo"] += 1
                        game_state["beans_collected"] += 1
                        game_state["max_combo"] = max(game_state["max_combo"], game_state["combo"])

                        # 检查成就
                        if game_state["beans_collected"] == 1:
                            check_achievement("first_blood")

                        # 速度恶魔成就
                        current_time = time.time()
                        if current_time - last_bean_time < 5:
                            combo_beans_count += 1
                            if combo_beans_count >= 10:
                                check_achievement("speed_demon")
                                combo_beans_count = 0
                        else:
                            combo_beans_count = 1
                        last_bean_time = current_time

                    elif obj["type"] == "bomb":
                        if "shield" in game_state["powerups"] and game_state["powerups"]["shield"] > 0:
                            # 护盾抵消伤害
                            game_state["powerups"]["shield"] = 0
                        else:
                            game_state["score"] -= 5
                            game_state["damage_timer"] = 10
                            game_state["combo"] = 0
                            game_state["bombs_hit"] += 1

                    elif obj["type"] == "powerup":
                        powerup_type = obj["powerup_type"]
                        game_state["powerups"][powerup_type] = POWERUP_TYPES[powerup_type]["duration"] * 30
                        socketio.emit("powerup_collected", {
                            "type": powerup_type,
                            "name": POWERUP_TYPES[powerup_type]["name"]
                        })

                    continue

                if obj["y"] > height:
                    # 物品掉出屏幕，重置连击
                    if obj["type"] == "bean":
                        game_state["combo"] = 0
                    continue

                # 绘制物品
                if obj["type"] == "powerup":
                    powerup_type = obj["powerup_type"]
                    if powerup_type in IMAGES:
                        overlay_image(frame, IMAGES[powerup_type], obj["x"], obj["y"])
                    else:
                        powerup_config = POWERUP_TYPES[obj["powerup_type"]]
                        cv2.circle(frame, (obj["x"], obj["y"]), 20, powerup_config["color"], -1)
                        cv2.circle(frame, (obj["x"], obj["y"]), 20, (255, 255, 255), 2)
                else:
                    item_img = IMAGES["bomb"] if obj["type"] == "bomb" else IMAGES["bean"]
                    overlay_image(frame, item_img, obj["x"], obj["y"])

                new_objects.append(obj)

            game_state["game_objects"] = new_objects

            # 6. 更新道具持续时间
            for powerup in list(game_state["powerups"].keys()):
                if game_state["powerups"][powerup] > 0:
                    game_state["powerups"][powerup] -= 1
                    if game_state["powerups"][powerup] <= 0:
                        del game_state["powerups"][powerup]

            # 7. 受伤特效
            if game_state["damage_timer"] > 0:
                red_overlay = np.zeros_like(frame)
                red_overlay[:] = (0, 0, 255)
                cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0, frame)
                game_state["damage_timer"] -= 1

            # 8. 绘制UI

            # 道具状态显示
            powerup_y = 30
            for powerup_type, frames_left in game_state["powerups"].items():
                seconds_left = frames_left // 30
                powerup_config = POWERUP_TYPES[powerup_type]
                text = f"{powerup_config['label']}: {seconds_left}s"
                cv2.putText(frame, text, (20, powerup_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, powerup_config["color"], 2)
                powerup_y += 35

            # 9. 推送数据到前端
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            socketio.emit("game_update", {
                "frame": frame_base64,
                "score": game_state["score"],
                "time_left": game_state["time_left"],
                "combo": game_state["combo"],
                "powerups": {k: v // 30 for k, v in game_state["powerups"].items()}
            })

            game_state["frame_count"] += 1
            time.sleep(1 / 30)  # 30 FPS

        # 释放资源
        if game_state["cap"]:
            game_state["cap"].release()


# SocketIO 事件监听
@socketio.on("start_game")
def handle_start_game(data):
    """接收前端的开始游戏指令"""
    if not game_state["running"]:
        difficulty = data.get("difficulty", "normal")
        game_state.update({
            "running": True,
            "paused": False,
            "score": 0,
            "frame_count": 0,
            "damage_timer": 0,
            "game_objects": [],
            "start_time": time.time(),
            "time_left": DIFFICULTY_CONFIGS[difficulty]["game_duration"],
            "last_mouth_x": CONFIG["VIDEO_WIDTH"] // 2,
            "last_mouth_y": CONFIG["VIDEO_HEIGHT"] // 2,
            "tilt_angle": 0,
            "latest_result": None,
            "difficulty": difficulty,
            "combo": 0,
            "max_combo": 0,
            "powerups": {},
            "bombs_hit": 0,
            "beans_collected": 0
        })
        Thread(target=game_loop, daemon=True).start()
        emit("game_started", {"status": "success", "difficulty": DIFFICULTY_CONFIGS[difficulty]["name"]})


@socketio.on("pause_game")
def handle_pause_game():
    """暂停游戏"""
    game_state["paused"] = not game_state["paused"]
    emit("game_paused", {"paused": game_state["paused"]})


@socketio.on("restart_game")
def handle_restart_game(data):
    """接收前端的重玩指令"""
    game_state["running"] = False
    time.sleep(0.5)
    handle_start_game(data)


@socketio.on("stop_game")
def handle_stop_game():
    """接收前端的停止游戏指令"""
    game_state["running"] = False
    emit("game_stopped", {"status": "success"})


@socketio.on("update_settings")
def handle_update_settings(data):
    """更新游戏设置"""
    game_state["settings"].update(data)
    emit("settings_updated", {"status": "success"})


@socketio.on("get_leaderboard")
def handle_get_leaderboard():
    """获取排行榜"""
    emit("leaderboard_data", {"leaderboard": leaderboard})


# Flask 路由
@app.route("/")
def index():
    """渲染前端页面"""
    return render_template("index.html")


@app.route("/api/difficulties")
def get_difficulties():
    """获取难度列表"""
    return jsonify(DIFFICULTY_CONFIGS)


@app.route("/api/achievements")
def get_achievements():
    """获取成就列表"""
    return jsonify(ACHIEVEMENTS)


# 启动服务
if __name__ == "__main__":
    print("=" * 50)
    print("吃豆人游戏服务器启动中...")
    print("=" * 50)
    print("本地访问: http://localhost:5000")
    print("局域网访问: http://你的IP:5000")
    print("=" * 50)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)