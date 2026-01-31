import cv2
import mediapipe as mp
import time
import random
import math
import numpy as np

# ================= 配置区域 =================
MODEL_PATH = 'face_landmarker.task' 

IMG_OPEN_PATH = "open.png"
IMG_CLOSE_PATH = "close.png"
IMG_BEAN_PATH = "bean.png"
IMG_BOMB_PATH = "bomb.png"

PLAYER_SIZE = 80     
ITEM_SIZE = 40       
MOUTH_OPEN_THRESHOLD = 0.01
EAT_DISTANCE = 50    
BOMB_CHANCE = 0.2    

# 【新增】游戏结束条件
GAME_DURATION = 30   # 游戏时长 (秒)
LOSE_SCORE = -1     # 分数低于这个值直接输
# ===========================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

def rotate_image(image, angle):
    """ 【新增】旋转图片函数 """
    if image is None: return None
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 执行旋转，背景保持透明 (BORDER_TRANSPARENT)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def overlay_image(background, overlay, x, y, size=None):
    if overlay is None: return background
    if size is not None: overlay = cv2.resize(overlay, (size, size))
    
    h_fg, w_fg = overlay.shape[:2]
    h_bg, w_bg = background.shape[:2]
    
    x1, y1 = x - w_fg // 2, y - h_fg // 2
    x2, y2 = x1 + w_fg, y1 + h_fg
    
    if x1 >= w_bg or y1 >= h_bg or x2 <= 0 or y2 <= 0: return background
    
    bg_x1 = max(0, x1); bg_y1 = max(0, y1)
    bg_x2 = min(w_bg, x2); bg_y2 = min(h_bg, y2)
    fg_x1 = max(0, -x1); fg_y1 = max(0, -y1)
    fg_x2 = fg_x1 + (bg_x2 - bg_x1)
    fg_y2 = fg_y1 + (bg_y2 - bg_y1)
    
    foreground = overlay[fg_y1:fg_y2, fg_x1:fg_x2, :3]
    alpha = overlay[fg_y1:fg_y2, fg_x1:fg_x2, 3] / 255.0
    
    background_roi = background[bg_y1:bg_y2, bg_x1:bg_x2]
    for c in range(3):
        background_roi[:, :, c] = (alpha * foreground[:, :, c] + 
                                   (1 - alpha) * background_roi[:, :, c])
    return background

def run_game():
    img_open = cv2.imread(IMG_OPEN_PATH, cv2.IMREAD_UNCHANGED)
    img_close = cv2.imread(IMG_CLOSE_PATH, cv2.IMREAD_UNCHANGED)
    img_bean = cv2.imread(IMG_BEAN_PATH, cv2.IMREAD_UNCHANGED)
    img_bomb = cv2.imread(IMG_BOMB_PATH, cv2.IMREAD_UNCHANGED)

    try:
        with open(MODEL_PATH, 'r') as f: pass
    except FileNotFoundError:
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        
        game_objects = [] 
        score = 0
        frame_count = 0
        damage_timer = 0
        
        # 运动与方向变量
        last_mouth_x = 0
        last_mouth_y = 0 # 【新增】记录Y坐标
        is_facing_right = True
        tilt_angle = 0   # 【新增】倾斜角度
        
        # 计时器
        start_time = time.time()
        is_game_over = False

        print("游戏开始！")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            height, width, _ = frame.shape
            frame = cv2.flip(frame, 1) 
            
            # 计算剩余时间
            elapsed_time = time.time() - start_time
            time_left = max(0, int(GAME_DURATION - elapsed_time))

            # --- 游戏结束判定 ---
            if time_left == 0 or score <= LOSE_SCORE:
                is_game_over = True

            if not is_game_over:
                # 仅在游戏进行时进行推理
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                timestamp = int(time.time() * 1000)
                landmarker.detect_async(mp_image, timestamp)

            mouth_center = (0, 0)
            is_mouth_open = False
            
            # --- 处理 AI 结果 ---
            if not is_game_over and latest_result and latest_result.face_landmarks:
                face = latest_result.face_landmarks[0]
                upper = face[13]
                lower = face[14]
                
                ux, uy = int(upper.x * width), int(upper.y * height)
                lx, ly = int(lower.x * width), int(lower.y * height)
                mouth_center = ((ux + lx) // 2, (uy + ly) // 2)
                
                # --- 核心：计算朝向与旋转 ---
                current_x = mouth_center[0]
                current_y = mouth_center[1]
                
                # 1. 运动方向判定 (8方向)
                dx = current_x - last_mouth_x
                dy = current_y - last_mouth_y
                
                # 移动阈值降为 8
                if abs(dx) > 8 or abs(dy) > 8:
                    # 计算运动角度 (y轴向下为正)
                    angle_deg = math.degrees(math.atan2(dy, dx))
                    
                    # 离散化为 8 个方向 (每 45 度一个)
                    # 将角度四舍五入到最近的 45 度倍数
                    # 比如 0, 45, 90, 135, 180, -135, -90, -45
                    target_angle = round(angle_deg / 45) * 45
                    
                    # cv2.getRotationMatrix2D 的角度是逆时针为正
                    # 我们的 angle_deg 是基于 y 轴向下计算的 (顺时针为正)
                    # 所以需要取反来获得正确的旋转角度
                    tilt_angle = -target_angle
                
                last_mouth_x = current_x
                last_mouth_y = current_y
                # -------------------------

                open_val = abs(upper.y - lower.y)
                if open_val > MOUTH_OPEN_THRESHOLD:
                    is_mouth_open = True
                
                # --- 绘制主角 ---
                current_img = img_open if is_mouth_open else img_close
                if current_img is not None:
                    # 使用旋转函数处理所有方向 (包括翻转)
                    # 假设原始素材是朝右的
                    current_img = rotate_image(current_img, tilt_angle)

                    overlay_image(frame, current_img, mouth_center[0], mouth_center[1], PLAYER_SIZE)
                else:
                    cv2.circle(frame, mouth_center, 30, (0,255,0), -1)

            # --- 物品逻辑 (仅在游戏进行时) ---
            if not is_game_over:
                if frame_count % 30 == 0:
                    obj_type = 'bomb' if random.random() < BOMB_CHANCE else 'bean'
                    game_objects.append({
                        'x': random.randint(50, width - 50), 
                        'y': 0, 
                        'type': obj_type
                    })

                for obj in game_objects[:]:
                    obj['y'] += 5
                    dist = 999
                    if mouth_center != (0,0):
                        dist = math.hypot(mouth_center[0] - obj['x'], mouth_center[1] - obj['y'])
                    
                    if dist < EAT_DISTANCE and is_mouth_open:
                        if obj['type'] == 'bean': score += 1
                        elif obj['type'] == 'bomb': 
                            score -= 5
                            damage_timer = 10 
                        game_objects.remove(obj)
                    elif obj['y'] > height:
                        game_objects.remove(obj)
                    else:
                        img_to_draw = img_bomb if obj['type'] == 'bomb' else img_bean
                        if img_to_draw is not None:
                            overlay_image(frame, img_to_draw, obj['x'], obj['y'], ITEM_SIZE)
                        else:
                            cv2.circle(frame, (obj['x'], obj['y']), 15, (0,255,255), -1)

            frame_count += 1

            # --- 特效与UI ---
            if damage_timer > 0:
                red_overlay = np.zeros_like(frame)
                red_overlay[:] = (0, 0, 255)
                cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0, frame)
                damage_timer -= 1

            # --- 1. 顶部 UI 背景条 (半透明) ---
            ui_overlay = frame.copy()
            cv2.rectangle(ui_overlay, (0, 0), (width, 80), (30, 30, 30), -1) # 深灰色背景
            alpha = 0.6
            cv2.addWeighted(ui_overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # --- 2. 绘制分数 (左侧) ---
            # 边框 + 填充
            score_text = f"SCORE: {score}"
            # 阴影效果
            cv2.putText(frame, score_text, (32, 52), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0), 2)
            color_score = (50, 255, 50) if score >= 0 else (50, 50, 255) # 亮绿或亮红
            cv2.putText(frame, score_text, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, color_score, 2)

            # --- 3. 绘制时间进度条 (右侧) ---
            bar_width = 200
            bar_height = 20
            bar_x = width - bar_width - 30
            bar_y = 30
            
            # 进度比例
            progress = max(0, time_left / GAME_DURATION)
            
            # 绘制边框
            cv2.rectangle(frame, (bar_x-2, bar_y-2), (bar_x + bar_width+2, bar_y + bar_height+2), (200, 200, 200), 2)
            
            # 绘制背景
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # 绘制进度 (根据时间变色: 绿 -> 黄 -> 红)
            if progress > 0.6: bar_color = (0, 255, 0)
            elif progress > 0.3: bar_color = (0, 255, 255)
            else: bar_color = (0, 0, 255)
            
            fill_width = int(bar_width * progress)
            if fill_width > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)
            
            # 时间文字
            time_text = f"{time_left}s"
            cv2.putText(frame, time_text, (bar_x + bar_width + 10, bar_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- 游戏结束画面 ---
            if is_game_over:
                # 模糊/暗化背景
                overlay = np.zeros_like(frame)
                overlay[:] = (0, 0, 0)
                cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
                
                # 居中显示 Game Over 面板
                cx, cy = width // 2, height // 2
                
                # 面板背景
                panel_w, panel_h = 400, 250
                px1, py1 = cx - panel_w//2, cy - panel_h//2
                px2, py2 = cx + panel_w//2, cy + panel_h//2
                
                # 白色边框 + 半透明深色底
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 3)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (50, 50, 50), -1) # 可以考虑再做一次半透明
                
                # 标题
                title = "GAME OVER"
                font = cv2.FONT_HERSHEY_TRIPLEX
                ts = cv2.getTextSize(title, font, 2, 3)[0]
                cv2.putText(frame, title, (cx - ts[0]//2, py1 + 80), font, 2, (100, 100, 255), 3)
                
                # 最终得分
                score_msg = f"Score: {score}"
                ss = cv2.getTextSize(score_msg, font, 1.2, 2)[0]
                cv2.putText(frame, score_msg, (cx - ss[0]//2, py1 + 150), font, 1.2, (255, 255, 255), 2)
                
                # 提示
                hint = "Final Score"
                hs = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
                # cv2.putText(frame, hint, (cx - hs[0]//2, py1 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
                
                # *新增* 重玩提示
                restart_msg = "Press 'R' to Restart"
                rs = cv2.getTextSize(restart_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.putText(frame, restart_msg, (cx - rs[0]//2, py2 - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
            # 统一显示
            cv2.imshow('Pacman Final', frame)
            
            # 统一按键处理
            key = cv2.waitKey(5) & 0xFF
            if key == 27: # ESC
                break
                
            if is_game_over:
                if key == ord('r') or key == ord('R'):
                    # 重置游戏状态
                    score = 0
                    frame_count = 0
                    game_objects = []
                    start_time = time.time()
                    is_game_over = False
                    game_objects = []
                    start_time = time.time()
                    is_game_over = False

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()