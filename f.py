import cv2
import mediapipe as mp
import pyautogui
import math
import threading
import time
from pynput.mouse import Button, Controller

# ====================== 全局可调配置区 ======================
# 屏幕分辨率 16:10
SCREEN_W = 2560
SCREEN_H = 1600
# 摄像头原生16:9
CAM_RAW_W = 1280
CAM_RAW_H = 720
# 裁切至16:10操控区域
# 四周预留空隙像素，自行修改数值控制边框宽度
PADDING = 40
# 目标比例 16:10
RATIO_W, RATIO_H = 16, 10

# 扣除四周空隙后的可用画布
usable_w = CAM_RAW_W - 2 * PADDING
usable_h = CAM_RAW_H - 2 * PADDING

# 按16:10比例计算适配可用区域的画面尺寸
max_h_from_w = usable_w * RATIO_H / RATIO_W
max_w_from_h = usable_h * RATIO_W / RATIO_H

if max_h_from_w <= usable_h:
    TARGET_CROP_W = int(usable_w)
    TARGET_CROP_H = int(max_h_from_w)
else:
    TARGET_CROP_W = int(max_w_from_h)
    TARGET_CROP_H = int(usable_h)

# 可用区域内居中偏移
offset_x = (usable_w - TARGET_CROP_W) // 2
offset_y = (usable_h - TARGET_CROP_H) // 2

# 最终裁剪坐标（兼容原有变量名，直接替换原有代码）
CROP_LEFT = PADDING + offset_x
CROP_TOP = PADDING + offset_y
CROP_RIGHT = CROP_LEFT + TARGET_CROP_W
CROP_BOTTOM = CROP_TOP + TARGET_CROP_H

CLICK_THRESHOLD = 50        # 点击灵敏度
PINCH_LOCK_FRAME = 1        # 捏合多少帧锁定长按拖拽
SMOOTH_MOVE = 0.1           # 移动光标平滑系数
SMOOTH_PINCH = 0.9          # 捏合拖拽强防抖系数
WINDOW_NAME = "Gesture Mouse Dev Preview"
# ===========================================================

# MediaPipe 手部初始化
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

mouse = Controller()

# 全局状态变量（线程共享）
cap = None
current_cam_id = 0
prev_mx, prev_my = 0, 0
pinch_count = 0
show_preview = True
# 线程安全标记：拖拽锁定状态
drag_lock_flag = False
program_running = True  # 程序总开关

def mouse_control_thread():
    global drag_lock_flag, program_running
    left_pressed = False
    while program_running:
        if drag_lock_flag:
            if not left_pressed:
                mouse.press(Button.left)
                left_pressed = True
        else:
            if left_pressed:
                mouse.release(Button.left)
                left_pressed = False
        time.sleep(0.01)

def safe_cleanup():
    """统一释放所有资源，打包/IDLE通用"""
    global program_running
    program_running = False
    time.sleep(0.15)
    mouse.release(Button.left)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def open_camera(cam_id):
    """打开指定摄像头 MJPG 720P"""
    global cap, prev_mx, prev_my, pinch_count
    if cap is not None:
        cap.release()
    new_cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    new_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RAW_W)
    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RAW_H)
    if not new_cap.isOpened():
        new_cap.release()
        return False
    prev_mx, prev_my = 0, 0
    pinch_count = 0
    cap = new_cap
    current_cam_id = cam_id
    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return True

def scan_all_cameras(max_scan=5):
    """扫描全部可用摄像头ID"""
    available = []
    for dev_id in range(max_scan):
        temp = cv2.VideoCapture(dev_id, cv2.CAP_DSHOW)
        if temp.isOpened():
            available.append(dev_id)
            temp.release()
    return available

if __name__ == "__main__":
    try:
        # 提前创建窗口，打包无控制台必备，避免空句柄
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)

        cam_list = scan_all_cameras()
        open_camera(cam_list[0])

        # 启动鼠标子线程
        mouse_thread = threading.Thread(target=mouse_control_thread, daemon=True)
        mouse_thread.start()


        # 主循环
        while cap.isOpened() and program_running:
            ret, frame = cap.read()
            if not ret:

                open_camera(current_cam_id)
                continue

            frame = cv2.flip(frame, 1)
            crop_frame = frame[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT]
            rgb_img = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_img)

            temp_drag_state = False

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_info, hand_land in zip(result.multi_handedness, result.multi_hand_landmarks):
                    hand_label = hand_info.classification[0].label
                    if hand_label != "Right":
                        continue

                    if show_preview:
                        mp_draw.draw_landmarks(crop_frame, hand_land, mp_hands.HAND_CONNECTIONS)

                    index_tip = hand_land.landmark[8]
                    thumb_tip = hand_land.landmark[4]

                    ix = int(index_tip.x * TARGET_CROP_W)
                    iy = int(index_tip.y * TARGET_CROP_H)
                    tx = int(thumb_tip.x * TARGET_CROP_W)
                    ty = int(thumb_tip.y * TARGET_CROP_H)

                    mid_x_norm = (index_tip.x + thumb_tip.x) / 2
                    mid_y_norm = (index_tip.y + thumb_tip.y) / 2
                    raw_mx = mid_x_norm * SCREEN_W
                    raw_my = mid_y_norm * SCREEN_H
                    

                    pinch_dist = math.hypot(ix - tx, iy - ty)

                    if show_preview:
                        cv2.line(crop_frame, (ix, iy), (tx, ty), (0, 255, 0), 2)
                        cv2.putText(crop_frame, f"Dist:{int(pinch_dist)}",
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    if pinch_dist < CLICK_THRESHOLD:
                        curr_mx = prev_mx + (raw_mx - prev_mx) * SMOOTH_PINCH
                        curr_my = prev_my + (raw_my - prev_my) * SMOOTH_PINCH
                        pinch_count += 1
                        if pinch_count >= PINCH_LOCK_FRAME:
                            temp_drag_state = True
                            if show_preview:
                                dot_x = int(mid_x_norm * TARGET_CROP_W)
                                dot_y = int(mid_y_norm * TARGET_CROP_H)
                                cv2.circle(crop_frame, (dot_x, dot_y), 12, (0, 0, 255), -1)
                    else:
                        curr_mx = prev_mx + (raw_mx - prev_mx) * SMOOTH_MOVE
                        curr_my = prev_my + (raw_my - prev_my) * SMOOTH_MOVE
                        pinch_count = 0

                    prev_mx, prev_my = curr_mx, curr_my
                    mouse.position = (curr_mx, curr_my)

            drag_lock_flag = temp_drag_state

            # 预览开启才绘图+imshow，关闭仅停止渲染，不销毁窗口
            if show_preview:
                cv2.rectangle(frame, (CROP_LEFT, CROP_TOP), (CROP_RIGHT, CROP_BOTTOM), (0, 255, 255), 3)
                drag_text = "DRAG LOCKED" if drag_lock_flag else "Free Move"
                preview_text = "Preview ON | E=关闭渲染"
                cv2.putText(frame, f"Cam:{current_cam_id} | {drag_text} | {preview_text}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                frame[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT] = crop_frame
                cv2.imshow(WINDOW_NAME, frame)
                # 检测窗口关闭，加异常捕获防打包空指针
                try:
                    prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                    if prop < 1:
                        break
                except Exception:
                    break

            # 按键响应
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            if key == ord('e'):
                show_preview = not show_preview
                stat = "开启" if show_preview else "关闭"
            elif key == ord('1'):
                open_camera(0)
            elif key == ord('2'):
                open_camera(1)
            elif key == ord('3'):
                open_camera(2)

        safe_cleanup()

    except KeyboardInterrupt:
        safe_cleanup()
