import cv2
import mediapipe as mp
import pyautogui
import math
import threading
import time
import numpy as np
from pynput.mouse import Button, Controller

# ====================== 全局可调配置区 ======================
# 投影HDMI输出分辨率（匹配极米R2S Pro）
SCREEN_W = 2560
SCREEN_H = 1600
# 摄像头原生分辨率
CAM_RAW_W = 1281
CAM_RAW_H = 720
# 裁切16:10操控区域
TARGET_CROP_W = int(CAM_RAW_H / 10 * 16)
TARGET_CROP_H = CAM_RAW_H
crop_side = (CAM_RAW_W - TARGET_CROP_W) // 2
CROP_LEFT = crop_side
CROP_RIGHT = CAM_RAW_W - crop_side
CROP_TOP = 0
CROP_BOTTOM = CAM_RAW_H

CLICK_THRESHOLD = 50
PINCH_LOCK_FRAME = 1
RIGHT_PINCH_FRAME = 1
SMOOTH_MOVE = 0.7
SMOOTH_PINCH = 0.9
WINDOW_NAME = "Gesture Mouse Projector"

# 四点透视矫正变量
calib_points = []
transform_matrix = None
calibrating = False
# 视频镜像开关，默认开启
flip_video = True
# ===========================================================

# MediaPipe手部初始化
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)
mouse = Controller()

# 全局状态
cap = None
current_cam_id = 0
prev_mx, prev_my = 0, 0
pinch_count = 0
right_pinch_count = 0
right_click_flag = False
show_preview = True
drag_lock_flag = False
program_running = True

def mouse_control_thread():
    """鼠标按键独立线程，稳定拖拽/右键长按"""
    global drag_lock_flag, program_running, right_click_flag
    left_pressed = False
    right_pressed = False
    while program_running:
        # 左键拖拽控制
        if drag_lock_flag:
            if not left_pressed:
                mouse.press(Button.left)
                left_pressed = True
        else:
            if left_pressed:
                mouse.release(Button.left)
                left_pressed = False
        # 右键控制
        if right_click_flag:
            if not right_pressed:
                mouse.press(Button.right)
                right_pressed = True
        else:
            if right_pressed:
                mouse.release(Button.right)
                right_pressed = False
        time.sleep(0.01)

def safe_cleanup():
    """统一释放摄像头、鼠标、窗口资源，打包无崩溃"""
    global program_running
    program_running = False
    time.sleep(0.15)
    mouse.release(Button.left)
    mouse.release(Button.right)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("资源全部释放完成")

def open_camera(cam_id):
    """打开指定摄像头，设置MJPG分辨率"""
    global cap, prev_mx, prev_my, pinch_count, right_pinch_count
    if cap is not None:
        cap.release()
    new_cap = cv2.VideoCapture(cam_id)
    new_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RAW_W)
    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RAW_H)
    if not new_cap.isOpened():
        print(f"摄像头 {cam_id} 打开失败")
        new_cap.release()
        return False
    prev_mx, prev_my = 0, 0
    pinch_count = 0
    right_pinch_count = 0
    cap = new_cap
    current_cam_id = cam_id
    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"成功切换摄像头{cam_id}，分辨率 {real_w} × {real_h}")
    return True

def scan_all_cameras(max_scan=5):
    """扫描所有可用摄像头ID"""
    dev_list = []
    for dev in range(max_scan):
        temp_cap = cv2.VideoCapture(dev)
        if temp_cap.isOpened():
            dev_list.append(dev)
            temp_cap.release()
    return dev_list

def mouse_click_event(event, x_win, y_win, flags, param):
    """标定鼠标回调：修正窗口坐标偏移，标点与点击位置完全重合"""
    global calib_points, calibrating, transform_matrix
    if event == cv2.EVENT_LBUTTONDOWN and calibrating:
        # 窗口坐标转裁切图局部坐标
        x_crop = x_win - CROP_LEFT
        y_crop = y_win
        # 仅记录黄色框内有效点
        if 0 <= x_crop < TARGET_CROP_W and 0 <= y_crop < TARGET_CROP_H:
            if len(calib_points) < 4:
                calib_points.append([x_crop, y_crop])
                print(f"标记第{len(calib_points)}个角点：{x_crop},{y_crop}")
        # 四点集齐生成透视矫正矩阵
        if len(calib_points) == 4:
            src = np.array(calib_points, dtype=np.float32)
            dst = np.array([
                [0, 0],
                [TARGET_CROP_W, 0],
                [TARGET_CROP_W, TARGET_CROP_H],
                [0, TARGET_CROP_H]
            ], dtype=np.float32)
            transform_matrix = cv2.getPerspectiveTransform(src, dst)
            calibrating = False
            print("四点标定完成，透视矫正生效，按C重新标定")

def warp_point(x, y):
    """透视坐标矫正函数"""
    global transform_matrix
    if transform_matrix is None:
        return x, y
    pt = np.array([[[x, y]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, transform_matrix)
    return float(res[0][0][0]), float(res[0][0][1])

if __name__ == "__main__":
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, mouse_click_event)
        cv2.waitKey(1)

        cam_list = scan_all_cameras()
        print("===== 投影仪隔空手势鼠标（镜像修复完整版） =====")
        print("手势：拇指+食指=左键拖拽 | 拇指+小指=右键长按")
        print("快捷键：")
        print(" W 切换画面镜像 | C 四点透视标定 | E 开关预览")
        print(" 1/2/3 切换摄像头 | q 退出程序")
        if len(cam_list) == 0:
            print("未检测到摄像头，程序退出")
            exit()
        open_camera(cam_list[0])

        # 启动鼠标控制子线程
        mouse_thread = threading.Thread(target=mouse_control_thread, daemon=True)
        mouse_thread.start()
        print(f"摄像头尺寸 {CAM_RAW_W}×{CAM_RAW_H}，操控裁切区 {TARGET_CROP_W}×{TARGET_CROP_H}")
        print("标定提示：按C开启标定，仅在画面黄色方框内点击四角，顺序：左上→右上→右下→左下")

        while cap.isOpened() and program_running:
            ret, frame = cap.read()
            if not ret:
                print("画面丢失，重新连接摄像头")
                open_camera(current_cam_id)
                continue

            # 画面镜像显示（仅预览画面翻转）
            display_frame = frame.copy()
            if flip_video:
                display_frame = cv2.flip(display_frame, 1)

            # 裁切有效操控区域
            crop_frame = display_frame[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT].copy()
            rgb_img = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_img)
            temp_drag_state = False
            temp_right_state = False

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_info, hand_land in zip(result.multi_handedness, result.multi_hand_landmarks):
                    hand_label = hand_info.classification[0].label
                    if hand_label != "Right":
                        continue
                    # 绘制手部骨架
                    if show_preview:
                        mp_draw.draw_landmarks(crop_frame, hand_land, mp_hands.HAND_CONNECTIONS)

                    # 获取三根手指归一化坐标
                    index_tip = hand_land.landmark[8]
                    thumb_tip = hand_land.landmark[4]
                    pinky_tip = hand_land.landmark[20]

                    # 转换为裁切图像素坐标
                    ix = int(index_tip.x * TARGET_CROP_W)
                    iy = int(index_tip.y * TARGET_CROP_H)
                    tx = int(thumb_tip.x * TARGET_CROP_W)
                    ty = int(thumb_tip.y * TARGET_CROP_H)
                    px = int(pinky_tip.x * TARGET_CROP_W)
                    py = int(pinky_tip.y * TARGET_CROP_H)

                    # 【核心修复】镜像开启时反转X坐标，鼠标移动方向与视觉同步
                    if flip_video:
                        ix = TARGET_CROP_W - ix
                        tx = TARGET_CROP_W - tx
                        px = TARGET_CROP_W - px

                    # 捏合距离计算
                    pinch_dist = math.hypot(ix - tx, iy - ty)
                    right_pinch_dist = math.hypot(px - tx, py - ty)

                    # 计算手指中点（矫正前裁切图坐标）
                    mid_x = (ix + tx) / 2
                    mid_y = (iy + ty) / 2
                    # 透视矫正坐标
                    warp_mid_x, warp_mid_y = warp_point(mid_x, mid_y)

                    # 映射到投影全屏分辨率，镜像同步反转X轴
                    if flip_video:
                        raw_mx = (1.0 - warp_mid_x / TARGET_CROP_W) * SCREEN_W
                    else:
                        raw_mx = warp_mid_x / TARGET_CROP_W * SCREEN_W
                    raw_my = warp_mid_y / TARGET_CROP_H * SCREEN_H

                   
                    # 右键逻辑（拇指+小指）
                    if right_pinch_dist < CLICK_THRESHOLD:
                        curr_mx = prev_mx + (raw_mx - prev_mx) * SMOOTH_PINCH
                        curr_my = prev_my + (raw_my - prev_my) * SMOOTH_PINCH
                        right_pinch_count += 1
                        pinch_count = 0
                        if right_pinch_count >= RIGHT_PINCH_FRAME:
                            temp_right_state = True
                            if show_preview:
                                dot_x = int((ix + px) / 2)
                                dot_y = int((iy + py) / 2)
                                cv2.circle(crop_frame, (dot_x, dot_y), 12, (255, 0, 255), -1)
                    # 左键拖拽逻辑（拇指+食指）
                    elif pinch_dist < CLICK_THRESHOLD:
                        curr_mx = prev_mx + (raw_mx - prev_mx) * SMOOTH_PINCH
                        curr_my = prev_my + (raw_my - prev_my) * SMOOTH_PINCH
                        pinch_count += 1
                        right_pinch_count = 0
                        if pinch_count >= PINCH_LOCK_FRAME:
                            temp_drag_state = True
                            if show_preview:
                                dot_x = int(mid_x)
                                dot_y = int(mid_y)
                                cv2.circle(crop_frame, (dot_x, dot_y), 12, (0, 0, 255), -1)
                    # 自由移动模式
                    else:
                        curr_mx = prev_mx + (raw_mx - prev_mx) * SMOOTH_MOVE
                        curr_my = prev_my + (raw_my - prev_my) * SMOOTH_MOVE
                        pinch_count = 0
                        right_pinch_count = 0
                    # 更新鼠标位置
                    prev_mx, prev_my = curr_mx, curr_my
                    mouse.position = (curr_mx, curr_my)
            drag_lock_flag = temp_drag_state
            right_click_flag = temp_right_state

            # 预览窗口渲染
            if show_preview:
                # 绘制黄色操控框
                cv2.rectangle(display_frame, (CROP_LEFT, CROP_TOP), (CROP_RIGHT, CROP_BOTTOM), (0, 255, 255), 3)
                # 状态文字
                flip_text = "FLIP:ON(W关闭)" if flip_video else "FLIP:OFF(W开启)"
                drag_text = "DRAG LOCKED" if drag_lock_flag else "Free Move"
                right_text = "RIGHT_CLICK" if right_click_flag else ""
                calib_text = "CALIBRATING(黄框内点四角)" if calibrating else "Calib OK" if transform_matrix is not None else "No Calib(PRESS C)"
                cv2.putText(display_frame, f"{flip_text} | Cam:{current_cam_id} | {drag_text} | {right_text} | {calib_text}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 绘制标定红点编号
                for idx, (cx, cy) in enumerate(calib_points):
                    cv2.circle(crop_frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                    cv2.putText(crop_frame, str(idx+1), (int(cx)+5, int(cy)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                # 裁切画面贴回主窗口
                display_frame[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT] = crop_frame
                cv2.imshow(WINDOW_NAME, display_frame)
                # 检测手动关闭窗口
                try:
                    prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                    if prop < 1:
                        print("手动关闭预览窗口，退出程序")
                        break
                except Exception:
                    break

            # 按键响应
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                print("按下q，安全退出")
                break
            elif key == ord('e'):
                show_preview = not show_preview
                stat = "开启" if show_preview else "关闭"
                print(f"预览渲染已{stat}")
            elif key == ord('w'):
                flip_video = not flip_video
                stat = "开启" if flip_video else "关闭"
                print(f"画面镜像已{stat}，鼠标方向自动同步反转")
            elif key == ord('c'):
                calib_points.clear()
                transform_matrix = None
                calibrating = True
                print("\n===== 四点标定模式 =====")
                print("仅在黄色方框内点击，顺序：左上 → 右上 → 右下 → 左下")
            elif key == ord('1'):
                open_camera(0)
            elif key == ord('2'):
                open_camera(1)
            elif key == ord('3'):
                open_camera(2)
        safe_cleanup()
    except KeyboardInterrupt:
        print("\n检测Ctrl+C强制退出，释放资源")
        safe_cleanup()
