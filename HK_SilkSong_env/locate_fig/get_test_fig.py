import dxcam
import pygetwindow as gw
import cv2
import time

# 1️⃣ 找到 Hollow Knight 窗口
window_title = "Hollow Knight Silksong"
windows = gw.getWindowsWithTitle(window_title)
if not windows:
    raise RuntimeError(f"未找到名为“{window_title}”的窗口，请确认游戏已启动且标题匹配。")

window = windows[0]
left, top, width, height = window.left, window.top, window.width, window.height

print(f"找到窗口: {window_title} ({left},{top},{width},{height})")

if window.isMinimized:
    window.restore()
    time.sleep(0.5)
window.activate()
time.sleep(0.1)
# 2️⃣ 创建 dxcam 实例
camera = dxcam.create(output_idx=0)

# 3️⃣ 使用窗口区域截图
frame = camera.grab(region=(left+13, top+58, left + width-13, top+58 + height-71))
print(f"frame shape: {frame.shape}")
frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

print(f"frame shape: {frame.shape}")
# cv2.imshow("frame", frame)
# cv2.waitKey(0)
# 4️⃣ 保存图片
cv2.imwrite("HK_SilkSong_env/locate_fig/hollow_knight_window.png", frame)
print("✅ 截图已保存为 hollow_knight_window.png")
