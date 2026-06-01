import time
import pyautogui as pag
import pygetwindow as gw

# 1. 找到窗口
# 注意替换成你窗口标题中 *包含* 的字符串，比如 "Hollow Knight", "Chrome" 等
window_title_keyword = "Hollow Knight Silksong"
windows = gw.getWindowsWithTitle(window_title_keyword)

if not windows:
    raise RuntimeError("找不到这个窗口，检查一下标题关键字对不对")

win = windows[0]

# 确保窗口是正常状态（非最小化）
if win.isMinimized:
    win.restore()
win.activate()   # 激活到前台
time.sleep(0.5)  # 等它切过来

# 2. 获取窗口的位置和大小
left, top, width, height = win.left, win.top, win.width, win.height
print("窗口位置: ", left, top, width, height)

# 3. 截取窗口区域的截图
window_img = pag.screenshot(region=(left+13, top+58, 1280, 720))

# 4. 在窗口截图中查找 icon.png
# 先准备一个小图标图片 icon.png，内容就是你要点击的那个按钮/图标
icon_path = "HK_SilkSong_env/locate_fig/first_sinner.png"

box = pag.locate(icon_path, window_img, confidence=0.8)  # confidence 需要 opencv 支持
if box is None:
    raise RuntimeError("在窗口中找不到图标，可能是图标样子不一样或被遮挡了")

# box 是一个类似 Box(left, top, width, height) 的对象，坐标相对 *窗口截图*
rel_x = box.left + box.width // 2   # 图标中心相对窗口左上角的 x
rel_y = box.top + box.height // 2   # 图标中心相对窗口左上角的 y

print("相对窗口坐标: ", rel_x, rel_y)

# 5. 换算成屏幕上的绝对坐标
abs_x = left + 13 + rel_x
abs_y = top + 58 + rel_y
print("屏幕绝对坐标: ", abs_x, abs_y)

# 6. 点击
# pag.moveTo(abs_x, abs_y, duration=0.2)
# pag.click()
