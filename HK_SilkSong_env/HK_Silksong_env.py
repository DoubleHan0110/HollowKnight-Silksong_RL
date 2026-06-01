import gymnasium as gym
import pygetwindow as gw
import pyautogui
import dxcam
import time
import cv2
import keyboard
import sys
import os
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from .sk_mod_event_class import ModEventClient
from env_wrapper import LifeLossInfo

class HK_Silksong_Env(gym.Env):
    def __init__(self):
        
        self.window = None
        self.camera = None

        self.process_time = 0.0

        # 菜单阈值
        self.godhome_menu_threshold = 0.99
        self.boss_room_threshold = 0.9
        self.boss_name_threshold = 0.98

        self.gap = 1.0 / 9.0
        self._prev_time = None

        self.KEYMAP = {
            0: ('a', 'move_left', 'hold'),      # 按住左
            1: ('d', 'move_right', 'hold'),     # 按住右
            2: ('w', 'look_up', 'hold'),        # 按住上（向上看）
            3: ('s', 'look_down', 'hold'),     # 按住下（向下看）
            4: ('j', 'attack', 'hold'),     # 攻击（瞬间）
            5: ('k', 'dash', 'hold'),        # 冲刺（瞬间）
            6: ('space', 'jump', 'hold'),       # 跳跃（按住）
            7: ('h', 'fight_back', 'hold'),   # 反击（瞬间）
        }
        self.action_keys = [self.KEYMAP[i][0] for i in range(len(self.KEYMAP))]
        self.num_actions = len(self.action_keys)
        self.instant_keys = {}
        self._key_states = np.zeros(self.num_actions, dtype=np.int8)

        # 定义动作空间
        self.action_space = gym.spaces.MultiBinary(self.num_actions)

        self.lives_info = None

        self.boss_targets = None
        self.boss_hp = None
        self._episode_frame_number = 0

        self._setup_windows()

        # 定义观测空间
        H, W = self.window.height, self.window.width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)

        self.mod_event_client = ModEventClient()

    def reset(self, seed=None, options=None):
        """重置环境，重新进入boss房间"""
        super().reset(seed=seed)

        self._cleanup_keys()
        
        # 重新进入boss房间
        self._enter_boss_room()

        self._prev_time = time.time()

        self.mod_event_client.reset(last_check_time=self._prev_time)
        
        obs = self._get_latest_frame()
        
        self.boss_targets = {"First Weaver"}
        self.boss_hp = 1300
        self.lives_info = 10
        self._episode_frame_number = 0
        info = {}
        info["lives"] = self.lives_info
        info["episode_frame_number"] = self._episode_frame_number
        # print("reset成功,开始动作")
        return obs, info  
    


    def step(self, action):

        self._execute_actions(action)
        # 控制采样频率
        if self._prev_time is not None:
            elapsed = time.time() - self._prev_time
            sleep_time = self.gap - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        # process_gap = time.time() - self.process_time
        # print(f"process_gap: {process_gap}")

        self._prev_time = time.time()

        # self.process_time = time.time()   

        obs = self._get_latest_frame()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        boss_hp_last = self.boss_hp

        current_time = self._prev_time
        # print(f"current_time: {current_time}")
        events = self.mod_event_client.get_events_since_last_check(current_time=current_time)


        if events["hits"]:
            # reward += 1
            # print(f"length of hits: {len(events['hits'])}")
            for ev in events["hits"]:
                name = ev.get("entity", "")
                self.boss_hp = ev.get("boss_hp", 0)
                reward += 1
                if self.boss_hp > boss_hp_last:
                    reward -= 7
                if name in self.boss_targets and self.boss_hp <= 0:
                    self.boss_targets.remove(name)
                    # print(f"[Mantis] defeated: {name}, remaining targets: {len(self.boss_targets)}")
            # print(f"hit the boss, reward: {reward}, boss_hp: {self.boss_hp}")       

        if events["damages"]:
            self.lives_info = events["damages"][-1].get("hornet_hp", 0)
            # print(f"got damaged, damage this time: {total_damage}, lives: {self.lives_info}")
            # print(f"got damaged, lives: {self.lives_info}")
            
        info["lives"] = self.lives_info
        self._episode_frame_number += 1
        info["episode_frame_number"] = self._episode_frame_number

        if obs is None:
            obs = np.zeros((self.window.height, self.window.width, 3), dtype=np.uint8)
            truncated = True
        
        if self.lives_info == 0: 
            terminated = True
        if len(self.boss_targets) == 0:
            self._cleanup_keys()
            time.sleep(6.0)
            self._recover_hp()
            terminated = True
        
        if terminated or truncated:
            self._cleanup_keys()
            time.sleep(5.0)

        return obs, reward, terminated, truncated, info

    def _setup_windows(self):
        """初始化窗口和camera"""
        # 查找空洞骑士窗口
        windows = gw.getWindowsWithTitle("Hollow Knight Silksong")
        self.window = windows[0]
        self.window.activate()
        time.sleep(0.1)

        self.camera = dxcam.create(output_idx=0)
        
    
    def _get_latest_frame(self):
        """获取最新帧"""

        screen_width = self.camera.width
        screen_height = self.camera.height
        left = max(0, self.window.left+13)
        top = max(0, self.window.top+58)
        right = min(screen_width, self.window.left + self.window.width - 13)
        bottom = min(screen_height, self.window.top + 58 + self.window.height - 71)
        
        region = (left, top, right, bottom)

        for attempt in range(5):
            frame = self.camera.grab(region=region)
            if frame is not None:
                break
            time.sleep(0.2)
            self.window.activate()
        
        if frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # print(f"frame shape: {frame.shape}")
        return frame

    def _enter_boss_room(self):
        """重新进入boss房间"""
        # print("正在重新进入boss房间...")
        time.sleep(1.0)

        # print("调出godhome菜单并进入")
        self._enter_godhome_menu()

        # print("走入boss房间")
        self._walk_till_boss_room()

        # print("进入wait_for_loading")
        self._wait_for_loading()
    
    def _enter_godhome_menu(self):
        """调出godhome菜单并进入"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_image_path = os.path.join(base_dir, "locate_fig", "godhome_menu_icon.png")
        icon_image = cv2.imread(icon_image_path, cv2.IMREAD_COLOR)
        while True:
            keyboard.send('b')
            time.sleep(1.0)
            frame = self._get_latest_frame()
            res = cv2.matchTemplate(frame, icon_image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            # print(f"max_val: {max_val}")
            if max_val > self.godhome_menu_threshold:
                break

        # 点击boss icon
        abs_x = self.window.left + 13 + 519
        abs_y = self.window.top + 58 + 98
        pyautogui.click(abs_x, abs_y)
        time.sleep(4.0)

    def _walk_till_boss_room(self):
        """走入boss房间"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        check_image_path = os.path.join(base_dir, "locate_fig", "boss_room_check.png")
        check_icon_image = cv2.imread(check_image_path, cv2.IMREAD_COLOR)

        check_image_path2 = os.path.join(base_dir, "locate_fig", "boss_room_check_2.png")
        check_icon_image2 = cv2.imread(check_image_path2, cv2.IMREAD_COLOR)

        skill_image_path = os.path.join(base_dir, "locate_fig", "skill_enter.png")
        skill_enter_image = cv2.imread(skill_image_path, cv2.IMREAD_COLOR)
        # 往右走到boss门口
        while True:
            keyboard.press('d')
            time.sleep(0.9)
            keyboard.release('d')
            frame = self._get_latest_frame()
            res = cv2.matchTemplate(frame, check_icon_image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            # print(f"max_val: {max_val}")
            if max_val > self.boss_room_threshold:
                break
            # time.sleep(0.1)

        # 触发boss房间对话
        while True:
            keyboard.press('w')
            time.sleep(0.1)
            keyboard.release('w')
            time.sleep(1.0)
            frame = self._get_latest_frame()
            res = cv2.matchTemplate(frame, check_icon_image2, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            # print(f"max_val: {max_val}")
            if max_val > self.boss_room_threshold:
                break
        
        while True:
            keyboard.press("space")
            time.sleep(0.1)
            keyboard.release("space")
            time.sleep(1.5)
            frame = self._get_latest_frame()
            res = cv2.matchTemplate(frame, skill_enter_image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            # print(f"max_val: {max_val}")
            if max_val > self.boss_room_threshold:
                break
        keyboard.press('l')
        time.sleep(0.2)
        keyboard.release('l')
        time.sleep(3.0)

    def _wait_for_loading(self):
        """等待游戏加载完成"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        boss_name_path = os.path.join(base_dir, "locate_fig", "boss_name.png")
        boss_name_image = cv2.imread(boss_name_path, cv2.IMREAD_COLOR)

        ready = False
        is_loading = False
        
        while True:
            frame = self._get_latest_frame()
            if frame is None:
                print("报错：当前截屏失败，等待空格键继续：")
                keyboard.wait('space')

                break
            res = cv2.matchTemplate(frame, boss_name_image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            # print(f"max_val: {max_val}")
            if max_val > self.boss_name_threshold:
                # print("找到boss名字!!!!!")
                break
            
            time.sleep(0.3)
        
        # 额外等待确保完全准备好
        time.sleep(1.5)

    def _recover_hp(self):
        # """恢复hp"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # icon_image_path = os.path.join(base_dir, "locate_fig", "godhome_menu_icon.png")
        # icon_image = cv2.imread(icon_image_path, cv2.IMREAD_COLOR)

        rest_icon_path = os.path.join(base_dir, "locate_fig", "sleep.png")
        rest_icon = cv2.imread(rest_icon_path, cv2.IMREAD_COLOR)

        # 进入godhome菜单
        keyboard.send('b')
        time.sleep(3.0)

        # 点击boss icon
        abs_x = self.window.left + 13 + 1142
        abs_y = self.window.top + 58 + 35
        pyautogui.click(abs_x, abs_y)
        time.sleep(4.0)

        while True:
            keyboard.press("d")
            time.sleep(0.08)
            keyboard.release("d")
            if self._check_icon(rest_icon):
                time.sleep(0.5)
                keyboard.press("w")
                time.sleep(0.1)
                keyboard.release("w")
                break
        time.sleep(2.0)

    def _check_icon(self, icon, threshold = 0.8):
        frame = self._get_latest_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame is None:
            frame = np.zeros((self.window.height, self.window.width, 3), dtype=np.uint8)
            print("[WARN] Empty frame in _check_icon")
        res = cv2.matchTemplate(frame, icon, cv2.TM_CCOEFF_NORMED)
        if res is None:
            raise RuntimeError("Template matching failed")
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > 0.8:
            return True
        else:
            return False



    def _execute_actions(self, action):
        """执行动作"""
        action = np.array(action, dtype=np.int8).flatten()

        for i in range(self.num_actions):
            key_name, _, _ = self.KEYMAP[i]
            new_state = action[i]
            old_state = self._key_states[i]
            
            if i in self.instant_keys:
                if new_state == 1:
                    keyboard.send(key_name)
            else:
                if new_state == 1 and old_state == 0:
                    keyboard.press(key_name)
                elif new_state == 0 and old_state == 1:
                    keyboard.release(key_name)

        self._key_states = action.copy()

    def _cleanup_keys(self):
        """清理可能按住的按键"""
        keys_to_release = ['a', 'd', 'w', 's', 'j', 'k', 'space', 'h']
        for key in keys_to_release:
            keyboard.release(key)

        self._key_states = np.zeros(self.num_actions, dtype=np.int8)
    
    def _cleanup(self):
        """清理camera"""
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
            self.camera = None
    


if __name__ == "__main__":
    env = HK_Silksong_Env()
    env._recover_hp()
    # env = gym.wrappers.ResizeObservation(env, shape = (64, 64))
    # env = LifeLossInfo(env)
    # episode =2
    # for i in range(episode):
    #     obs, info = env.reset()
    #     while True:
    #         action = env.action_space.sample()
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         if terminated:
    #             print(f"episode {i} terminated")
    #             break
    #         if truncated:
    #             break
    #     print(f"episode {i} finished")
    # env.reset()
    # env._is_challenge_menu(env._get_latest_frame())
