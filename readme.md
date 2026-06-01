# Knight Dreamer: Imagination-Based RL Agent

本项目将基于 STORM（Stochastic Transformer-based World Model）的想象式强化学习方法应用到《Hollow Knight》和《Hollow Knight: Silksong》的 Boss 战场景中。项目通过屏幕截图获得视觉观测，用键盘模拟执行动作，并通过自定义 Mod 暴露命中、受伤、血量等事件信号，从而把真实游戏封装成 Gymnasium 风格环境进行训练和评估。


## 项目结构

```text
.
├── agents.py                    # Actor-Critic agent
├── train_HK.py                  # Hollow Knight / Silksong 训练入口
├── eval_HK.py                   # 评估入口
├── env_wrapper.py               # Gymnasium wrapper，包括动作空间转换和life-loss 标记
├── replay_buffer.py             # 经验回放与轨迹缓存
├── utils.py                     # 配置、随机种子、TensorBoard 日志
├── config_files/                # Hollow Knight / Silksong 配置
├── HollowKnight_env/            # Hollow Knight 环境封装与 Mod 事件客户端
├── HK_SilkSong_env/             # Silksong 环境封装与 Mod 事件客户端
├── sub_models/                  # World model、Transformer、loss 等网络模块
├── ckpt/                        # 训练检查点
├── runs/                        # TensorBoard 日志
├── eval_result/                 # 评估结果
```

## 环境要求

推荐环境：

- Windows 10/11
- Python 3.10
- NVIDIA GPU + CUDA 版本的 PyTorch
- 游戏窗口模式，分辨率固定为 1280 x 720
- Hollow Knight 或 Hollow Knight: Silksong 已启动并处于可进入 Boss 战的状态

游戏侧依赖：

- Hollow Knight：需要自定义 Mod 通过 HTTP 向 Python 环境发送命中、受伤、Boss 血量等事件。
- Hollow Knight: Silksong：需要 Godhome 相关 Mod、BepInEx 5、Configuration Manager，以及项目配套的事件上报插件。

Python 依赖安装：

```powershell
python -m pip install -r requirements.txt
```

如果需要 CUDA 版 PyTorch，建议按本机 CUDA 版本参考 PyTorch 官网命令安装 `torch`、`torchvision`、`torchaudio`，再安装其余依赖。

## 操作方法

### 1. 准备游戏窗口

1. 启动对应游戏。
2. 使用窗口模式，确保窗口标题能被脚本识别：
   - `Hollow Knight`
   - `Hollow Knight Silksong`
3. 将分辨率固定为 1280 x 720。
4. 确认 Mod 服务已运行，Python 侧可以通过 `ModEventClient` 获取事件。

环境封装会使用 `dxcam` 截取游戏窗口，用 `keyboard` / `pyautogui` 模拟按键和点击，并以约 9 FPS 的频率采样。

### 2. 训练 Hollow Knight

```powershell
python -u train_HK.py `
  -n "HollowKnight_2L256D4H_200k_seed1_9FPS_bs64_twls2_ibl32" `
  -seed 1 `
  -config_path "config_files/STORM_HK.yaml" `
  -trajectory_path "D_TRAJ/HollowKnight.pkl" `
  -env_name "HollowKnight"
```

### 3. 训练 Hollow Knight: Silksong

也可以直接运行项目中的批处理脚本：

```powershell
.\train_HK.bat
```

等价命令示例：

```powershell
python -u train_HK.py `
  -n "HollowKnight_Silksong_2L256D4H_200k_seed1_9FPS_bs64_twls2_ibl32" `
  -seed 1 `
  -config_path "config_files/STORM_HK_Silksong.yaml" `
  -trajectory_path "D_TRAJ/HollowKnight_Silksong.pkl" `
  -env_name "HollowKnight_Silksong"
```

断点续训：

```powershell
python -u train_HK.py `
  -n "HollowKnight_Silksong_2L256D4H_200k_seed1_9FPS_bs64_twls2_ibl32" `
  -seed 1 `
  -config_path "config_files/STORM_HK_Silksong.yaml" `
  -trajectory_path "D_TRAJ/HollowKnight_Silksong.pkl" `
  -env_name "HollowKnight_Silksong" `
  -resume_step 410192
```

训练输出：

- 模型权重：`ckpt/<run_name>/world_model_<step>.pth`
- Agent 权重：`ckpt/<run_name>/agent_<step>.pth`
- Replay buffer：`ckpt/<run_name>/replay_buffer_<step>.pkl`
- 训练状态：`ckpt/<run_name>/train_state_<step>.pkl`
- TensorBoard 日志：`runs/<run_name>/`

查看 TensorBoard：

```powershell
tensorboard --logdir runs
```

### 4. 评估模型

运行批处理脚本：

```powershell
.\eval_HK.bat
```

或手动执行：

```powershell
python eval_HK.py `
  -run_name "HollowKnight_Silksong_2L256D4H_200k_seed1_9FPS_bs64_twls2_ibl32" `
  -env_name "HollowKnight_Silksong" `
  -config_path "config_files/STORM_HK_Silksong.yaml" `
  -num_episode 10
```

指定某个 checkpoint：

```powershell
python eval_HK.py `
  -run_name "HollowKnight_Silksong_2L256D4H_200k_seed1_9FPS_bs64_twls2_ibl32" `
  -env_name "HollowKnight_Silksong" `
  -config_path "config_files/STORM_HK_Silksong.yaml" `
  -num_episode 10 `
  -step 400431
```

评估结果会保存到 `eval_result/<run_name>_step<step>.csv`。

## 方法原理

项目采用基于世界模型的强化学习流程，核心思想是先用真实游戏交互训练一个可预测未来的 latent dynamics model，再让 Actor-Critic agent 在世界模型生成的 imagined rollouts 上学习策略，以减少真实游戏采样需求。

### 数据采集

每一步采集元组：

```text
(observation, action, reward, done)
```

- `observation`：游戏窗口截图，训练前缩放到 64 x 64 RGB。
- `action`：多按键二进制组合，经 `MultiBinaryToDiscreteWrapper` 转为离散动作。
  - Hollow Knight：7 个按键，离散动作数 `2^7 = 128`。
  - Silksong：8 个按键，离散动作数 `2^8 = 256`。
- `reward`：命中 Boss 时给予正奖励。
- `done`：角色死亡、Boss 被击败或环境截屏异常时终止。
- `life_loss`：角色受伤会被额外标记，并在 world model 训练中作为终止信号处理。

Replay buffer 默认容量为 150,000 条样本，采样频率约 9 FPS。

### World Model

World model 位于 `sub_models/world_models.py`，整体参考 STORM：

- CNN encoder 将 64 x 64 图像编码为离散随机 latent。
- Transformer sequence model 根据历史 latent 和动作预测后续 hidden state。
- 模型同时预测：
  - 下一步 prior latent distribution
  - reward
  - continuation / termination
  - reconstructed observation
- 训练损失包括图像重建、reward symlog two-hot、continuation BCE，以及 posterior/prior KL 项。

在 imagination 阶段，模型先用 replay buffer 中的短上下文初始化状态，然后自回归生成未来 latent、reward、termination。

### Actor-Critic

Agent 位于 `agents.py`。Actor 和 Critic 都是 MLP：

- 输入为 world model latent 和 Transformer hidden feature 的拼接。
- Actor 输出离散动作分布。
- Critic 输出状态价值。
- 使用 lambda-return、entropy regularization 和 EMA critic regularization 训练。

默认关键配置：

- 图像尺寸：64 x 64
- Transformer：2 层，hidden dim 256，4 heads
- Agent hidden dim：512
- `gamma = 0.985`
- `lambda = 0.95`
- imagination context length：8
- imagination rollout length：32

## 实验结果

报告中的主要结论：

- Hollow Knight 的 Sisters of Battle / Mantis Lord 类 Boss 训练约 250k steps，评估达到 90% 胜率。
- Hollow Knight: Silksong 的 Cogwork Dancers 训练约 400k steps，评估达到 8/10 胜率，并测试中有无伤通关。
- 对更高难度、背景变化更大、攻击频率更高的 Silksong Boss，例如 First Sinner 和 Lost Lace，world model 对复杂分布建模仍不稳定，未能得到稳定通关策略

视频演示：

- [Sisters of Battle / Hollow Knight Demo](https://www.youtube.com/watch?v=bDEW-dGeLl8&t=7s)
- [Cogwork Dancers / Silksong Demo](https://www.youtube.com/watch?v=Fy8mR5C2TKs)


## 注意事项

- 该项目强依赖真实游戏窗口、输入焦点和 Mod 事件服务；训练时不要移动窗口或切换焦点
- `keyboard` 在 Windows 上可能需要管理员权限。
- 目前训练只支持单环境采样，无法像 Atari benchmark 那样并行加速真实游戏
