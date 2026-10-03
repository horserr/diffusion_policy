# Diffusion Policy 代码树与文件作用说明

> 本文件梳理整个仓库的结构，逐一说明每个核心文件的作用。
> 阅读顺序建议：先看「核心入口」，再看「方法论核心」，最后按需查看各任务/环境模块。

---

## 0. 顶层入口文件（仓库根目录）

| 文件                                                            | 作用                                                                                                                  |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `train.py`                                                      | **训练入口**。使用 Hydra 读取 `diffusion_policy/config/` 下的配置，实例化对应的 `Workspace` 并调用 `run()` 开始训练。 |
| `eval.py`                                                       | **离线评估入口**。加载一个 checkpoint，在仿真环境里跑 rollout 并输出 `eval_log.json` 与视频。                         |
| `demo_pusht.py`                                                 | 仿真 Push-T 的交互式演示脚本。                                                                                        |
| `demo_real_robot.py` / `eval_real_robot.py`                     | 真机（UR5）数据采集演示 / 真机评估。                                                                                  |
| `ray_exec.py` / `ray_train_multirun.py` / `multirun_metrics.py` | 用 Ray 做多种子并行训练，并聚合多个 seed 的评测指标（对应论文 Table I/II/IV 的 `max` 与 `k_min_train_loss`）。        |
| `setup.py`                                                      | 包安装配置。                                                                                                          |
| `conda_environment.yaml`                                        | 仿真环境依赖（Linux + GPU）。`_real` 为真机依赖，`_macos` 仅供开发。                                                  |

---

## 1. `diffusion_policy/` 主包

### 1.1 方法论核心（**理解 Diffusion Policy 最关键的部分**）

#### `policy/` —— 策略（模型前向 + 扩散采样 + 损失）

| 文件                                                                 | 作用                                                                                                                                                                                                              |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `policy/base_lowdim_policy.py`                                       | 低维策略基类：定义 `predict_action` / `compute_loss` / `set_normalizer` 接口，并把模型移动到指定 device/dtype。                                                                                                   |
| `policy/base_image_policy.py`                                        | 图像策略基类：同上，观察为图像字典。                                                                                                                                                                              |
| `policy/diffusion_unet_lowdim_policy.py`                             | **CNN-based Diffusion Policy（低维版，核心）**。实现条件 DDPM：训练时加噪 + 预测噪声，推理时从高斯噪声逐步去噪生成动作序列。支持 `obs_as_local_cond`（FiLM）、`obs_as_global_cond`、inpainting 三种条件注入方式。 |
| `policy/diffusion_unet_image_policy.py`                              | **CNN-based Diffusion Policy（图像版，核心）**。与低维版几乎一致，但观察先经 `MultiImageObsEncoder` 编码成特征向量，再作为 global cond 注入 UNet。                                                                |
| `policy/diffusion_transformer_lowdim_policy.py`                      | **Transformer-based Diffusion Policy（低维版，核心）**。噪声动作序列作为 token，观察经 cross-attention 注入，实现 causal attention。                                                                              |
| `policy/diffusion_transformer_hybrid_image_policy.py`                | Transformer 骨干 + 图像观察（hybrid：图像特征 + 低维 proprioception）。                                                                                                                                           |
| `policy/diffusion_unet_hybrid_image_policy.py`                       | CNN 骨干 + 图像观察（hybrid）。                                                                                                                                                                                   |
| `policy/diffusion_unet_video_policy.py`                              | 视频输入变体（论文未重点使用）。                                                                                                                                                                                  |
| `policy/ibc_dfo_lowdim_policy.py` / `ibc_dfo_hybrid_image_policy.py` | 基线方法 **IBC**（能量模型 + DFO 优化采样）。                                                                                                                                                                     |
| `policy/robomimic_lowdim_policy.py` / `robomimic_image_policy.py`    | 基线方法 **LSTM-GMM（BC-RNN）**，来自 RoboMimic。                                                                                                                                                                 |
| `policy/bet_lowdim_policy.py`                                        | 基线方法 **BET**。                                                                                                                                                                                                |

#### `model/diffusion/` —— 扩散模型骨干网络

| 文件                                           | 作用                                                                                                                                  |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `model/diffusion/conditional_unet1d.py`        | **1D 时序 U-Net**（论文 Fig.1b 的 CNN 骨干）。用 1D 卷积处理动作序列，以 FiLM 方式注入扩散步 k 与观察特征。                           |
| `model/diffusion/conv1d_components.py`         | 1D 卷积基础组件：`Conv1dBlock`、`Downsample1d`、`Upsample1d`。                                                                        |
| `model/diffusion/transformer_for_diffusion.py` | **Time-series Diffusion Transformer**（论文 Fig.1c）。基于 minGPT 的 decoder-only 结构，支持 causal attention、观察 cross-attention。 |
| `model/diffusion/positional_embedding.py`      | 正弦位置编码（扩散步 k 用）。                                                                                                         |
| `model/diffusion/mask_generator.py`            | 生成 inpainting 条件 mask：训练时决定哪些时间步的观察/动作作为已知条件。                                                              |
| `model/diffusion/ema_model.py`                 | **指数滑动平均（EMA）**：对模型权重做 EMA，推理用 EMA 模型（DDPM 标准做法）。                                                         |

#### `model/vision/` —— 视觉编码器

| 文件                                      | 作用                                                                                                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model/vision/multi_image_obs_encoder.py` | **多视角视觉编码器（核心）**。对每个 RGB 相机用独立的 ResNet 编码，可替换 BatchNorm→GroupNorm，可做随机裁剪、ImageNet 归一化；最后把图像特征与低维 proprioception 拼接。 |
| `model/vision/model_getter.py`            | 根据名字构造 ResNet（`get_resnet`，论文用 ResNet-18 无预训练）。                                                                                                         |
| `model/vision/crop_randomizer.py`         | 随机裁剪（训练增强）与位置编码。                                                                                                                                         |

#### `model/common/` —— 通用工具

| 文件                                                            | 作用                                                                                                          |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `model/common/normalizer.py`                                    | **`LinearNormalizer`**：对 obs/action 做线性归一化（scale+bias），是最常见的 bug 来源，论文/README 特别提醒。 |
| `model/common/rotation_transformer.py`                          | 旋转表示变换（axis-angle ↔ rotation_6d 等，用于 robomimic 绝对动作）。                                        |
| `model/common/lr_scheduler.py`                                  | 学习率调度（cosine 等）。                                                                                     |
| `model/common/module_attr_mixin.py` / `dict_of_tensor_mixin.py` | 给模块加 `.device`/`.dtype` 属性、字典操作 mixin。                                                            |
| `model/common/shape_util.py` / `tensor_util.py`                 | 张量形状/类型工具。                                                                                           |

#### `model/bet/` —— BET 基线所需库

minGPT、latent generator、k-means 离散化等（仅 BET 基线使用）。

---

### 1.2 任务侧（每个任务 = Dataset + EnvRunner + config）

#### `dataset/` —— 数据集（把第三方数据适配到统一接口）

| 文件                                                                  | 作用                                                                                                                                                        |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset/base_dataset.py`                                             | **数据集接口定义（核心）**：`BaseLowdimDataset` / `BaseImageDataset`，规定 `__getitem__` 返回 `{obs, action}`，`get_normalizer()` 返回 `LinearNormalizer`。 |
| `dataset/pusht_image_dataset.py`                                      | **Push-T 图像数据集**。读 `pusht_cchi_v7_replay.zarr`，输出 `image + agent_pos` 观察与 2D 位置动作。                                                        |
| `dataset/pusht_dataset.py`                                            | Push-T 关键点（9 个 keypoints）低维数据集。                                                                                                                 |
| `dataset/robomimic_replay_image_dataset.py`                           | **RoboMimic 图像数据集（核心）**。把 `.hdf5` 转成 `ReplayBuffer`（可缓存为 `.zarr.zip`），处理绝对动作、旋转表示、normalizer。                              |
| `dataset/robomimic_replay_lowdim_dataset.py`                          | RoboMimic 低维（state）数据集。                                                                                                                             |
| `dataset/blockpush_lowdim_dataset.py`                                 | Block Push 低维数据集。                                                                                                                                     |
| `dataset/kitchen_lowdim_dataset.py` / `kitchen_mjl_lowdim_dataset.py` | Franka Kitchen 低维数据集。                                                                                                                                 |
| `dataset/real_pusht_image_dataset.py`                                 | 真机 Push-T 数据集（zarr）。                                                                                                                                |
| `dataset/mujoco_image_dataset.py`                                     | 通用 mujoco 图像数据集。                                                                                                                                    |

#### `env_runner/` —— 评测器（执行 policy rollout 并产出指标）

| 文件                                                                  | 作用                                                             |
| --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `env_runner/base_lowdim_runner.py` / `base_image_runner.py`           | runner 基类：管理多环境、视频记录、日志。                        |
| `env_runner/pusht_image_runner.py` / `pusht_keypoints_runner.py`      | Push-T 的评测 runner（图像/关键点），计算 target coverage 得分。 |
| `env_runner/robomimic_image_runner.py` / `robomimic_lowdim_runner.py` | **RoboMimic 评测 runner（核心）**，并行跑 N 个 env 计算成功率。  |
| `env_runner/blockpush_lowdim_runner.py` / `kitchen_lowdim_runner.py`  | Block Push / Kitchen 评测 runner。                               |
| `env_runner/real_pusht_image_runner.py`                               | 真机 Push-T 评测。                                               |

#### `env/` —— 仿真环境

| 子目录               | 作用                                                                                                                                                                                     |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `env/pusht/`         | **Push-T 环境（Pymunk 物理引擎）**：`pusht_env.py`（底层）、`pusht_image_env.py`（图像观察包装）、`pusht_keypoints_env.py`（9 关键点观察）、`pymunk_keypoint_manager.py`（关键点管理）。 |
| `env/robomimic/`     | RoboMimic 环境的 gym 包装（`robomimic_image_wrapper.py` / `robomimic_lowdim_wrapper.py`）。                                                                                              |
| `env/block_pushing/` | 多模态方块推块环境（PyBullet）+ 脚本 oracle。                                                                                                                                            |
| `env/kitchen/`       | Franka Kitchen 环境 + Relay Policy Learning 第三方代码。                                                                                                                                 |

#### `config/` —— Hydra 配置（训练任务的具体参数）

| 文件                                                                               | 作用                                                                                                                                          |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `config/train_diffusion_unet_image_workspace.yaml`                                 | **CNN 图像策略的训练主配置**（horizon=16, To=2, Ta=8, num_train_timesteps=100, ResNet18 等）。                                                |
| `config/train_diffusion_unet_lowdim_workspace.yaml`                                | CNN 低维策略训练主配置。                                                                                                                      |
| `config/train_diffusion_transformer_lowdim_workspace.yaml`                         | Transformer 低维策略训练配置。                                                                                                                |
| `config/train_diffusion_unet_hybrid_workspace.yaml`                                | CNN + 图像 hybrid 配置。                                                                                                                      |
| `config/train_diffusion_transformer_hybrid_workspace.yaml`                         | Transformer + 图像 hybrid 配置。                                                                                                              |
| `config/train_bet_lowdim_workspace.yaml` / `train_ibc_dfo_*` / `train_robomimic_*` | 基线方法（BET/IBC/LSTM-GMM）训练配置。                                                                                                        |
| `config/task/*.yaml`                                                               | **每个任务的具体定义**（shape_meta、dataset、env_runner）。例如 `task/pusht_image.yaml`、`task/can_image.yaml`、`task/square_image.yaml` 等。 |

---

### 1.3 支撑模块

| 目录/文件                         | 作用                                                                                                                                                                                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `workspace/`                      | **`Workspace`（训练/评测生命周期管理，核心）**。`base_workspace.py` 提供 checkpoint 保存/加载（dill + state*dict）；各 `train*\*\_workspace.py` 实现完整训练循环。                                                                           |
| `common/`                         | 通用工具：`replay_buffer.py`（zarr 数据存储）、`sampler.py`（`SequenceSampler` 序列采样与 padding）、`checkpoint_util.py`（TopK checkpoint 管理）、`json_logger.py`、`pytorch_util.py`、`normalize_util.py`、`robomimic_config_util.py` 等。 |
| `gym_util/`                       | 并行环境：`async_vector_env.py`（多进程）、`sync_vector_env.py`、`multistep_wrapper.py`（multi-step 跳过）、`video_recording_wrapper.py`。                                                                                                   |
| `codecs/imagecodecs_numcodecs.py` | Jpeg2000 图像压缩 codec（用于 zarr 图像压缩）。                                                                                                                                                                                              |
| `real_world/`                     | 真机代码：RealSense 相机、UR5 RTDE 控制、SpaceMouse 遥操作、数据转换。                                                                                                                                                                       |
| `shared_memory/`                  | 无锁共享内存队列/环形缓冲（真机多进程数据流）。                                                                                                                                                                                              |
| `scripts/`                        | 数据转换/评估辅助脚本（robomimic 数据转换、BET 数据生成、真机指标计算等）。                                                                                                                                                                  |

---

## 2. 数据流向总览（训练一帧数据的完整路径）

```
config/task/<task>.yaml  ──►  dataset/<task>_dataset.py   (读取 zarr/hdf5 → 采样序列)
                                        │
                                        ▼  {obs, action} 批数据
workspace/train_*_workspace.py  ──►  policy/<method>_policy.compute_loss()
                                        │  归一化 → 加噪 → 网络预测噪声 → MSE loss
                                        ▼
policy 内部 model/  (ConditionalUnet1D 或 TransformerForDiffusion)
```

推理（rollout）路径：

```
env_runner/<task>_runner.run(policy)
        │  每 Ta 步调用一次 policy.predict_action(obs)
        ▼
policy.predict_action(): 归一化观察 → 高斯噪声 → K 步去噪 → 反归一化 → 取前 Ta 步动作
        ▼
gym 环境 step() 执行动作 → 回到 runner 记录指标/视频
```

---

## 3. 论文术语 ↔ 代码变量对照表

| 论文术语                        | 代码变量                                                   | 含义                                  |
| ------------------------------- | ---------------------------------------------------------- | ------------------------------------- |
| Observation Horizon $T_o$       | `n_obs_steps` / `To`                                       | 输入多少步观察                        |
| Action Prediction Horizon $T_p$ | `horizon` / `T`                                            | 一次预测多长的动作序列                |
| Action Execution Horizon $T_a$  | `n_action_steps` / `Ta`                                    | 实际执行多少步（receding horizon）    |
| 去噪迭代次数 $K$                | `num_train_timesteps` / `num_inference_steps`              | 训练 100 步、推理可用 DDIM 减到 10 步 |
| 噪声预测网络 $\epsilon_\theta$  | `model`（`ConditionalUnet1D` / `TransformerForDiffusion`） | 预测噪声/分数的网络                   |
| 噪声调度                        | `noise_scheduler`（`DDPMScheduler`，`squaredcos_cap_v2`）  | 论文中的 Square Cosine Schedule       |
| 条件 $p(A_t\|O_t)$              | `obs_as_global_cond` / `obs_as_local_cond` / inpainting    | 三种观察条件注入方式                  |

> 关键默认超参（`config/train_diffusion_unet_*_workspace.yaml`）：
> `horizon=16`, `n_obs_steps=2`, `n_action_steps=8`, `num_train_timesteps=100`,
> `beta_schedule=squaredcos_cap_v2`, 图像用 ResNet-18（无预训练），`down_dims=[512,1024,2048]`（图像）/`[256,512,1024]`（低维）。
