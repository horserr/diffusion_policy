# Push-T 与 RoboMimic 实验复现指南

> 目标：在仿真环境复现论文 Table I（low-dim state policy）与 Table II（image visual policy）中
> **Push-T** 与 **RoboMimic** 的结果。
> 官方原始说明见仓库根目录 `README.md`，本文件补充了任务/配置的映射与常见坑。

---

## 1. 环境准备（Linux + NVIDIA GPU）

```bash
# 1) 系统依赖（mujoco）
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# 2) 创建 conda 环境（推荐 mamba，更快）
mamba env create -f conda_environment.yaml   # 或 conda env create -f conda_environment.yaml

# 3) 激活环境 + wandb 登录
conda activate robodiff
wandb login
```

> 注意：`conda_environment.yaml` 环境名为 `robodiff`；`conda_environment_macos.yaml` 仅供开发，
> `conda_environment_real.yaml` 是真机环境。

---

## 2. 下载训练数据

```bash
mkdir -p data && cd data

# Push-T（图像 + 关键点都在一个 zip 里）
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip

# RoboMimic（包含 lift/can/square/transport/toolhang 的 ph 与 mh 数据集）
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic.zip
unzip robomimic.zip && rm -f robomimic.zip

cd ..
```

解压后应有：

```
data/pusht/pusht_cchi_v7_replay.zarr
data/robomimic/datasets/<task>/<dataset_type>/image.hdf5    # 图像
data/robomimic/datasets/<task>/<dataset_type>/low_dim.hdf5  # 低维 state
```

其中 `<task>` ∈ {lift, can, square, transport, toolhang}，`<dataset_type>` ∈ {ph, mh}。

> 注意：`conda_environment.yaml` 里的 `imagecodecs`/zarr 代码需要在导入时注册 codec，
> 仓库已在 `dataset/robomimic_replay_image_dataset.py` 顶部调用 `register_codecs()`。

---

## 3. 任务 ↔ 配置文件对照表

训练命令统一形式：

```bash
python train.py --config-name=<workspace配置> task=<task配置> [其他覆盖项]
```

### 3.1 Push-T

| 观察类型       | workspace 配置                          | task 配置      | 备注                  |
| -------------- | --------------------------------------- | -------------- | --------------------- |
| 图像           | `train_diffusion_unet_image_workspace`  | `pusht_image`  | 论文主结果，ResNet-18 |
| 关键点（低维） | `train_diffusion_unet_lowdim_workspace` | `pusht_lowdim` | 9 个 2D keypoints     |

Push-T 指标是 **target coverage（IoU）**，不是成功率（`env_runner/pusht_image_runner.py` 里 `legacy_test`）。

### 3.2 RoboMimic（每个任务 × {ph, mh} × {state, image}）

task 配置命名规则：`<task>_<lowdim|image>`，例如 `square_image`、`can_lowdim`、`lift_image`。

| RoboMimic 任务 | 低维 task 配置     | 图像 task 配置    |
| -------------- | ------------------ | ----------------- |
| Lift           | `lift_lowdim`      | `lift_image`      |
| Can            | `can_lowdim`       | `can_image`       |
| Square         | `square_lowdim`    | `square_image`    |
| Transport      | `transport_lowdim` | `transport_image` |
| Tool Hang      | `tool_hang_lowdim` | `tool_hang_image` |

- 低维用 `train_diffusion_unet_lowdim_workspace`；
- 图像用 `train_diffusion_unet_image_workspace`；
- Transformer 骨干分别用 `train_diffusion_transformer_lowdim_workspace` / `train_diffusion_transformer_hybrid_workspace`。
- 数据集类型（ph/mh）由 task 配置里的 `dataset_path` 决定，例如 `can_image.yaml` 里
  `dataset_path: data/robomimic/datasets/${task.task_name}/${task.dataset_type}/image.hdf5`，
  想跑 mh 就覆盖 `task.dataset_type=mh`。

---

## 4. 单 seed 训练

### 4.1 Push-T（图像，CNN，复现论文 Table II）

```bash
python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=pusht_image \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

### 4.2 RoboMimic square（图像，CNN）

```bash
python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=square_image \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

### 4.3 RoboMimic square（低维 state，CNN）

```bash
python train.py \
  --config-name=train_diffusion_unet_lowdim_workspace \
  task=square_lowdim \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

### 4.4 换数据集类型（例如 can 的 multi-human 图像）

```bash
python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=can_image \
  task.dataset_type=mh \
  training.seed=42 \
  training.device=cuda:0
```

训练输出目录结构（README 中的例子）：

```
data/outputs/<date>/<time>_<name>_<task>/
├── checkpoints/           # epoch=XXXX-test_mean_score=Y.YYY.ckpt + latest.ckpt
├── .hydra/                # config.yaml / hydra.yaml / overrides.yaml
├── logs.json.txt          # 每一步的 train_loss / val_loss / 评测指标
├── media/                 # 每隔 rollout_every 生成的 rollout 视频
└── train.log
```

- 评测指标以 `test/mean_score`（成功路或 coverage）记录在 wandb 与 `logs.json.txt`。
- 每 50 个 epoch 评测一次（`rollout_every=50`），并保存 top-5 checkpoint（按 `test_mean_score`）。

---

## 5. 多种子并行训练（复现论文 3-seed 均值）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3

python ray_train_multirun.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=pusht_image \
  --seeds=42,43,44 \
  --monitor_key=test/mean_score \
  multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
  multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'
```

聚合后的指标由 `multirun_metrics.py` 输出到 wandb 项目 `diffusion_policy_metrics`。
论文 Table I/II 报告的是：

- `(max performance) / (average of last 10 checkpoints)`；
- `max` 对应 `max` 聚合，`average of last 10 checkpoints` 对应 `k_min_train_loss` 聚合；
- 3 个训练 seed × 50 个环境初始条件（共 150 次 rollout）。

---

## 6. 离线评估已有 checkpoint

```bash
python eval.py \
  --checkpoint data/outputs/.../checkpoints/epoch=0550-test_mean_score=0.969.ckpt \
  --output_dir data/eval_output \
  --device cuda:0
```

输出 `eval_log.json`（含 `test/mean_score` 等指标）与 `media/*.mp4` 视频。

也可以直接从官方下载预训练 checkpoint 评估：

```
https://diffusion-policy.cs.columbia.edu/data/experiments/<image|low_dim>/<task>/diffusion_policy_cnn/train_0/checkpoints/<name>.ckpt
```

---

## 7. 关键超参与「配方」（复现成功的关键）

| 超参                       | 图像 CNN                                      | 低维 CNN         | 说明                            |
| -------------------------- | --------------------------------------------- | ---------------- | ------------------------------- |
| `horizon`                  | 16                                            | 16               | 动作预测长度 $T_p$              |
| `n_obs_steps`              | 2                                             | 2                | 观察长度 $T_o$                  |
| `n_action_steps`           | 8                                             | 8                | 执行长度 $T_a$                  |
| `num_train_timesteps`      | 100                                           | 100              | 训练扩散步数                    |
| `num_inference_steps`      | 100                                           | 100              | 推理步数（DDIM 可减到 10 加速） |
| `beta_schedule`            | `squaredcos_cap_v2`                           | 同               | Square Cosine 调度              |
| `down_dims`                | `[512,1024,2048]`                             | `[256,512,1024]` | UNet 通道                       |
| `kernel_size` / `n_groups` | 5 / 8                                         | 5 / 8            | 卷积核 / GroupNorm 组数         |
| `obs_encoder`              | ResNet-18, GroupNorm, crop 76                 | —                | 视觉编码器                      |
| `batch_size`               | 64                                            | 256              | dataloader                      |
| `lr` / `lr_scheduler`      | 1e-4 / cosine+500warmup                       | 同               |                                 |
| `use_ema`                  | True                                          | True             | EMA 模型用于评测                |
| `num_epochs`               | 8000（图像）/5000（低维，paper 说 3000/4500） | 同               |                                 |

> 论文原文：state-based 任务训练 4500 epochs，image-based 3000 epochs；
> 但仓库默认配置里 `num_epochs` 是 8000（图像）/5000（低维），实际按 `rollout_every=50` 周期性保存
> 并用 top-k checkpoint 选最优。若想严格对齐论文，可用
> `training.num_epochs=3000`（图像）/`4500`（低维）覆盖。

---

## 8. 常见坑与排查

1. **RoboMimic `.hdf5` 首次加载慢**：数据集会被转成 zarr 并缓存为
   `image.hdf5.zarr.zip`（`use_cache=True`）。第一次会 `Acquiring lock on cache.` 并生成缓存，之后秒加载。
2. **归一化问题**：`LinearNormalizer`（`model/common/normalizer.py`）是最常见的 bug 来源。
   若训练 loss 不收敛，先打印每个 key 的 `scale`/`bias` 检查。
3. **robomimic 动作空间**：默认 `abs_action=False`（直接用数据集里已归一化的 delta 动作）；
   若 `abs_action=True` 会用 `rotation_transformer` 把 axis-angle 转成 `rotation_6d`。
   低维 CNN 配置里 `oa_step_convention=True`，注意它会影响取动作的起始索引（`start=To-1` vs `To`）。
4. **图像策略 eval 需要 16 核 64GB**：`can_image.yaml` 里注释说明 `n_envs=28`、每个 env 约 1GB，
   资源不足时减小 `task.env_runner.n_test` 或 `n_envs`。
5. **Push-T 指标是 IoU/coverage**：`test/mean_score` 是 target 覆盖率，不是成功率。
6. **多进程 + OpenGL**：`gym_util/async_vector_env.py` 用 `fork`，robosuite 类环境初始化 OpenGL
   会导致子进程段错误，必要时提供 `dummy_env_fn`（见 README「Codebase Tutorial」）。
7. **EMA 与 BatchNorm 冲突**：必须用 GroupNorm（配置 `use_group_norm=True`），否则 EMA 会毁掉训练。

---

## 9. 最小复现清单（TL;DR）

```bash
# 环境
mamba env create -f conda_environment.yaml && conda activate robodiff && wandb login

# 数据
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip && unzip pusht.zip && rm pusht.zip
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic.zip && unzip robomimic.zip && rm robomimic.zip
cd ..

# Push-T 图像（CNN）
python train.py --config-name=train_diffusion_unet_image_workspace task=pusht_image training.seed=42 training.device=cuda:0

# RoboMimic square 图像（CNN）
python train.py --config-name=train_diffusion_unet_image_workspace task=square_image training.seed=42 training.device=cuda:0

# RoboMimic square 低维（CNN）
python train.py --config-name=train_diffusion_unet_lowdim_workspace task=square_lowdim training.seed=42 training.device=cuda:0
```
