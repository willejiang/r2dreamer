# R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides a PyTorch implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a computationally efficient world model that achieves high performance on continuous control benchmarks. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies. This repository is tested with Ubuntu 24.04 and Python 3.11.

If you prefer Docker, follow [`docs/docker.md`](docs/docker.md).

```bash
# Installing via a virtual env like uv is recommended.
pip install -r requirements.txt
```

Run training on default settings:

```bash
python3 train.py logdir=./logdir/test
```

Monitoring results:
```bash
tensorboard --logdir ./logdir
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro
python3 train.py model.rep_loss=r2dreamer
```

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Available Benchmarks
At the moment, the following benchmarks are available in this repository.

| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Image | Continuous | 1M | Robotic manipulation with complex contact interactions.|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [DMC Subtle](envs/dmc_subtle.py) | Image | Continuous |1M| DeepMind Control Suite with tiny task-relevant objects. |
| [Atari 100k](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

Use Hydra to select a benchmark and a specific task using `env` and `env.task`, respectively.

```bash
python3 train.py ... env=dmc_vision env.task=dmc_walker_walk
```

## Headless rendering

If you run MuJoCo-based environments (DMC / MetaWorld) on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```
## Full experiment use case for LEVERAGING HUMAN DEMONSTRATIONS FOR SURPRISE-PRIORITIZED REPLAY IN DREAMER-STYLE AGENTS FOR NETHACK
The experiment is run in google colab A100 80GB GPU. Highly recommend to do the research in notebook.


First, clone the repo and pip install all requirements in the notebook
```bash
!git clone https://github.com/willejiang/r2dreamer.git
!cd /content/r2dreamer
!pip install -r requirements.txt
```

Next, if you want to train the inverse model on [nle data](https://github.com/facebookresearch/nle), run the following commands and create a small python script in the notebook
```bash
%cd /content/
!mkdir -p nld-aa
!curl -o nld-aa/nld-aa-dir-aa.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-aa.zip
!mkdir /content/nle_data
!unzip -q /content/nld-aa/nld-aa-dir-aa.zip -d /content/nle_data/nld-aa

%cd /content/
!mkdir -p nld-nao
!curl -o nld-nao/nld-nao-dir-bf.zip  https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-bf.zip
!unzip -q nld-nao/nld-nao-dir-bf.zip -d /content/nle_data/nld-nao-bf
%cd /content/
!curl -o nld-nao/nld-nao_xlogfiles.zip https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao_xlogfiles.zip
!unzip -q nld-nao/nld-nao_xlogfiles.zip -d /content/nle_data/nld-nao-bf/nld-nao-unzipped
```

```bash
import shutil
import nle.dataset as nld
import numpy as np

shutil.rmtree("/root/.nle", ignore_errors=True)

nld.db.create()

nld.add_nledata_directory(
    "/content/nle_data/nld-aa/nle_data",
    "nld-aa-v0",
)

nld.add_altorg_directory(
    "/content/nle_data/nld-nao-bf/nld-nao-unzipped",
    "nld-nao-v0",
)

aa_dataset = nld.TtyrecDataset("nld-aa-v0", batch_size=1)
mb_aa = next(iter(aa_dataset))

print("AA sample:")
for k, v in mb_aa.items():
    arr = np.asarray(v)
    print(k, arr.shape, arr.dtype)

nao_dataset = nld.TtyrecDataset("nld-nao-v0", batch_size=1)
mb_nao = next(iter(nao_dataset))

print("\nNAO sample:")
for k, v in mb_nao.items():
    arr = np.asarray(v)
    print(k, arr.shape, arr.dtype)
```

To pretrain an agent in nethack, do the following
```bash
%cd /content/r2dreamer/
!python3 train.py \
  logdir=/content/drive/MyDrive/logdir/nethack_debug \
  env=nethack \
  env.task=nethack_Score \
  env.env_num=1 \
  env.eval_episode_num=2 \
  env.steps=20000 \
  env.time_limit=1000 \
  batch_size=8 \
  batch_length=32 \
  model=size12M \
  device=cuda:0
```

Training an inverse model is suggested
```bash
%cd /content/r2dreamer/
!python train_inverse_nld_aa_23.py \
  --dataset_name nld-aa-v0 \
  --steps 3000 \
  --lr 1e-4 \
  --hidden 512 \
  --hidden2 512 \
  --seed 0 \
  --include_reward \
  --log_every 100 \
  --max_batches 500 \
  --batch_size 256 \
  --save_path /content/drive/MyDrive/nld_aa_inverse_image_only_23.npz
```
remember to change the dataset name to the same name as was previously created, and remember to use aa data, which contains action.


The baseline for surprise prediction error is like the following
```bash
!python /content/r2dreamer/surprise_r2.py \
  --repo_dir /content/r2dreamer \
  --checkpoint /content/drive/MyDrive/logdir/nethack_debug/latest.pt \
  --data /content/nle_data/nld-nao-bf/nld-nao-unzipped \
  --inverse_model /content/drive/MyDrive/nld_aa_inverse_image_only_23.npz \
  --output /content/drive/MyDrive/nao_surprise_32horizon.npz \
  --override env=nethack \
  --override env.task=nethack_Score \
  --override env.env_num=1 \
  --override env.eval_episode_num=1 \
  --override model=size12M \
  --override device=cuda:0 \
  --seq_length 32 \
  --horizon 32 \
  --metric mse \
  --limit 1000 \
  --action_mode inverse \
  --random_seed 0
```

Fine tune the world model
```bash
!python /content/r2dreamer/rssm_training_human_surprise.py \
  --repo_dir /content/r2dreamer \
  --checkpoint /content/drive/MyDrive/logdir/nethack_debug/latest.pt \
  --surprise_npz /content/drive/MyDrive/nao_surprise_32horizon.npz \
  --data /content/nle_data/nld-nao-bf/nld-nao-unzipped \
  --inverse_model /content/drive/MyDrive/nld_aa_inverse_image_only_23.npz \
  --output_checkpoint /content/drive/MyDrive/logdir/nethack_debug/latest_surprise_200ft.pt \
  --override env=nethack \
  --override env.task=nethack_Score \
  --override env.env_num=1 \
  --override env.eval_episode_num=1 \
  --override model=size12M \
  --override device=cuda:0 \
  --seq_length 32 \
  --top_k 200 \
  --steps 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --rollout_horizon 32 \
  --save_every 50
```

Then one may use the fine tuned world model to run test on the dataset
```bash
!python /content/r2dreamer/surprise_r2.py \
  --repo_dir /content/r2dreamer \
  --checkpoint /content/drive/MyDrive/logdir/nethack_debug/latest_surprise_200ft.pt \
  --data /content/nle_data/nld-nao-bf/nld-nao-unzipped \
  --inverse_model /content/drive/MyDrive/nld_aa_inverse_image_only_23.npz \
  --output /content/drive/MyDrive/nao_surprise_after_trained.npz \
  --override env=nethack \
  --override env.task=nethack_Score \
  --override env.env_num=1 \
  --override env.eval_episode_num=1 \
  --override model=size12M \
  --override device=cuda:0 \
  --seq_length 32 \
  --horizon 32 \
  --metric mse \
  --limit 1000 \
  --action_mode inverse \
  --random_seed 0
```
## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[dreamerv3-torch]: https://github.com/NM512/dreamerv3-torch
