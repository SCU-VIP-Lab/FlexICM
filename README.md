# FlexICM: A Flexible Image Coding for Machines Framework

Official codebase for the paper **FlexICM: A Flexible Image Coding for Machines Framework** (Tianma Shen, Ying Liu).

Built on the **TIC (Transformer-based Image Compression)** base codec, this repository implements:

- **TAIC (Base Layer)**: five single-task codecs that decode task intermediate features `h` **without** full image reconstruction
- **C-TAIC (Extension Layer)**: three multi-task scenarios that condition on the base-layer latent \(\hat{y}_b\) via cross-attention

## Five Tasks and Three Scenarios

### TAIC (five task codecs)

| Task | Teacher / Task Network | Feature Alignment | Metric |
|------|------------------------|-------------------|--------|
| Object Detection | Faster R-CNN + **Swin-B** | FPN `P2..P6` (Eq. 2) | mAP-bbox |
| Semantic Segmentation | UPerNet + **Swin-B** | FPN `P2..P6` | mIoU |
| Instance Segmentation | Mask R-CNN + **Swin-B** | FPN `P2..P6` | mAP-mask |
| Panoptic Segmentation | MaskFormer + **Swin-B** | Stages `F1..F4` (Eq. 3) | PQ |
| Pose Estimation | **HigherHRNet** | Stages `F1..F4` | mAP-OKS |

### C-TAIC (three scenarios)

| Scenario | Base Layer | Extension Layer |
|----------|------------|-----------------|
| **s1** | Object Detection | Instance Segmentation |
| **s2** | Semantic Segmentation | Panoptic Segmentation |
| **s3** | Object Detection | Pose Estimation |

---

## Environment Setup

> **Important:** Codec training **requires** task networks (teachers) to be available.
> The loss \(D\) is computed from frozen teacher features, so you cannot train TAIC / C-TAIC
> with only the codec packages. Install the teacher stack in **Task networks (teachers)** before the first training run.

### Recommended environment

- Ubuntu / RHEL, **CUDA 11.7+**, single **NVIDIA A100** (paper setting)
- Python **3.8–3.10**
- PyTorch **≥ 1.12** (2.0+ recommended)

```bash
conda create -n flexicm python=3.9 -y
conda activate flexicm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Core codec dependencies

| Package | Role |
|---------|------|
| `compressai` | EntropyBottleneck / GaussianConditional / conv-deconv |
| `timm` | **Required** Swin-B teacher backbone for feature alignment |
| `PyYAML` | Training configs |

### Task networks (teachers) — **required before training**

Teachers are already implemented in `flexicm/tasks/` and are constructed automatically by
`scripts/train_taic.py` / `scripts/train_ctaic.py` via `build_teacher(...)`.
You still must install their runtime dependencies and allow pretrained weights to download.

| Task | Teacher used in training | What you need installed |
|------|--------------------------|-------------------------|
| Detection / Instance / Semantic / Panoptic | Swin-B backbone (+ FPN or stages) via `timm` | `timm` (from `requirements.txt`); first run downloads ImageNet-pretrained Swin-B |
| Pose | HigherHRNet-style HRNet stem (original HRNet, not Swin) | Implemented in-repo; no extra package beyond PyTorch |

Checklist before training:

1. `pip install -r requirements.txt` (includes `timm`)
2. Machine can reach the internet **or** you have cached `timm` Swin-B weights (for the four Swin tasks)
3. Verify teachers import cleanly:

```bash
python -c "from flexicm.tasks import build_teacher; build_teacher('detection'); print('teachers ok')"
```

Without a working teacher, training will fail when computing the feature-alignment term \(D\).

### Task heads for metric evaluation

To evaluate paper metrics (mAP / mIoU / PQ / OKS) with full task heads, also install:

```bash
pip install -U openmim
mim install mmengine mmcv
mim install mmdet mmsegmentation mmpose
# or Detectron2 (alternative for detection / instance evaluation)
```

Recommended official weights (same model families as the paper):

- **Faster / Mask R-CNN + Swin-B**: MMDetection Model Zoo
- **UPerNet + Swin-B**: MMSegmentation Model Zoo
- **MaskFormer + Swin-B**: MMDetection / Mask2Former
- **HigherHRNet**: MMPose Model Zoo (**HRNet backbone**)

These full heads are **not** required to start codec training; they are for final rate–accuracy evaluation.

---

## Repository Layout

```
FlexICM/
├── FlexICM.pdf                 # paper
├── requirements.txt
├── README.md
├── configs/
│   ├── taic/                   # five single-task configs
│   │   ├── detection.yaml
│   │   ├── semantic.yaml
│   │   ├── instance.yaml
│   │   ├── panoptic.yaml
│   │   └── pose.yaml
│   └── ctaic/                  # three multi-task scenarios
│       ├── s1_det_instance.yaml
│       ├── s2_sem_panoptic.yaml
│       └── s3_det_pose.yaml
├── scripts/
│   ├── download_base_codecs.sh
│   ├── train_taic.py
│   └── train_ctaic.py
├── flexicm/
│   ├── models/                 # TAIC / C-TAIC / SFMA / TaskConnector / Conditional
│   ├── layers/                 # RSTB / WindowAttention (same lineage as AdaptiveICMH)
│   ├── tasks/                  # teachers + feature-alignment losses
│   ├── data/                   # COCO / COCO-WholeBody image loading
│   └── utils/
└── checkpoints/                # placeholder tree for base / TAIC / C-TAIC weights
                                # see checkpoints/README.md
```

Eval configs (stub until full metrics are implemented): `configs/eval/`.
Eval entry points: `scripts/eval_taic.py`, `scripts/eval_ctaic.py` (currently only check that real checkpoints replaced `PLACEHOLDER` files).

---

## Dataset Preparation

### COCO-2017 (detection / instance / semantic / panoptic)

```text
/data/coco2017/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── panoptic_train2017.json
    ├── panoptic_val2017.json
    ├── panoptic_train2017/          # PNG
    ├── panoptic_val2017/
    ├── stuff_train2017.json         # semantic / stuff (if used)
    └── stuff_val2017.json
```

Download:

```bash
# images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
# annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
```

Set in the corresponding YAML:

```yaml
dataset_path: "/data/coco2017"
```

### COCO-WholeBody (pose estimation)

Pose uses the same COCO `train2017/val2017` images plus WholeBody keypoint annotations:

- Project page: [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody)
- Place JSON files under `annotations/`; evaluate with MMPose HigherHRNet + WholeBody configs

Codec **training** only needs images for feature alignment, so `train2017` images are sufficient for that stage.

### Data processing (aligned with task networks)

The paper requires **codec training preprocessing to match task-network preprocessing**. Defaults in this repo:

1. **Codec input**: RGB, `ToTensor()` → `[0,1]`; training uses `Resize → RandomCrop(256) → RandomHorizontalFlip`
2. **Inside the teacher**: ImageNet mean/std normalization (consistent with Swin / HRNet pretraining)
3. **Spatial alignment**: TIC requires spatial size divisible by **256** (256 crop for training; pad at inference)

If you use official MMDet/MMSeg pipelines (short-side resize, normalization, etc.), ensure:

- Teacher feature extraction uses the **same normalize / resize logic** as that task network
- Codec and teacher see geometrically consistent tensors (same crop / same pad)

Edit points: `flexicm/data/datasets.py`, `flexicm/tasks/swin_teacher.py`, `flexicm/tasks/__init__.py` (HigherHRNet).

---

## Base Codec (TIC) Checkpoints

The paper uses the same TIC pretrained weights as AdaptiveICMH / TransTIC:

| Quality | λ (paper) | Checkpoint |
|:-------:|:---------:|------------|
| 1 | 0.0035 | [base_codec_1](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_1.pth.tar) |
| 2 | 0.0067 | [base_codec_2](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_2.pth.tar) |
| 3 | 0.0130 | [base_codec_3](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_3.pth.tar) |
| 4 | 0.0250 | [base_codec_4](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_4.pth.tar) |

```bash
bash scripts/download_base_codecs.sh
# downloads into checkpoints/base_codec/base_codec_{1,2,3,4}.pth.tar
```

Config example:

```yaml
base_codec: "./checkpoints/base_codec/base_codec_1.pth.tar"
quality_level: 1
lmbda: 0.0035
```

For each bitrate point, switch the matching `base_codec_k` and `lmbda`.

Trained TAIC / C-TAIC weights for eval should be placed under `checkpoints/taic/` and
`checkpoints/ctaic/` (see `checkpoints/README.md`). Until then, each quality folder
contains a `PLACEHOLDER` file.


---

## Training

Paper settings:

- Optimizer: **AdamW**, `lr=1e-4`
- TAIC: `batch_size=80`, `epochs=35`
- C-TAIC: `batch_size=40`, `epochs=40`
- `λ ∈ {0.0035, 0.0067, 0.0130, 0.0250}`

> If GPU memory is insufficient, reduce `batch_size` (optionally use gradient accumulation to approximate the paper effective batch).

### Train five TAIC models

```bash
# edit dataset_path / base_codec / lmbda / gpu_id in configs/taic/*.yaml as needed
python scripts/train_taic.py -c configs/taic/detection.yaml
python scripts/train_taic.py -c configs/taic/semantic.yaml
python scripts/train_taic.py -c configs/taic/instance.yaml
python scripts/train_taic.py -c configs/taic/panoptic.yaml
python scripts/train_taic.py -c configs/taic/pose.yaml
```

Trainable modules: **encoder SFMA + Task Connector**; TIC trunk is frozen.

### Train three C-TAIC scenarios

Requires a trained **base TAIC** checkpoint (to provide \(\hat{y}_b\)) and Stage-1 weights for the extension task.

```bash
# ---- s1: det → instance ----
python scripts/train_ctaic.py -c configs/ctaic/s1_det_instance.yaml --stage 1
python scripts/train_ctaic.py -c configs/ctaic/s1_det_instance.yaml --stage 2

# ---- s2: semantic → panoptic ----
python scripts/train_ctaic.py -c configs/ctaic/s2_sem_panoptic.yaml --stage 1
python scripts/train_ctaic.py -c configs/ctaic/s2_sem_panoptic.yaml --stage 2

# ---- s3: det → pose ----
python scripts/train_ctaic.py -c configs/ctaic/s3_det_pose.yaml --stage 1
python scripts/train_ctaic.py -c configs/ctaic/s3_det_pose.yaml --stage 2
```

Stage meanings:

| Stage | Mode | Trainable modules | `ŷ_b` |
|:-----:|------|-------------------|-------|
| 1 | TAIC mode | SFMA + Task Connector | not used |
| 2 | C-TAIC mode | Prompt Generator + Condition Generator | from frozen base TAIC AD output |

Check these config fields:

```yaml
base_taic_checkpoint:  # trained TAIC for the base task
taic_init:             # extension-task TAIC (optional Stage-1 init)
stage1_checkpoint:     # Stage-1 result loaded in Stage 2
```

---

## Code Map to the Paper

| Paper component | Code location |
|-----------------|---------------|
| SFMA | `flexicm/models/sfma.py` |
| Task Connector | `flexicm/models/task_connector.py` |
| TAIC | `flexicm/models/taic.py` |
| C-TAIC + two-stage freeze | `flexicm/models/ctaic.py` |
| Prompt / Mask / Cd | `flexicm/models/conditional.py` |
| Cross-attention (Q from features; K/V include prompts) | `flexicm/models/cross_attention.py` |
| \(R+\lambda D\) | `flexicm/tasks/losses.py` |
| Five teachers | `flexicm/tasks/__init__.py` |


---

## Citation

If you use this code or the paper, please cite FlexICM and acknowledge the base works:

- TIC: Lu et al., Transformer-based Image Compression
- TransTIC / AdaptiveICMH: task-adaptive SFMA tuning
