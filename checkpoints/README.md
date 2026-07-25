# Checkpoints Layout (Placeholders)

Put pretrained / trained weights here before running **test / eval**.
Training still writes to `logs/` by default; after training, copy (or symlink) best
checkpoints into this tree so eval configs have a stable path.

> Files named `PLACEHOLDER` are not real weights. Replace each with the matching
> `.pth.tar` checkpoint, then update or keep the path expected by eval scripts.

## Directory map

```text
checkpoints/
├── base_codec/                 # frozen TIC (TransTIC / AdaptiveICMH)
│   ├── base_codec_1.pth.tar    # λ = 0.0035
│   ├── base_codec_2.pth.tar    # λ = 0.0067
│   ├── base_codec_3.pth.tar    # λ = 0.0130
│   └── base_codec_4.pth.tar    # λ = 0.0250
│
├── taic/                       # five single-task TAIC codecs
│   ├── detection/{1,2,3,4}/checkpoint_best_loss.pth.tar
│   ├── semantic/{1,2,3,4}/checkpoint_best_loss.pth.tar
│   ├── instance/{1,2,3,4}/checkpoint_best_loss.pth.tar
│   ├── panoptic/{1,2,3,4}/checkpoint_best_loss.pth.tar
│   └── pose/{1,2,3,4}/checkpoint_best_loss.pth.tar
│
└── ctaic/                      # three multi-task scenarios
    ├── s1_det_instance/
    │   ├── stage1/{1,2,3,4}/checkpoint_best_loss.pth.tar
    │   └── stage2/{1,2,3,4}/checkpoint_best_loss.pth.tar
    ├── s2_sem_panoptic/
    │   ├── stage1/{1,2,3,4}/checkpoint_best_loss.pth.tar
    │   └── stage2/{1,2,3,4}/checkpoint_best_loss.pth.tar
    └── s3_det_pose/
        ├── stage1/{1,2,3,4}/checkpoint_best_loss.pth.tar
        └── stage2/{1,2,3,4}/checkpoint_best_loss.pth.tar
```

Quality folders `{1,2,3,4}` match paper λ / TIC quality levels.

## Download base TIC codecs

```bash
bash scripts/download_base_codecs.sh
# downloads into checkpoints/base_codec/
```

## After training: copy into placeholders

```bash
# example: TAIC detection, quality 1
cp logs/taic_detection/1/checkpoint_best_loss.pth.tar \
   checkpoints/taic/detection/1/checkpoint_best_loss.pth.tar

# example: C-TAIC s1 stage2, quality 1
cp logs/ctaic_s1_stage2/1/checkpoint_best_loss.pth.tar \
   checkpoints/ctaic/s1_det_instance/stage2/1/checkpoint_best_loss.pth.tar
```

## Eval configs

See `configs/eval/` — they point to these placeholder paths.

Codec test (bpp / feature distortion):

```bash
python scripts/eval_taic.py -c configs/eval/taic_detection.yaml
python scripts/eval_ctaic.py -c configs/eval/ctaic_s1.yaml
```

Scripts refuse to run if a `PLACEHOLDER` file is still present or if the `.pth.tar` is missing.
Task rate–accuracy metrics are not implemented in these scripts yet.

## Task networks (for metric evaluation)

```text
checkpoints/task_networks/
├── detection/model.pth
├── instance/model.pth
├── semantic/model.pth
├── panoptic/model.pth
└── pose/model.pth
```

These are **official pretrained task networks** (not codec weights).
Required when running:

```bash
python scripts/eval_taic.py -c configs/eval/taic_detection.yaml --with-metrics
```

See `configs/task_networks/README.md` for config/checkpoint pairing.
