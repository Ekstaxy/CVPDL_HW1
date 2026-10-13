# CVPDL 2026 HW1 — object detection

Config-driven experiments for the 10-class drone-image detection homework. Every run is scored
locally the way Kaggle scores it: predictions are written in the submission CSV format and scored with
`torchmetrics` `MeanAveragePrecision(box_format="xywh", iou_type="bbox", class_metrics=True)`.

## Final submission

`submission_test.csv` is a Weighted Boxes Fusion (WBF) ensemble of six COCO-pretrained detectors fine-tuned
on the training set only. Each model predicts with sliced inference, and the six outputs are fused with WBF.

| Member | Training | Inference | val mAP50:95 |
|---|---|---|---|
| D-FINE-L 1536, slicing-aided fine-tune | `s3_03_dfine_l_1536_tileft`: s3_01 + 4 epochs on 640 px tiles | full image at 1920 + 512 px tiles | 0.4147 |
| D-FINE-L 1536 | `s3_01_dfine_l_1536`: 72 epochs, batch 1, BatchNorm frozen | same | 0.4082 |
| D-FINE-L 1536, short schedule | `s3_02_dfine_l_1536_short`: 32 epochs | same | 0.4078 |
| D-FINE-L 1280 | `s2_05_dfine_l_1280`: 72 epochs, batch 2 | full image at 1600 + 512 px tiles | 0.4009 |
| YOLO11l 1280 | `s2_02_yolo11l_1280_aug`: 100 epochs | full image at 1600 + 640 px tiles, h-flip | 0.3784 |
| YOLO11m 1536 | `s2_04_yolo11m_1536_aug`: 100 epochs | full image + 800 px tiles, h-flip | 0.3771 |
| **WBF ensemble** (IoU 0.7, equal weights) | | | **0.4354** (Kaggle public 0.38642) |

Tiles overlap by 20 % and are resized to the model's training size. Views of one model are merged by
class-wise NMS (IoU 0.6). The final CSV keeps at most 100 boxes per class per image, confidence ≥ 0.001.

### Reproduce

1. Set up the environment and the D-FINE code:

   ```bash
   scripts/setup_env.sh                                   # .venv with requirements.txt (torch 2.5.1, ultralytics 8.4.173, gdown)
   git clone https://github.com/Peterande/D-FINE third_party/D-FINE
   git -C third_party/D-FINE checkout 956d170
   ```

2. Put the homework data in `dataset/` (`images/{train,val,test}`, `annotations/instances_{train,val}.json`,
   `labels/`, `sample_submission.csv`), exactly as downloaded.

3. Run one of:

   **A. Inference only, with the trained weights** (about 2 h on one RTX 4090)

   ```bash
   scripts/download_weights.sh           # the six trained models from Google Drive (gdown, 2.1 GB, MD5-checked)
   scripts/reproduce_final.sh            # predict val + test with each model, then WBF
   ```

   **B. Train from scratch, then infer** (about 52 h)

   ```bash
   wget -P checkpoints https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth
   # yolo11l.pt and yolo11m.pt download themselves into checkpoints/ on first use
   scripts/reproduce_final.sh --train
   ```

   `--train` trains the six models in order with `scripts/run_experiment.sh` (train, then val CSV + mAP, then
   test CSV), so `s3_03` can continue from `s3_01`'s `best.pt`. It writes the weights to `results/<id>/weights/`,
   over the downloaded ones if they are there. Then it runs the same inference and WBF as A.

   Both write `submission_test.csv` to the repo root. The val CSV and its score are in `results/_ensemble/final/`.
   With `TAG_SUFFIX=_repro` the member outputs are written next to the existing ones instead of over them.

Training is seeded (`seed: 0`, deterministic mode). GPU kernels can still differ slightly between runs,
so a retrained model may differ in the third decimal of mAP.

## Setup for other experiments

```bash
wget -P checkpoints https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{s,m}_coco.pth   # ablations only
```

The dataset goes in `dataset/` (the layout from the homework slide). YOLO weights download
themselves into `checkpoints/`. Pretrained weights are listed in `docs/PRETRAINED.md`.

## Run

```bash
scripts/run_experiment.sh s1_03_yolo11s_1280          # train -> val CSV + mAP -> test CSV
nohup setsid scripts/run_queue.sh configs/queues/stage1.txt > results/queue_stage1.log 2>&1 &
.venv/bin/python scripts/summarize.py                 # rebuild docs/EXPERIMENTS.md
```

Single steps:

```bash
.venv/bin/python train.py    --config configs/<id>.yaml [train.batch=4 ...]
.venv/bin/python evaluate.py --config results/<id>/config.yaml [--weights <file>] [--tag hflip predict.hflip=true]
.venv/bin/python test.py     --config results/<id>/config.yaml [--weights <file>] [--tag hflip]
.venv/bin/python scripts/tune_postprocess.py --top 2 --test-best     # inference sweep, no training
scripts/sweep_slice.sh <id>                                          # tile size x overlap sweep for one run
.venv/bin/python scripts/ensemble.py --name ens1 --members <id_a>/<tag> <id_b>/<tag>   # WBF, grid on val
.venv/bin/python scripts/ensemble_classwise.py --name cw1 --members <id_a>/<tag> <id_b>/<tag>
.venv/bin/python scripts/visualize_predictions.py --csv results/<id>/val_pred.csv --split val --gt
.venv/bin/python scripts/audit_dataset.py
.venv/bin/pytest tests/
```

Re-running an experiment is safe: finished steps are skipped and an interrupted training resumes.
To redo a run from scratch, delete `results/<id>/`.

## Layout

| Path | Content |
|---|---|
| `configs/` | `base.yaml` defaults; one YAML per experiment (`base:` parent + overrides); `models/` custom architectures; `queues/` run lists |
| `models/` | model adapters: `yolo.py` (Ultralytics), `dfine.py` (official D-FINE repo; adds frozen BatchNorm and tile training) |
| `utils/` | config, data (tiles, repeat-factor sampling, tile COCO json), postprocess (TTA, slicing, NMS/WBF), metrics + CSV, visualize, logger |
| `train.py`, `evaluate.py`, `test.py` | training, validation scoring, test submission |
| `scripts/` | environment, experiment/queue runners, audit, inference sweeps, ensembles, visualization, summary, `download_weights.sh`, `reproduce_final.sh` |
| `tests/` | checks for the CSV format, the metric, slicing and merging |
| `docs/` | `EXPERIMENTS.md` (generated results table), `PRETRAINED.md`, `kaggle_scores.yaml`, notes, report draft |
| `checkpoints/` | pretrained weights only |
| `results/<id>/` | everything a run produces: `config.yaml`, `env.txt`, logs, `metrics.json`, `val_pred.csv`, `test_submission.csv`, `plots/`, `weights/{best,last}.pt`, `eval_<tag>/` inference variants |
| `results/_ensemble/<name>/` | ensemble outputs: `config.yaml`, `metrics.json`, `val_pred.csv`, `test_submission.csv`, `grid.csv` |

## Rules built into the code

- Training uses the train split only, including the tiles. The validation set is used only for checkpoint
  selection and for choosing inference and ensemble settings. Test images are only predicted.
- Pretrained weights are COCO only (`docs/PRETRAINED.md`). Nothing trained on aerial or VisDrone-like data.
- Every process is capped at 11.5 GiB of GPU memory (`vram_cap_gb`), and the peak is recorded per run.
  For the six final models the largest peaks are 10.7 GB in training and 10.2 GB in inference.
- Submissions keep at most 100 detections per class per image (what the metric scores), confidence ≥ 0.001.
