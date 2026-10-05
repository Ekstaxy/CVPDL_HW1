# CVPDL 2026 HW1 — object detection

Config-driven experiments for the 10-class drone-image detection homework. Every run is scored
locally the way Kaggle scores it: predictions are written in the submission CSV format and scored with
`torchmetrics` `MeanAveragePrecision(box_format="xywh", iou_type="bbox", class_metrics=True)`.

## Setup

```bash
scripts/setup_env.sh                       # creates .venv and installs requirements.txt
git clone https://github.com/Peterande/D-FINE third_party/D-FINE      # only for D-FINE runs
wget -P checkpoints https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth
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
.venv/bin/python evaluate.py --config results/<id>/config.yaml [--tag hflip predict.hflip=true]
.venv/bin/python test.py     --config results/<id>/config.yaml [--tag hflip]
.venv/bin/python scripts/tune_postprocess.py --top 2 --test-best     # inference sweep, no training
.venv/bin/python scripts/ensemble.py --name ens1 --runs <id_a> <id_b>/eval_hflip
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
| `models/` | model adapters: `yolo.py` (Ultralytics), `dfine.py` (official D-FINE repo) |
| `utils/` | config, data (tiles, repeat-factor sampling), postprocess (TTA, slicing, NMS/WBF), metrics + CSV, visualize, logger |
| `train.py`, `evaluate.py`, `test.py` | training, validation scoring, test submission |
| `scripts/` | environment, experiment/queue runners, audit, inference sweep, ensemble, visualization, summary |
| `tests/` | checks for the CSV format, the metric, slicing and merging |
| `docs/` | `EXPERIMENTS.md` (generated results table), `PRETRAINED.md`, notes |
| `checkpoints/` | pretrained weights only |
| `results/<id>/` | everything a run produces: `config.yaml`, `env.txt`, logs, `metrics.json`, `val_pred.csv`, `test_submission.csv`, `plots/`, `weights/{best,last}.pt`, `eval_<tag>/` inference variants |

## Rules built into the code

- The validation set is used only for checkpoint selection and scoring; test images are only predicted.
- Every process is capped at 11.5 GiB of GPU memory (`vram_cap_gb`), and the peak is recorded per run.
- Submissions keep at most 100 detections per class per image (what the metric scores), confidence ≥ 0.001.
