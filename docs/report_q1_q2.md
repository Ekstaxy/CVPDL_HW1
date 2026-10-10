# HW1 report draft: Q1 and Q2

<!--
Final numbers as of 2026-10-11. Final submission: 6-model WBF ensemble, val 0.4354, Kaggle public 0.38642
(results/_ensemble/wbf_tileft, reproduced by scripts/reproduce_final.sh).
All "val" numbers: mAP@[.5:.95] on the 700 val images, computed from the Kaggle-format CSV
with the TA's torchmetrics call (box_format="xywh"). Val was used only for model selection.
Page budget: Q1 about 1.2 pages, Q2 about 1.8 pages; Q3 (4 images) needs the rest.
-->

## Q1. Method

### Overview

The final submission is an ensemble of six COCO-pretrained detectors fine-tuned on the training set:
four DETR-style **D-FINE-L** models and two CNN **YOLO11** models (l and m).
Each model predicts with **sliced inference**, and the six outputs are fused with
**Weighted Boxes Fusion (WBF)**.

```
image ─┬─ D-FINE-L @1536, slicing-aided fine-tune ─┐  each model: full image (1.25x train size)
       ├─ D-FINE-L @1536                           │  + 512 px tiles (20 % overlap) at train size,     ┌──────────────┐
       ├─ D-FINE-L @1536, 32-epoch schedule        ├─ merged by class-wise NMS (IoU 0.6).             ─┤ WBF IoU 0.7  ├─ top-100 / class / image
       ├─ D-FINE-L @1280                           │  YOLO11l: 640 px tiles + h-flip;                  └──────────────┘
       ├─ YOLO11l  @1280                           │  YOLO11m: 800 px tiles + h-flip, full image at 1x
       └─ YOLO11m  @1536 ──────────────────────────┘
```

> Figure: insert D-FINE Fig. 2 (FDR + GO-LSD) from Peng et al., *D-FINE: Redefine Regression Task in DETRs
> as Fine-grained Distribution Refinement* (ICLR 2025), next to the pipeline sketch above.

### Models

**D-FINE** (main model, strongest single detector). An HGNetv2-B4 backbone feeds a hybrid encoder
(intra-scale attention + cross-scale CNN fusion) and a transformer decoder with 500 object queries
(300 in the COCO default; raised because val images contain up to 256 objects, with a median of 42).
D-FINE predicts each box edge as a probability distribution over offsets. Each decoder layer refines the
distributions of the previous layer (Fine-grained Distribution Refinement), and the last layer teaches the
earlier ones through self-distillation (GO-LSD). It is NMS-free: one query per object, matched one-to-one
by the Hungarian algorithm.

**YOLO11** (second model family, adds diversity to the ensemble). A CSP backbone (C3k2 + C2PSA attention
block), a PAN neck and an anchor-free decoupled head at strides 8/16/32, with Task-Aligned label
assignment and NMS at inference.

### Loss functions

| Model | Loss |
|---|---|
| D-FINE | $L = 1\cdot L_{VFL} + 5\cdot L_{1} + 2\cdot L_{GIoU} + 0.15\cdot L_{FGL} + 1.5\cdot L_{DDF}$, applied to every decoder layer, the encoder proposals and the denoising queries. VFL = varifocal classification loss (α 0.75, γ 2). FGL = fine-grained localization loss on the edge distributions. DDF = decoupled distillation focal loss (GO-LSD). Matching cost: 2·class + 5·L1 + 2·GIoU. |
| YOLO11 | $L = 7.5\cdot L_{CIoU} + 0.5\cdot L_{BCE\text{-}cls} + 1.5\cdot L_{DFL}$ (distribution focal loss on box edges) |

### Training settings

| | D-FINE-L 1536 (×3) | D-FINE-L 1280 | YOLO11l | YOLO11m |
|---|---|---|---|---|
| Input size | 1536 | 1280 | 1280 | 1536 |
| Epochs | 72 / 32 / 72 + 4 on tiles (last 8, 8, 1 without strong aug) | 72 (last 8 without strong aug) | 100 (mosaic off last 10) | 100 (mosaic off last 10) |
| Batch | 1, BatchNorm statistics frozen | 2 | 2 | 2 |
| Optimizer | AdamW, lr 5e-5 (backbone 2.5e-5), constant after 500 warm-up iterations, wd 1.25e-4, EMA 0.9999, grad clip 0.1 | same, lr 1e-4 (backbone 5e-5) | AdamW lr 7.1e-4, cosine decay, 3 warm-up epochs, wd 5e-4 | same |
| Augmentation | photometric distortion, random IoU crop, h-flip, multi-scale | same | mosaic, mixup 0.15, scale 0.7, HSV-V 0.5, h-flip | same |
| Train time (RTX 4090) | 18.1 h / 8.7 h / +8.0 h | 9.0 h | 4.1 h | 4.1 h |
| Peak VRAM | 8.3 GB | 10.4 GB | 10.6 GB | 10.7 GB |

The third D-FINE-L 1536 model starts from the first one and is trained for 4 more epochs on 640 px crops
of the training images (20 % overlap, 27,245 tiles) mixed with the 4,500 full images, each resized to 1536 px
(slicing-aided fine-tuning, as in SAHI).

**Inference.** conf ≥ 0.001, at most 100 boxes per class per image (the metric's per-class cap).
Each model sees the full image plus overlapping tiles. Every tile is resized to the training size, so a 512 px
tile is enlarged 3× for the 1536 px models. The views of one model are merged by class-wise NMS (IoU 0.6).
The tile size and the full-image scale were chosen per model on val (Q2). The six models are fused with WBF,
IoU 0.7, equal weights and averaged confidence. Every process is capped at 11.5 GB of VRAM (peak 10.2 GB in
inference); models run one after another, never together.

### Pretrained weights

All weights come from COCO train2017. None are from aerial, drone or VisDrone-like data.

| Weights | Source |
|---|---|
| `dfine_l_coco.pth` (also `dfine_s_coco.pth`, `dfine_m_coco.pth` in ablations) | github.com/Peterande/D-FINE releases |
| `yolo11l.pt`, `yolo11m.pt` (also `yolo11s`, `yolo12s`, `yolo26s` in ablations) | github.com/ultralytics/assets releases |

---

## Q2. Strategies and improvements

**Data analysis first** (training set): 60 % of the boxes are small (< 32² px; median box side 26 px),
there are 43 objects per image (median), class counts range from 97,953 (car) to 2,280 (awning-tricycle),
and about 13 % of the images are dark (mean brightness < 60). These three properties shaped what we tried.

### Main ablation (one change at a time, val mAP@[.5:.95], default full-image inference)

| # | Change | Parent | val | Δ |
|---|---|---|---|---|
| 1 | YOLO11s, 640 px (baseline) | – | 0.1642 | – |
| 2 | Resolution 640 → 1024 | 1 | 0.2509 | **+0.0867** |
| 3 | Resolution 1024 → 1280 | 2 | 0.2832 | +0.0323 |
| 4 | + stronger aug (mixup, scale, brightness) | 3 | 0.2907 | +0.0074 |
| 5 | + P2 (stride-4) head | 3 | 0.2785 | −0.0048 |
| 6 | + repeat-factor sampling of rare classes | 3 | 0.2798 | −0.0034 |
| 7 | + tiled training (800 px crops) | 3 | 0.2817 | −0.0015 |
| 8 | YOLO12s / YOLO26s instead of YOLO11s | 3 | 0.2904 / 0.2946 | +0.007 / +0.011 |
| 9 | Model size s → m | 3 | 0.3243 | **+0.0410** |
| 10 | m → l, + aug, 100 epochs | 9 | 0.3377 | +0.0134 |
| 11 | YOLO11m at 1536 px, + aug | 9 | 0.3493 | +0.0250 |
| 12 | D-FINE-M, 1280 px | 9 | 0.3501 | +0.0258 |
| 13 | D-FINE-L, 1280 px | 12 | 0.3667 | +0.0166 |
| 14 | D-FINE-L, 1536 px (batch 1, BatchNorm frozen) | 13 | 0.3771 | +0.0104 |
| 15 | 72 → 32 epochs | 14 | 0.3773 | +0.0002 (half the training time) |
| 16 | + 4 epochs on 640 px tiles | 14 | 0.3705 | −0.0066 (but +0.0065 with sliced inference, below) |

### Inference-time strategies (same weights)

| Strategy | YOLO11m-1280 | YOLO11l-1280 | D-FINE-L-1280 |
|---|---|---|---|
| Default | 0.3243 | 0.3377 | 0.3667 |
| H-flip TTA (NMS) | 0.3355 | 0.3491 | 0.3702 |
| Multi-scale + h-flip | 0.3545 | 0.3677 | 0.3833 |
| Sliced 800 px + full image | 0.3551 | 0.3728 | 0.3897 |
| Sliced + h-flip | **0.3581** | 0.3745 | 0.3890 |
| Tuned tiles + 1.25× full image | – | **0.3784** (640 px, h-flip) | **0.4009** (512 px) |
| Changing NMS IoU 0.6 → 0.5 / 0.7 | ±0.001 | ±0.001 | 0.0000 |

**Tile size matters, overlap does not** (D-FINE-L-1536, full image + tiles, 20 % overlap):

| Tile | 1024 px | 800 px | 640 px | 512 px | 512 px + full image at 1.25× |
|---|---|---|---|---|---|
| val | 0.3932 | 0.4014 | 0.4064 | 0.4067 | **0.4082** |

Overlap 0.15 / 0.2 / 0.25 / 0.3 changed val by at most 0.0007 for every tile size. Smaller tiles helped
D-FINE, but not YOLO11m-1536 (800 px 0.3771, 640 px 0.3759, 512 px 0.3634), so the tile size is chosen per model.

### Ensemble

| Ensemble (WBF, IoU 0.7, equal weights) | val | Kaggle public |
|---|---|---|
| D-FINE-L-1280 alone (sliced) | 0.3897 | 0.35387 |
| 4 models: D-FINE-L/M-1280, YOLO11l-1280, YOLO11m-1536 (800 px tiles) | 0.4213 | 0.37640 |
| D-FINE-L-1536 replaces D-FINE-M | 0.4262 | 0.37997 |
| tile size and full-image scale tuned per model | 0.4306 | 0.38324 |
| + D-FINE-L-1536 with the 32-epoch schedule (5 models) | 0.4331 | 0.38498 |
| + D-FINE-L-1536 with slicing-aided fine-tuning (6 models, final) | **0.4354** | **0.38642** |

Leaving one member out of the final ensemble always lowers val. The two weakest single models matter most,
because they are the only CNNs:

| Left out | YOLO11l | YOLO11m | D-FINE tile fine-tune | D-FINE-L-1280 | D-FINE 32-ep | D-FINE-L-1536 |
|---|---|---|---|---|---|---|
| val | 0.4326 (−0.0028) | 0.4331 (−0.0023) | 0.4332 (−0.0022) | 0.4346 (−0.0008) | 0.4347 (−0.0007) | 0.4350 (−0.0004) |

Pairs tell the same story: two YOLOs fused gave 0.3992 and two D-FINEs 0.3998, while one of each gave 0.4117.

### Small objects (the most important issue)

- **Input resolution was the largest single gain**: 640 → 1280 px raised val from 0.164 to 0.283 (+73 %), and
  mAP-small from 0.090 to 0.205. At 640 px a 26 px object shrinks to about 12 px, below what the stride-8 head
  resolves. 1280 → 1536 px added another +0.010 to +0.025.
- **Sliced inference** gave a further +0.023 to +0.066 on every model, at no training cost. Each tile is
  enlarged to the network input, so small objects are seen at 2–3× their size. The full image is kept as one
  extra view (at 1.25× for D-FINE) so that large objects cut by tile borders are still found.
- **Slicing-aided fine-tuning.** Our first tiled training (row 7, YOLO11s) did not help full-image inference, but
  that model gained the most from sliced inference (+0.066 vs +0.031 for YOLO11m), because training and inference
  then see objects at the same scale. Fine-tuning D-FINE-L-1536 for 4 epochs on 640 px tiles repeated this: full-image
  val dropped from 0.3771 to 0.3705, but sliced val rose from 0.4082 to **0.4147**, the best single model.
- **What did not help:** an extra P2 (stride-4) head improved mAP-small slightly (0.205 → 0.211) but lost overall
  mAP (−0.005 for s, −0.009 for m) and cost 1.7× training time.
- **D-FINE** was the best family for small objects. Its distribution-based box refinement gives tighter boxes
  (D-FINE-L mAP75 0.388 vs 0.356 for YOLO11l at the same 1280 px), which matters because mAP@[.5:.95] punishes
  a 2 px error on a 20 px box heavily. The final ensemble reaches mAP-small 0.366 (mAP-medium 0.537, large 0.577).

### Night images

Only brightness augmentation was used (HSV-V gain 0.5, plus D-FINE's photometric distortion); there is no
separate night model. Splitting val by mean brightness (< 60 → night, 32 of 700 images):

| Model | day | night |
|---|---|---|
| YOLO11s-1280 | 0.2834 | 0.2749 |
| + stronger aug (incl. brightness) | 0.2903 (+0.007) | 0.3038 (**+0.029**) |
| D-FINE-L-1536, tile fine-tune, sliced | 0.4157 | 0.3772 |
| Final ensemble | 0.4365 | 0.3947 |

The augmentation helped night images about four times more than day images. The final model still loses
0.04 at night. Val has only 32 night images, so the night numbers are noisy.

### Class imbalance

The metric is a **macro** average over classes, so awning-tricycle (1 % of the boxes) counts as much as car (41 %).

- **Repeat-factor sampling** (LVIS-style, threshold 0.5: images with rare classes are repeated) did not help
  (−0.003 overall). Tricycle improved (0.227 → 0.237), but awning-tricycle dropped (0.183 → 0.162),
  probably because the repeated images also repeat the frequent classes in them, which leads to overfitting.
- **Scaling the model helped the rare and small classes the most.** From YOLO11s-1280 to D-FINE-L-1280:
  bicycle 0.152 → 0.219 (+44 %), awning-tricycle 0.183 → 0.257 (+40 %), people 0.160 → 0.242 (+51 %),
  but car only 0.539 → 0.593 (+10 %).
- **Class-wise ensemble weights did not help.** Weighting each model per class by its own per-class AP gained
  only +0.0005 in a two-fold split of val (weights from one half, scored on the other), within the noise
  between the halves (0.002). The models rank the classes almost the same way, so equal weights were kept.
- In the final model the weakest classes are bicycle 0.269, people 0.299 and awning-tricycle 0.321.
  The best are bus 0.657 and car 0.636.

### Other observations

- **Batch 1 needs frozen BatchNorm.** D-FINE-L at 1536 px only fits 12 GB with batch 1. With live BatchNorm,
  statistics from a single image made training diverge (NaN loss) twice, even with the learning rate halved.
  Freezing BatchNorm statistics at their COCO values fixed it (val after the first epoch 0.265, vs 0.232 at 1280 px).
- **Most of D-FINE's training is not needed.** All three D-FINE runs peaked mid-training (epochs 20–44), drifted
  down by 0.004–0.015, and the final epochs without strong augmentation only brought them back. A 32-epoch
  schedule matched the 72-epoch one (0.3773 vs 0.3771) in half the time, and still added +0.0025 to the
  ensemble as an independently trained member.
- **WBF suits model ensembles, not D-FINE flip TTA.** Merging D-FINE's original and flipped predictions with WBF
  lost 0.046, while NMS gained 0.004. D-FINE outputs 500 one-to-one predictions per view. WBF averages the
  confidence over views, which penalizes correct boxes found in only one view. Across different models, the
  same WBF gave +0.021 over the best member.
- **Validation tracks Kaggle consistently.** The public score is 0.024–0.049 below val for all nine
  submissions (the gap grows for ensembles, whose settings were chosen on val), and the order is the same
  on val and on Kaggle. Choosing models on val, not on the public board, was therefore safe.
