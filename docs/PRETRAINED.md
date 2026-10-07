# Pretrained weights used

Only general-purpose checkpoints (COCO detection). Nothing trained on drone / aerial / VisDrone-like data.
Files are downloaded into `checkpoints/`.

| File | Model | Trained on | Source |
|---|---|---|---|
| `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt` | YOLO11 s / m / l | COCO train2017 | https://github.com/ultralytics/assets/releases (auto-downloaded by Ultralytics) |
| `yolo12s.pt` | YOLO12 s | COCO train2017 | same |
| `yolo26s.pt` | YOLO26 s | COCO train2017 | same |
| `yolo26n.pt` | YOLO26 n | COCO train2017 | same; only used by Ultralytics' automatic mixed-precision self-check, never for training or inference |
| `dfine_s_coco.pth`, `dfine_m_coco.pth`, `dfine_l_coco.pth` | D-FINE S / M / L (HGNetv2 B0 / B2 / B4) | COCO train2017 | https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{s,m,l}_coco.pth |

Not used: Objects365 checkpoints, DINOv3 backbones, any VisDrone / UAV checkpoint.
Update this table whenever a new weight file is added.
