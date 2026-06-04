# M6-09 Assessment — Cat Detection v2

## Image for leaderboard
docker pull gasimova51/cat-detector:final
Image: gasimova51/cat-detector:final
Student: Khavar Gasimova

## Week-1 Baseline Results
- mAP@0.5: 0.915
- mAP@0.5:0.95: 0.728
- Variant: yolo26s
- Epochs: 30

## Run inference
```bash
docker run --rm gasimova51/cat-detector:final info

docker run --rm \
  -v /path/to/images:/data/input:ro \
  -v /path/to/output:/data/output \
  gasimova51/cat-detector:final predict
```
