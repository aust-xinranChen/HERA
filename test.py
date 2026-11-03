from ultralytics import YOLO
import torch
from thop import profile
def main():
    model = YOLO('runs/train/OGSOD-2.0/../weights/best.pt')
    metrics = model.val(split='test', imgsz=256, device=7, batch=1, workers=4,data='ultralytics/cfg/datasets/OGSOD-2.0.yaml', project='runs/test/OGSOD-2.0', name='..')
    map75 = metrics.box.map75
    map50 = metrics.box.map50
    map = metrics.box.map

    print(f"mAP@0.5: {map50:.5f}")
    print(f"mAP@0.75: {map75:.5f}")
    print(f"mAP@0.5:0.95: {map:.5f}")

  
if __name__ == '__main__':
    main()
