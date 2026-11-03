from ultralytics import YOLO

def main():
    data ='ultralytics/cfg/datasets/OGSOD-2.0.yaml'
    cfg = 'ultralytics/cfg/models/benchmark/yolo11-obb-HERA.yaml'
    model = YOLO(cfg)    
    project = 'runs/train/OGSOD-2.0/'
    name = 'best'
    model.train(data=data, epochs=300, batch=64, imgsz=256, name=name, resume=False, device=[0], project=project) 
    # Single-GPU training: device=0 
    # Multi-GPU training: device=[0,1]'
    # resume train setting 'resume=True' and 'cfg = 'runs/train/M4-SAR/yolo11-obb-E2E-OSDet-M4-SAR-win2/weights/last.pt''

if __name__ == '__main__':
    main()
