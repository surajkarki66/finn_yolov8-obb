python val.py --weights runs/yolov8-obb/train/best.pt --cfg configs/models/yolov8-obb.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json  --project runs/yolov8-obb
#python val.py --weights runs/yolov8-obb-quant-4w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-4w4a
