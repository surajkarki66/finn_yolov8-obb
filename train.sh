#python train.py --cfg configs/models/yolov8-obb.yaml --data data/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 400 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python train.py --cfg configs/models/yolov8-obb.yaml --data data/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 400 --batch 32 --imgsz 416 --hyp configs/hyp/hyp2.yaml --project runs/yolov8-obb-2

# QAT
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/data.yaml --weights ./best.pt --epochs 400 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales --fresh-ema
python3 train.py --cfg configs/models/quant/quantyolov8_obb_8w8a_common_act.yaml --data data/data.yaml --weights ./best.pt --epochs 400 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-8w8a --train-quant-scales --fresh-ema --no-amp
