#python train.py --cfg configs/models/yolov8-obb.yaml --data data/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python train.py --cfg configs/models/yolov8-obb.yaml --data data/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp2.yaml --project runs/yolov8-obb-2

# QAT
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_2w2a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-2w2a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_6w6a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-6w6a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_2w4a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-2w4a --train-quant-scales
python3 train.py --cfg configs/models/quant/quantyolov8_obb_8w3a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-8w3a --train-quant-scales


