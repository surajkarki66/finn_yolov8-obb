#python train.py --cfg configs/models/yolov8-obb.yaml --data data/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python train.py --cfg configs/models/yolov8-obb.yaml --data data/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp2.yaml --project runs/yolov8-obb-2

# HPD2
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_2w2a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-2w2a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_6w6a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-6w6a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_2w4a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-2w4a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_8w3a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-8w3a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_3w5a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-3w5a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w2a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w2a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_6w4a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-6w4a --train-quant-scales

# DOTA
#python train.py --cfg configs/models/yolov8-obb.yaml --data data/dota/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/dota/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_2w2a_common_act.yaml --data data/dota/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-2w2a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_6w6a_common_act.yaml --data data/dota/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-6w6a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_2w4a_common_act.yaml --data data/dota/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-2w4a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_8w3a_common_act.yaml --data data/dota/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-8w3a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_3w5a_common_act.yaml --data data/dota/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-3w5a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w2a_common_act.yaml --data data/dota/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w2a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_6w4a_common_act.yaml --data data/dota/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-6w4a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_8w8a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-8w8a --train-quant-scales

# DIOR
#python train.py --cfg configs/models/yolov8-obb.yaml --data data/YOLODIOR-R/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_2w2a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-2w2a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_6w6a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-6w6a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_2w4a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-2w4a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_8w3a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-8w3a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_3w5a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-3w5a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w2a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w2a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_6w4a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-6w4a --train-quant-scales
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_8w8a_common_act.yaml --data data/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-8w8a --train-quant-scales

# HABOF
#python3 train.py --cfg configs/models/yolov8-obb.yaml --data data/HABOF/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/HABOF/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales

# CEPDOF
#python3 train.py --cfg configs/models/yolov8-obb.yaml --data data/CEPDOF/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/CEPDOF/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales

# WEPDTOF
#python3 train.py --cfg configs/models/yolov8-obb.yaml --data data/WEPDTOF/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/WEPDTOF/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales

# fisheye8k
#python3 train.py --cfg configs/models/yolov8-obb.yaml --data data/fisheye8k/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/fisheye8k/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales

# PMOF
#python3 train.py --cfg configs/models/yolov8-obb.yaml --data data/data_416/data.yaml --weights pretrained/yolov8n-obb.pt --epochs 300 --batch 64 --imgsz 416 --hyp configs/hyp/hyp.yaml
#python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/data_416/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a --train-quant-scales
python3 train.py --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/data_416/data.yaml --epochs 300 --batch 32 --imgsz 416 --hyp configs/hyp/hyp.yaml --project runs/yolov8-obb-quant-4w4a_finetune --train-quant-scales --weights runs/yolov8-obb/train/best.pt
