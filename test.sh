#python3 val.py --weights runs/yolov8-obb/train/best.pt --cfg configs/models/yolov8-obb.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json  --project runs/yolov8-obb
#python3 val.py --weights runs/yolov8-obb-2/train/best.pt --cfg configs/models/yolov8-obb.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json  --project runs/yolov8-obb-2

#HPD
#python3 val.py --weights runs/yolov8-obb-quant-4w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-4w4a
#python3 val.py --weights runs/yolov8-obb-quant-8w3a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_8w3a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-8w3a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-6w6a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_6w6a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-6w6a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-2w2a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_2w2a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-2w2a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-2w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_2w4a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-2w4a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-3w5a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_3w5a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-3w5a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-6w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_6w4a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-6w4a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-4w2a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_4w2a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-4w2a --conf 0.01

#DOTA
#python3 val.py --weights runs/yolov8-obb-quant-4w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/dota/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-4w4a
#python3 val.py --weights runs/yolov8-obb-quant-8w3a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_8w3a_common_act.yaml --data data/dota/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-8w3a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-6w6a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_6w6a_common_act.yaml --data data/dota/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-6w6a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-2w2a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_2w2a_common_act.yaml --data data/dota/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-2w2a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-2w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_2w4a_common_act.yaml --data data/dota/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-2w4a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-3w5a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_3w5a_common_act.yaml --data data/dota/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-3w5a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-6w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_6w4a_common_act.yaml --data data/dota/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-6w4a --conf 0.01
#python3 val.py --weights runs/yolov8-obb-quant-4w2a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_4w2a_common_act.yaml --data data/dota/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-4w2a --conf 0.01

#DIOR-R
python3 val.py --weights runs/yolov8-obb-quant-4w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-4w4a --conf 0.01
python3 val.py --weights runs/yolov8-obb-quant-8w3a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_8w3a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-8w3a --conf 0.01
python3 val.py --weights runs/yolov8-obb-quant-6w6a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_6w6a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-6w6a --conf 0.01
python3 val.py --weights runs/yolov8-obb-quant-2w2a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_2w2a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-2w2a --conf 0.01
python3 val.py --weights runs/yolov8-obb-quant-2w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_2w4a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-2w4a --conf 0.01
python3 val.py --weights runs/yolov8-obb-quant-3w5a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_3w5a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-3w5a --conf 0.01
python3 val.py --weights runs/yolov8-obb-quant-6w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_6w4a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-6w4a --conf 0.01
python3 val.py --weights runs/yolov8-obb-quant-4w2a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_4w2a_common_act.yaml --data data/data.yaml --batch 32 --imgsz 416 --save-json --project runs/yolov8-obb-quant-4w2a --conf 0.01
