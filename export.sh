#python3 export.py --weights runs/yolov8-obb-quant-4w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --nc 1 --input_shape 416 416 
#python3 export.py --weights runs/yolov8-obb-quant-2w2a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_2w2a_common_act.yaml --nc 1 --input_shape 416 416 

#python3 export.py --weights runs/yolov8-obb-quant-8w3a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_8w3a_common_act.yaml --nc 1 --input_shape 416 416
#python3 export.py --weights runs/yolov8-obb-quant-6w6a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_6w6a_common_act.yaml --nc 1 --input_shape 416 416
#python3 export.py --weights runs/yolov8-obb-quant-2w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_2w4a_common_act.yaml --nc 1 --input_shape 416 416 
#python3 export.py --weights runs/yolov8-obb-quant-3w5a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_3w5a_common_act.yaml --nc 1 --input_shape 416 416
#python3 export.py --weights runs/yolov8-obb-quant-6w4a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_6w4a_common_act.yaml --nc 1 --input_shape 416 416
#python3 export.py --weights runs/yolov8-obb-quant-4w2a/train/best.pt --cfg configs/models/quant/quantyolov8_obb_4w2a_common_act.yaml --nc 1 --input_shape 416 416


python3 export.py --weights ./best.pt --cfg configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml --nc 5 --input_shape 416 416 