# Dataset for YOLOv8-OBB

This folder is the default place for dataset and `data.yaml`. Training, validation, and export use a single **data YAML** that points to your images and defines classes.

## Directory layout

Use the standard **images / labels** layout so the loader can resolve label files from image paths:

```
path/to/dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   └── ...
│   └── val/
│       ├── img1.txt
│       └── ...
└── data.yaml
```

Label paths are derived from image paths by swapping the `images` segment for `labels` and changing the extension to `.txt`. So `.../images/train/img1.jpg` → `.../labels/train/img1.txt`.

**Supported image extensions:** `bmp`, `jpeg`, `jpg`, `png`, `tif`, `tiff`, `webp`.

## data.yaml

Example:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 15
names: {0: class0, 1: class1, 2: class2, ...}
```

| Key    | Required | Description |
|--------|----------|-------------|
| `path` | no*      | Root directory; paths in `train`/`val` are relative to this. If omitted, root is the directory of the YAML file. |
| `train`| yes      | Training images: directory path (relative to `path`) or list of paths. |
| `val`  | no       | Validation images; used for mAP during training and by `val.py`. |
| `nc`   | yes      | Number of classes (integer). |
| `names`| no       | Class names: dict `{0: "name0", 1: "name1", ...}` or list `["name0", "name1", ...]`. |

\* If `path` is omitted, `train` and `val` are resolved relative to the YAML’s parent directory.

## OBB label format

One **.txt** file per image. One line per object:

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

- **class_id**: integer index (0 to `nc - 1`).
- **x1 y1 … x4 y4**: four corners of the oriented box, **normalized** in `[0, 1]` (fraction of image width/height).

Order of corners is arbitrary but must be consistent (e.g. clockwise). Same format as [Ultralytics OBB](https://docs.ultralytics.com/datasets/obb/).

Example (one object, class 0, quadrilateral in normalized coords):

```
0 0.25 0.30 0.45 0.28 0.48 0.55 0.22 0.52
```

Images with no objects can have an empty `.txt` file or no file (missing labels are treated as empty).
