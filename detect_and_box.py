"""
读取 dataset/webImage 下的图片，用 best_cigarette.pt (YOLOv7) 检测香烟，
再用 OpenCV YuNet (基于深度学习的人脸检测器) 定位画面里的人脸
(作为"抽烟的人"的标记)，把两种检测框画在图片上，另存到一个新目录里。

之前试过 dlib 正脸检测 + Haar 侧脸级联的组合，但抽烟照片经常是极端侧脸/
逆光剪影，这类经典方法召回率还是不够。YuNet 是轻量级 DNN 人脸检测器，
对各种角度、遮挡的鲁棒性明显更好，21 张测试图里能测到的人脸数从 26 提升到 43，
且不需要装 dlib，只需要一个 ~230KB 的 onnx 模型文件(已放在仓库根目录:
face_detection_yunet_2023mar.onnx，来自 OpenCV Zoo)。

用法:
    python detect_and_box.py
    python detect_and_box.py --source dataset/webImage --output dataset/webImage_boxed --conf-thres 0.25
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

# best_cigarette.pt 是 torch<2.6 时代导出的 checkpoint，里面存了完整的 Model 对象，
# 在 torch>=2.6 下默认的 weights_only=True 会拒绝加载。这里权重来自本仓库自带的文件，可信任，
# 所以强制用 weights_only=False 来兼容新版 torch。
_orig_torch_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_load

from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_coords  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CIGARETTE_COLOR = (0, 0, 255)   # 红色 - 香烟
FACE_COLOR = (0, 255, 0)        # 绿色 - 抽烟的人(人脸)


def draw_box(img, xyxy, color, label):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    if label:
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def load_yolo_model(weights, device, img_size):
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    if device.type != "cpu":
        model.half()
    model.eval()
    return model, stride, img_size


def detect_cigarettes(model, stride, img_size, device, im0, conf_thres, iou_thres):
    img = letterbox(im0, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)

    img_t = torch.from_numpy(img).to(device)
    img_t = img_t.half() if device.type != "cpu" else img_t.float()
    img_t /= 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_t)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

    boxes = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_t.shape[2:], pred[:, :4], im0.shape).round()
        for *xyxy, conf, cls in pred:
            boxes.append((xyxy, float(conf), int(cls)))
    return boxes


def load_face_detector(model_path, conf_thres=0.6, nms_thres=0.3):
    return cv2.FaceDetectorYN.create(model_path, "", (320, 320), conf_thres, nms_thres)


def detect_faces(face_detector, im0):
    h, w = im0.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(im0)
    if faces is None:
        return []
    boxes = []
    for f in faces:
        x, y, bw, bh = f[:4]
        boxes.append((x, y, x + bw, y + bh))
    return boxes


def main(opt):
    source = Path(opt.source)
    out_dir = Path(opt.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"No images found in {source}")
        return

    device = select_device(opt.device)
    model, stride, img_size = load_yolo_model(opt.weights, device, opt.img_size)
    names = model.module.names if hasattr(model, "module") else model.names

    face_detector = load_face_detector(opt.face_model, opt.face_conf_thres)

    for img_path in images:
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            print(f"[skip] failed to read {img_path}")
            continue

        cig_boxes = detect_cigarettes(model, stride, img_size, device, im0, opt.conf_thres, opt.iou_thres)
        for xyxy, conf, cls in cig_boxes:
            draw_box(im0, xyxy, CIGARETTE_COLOR, f"{names[cls]} {conf:.2f}")

        face_boxes = detect_faces(face_detector, im0)
        for xyxy in face_boxes:
            draw_box(im0, xyxy, FACE_COLOR, "smoker face")

        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), im0)
        print(f"{img_path.name}: {len(cig_boxes)} cigarette(s), {len(face_boxes)} face(s) -> {save_path}")

    print(f"Done. Annotated images saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="dataset/webImage", help="输入图片目录")
    parser.add_argument("--output", type=str, default="dataset/webImage_boxed", help="输出图片目录")
    parser.add_argument("--weights", type=str, default="best_cigarette.pt", help="YOLOv7 权重路径")
    parser.add_argument("--img-size", type=int, default=640, help="推理尺寸")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IOU 阈值")
    parser.add_argument("--device", default="", help="cuda device, 例如 0 或 cpu")
    parser.add_argument("--face-model", type=str, default="face_detection_yunet_2023mar.onnx", help="YuNet人脸检测onnx模型路径")
    parser.add_argument("--face-conf-thres", type=float, default=0.6, help="人脸检测置信度阈值")
    opt = parser.parse_args()
    main(opt)
