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
FACE_COLOR = (0, 255, 0)        # 绿色 - 普通人脸
SMOKER_COLOR = (0, 165, 255)    # 橙色 - 被判定为抽烟者的人脸


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


NOSE, LEYE, REYE, LWRIST, RWRIST = 0, 1, 2, 9, 10
KPT_CONF_THRES = 0.3


def load_pose_model(weights="yolov8n-pose.pt"):
    from ultralytics import YOLO

    return YOLO(weights)


def detect_poses(pose_model, im0, conf_thres=0.25):
    # 每个人一次性给出人体框 + 17 个 COCO 关键点(含手腕、鼻子/眼睛)，
    # 关键点天然按人分组，手腕和这个人自己的脸是绑在一起的，不用再靠距离去猜。
    result = pose_model.predict(im0, conf=conf_thres, verbose=False)[0]
    persons = []
    if result.keypoints is not None and result.boxes is not None and len(result.boxes):
        kpts = result.keypoints.data.cpu().numpy()  # (N, 17, 3): x, y, conf
        boxes = result.boxes.xyxy.cpu().numpy()      # (N, 4)
        for box, kp in zip(boxes, kpts):
            persons.append({"bbox": tuple(box), "kpts": kp})
    return persons


def _nearest_face_to_point(point, face_boxes):
    best_idx, best_dist = None, None
    for idx, (fx1, fy1, fx2, fy2) in enumerate(face_boxes):
        fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        dist = ((point[0] - fcx) ** 2 + (point[1] - fcy) ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            best_idx, best_dist = idx, dist
    return best_idx


def _match_via_nearest_face(cig_center, face_boxes, max_norm_dist):
    # 几何就近关联(备用方案): 直接找离香烟最近的人脸，用人脸框对角线归一化距离。
    # 局限: 如果有人把香烟举得离自己的脸很远、却离别人的脸更近，会关联错——
    # 这正是姿态方案要解决的问题，所以只在拿不到姿态数据时才兜底用它。
    best_idx, best_norm_dist = None, None
    for idx, (fx1, fy1, fx2, fy2) in enumerate(face_boxes):
        fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        diag = ((fx2 - fx1) ** 2 + (fy2 - fy1) ** 2) ** 0.5
        dist = ((cig_center[0] - fcx) ** 2 + (cig_center[1] - fcy) ** 2) ** 0.5
        norm_dist = dist / max(diag, 1e-6)
        if best_norm_dist is None or norm_dist < best_norm_dist:
            best_idx, best_norm_dist = idx, norm_dist
    if best_idx is not None and best_norm_dist <= max_norm_dist:
        return best_idx
    return None


def _match_via_pose(cig_center, persons, face_boxes, max_norm_dist):
    # 香烟离哪个人的手腕关键点最近(按这个人的人体框对角线归一化)，就归到那个人身上，
    # 再用同一个人的鼻子/眼睛关键点去找对应的 YuNet 人脸框——手腕和脸从一开始就是同一个人的，
    # 不会像纯人脸距离法那样在"手伸远了"时误判成旁边的人。
    best_person, best_norm_dist = None, None
    for person in persons:
        bx1, by1, bx2, by2 = person["bbox"]
        diag = ((bx2 - bx1) ** 2 + (by2 - by1) ** 2) ** 0.5
        for wrist_idx in (LWRIST, RWRIST):
            wx, wy, wc = person["kpts"][wrist_idx]
            if wc < KPT_CONF_THRES:
                continue
            dist = ((cig_center[0] - wx) ** 2 + (cig_center[1] - wy) ** 2) ** 0.5
            norm_dist = dist / max(diag, 1e-6)
            if best_norm_dist is None or norm_dist < best_norm_dist:
                best_norm_dist, best_person = norm_dist, person

    if best_person is None or best_norm_dist > max_norm_dist:
        return None

    nose_x, nose_y, nose_c = best_person["kpts"][NOSE]
    if nose_c >= KPT_CONF_THRES:
        ref = (nose_x, nose_y)
    else:
        leye, reye = best_person["kpts"][LEYE], best_person["kpts"][REYE]
        if leye[2] >= KPT_CONF_THRES and reye[2] >= KPT_CONF_THRES:
            ref = ((leye[0] + reye[0]) / 2, (leye[1] + reye[1]) / 2)
        else:
            bx1, by1, bx2, by2 = best_person["bbox"]
            ref = ((bx1 + bx2) / 2, by1 + (by2 - by1) * 0.15)  # 兜底：人体框顶部当作头部位置

    return _nearest_face_to_point(ref, face_boxes)


def find_smoker_faces(cig_boxes, face_boxes, persons=None, wrist_max_dist=0.5, face_max_dist=2.0):
    smoker_indices = set()
    for xyxy, conf, cls in cig_boxes:
        cx1, cy1, cx2, cy2 = [float(v) for v in xyxy]
        cig_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)

        matched_idx = None
        if persons:
            matched_idx = _match_via_pose(cig_center, persons, face_boxes, wrist_max_dist)
        if matched_idx is None:
            matched_idx = _match_via_nearest_face(cig_center, face_boxes, face_max_dist)
        if matched_idx is not None:
            smoker_indices.add(matched_idx)
    return smoker_indices


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
    pose_model = load_pose_model(opt.pose_weights) if opt.smoker_method == "pose" else None

    for img_path in images:
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            print(f"[skip] failed to read {img_path}")
            continue

        cig_boxes = detect_cigarettes(model, stride, img_size, device, im0, opt.conf_thres, opt.iou_thres)
        for xyxy, conf, cls in cig_boxes:
            draw_box(im0, xyxy, CIGARETTE_COLOR, f"{names[cls]} {conf:.2f}")

        face_boxes = detect_faces(face_detector, im0)
        persons = detect_poses(pose_model, im0) if pose_model is not None else None
        smoker_indices = find_smoker_faces(cig_boxes, face_boxes, persons, opt.wrist_max_dist, opt.smoker_max_dist)
        for idx, xyxy in enumerate(face_boxes):
            if idx in smoker_indices:
                draw_box(im0, xyxy, SMOKER_COLOR, "SMOKER")
            else:
                draw_box(im0, xyxy, FACE_COLOR, "face")

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
    parser.add_argument("--smoker-max-dist", type=float, default=2.0, help="[geo兜底]香烟到人脸中心的最大归一化距离(按人脸框对角线归一化)，超过则不判定为该人抽烟")
    parser.add_argument("--smoker-method", choices=["pose", "geo"], default="pose", help="抽烟者关联方式: pose=YOLOv8-pose手腕关键点(推荐) geo=纯人脸距离")
    parser.add_argument("--pose-weights", type=str, default="yolov8n-pose.pt", help="YOLOv8-pose 权重(首次运行会自动下载)")
    parser.add_argument("--wrist-max-dist", type=float, default=0.5, help="[pose]香烟到手腕的最大归一化距离(按人体框对角线归一化)")
    opt = parser.parse_args()
    main(opt)
