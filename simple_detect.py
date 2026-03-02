import torch
import os
import cv2
import numpy as np
import onnxruntime
import argparse
import time

# ====================== 1. 自定义配置（根据你的模型修改） ======================
# 替换为你的检测类别，顺序必须和训练时一致
CLASSES = ['basketball']  # 示例，改成自己的
# 模型输入尺寸（和导出ONNX时的尺寸一致，默认640x640）
INPUT_SIZE = (640, 640)
# 检测阈值（可根据效果调整）
CONF_THRES = 0.5
IOU_THRES = 0.5


class YOLOV5_ONNX:
    def __init__(self, onnx_path):
        """初始化ONNX模型"""
        # 加载ONNX模型（支持CPU/GPU，若有GPU会自动使用）
        self.session = onnxruntime.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用GPU
        )
        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, img):
        """图像预处理：resize、格式转换、归一化、增加维度"""
        img_resized = cv2.resize(img, INPUT_SIZE)
        # BGR转RGB、HWC转CHW、归一化、增加batch维度
        img_processed = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_processed = img_processed.astype(np.float32) / 255.0
        img_processed = np.expand_dims(img_processed, axis=0)
        return img_processed, img_resized

    def infer(self, img):
        """执行推理：输入原始图像，返回检测结果和resize后的图像"""
        img_tensor, img_resized = self.preprocess(img)
        # 执行ONNX推理
        pred = self.session.run([self.output_name], {self.input_name: img_tensor})[0]
        return pred, img_resized


# 非极大值抑制（NMS）
def nms(dets, thresh):
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        # 计算相交区域
        xx1 = np.maximum(x1[i], x1[index[1:]])
        yy1 = np.maximum(y1[i], y1[index[1:]])
        xx2 = np.minimum(x2[i], x2[index[1:]])
        yy2 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # 计算IOU
        overlap = (w * h) / (areas[i] + areas[index[1:]] - w * h)
        # 保留IOU小于阈值的框
        index = index[np.where(overlap <= thresh)[0] + 1]
    return keep


# 过滤低置信度框并执行NMS
def postprocess(pred, conf_thres=0.5, iou_thres=0.5):
    pred = np.squeeze(pred)
    # 过滤置信度低于阈值的框
    valid_mask = pred[..., 4] > conf_thres
    pred = pred[valid_mask]
    if len(pred) == 0:
        return np.array([])

    # 获取类别
    cls = np.argmax(pred[..., 5:], axis=1)
    pred[..., 5] = cls

    # 坐标转换：xywh -> xyxy
    pred[:, :4] = xywh2xyxy(pred[:, :4])

    # 按类别执行NMS
    output = []
    for cls_id in np.unique(cls):
        cls_mask = pred[..., 5] == cls_id
        cls_pred = pred[cls_mask]
        keep = nms(cls_pred, iou_thres)
        output.append(cls_pred[keep])

    if len(output) > 0:
        return np.concatenate(output)
    return np.array([])


# 坐标转换：xywh -> xyxy
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


# 绘制检测结果
def draw_results(img, dets):
    if len(dets) == 0:
        return img

    boxes = dets[:, :4].astype(np.int32)
    scores = dets[:, 4]
    cls_ids = dets[:, 5].astype(np.int32)

    for box, score, cls_id in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = box
        # 绘制检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制类别和置信度
        label = f"{CLASSES[cls_id]} {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


# 检测单张图片
def detect_image(model, img_path, save_path="result.jpg"):
    print(f"正在检测图片：{img_path}")
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图片 {img_path}")
        return
    # 推理
    pred, img_resized = model.infer(img)
    # 后处理
    dets = postprocess(pred, CONF_THRES, IOU_THRES)
    # 绘制结果（将resize后的结果映射回原图尺寸）
    h, w = img.shape[:2]
    scale_x = w / INPUT_SIZE[0]
    scale_y = h / INPUT_SIZE[1]
    if len(dets) > 0:
        dets[:, [0, 2]] *= scale_x
        dets[:, [1, 3]] *= scale_y
    img = draw_results(img, dets)
    # 保存并显示结果
    cv2.imwrite(save_path, img)
    print(f"检测完成，结果已保存至：{save_path}")
    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 检测视频
def detect_video(model, video_path, save_path="result_video.mp4"):
    # 打开视频（0为摄像头，其他为视频文件路径）
    cap = cv2.VideoCapture(video_path if video_path != "0" else 0)
    if not cap.isOpened():
        print(f"错误：无法打开视频/摄像头 {video_path}")
        return

    # 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 初始化视频写入器（仅保存文件时使用）
    out = None
    if video_path != "0" and save_path:
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print("开始检测（按 'q' 退出）...")
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        pred, img_resized = model.infer(frame)
        # 后处理
        dets = postprocess(pred, CONF_THRES, IOU_THRES)
        # 绘制结果（映射回原图尺寸）
        scale_x = width / INPUT_SIZE[0]
        scale_y = height / INPUT_SIZE[1]
        if len(dets) > 0:
            dets[:, [0, 2]] *= scale_x
            dets[:, [1, 3]] *= scale_y
        frame = draw_results(frame, dets)

        # 计算并显示FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_real = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps_real:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 显示帧
        cv2.imshow("Video Detection", frame)
        # 保存帧（仅视频文件）
        if out:
            out.write(frame)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"检测完成！共处理 {frame_count} 帧，平均FPS：{fps_real:.1f}")
    if video_path != "0" and save_path:
        print(f"结果视频已保存至：{save_path}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YOLOv5 ONNX 推理程序（支持图片/视频/摄像头）")
    parser.add_argument("--model", required=True, help="ONNX模型路径，如：yolov5s.onnx")
    parser.add_argument("--type", required=True, choices=['image', 'video', 'camera'],
                        help="检测类型：image（图片）、video（视频）、camera（摄像头）")
    parser.add_argument("--input", help="输入路径：图片/视频路径；摄像头填0（默认0）", default="0")
    parser.add_argument("--output", help="输出路径：图片/视频保存路径", default="result.jpg")

    args = parser.parse_args()

    # 加载模型
    print(f"正在加载模型：{args.model}")
    try:
        model = YOLOV5_ONNX(args.model)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败：{e}")
        return

    # 根据类型执行检测
    if args.type == "image":
        detect_image(model, args.input, args.output)
    elif args.type == "video":
        detect_video(model, args.input, args.output)
    elif args.type == "camera":
        detect_video(model, "0")  # 摄像头固定用0


if __name__ == "__main__":
    main()