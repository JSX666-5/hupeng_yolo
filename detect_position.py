import cv2
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs
import os

os.environ["ORT_NO_CPU_AFFINITY"] = "1"

class YOLORealSense:
    def __init__(self, model_path):
        # 初始化 YOLO
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

        # 初始化 RealSense（只配置一次）
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

    def get_frame_data(self):
        """
        【真正实时】
        获取一帧 → 检测 → 直接返回所有目标数据
        返回：[ ((cx, cy), depth, confidence), ... ]
        无目标返回空列表
        """
        # ----------------------- 关键：只等一帧 -----------------------
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return []

        frame = np.asanyarray(color_frame.get_data())
        # -------------------------------------------------------------

        # YOLO 预处理
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        input_tensor = np.expand_dims(img, axis=0)

        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        pred = outputs[0][0]
        h, w = frame.shape[:2]

        boxes = []
        scores = []
        for i in range(pred.shape[0]):
            conf = pred[i, 4]
            if conf < 0.2:
                continue

            x1, y1, x2, y2 = pred[i, :4]
            x1 = int(x1 * w / 640)
            y1 = int(y1 * h / 640)
            x2 = int(x2 * w / 640)
            y2 = int(y2 * h / 640)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))

        # NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.2, 0.6)
            if len(indices):
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]

        # 返回数据：坐标 + 深度 + 置信度
        result = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            dist = depth_frame.get_distance(cx, cy)
            result.append( ((cx, cy), dist, score) )

        return result

    def release(self):
        self.pipeline.stop()

if __name__ == "__main__":
    
    detector = YOLORealSense("person_origin.onnx")
    while True:
    # 每一次调用 = 最新一帧数据
        targets = detector.get_frame_data()

        for target in targets:
           (cx, cy), distance, conf = target
           print(f"坐标({cx},{cy}) 距离 {distance:.2f}m 置信度 {conf:.2f}")


    detector.release()