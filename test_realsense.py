import cv2
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs  # 新增：深度相机依赖

class YOLO26ONNX:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.7):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 加载YOLO模型
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # ===================== 新增：初始化RealSense相机 =====================
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置与YOLO同分辨率（关键：深度图 & 彩色图必须对齐）
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 启动相机
        self.pipeline.start(self.config)
        # =================================================================

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, outputs, orig_shape):
        pred = outputs[0][0]
        h, w = orig_shape[:2]
        boxes = []
        scores = []

        for i in range(pred.shape[0]):
            conf = pred[i, 4]
            if conf < 0.15:
                continue

            x1 = pred[i, 0]
            y1 = pred[i, 1]
            x2 = pred[i, 2]
            y2 = pred[i, 3]

            x1 = int(x1 * w / 640)
            y1 = int(y1 * h / 640)
            x2 = int(x2 * w / 640)
            y2 = int(y2 * h / 640)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.15, 0.7)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]

        return boxes, scores

    def run(self):
        print("开始检测 + 深度测距（按 q 退出）...")
        
        try:
            while True:
                # ===================== 从相机获取 彩色帧 + 深度帧 =====================
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # 转成OpenCV格式
                frame = np.asanyarray(color_frame.get_data())
                # =================================================================

                # YOLO检测
                input_tensor = self.preprocess(frame)
                outputs = self.session.run(None, {self.input_name: input_tensor})
                boxes, scores = self.postprocess(outputs, frame.shape)

                # 画框 + 测距（核心修改）
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box

                    # ========= 计算检测框的中心点 =========
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # ========= 读取中心点的深度距离 =========
                    distance = depth_frame.get_distance(cx, cy)

                    # 画框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 显示：置信度 + 距离（单位：米）
                    cv2.putText(frame, f"Person {scores[i]:.2f} | {distance:.2f}m",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 在中心点画红点，方便观察
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                cv2.imshow("YOLO + RealSense 测距", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # 退出时关闭相机
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = YOLO26ONNX('person_origin.onnx')
    detector.run()  # 不再用普通摄像头，直接用RealSense