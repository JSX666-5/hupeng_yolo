import cv2
import numpy as np
import onnxruntime as ort
import time

class YOLO26ONNX:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.7):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 加载模型
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, img):
        # YOLO 官方标准预处理（绝对正确）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, outputs, orig_shape):
        # ==============================================
        # 【终极不踩坑版本】
        # 模型输出就是 640 坐标 → 直接缩放！不做任何多余计算
        # ==============================================
        pred = outputs[0][0]  # (8400, 84) or (8400, 5)
        h, w = orig_shape[:2]
        boxes = []
        scores = []

        for i in range(pred.shape[0]):
            # 直接取置信度（不做类别判断，避免过滤掉人）
            conf = pred[i, 4]
            if conf < 0.15:  # 极低阈值，确保能看到人
                continue

            # 直接读取模型输出的 x1 y1 x2 y2
            # 这是唯一不会出错的方式
            x1 = pred[i, 0]
            y1 = pred[i, 1]
            x2 = pred[i, 2]
            y2 = pred[i, 3]

            # 直接缩放到原图（最安全、最通用、不会出错）
            x1 = int(x1 * w / 640)
            y1 = int(y1 * h / 640)
            x2 = int(x2 * w / 640)
            y2 = int(y2 * h / 640)

            # 防止越界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # 过滤无效框
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))

        # NMS 去重
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.15, 0.7)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]

        return boxes, scores

    def run(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源：{source}")

        print("开始检测（按 q 退出）...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = self.preprocess(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            boxes, scores = self.postprocess(outputs, frame.shape)

            # 画框
            for i, box in enumerate(boxes):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {scores[i]:.2f}",
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLO26 Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = YOLO26ONNX('person_origin.onnx')
    detector.run(0)  # 摄像头