import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "person_origin.onnx"  # 你的模型文件名

# 检查文件是否存在
if not os.path.exists(MODEL_PATH):
    print(f"错误：找不到模型文件 '{MODEL_PATH}'")
else:
    print(f"正在加载模型: {MODEL_PATH} ...")
    # device=0 使用 GPU, verbose=False 关闭启动日志
    detector = YOLO(MODEL_PATH)
    print("模型加载成功！")

# ================= 2. 核心功能函数 =================
def get_centers_from_frame(frame, conf_thres=0.5):
    """
    输入单帧图像，输出画面中所有人的中心坐标列表。
    """
    if 'detector' not in globals():
        return []

    # 执行推理
    # stream=True 开启流模式（节省内存），verbose=False 关闭日志
    results = detector.predict(frame, conf=conf_thres, stream=True, verbose=False)
    
    centers = []
    
    for result in results:
        if result.boxes is not None:
            # 获取边界框坐标 [x1, y1, x2, y2]
            boxes = result.boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = box
                # 计算中心点
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers.append((cx, cy))
            
    return centers

# ================= 3. 视频流调用逻辑 =================
def run_video_inference(source=0):
    """
    打开视频源并实时调用检测函数。
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"无法打开视频源: {source}")
        return

    print(f"开始处理视频源: {source} ... (按 'q' 退出)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频播放完毕，退出。")
                break
                
            # 调用核心函数
            centers = get_centers_from_frame(frame)
            
            # 处理数据与显示
            if centers:
                # 打印第一个人的坐标
                print(f"检测到 {len(centers)} 人，中心坐标: {centers[0]}")
                
                # 在画面上画红点
                for cx, cy in centers:
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            else:
                print("未检测到人", end='\r')
            
            # 显示画面
            cv2.imshow('Real-time Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出。")


if __name__ == "__main__":
    run_video_inference(0)