from ultralytics import YOLO
import cv2


model = YOLO("person_fp16.onnx") 

# ===================== 2. 配置参数 =====================
CONF_THRESHOLD = 0.25  # 置信度
IOU_THRESHOLD = 0.45   # IOU
DEVICE = "cpu"           # 使用 GPU (如果是 0)，如果还是报 CUDA 错则改为 "cpu"

# ===================== 3. 执行推理 =====================
def run_camera():
    print("正在打开摄像头... (按 'q' 退出)")
    
    # 这里的 source=0 代表摄像头
    # stream=True 开启流模式，这对实时推理至关重要
    results = model.predict(
        source=0, 
        conf=CONF_THRESHOLD, 
        iou=IOU_THRESHOLD,
        device=DEVICE,
        show=True,   # 显示画面
        stream=True  # 开启流模式，极大提升 FPS
    )

    # 循环读取结果
    for result in results:
        # 按 'q' 退出的逻辑（虽然 show=True 会自动处理窗口，但为了规范可以保留）
        # 注意：在 stream 模式下，ultralytics 会自己管理窗口循环
        pass

if __name__ == "__main__":
    run_camera()
