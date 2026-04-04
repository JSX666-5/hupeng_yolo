from ultralytics import YOLO
import cv2

# ===================== 1. 加载你的训练好的模型 =====================
# 替换成你自己的模型路径（绝对路径/相对路径都可以）
model = YOLO("../person.pt")  # 你的预训练/训练后模型

# ===================== 2. 推理配置（可根据需求修改） =====================
CONF_THRESHOLD = 0.25  # 置信度阈值（低于该值不显示）
IOU_THRESHOLD = 0.45   # IOU 阈值（去重框）
SHOW_RESULT = True     # 是否显示推理画面
SAVE_RESULT = True     # 是否保存推理结果
DEVICE = "0"         # 推理设备：cpu / cuda(显卡) / 0(第一张显卡)

# ===================== 3. 执行推理 =====================

# -------------------- 方式1：单张图片推理 --------------------
def infer_image(image_path):
    print(f"正在推理图片: {image_path}")
    # 执行推理
    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        show=SHOW_RESULT,
        save=SAVE_RESULT,
        device=DEVICE
    )
    # 打印检测结果信息
    for result in results:
        print(f"检测到目标数量: {len(result.boxes)}")
        print(f"目标类别+置信度: {result.boxes.cls} {result.boxes.conf}")

# -------------------- 方式2：视频文件推理 --------------------
def infer_video(video_path):
    print(f"正在推理视频: {video_path}")
    results = model.predict(
        source=video_path,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        show=SHOW_RESULT,
        save=SAVE_RESULT,
        device=DEVICE
    )

# -------------------- 方式3：摄像头实时推理 --------------------
def infer_camera(camera_id=0):
    print("正在打开摄像头，按 q 退出...")
    results = model.predict(
        source=camera_id,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        show=SHOW_RESULT,
        save=SAVE_RESULT,
        device=DEVICE
    )

# ===================== 运行推理 =====================
if __name__ == "__main__":
    # 选择一种方式运行，注释掉其他方式
    # infer_image("zidane.jpg")       # 图片推理
    # infer_video("test.mp4")     # 视频推理
    infer_camera(0)             # 摄像头推理