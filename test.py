from ultralytics import YOLO

def export_model():

    model = YOLO("person.pt") 

    # 2. 导出为 ONNX 格式
    # opset=12 是常用的标准，imgsz=640 必须和你训练时一致
    model.export(format="onnx",half=True,simplify=True)
    
    print("✅ 导出成功！文件已保存为 best.onnx")

if __name__ == "__main__":
    export_model()