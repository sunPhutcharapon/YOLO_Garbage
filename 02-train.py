from ultralytics import YOLO

if __name__ == '__main__':

    # โหลดโมเดล Detection แบบ Pre-trained (เริ่มจากโมเดลที่เรียนรู้มาแล้วจะแม่นกว่าเริ่มจากศูนย์)
    model = YOLO('yolo26n.pt')

    results = model.train(
        # ไฟล์ data.yaml ที่ 01-export_dataset.py สร้างให้ในโฟลเดอร์ dataset
        data='C:\\Users\\Thinkpad\\OneDrive\\เอกสาร\\งานจารรุจจิ\\YOLO_Garbage\\data.yaml',
        epochs=20,
        imgsz=416,
        optimizer="MuSGD",
        device="cpu",        # ถ้ามี GPU เปลี่ยนเป็น 0
        patience=20,         # หยุดเองถ้า val ไม่ดีขึ้น 20 epochs ติดกัน

        # --- Data Augmentation ---
        degrees=15.0,        # สุ่มหมุนภาพ -15 ถึง +15 องศา
        shear=5.0,           # บิดภาพแบบเฉียง
        perspective=0.001,   # จำลองมุมมองที่ต่างออกไป

        fliplr=0.5,          # พลิกซ้าย-ขวา 50%
        flipud=0.2,          # ขยะวางกลับหัวได้ จึงเปิดพลิกบน-ล่างเล็กน้อย

        hsv_h=0.015,         # ปรับสี/ความสว่าง รับมือแสงต่างกัน
        hsv_s=0.5,
        hsv_v=0.4,

        mosaic=1.0,          # ต่อภาพ 4 ภาพเข้าด้วยกัน -> ได้ภาพที่มีขยะหลายชิ้น (สำคัญมาก เพราะรูปต้นฉบับมีชิ้นเดียว)
        mixup=0.1,           # ซ้อนภาพ 2 ภาพ ช่วยลด Overfitting
        close_mosaic=10      # ปิด Mosaic ใน 10 epochs สุดท้าย
    )
