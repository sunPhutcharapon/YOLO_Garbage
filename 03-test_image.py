from ultralytics import YOLO

# โหลดโมเดลที่ผ่านการฝึก (Trained Model)
model = YOLO(r"C:\Users\Thinkpad\OneDrive\เอกสาร\งานจารรุจจิ\YOLO_Garbage\runs\detect\train-3\weights\best.pt ")

# นำโมเดลไปทดสอบกับรูปภาพ
results = model.predict(
    r"C:\Users\Thinkpad\OneDrive\เอกสาร\งานจารรุจจิ\YOLO_Garbage\Garbage_Dataset_Classification\images\images\val\0071.jpg",  # เปลี่ยนเป็นรูปที่ต้องการทดสอบ
    conf=0.5,       # กำหนดค่า Confidence ขั้นต่ำที่ 50%
    save=True       # บันทึกภาพผลลัพธ์ที่ตรวจจับได้
)

# แสดงรายการวัตถุที่ตรวจพบ
r = results[0]
print(f"ตรวจพบ {len(r.boxes)} ชิ้น")
for box in r.boxes:
    print(f"  {r.names[int(box.cls)]:<10} {box.conf.item() * 100:.1f}%")

# แสดงภาพผลลัพธ์
r.show()
