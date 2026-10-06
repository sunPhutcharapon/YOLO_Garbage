# YOLO_Garbage

โปรเจกต์สำหรับฝึกโมเดล **YOLO26** เพื่อตรวจจับและจำแนกขยะจากภาพ วิดีโอ และกล้องแบบ Real-time โดยใช้ **Ultralytics YOLO** และ **Label Studio** สำหรับทำ Bounding Box

---
## ผู้จัดทำ

นาย อติโรจน์ กุหลั่น 67543210049-2<br>นาย พัชรพล สืบทายาท 67543210041-9

## 📂 โครงสร้างโปรเจกต์

```text
YOLO_Garbage/
├── README.md
├── 01-export_dataset.py
├── 02-train.py
├── 03-test_image.py
├── 04-test_video.py
├── 05-test-camera.py
├── data.yaml
├── labeling_config.xml
├── requirements.txt
├── yolo26n.pt
├── train_Garbage.mp4
├── Frame/
│   └── images/                         # เฟรมภาพสำหรับทำ Label
├── Garbage_Dataset_Classification/
│   ├── classes.txt
│   ├── data.yaml
│   └── images/
│       ├── images/train และ images/val # ภาพ Dataset
│       └── labels/train และ labels/val # YOLO annotations
└── runs/detect/                        # ผลลัพธ์จากการฝึกและทดสอบ
```

สคริปต์ `01-export_dataset.py` อ่านไฟล์ JSON จาก Label Studio ในโฟลเดอร์โปรเจกต์ แล้วแปลงและแบ่งภาพเป็น train/val ส่วน `02-train.py` ใช้ไฟล์ `data.yaml` ที่โฟลเดอร์หลัก

---

## 🏷️ Classes

ไฟล์ `data.yaml` ระบุคลาสที่โมเดลรู้จำ 6 ประเภท:

```text
cardboard, glass, metal, paper, plastic, trash
```

ตรวจสอบให้ Label Studio, `data.yaml` และ labels ใน Dataset ใช้ชื่อคลาสให้ตรงกัน

---

## ⚙️ ติดตั้ง

ต้องมี Python ติดตั้งอยู่ จากโฟลเดอร์โปรเจกต์ให้สร้างและเปิดใช้งาน Virtual Environment:

### Windows (Command Prompt)

```bat
py -3 -m venv env
.\env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎞️ เตรียมภาพจากวิดีโอ

ติดตั้ง FFmpeg หากยังไม่มี แล้วแยกเฟรมจากวิดีโอ ตัวอย่างคำสั่ง (แก้ชื่อไฟล์และจำนวนเฟรมต่อวินาทีตามต้องการ):

```bash
ffmpeg -i train_Garbage.mp4 -vf fps=2 Frame/images/%04d.jpg
```

`fps=2` หมายถึงดึงภาพประมาณ 2 เฟรมต่อวินาที ไฟล์ภาพจะถูกบันทึกใน `Frame/images/`

---

## 🏷️ ทำ Label ด้วย Label Studio


---

## 🔄 แปลง Label Studio JSON เป็น YOLO Dataset

ก่อนรัน ตรวจสอบค่าต้นสคริปต์ใน `01-export_dataset.py` ให้ตรงกับตำแหน่งโฟลเดอร์ของเครื่อง:

```python
IMAGES_DIR = Path(r"...\YOLO_Garbage\Frame\images")
OUTPUT_DIR = Path(r"...\YOLO_Garbage\Garbage_Dataset_Classification\images")
TRAIN_SPLIT = 0.8
SEED = 42
```

สคริปต์จะสร้างโฟลเดอร์ `images/train`, `images/val`, `labels/train`, `labels/val` และไฟล์ `data.yaml` กับ `classes.txt` ในตำแหน่ง Output โดยแบ่งข้อมูลตามสัดส่วน train 80% และ validation 20% (หากมีภาพเพียงพอ)

```bash
python 01-export_dataset.py
```

> สคริปต์สร้าง `data.yaml` ไว้ในโฟลเดอร์ Output ส่วน `02-train.py` อ้างถึง `data.yaml` ในโฟลเดอร์หลัก ให้ตรวจว่าไฟล์หลักชี้ไปยัง Dataset ที่ถูกต้องก่อนฝึก

---

## 🧠 ฝึก YOLO26

การตั้งค่าปัจจุบันอยู่ใน `02-train.py`:

```text
โมเดลเริ่มต้น    yolo26n.pt
Epochs           50
Image size       416
Optimizer        MuSGD
Device           CPU (เปลี่ยนเป็น 0 หากใช้ GPU ที่พร้อมใช้งาน)
Patience         20
```


```bash
python 02-train.py
```

ตำแหน่งผลลัพธ์ `runs/detect/...` ขึ้นกับชื่อโฟลเดอร์ที่ Ultralytics กำหนดในแต่ละครั้ง โดยไฟล์น้ำหนักที่ใช้ทดสอบมักอยู่ที่ `weights/best.pt`

---

## 🧪 ทดสอบโมเดล

### 1. ทดสอบภาพ

แก้ Path โมเดลและภาพใน `03-test_image.py` แล้วรัน:

```bash
python 03-test_image.py
```

ค่า Confidence ปัจจุบันคือ `0.5` และบันทึกผลด้วย `save=True`

![รูปผลลัพธ์การตรวจจับขยะ](<test_images/test1.png>)

### 2. ทดสอบวิดีโอ

กำหนด Path วิดีโอทดสอบและโมเดลใน `04-test_video.py` แล้วรัน:

```bash
python 04-test_video.py
```

ผลลัพธ์ถูกบันทึกโดย Ultralytics ภายใต้ `runs/detect/predict...`

![รูปผลลัพธ์การตรวจจับขยะจากวีดีโอ](<test_images/test2.png>)

### 3. ทดสอบกล้อง

ตรวจสอบ Path โมเดลใน `05-test-camera.py` แล้วรัน:

```bash
python 05-test-camera.py
```

โปรแกรมใช้กล้อง Webcam และกด `q` เพื่อออก

---

![รูปผลลัพธ์การตรวจจับจากกล้อง](<test_images/test3.png>)

## ⚠️ หมายเหตุ

- ตรวจสอบ Path ในไฟล์ Python ให้ตรงกับโฟลเดอร์ของเครื่องก่อนรัน
- ต้องมีไฟล์ JSON ที่ Export จาก Label Studio อยู่ในโฟลเดอร์เดียวกับ `01-export_dataset.py`
- ตรวจสอบให้ชื่อภาพใน JSON ตรงกับไฟล์ใน `Frame/images/`
- ตรวจสอบ `data.yaml` และชื่อคลาสก่อนเริ่ม Train
- อย่าเปลี่ยนไฟล์ Dataset หรือ Path ที่กำลังใช้ระหว่างการฝึก
