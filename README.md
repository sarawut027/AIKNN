# ระบบทำนายความเสี่ยงโรคหัวใจด้วยเทคนิค KNN (K-Nearest Neighbors)

เว็บแอปพลิเคชันภาษาไทยสำหรับวิเคราะห์ข้อมูลและจำแนกความเสี่ยงโรคหัวใจจากชุดข้อมูล `Heart3.csv` พัฒนาด้วย **Streamlit** และ **scikit-learn** พร้อมรองรับการ Deploy บน **Streamlit Community Cloud (`streamlit.io`)**

## โครงสร้างไฟล์ในโปรเจกต์
- `app.py` — หน้าเว็บแอปพลิเคชันหลัก (UI ภาษาไทย, Dashboard วิเคราะห์ข้อมูล, ประสิทธิภาพโมเดล KNN, ทำนายรายบุคคล และทำนายชุดข้อมูล CSV)
- `model_utils.py` — โมดูลจัดการข้อมูล, แทนที่ค่า `0` ผิดปกติด้วย Median, สร้าง Pipeline (Scaler + KNN), ประเมินผล และทำนายผล
- `Heart3.csv` — ชุดข้อมูลผู้ป่วยโรคหัวใจ (918 แถว, 12 คอลัมน์)
- `requirements.txt` — รายการแพ็กเกจ Python สำหรับติดตั้งบนเครื่องและ `streamlit.io`
- `.streamlit/config.toml` — การตั้งค่าธีมสีของหน้าเว็บ

## วิธีรันบนเครื่อง (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## วิธี Deploy บน streamlit.io
1. แตกไฟล์ `.zip` นี้แล้วอัปโหลดไฟล์ทั้งหมดขึ้น GitHub Repository ของคุณ
2. เข้าสู่ระบบที่ [share.streamlit.io](https://share.streamlit.io/)
3. กด **New app** เลือก Repository, Branch และระบุ Main file path เป็น `app.py`
4. กด **Deploy!**
