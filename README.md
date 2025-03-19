# Automatic VAR System 🔴🟢⚽

This project automatically generates the **VAR line** in soccer videos, ensuring precision in offside analysis.

---

## 🚀 Features
✅ **Refined Field Line Detection** → Filters noise and detects only relevant lines.  
✅ **Adaptive Homography** → Dynamically adjusts perspective based on detected field lines.  
✅ **Player Tracking with DeepSORT** → Keeps players identified throughout the match.  
✅ **Intelligent VAR Line Adjustment** → Compensates for field inclinations for greater accuracy.  
✅ **Video Processing for Matches** → Automatically saves the processed video with the VAR line drawn.  
✅ **Support for Full Matches or Clips** → Can be used for entire matches or specific highlights.  

---

## 💂️📚 **Project Structure**
```
VAR-Automatico/
│\── README.md
│\── requirements.txt
│\── src/
│   ├── main.py
│   ├── detect_players.py
│   ├── track_players.py
│   ├── detect_lines.py
│   ├── adjust_perspective.py
│   ├── refine_field_lines.py
│   ├── draw_var_line.py
│\── models/
│   └── yolov8.pt
│\── data/
│   └── match_video.mp4
```

---

## 📌 **Prerequisites**
- **Python 3.7+**
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort)

---

## 🔧 **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-user/VAR-Automatico.git
   cd VAR-Automatico
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 model and save it in the `models/` folder:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
   ```

5. Place a test video in the `data/` folder and name it `match_video.mp4`.

---

## ▶ **Run the Project**
```bash
python src/main.py
```
- The processed video will be saved as `output_video.mp4`.
- Press **q** to stop execution at any time.

---

## 🔜 **Next Steps**
✅ **Integration with Live Video Feeds** → The next goal is to enable real-time offside detection during match broadcasts.



