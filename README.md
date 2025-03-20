# Automatic VAR System

The Automatic VAR System automatically generates the VAR (Video Assistant Referee) line in soccer videos, ensuring accurate offside analysis through a series of advanced computer vision techniques.

---

## Overview

This project integrates multiple computer vision modules to analyze soccer videos. It detects field lines, corrects perspective, detects and tracks players, and intelligently calculates and draws the offside (VAR) line. An interactive analysis mode allows for manual selection and adjustment to further improve accuracy.

---

## 🚀 Features

- **Refined Field Line Detection**  
  - Uses Canny edge detection, Hough transform, and additional stabilization techniques (see `detect_lines.py` and `detect_field_lines.txt`) to accurately detect the field boundaries.
  - Further refines results with a filtering module (`refine_field_lines.py`).

- **Adaptive Homography & Perspective Correction**  
  - Corrects the camera perspective based on detected field lines using a caching mechanism to improve performance ([`adjust_perspective.py`](#)).

- **Player Detection and Tracking**  
  - **Detection:** Uses a YOLO (Ultralytics) model to detect players in each frame ([`detect_players.py`](#)).
  - **Tracking:** Leverages DeepSORT to track players across frames, ensuring consistent identification throughout the match ([`track_players.py`](#)).

- **Intelligent VAR Line Adjustment**  
  - Computes the offside line by drawing a line parallel to the field line, anchored to the defender’s position. The system adjusts the line’s vertical position based on the most advanced player and applies stabilization techniques to avoid jitter ([`draw_var_line.py`](#)).

- **Interactive Analysis Mode**  
  - Allows users to pause the video and select key points (field line, defender, attacker) to manually adjust the analysis.
  - Provides on-screen instructions and feedback for a streamlined analysis experience.

- **Video Processing and Output**  
  - Processes entire matches or selected clips.
  - Saves processed output (video or analysis snapshot) with the VAR line and player annotations.

---


🎥 Demonstration Video
Watch the Automatic VAR System in action:
📹 Video Demonstration


---

## 📂 Project Structure

```
Automatic-VAR-System/
├── README.md
├── requirements.txt
├── models/
│   └── yolov8.pt            # YOLOv8 model for player detection
├── data/
│   └── match_video.mp4      # Input soccer match video
└── src/
    ├── main.py              # Main execution file
    ├── detect_players.py    # Detects players using YOLO
    ├── track_players.py     # Tracks players using DeepSORT
    ├── detect_lines.py      # Basic field line detection (Canny + Hough)
    ├── detect_field_lines.txt  # Enhanced field line detection with stabilization
    ├── refine_field_lines.py    # Filters and refines detected field lines
    ├── adjust_perspective.py    # Corrects video perspective based on field lines
    └── draw_var_line.py         # Draws the VAR (offside) line and annotates frames
```

---

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-user/Automatic-VAR-System.git
   cd Automatic-VAR-System
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the YOLOv8 model:**
   - Either use the provided model in the `models/` folder or download it using:
     ```bash
     wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
     ```

5. **Place your test video:**
   - Put your soccer match video in the `data/` folder and name it `match_video.mp4`.

---

## ▶ Running the Project

Execute the main script to start processing the video:
```bash
python src/main.py
```

### Interactive Controls

- **Spacebar:** Pause/Resume video playback.
- **S:** Enter analysis mode (captures the current frame for manual point selection).
- **A/D:** Switch attacking direction (left/right).
- **R:** Reset analysis mode.
- **Q:** Quit the application (if in analysis mode and a VAR line has been calculated, an image snapshot is saved as `offside_analysis.jpg`).
- **. / , (when paused):** Advance or rewind one frame.

The processed video will display detected players, field lines, and the computed VAR line. In analysis mode, manual adjustments and annotations will be visible.

---

## 🔜 Next Steps

- **Live Video Integration:** Enable real-time offside detection during live broadcasts.
- **Enhanced Stabilization:** Further improve stability and accuracy for varying match conditions.
- **Extended Analytics:** Incorporate additional statistical insights from player positions and movements.

---

## License

*Specify your project license here (e.g., MIT License).*

---

This updated README reflects the current state of the project and its modules. Feel free to further adjust it to suit additional project changes or specific deployment requirements.

Happy coding!

