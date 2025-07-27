# AI Juggling Coach - The "Guardiola" Tactical Analyst

This project is a sophisticated football juggling counter and analyst that uses computer vision and multimodal AI to provide tactical feedback in the persona of Pep Guardiola.

It goes beyond simple counting by analyzing a player's performance, identifying key moments, and using Google's Gemini 1.5 Pro to generate world-class, qualitative feedback on technique, balance, and control.

---

## Features

- **Real-Time Juggle Counting:** Utilizes a robust Trajectory Engine to accurately count juggles in real-time.
- **Stateful Engine:** Intelligently distinguishes between preparatory ground rolls, the initial "lift-off," and a continuous aerial juggling sequence.
- **Dynamic "Kicking Zone":** Uses MediaPipe pose estimation to create a dynamic hip-to-foot zone that tracks the player's body, ensuring high accuracy regardless of player position.
- **AI-Powered Tactical Analysis:** At the end of a sequence, it compiles a "dossier" with stats and key visual frames (the first touch, last successful touch, and failure moment).
- **"Pep Guardiola" Persona:** This dossier is sent to the Gemini 1.5 Pro multimodal AI with a master prompt that instructs it to provide a concise, technical, and actionable analysis in the style of the world-class manager.

## How It Works

The application is built on a hybrid architecture to ensure both real-time performance and deep analysis:

1.  **The Real-Time Trajectory Engine (On-Device):** The main Python script uses OpenCV to process the video. YOLOv8 tracks the player and ball, while MediaPipe tracks the player's pose. This engine's only job is to count juggles in real-time and identify the start and end of a juggling sequence.
2.  **The Asynchronous AI Analyst (Cloud AI):** When the on-device engine detects a sequence has ended (e.g., a ground bounce), it triggers the AI analyst. It packages the performance data and key frames into a "dossier" and sends it to the Gemini 1.5 Pro API. The AI's response is then displayed on-screen.

## Setup and Installation

**1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/dynamic-juggling-coach.git
cd dynamic-juggling-coach
```

**2. Create and Activate a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
All required libraries are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**4. Configure Your API Key**
This project requires a Google Gemini API key with billing enabled for the `gemini-1.5-pro` model.
```bash
# Copy the template to create your personal config file
cp config.json.template config.json
```
Now, open the newly created `config.json` file and paste your API key:
```json
{
  "GOOGLE_API_KEY": "YOUR_API_KEY_HERE"
}
```

**5. Add Your Video**
Place your video file in the root of the project directory and make sure its name matches the `VIDEO_PATH` variable in `coach.py` (e.g., `Juggling.mov`).

## Usage

Once the setup is complete, run the application from your terminal:
```bash
python3 coach.py
```

## Technology Stack

- **Python 3**
- **OpenCV:** For video processing and drawing.
- **Ultralytics YOLOv8:** For state-of-the-art object detection (player and ball).
- **Supervision:** For streamlining the use of detection models.
- **MediaPipe:** For robust, real-time body pose estimation.
- **Google Gemini 1.5 Pro:** For the multimodal AI analysis and feedback generation.

## Future Work

- [ ] **Timestamped Feedback:** Enhance the AI prompt to return the index of the critical frame, allowing the final video to pause or highlight the exact moment of the identified flaw.
- [ ] **Shooting Analysis:** Expand the application to a new "Shooting" mode that analyzes shooting form, power, and accuracy against a goal.

## Acknowledgments

This project was inspired by the work on the [Gemini Bball project](https://github.com/farzaa/gemini-bball) by Farzaa 
& JuggleNet (https://github.com/Logan1904/JuggleNet) by Logan1904

---
