import cv2
import time
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv
import mediapipe as mp
import numpy as np
import os
import sys
import google.generativeai as genai
from PIL import Image
import json
import re

# --- Engineering Configuration ---
def load_config(filepath='config.json'):
    """
    Loads the API key directly from a JSON config file for robustness.
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
            return config.get("GOOGLE_API_KEY")
    except Exception as e:
        print(f"CRITICAL ERROR loading config.json: {e}")
        return None

VIDEO_PATH = 'Juggling.mov'
MODEL_NAME = "yolov8m.pt"

# --- Engine Parameters ---
TRACE_LENGTH = 5
JUGGLE_COOLDOWN = 0.4
GROUND_ZONE_PERCENT = 0.15
CORRECTIVE_STEP_THRESHOLD = 40.0

# --- UI/UX Parameters ---
FEEDBACK_DISPLAY_DURATION = 8.0

# --- Initialize Services ---
API_KEY = load_config()
GEMINI_MODEL = None
if not API_KEY:
    print("CRITICAL ERROR: GOOGLE_API_KEY not found in config.json.")
    sys.exit(1)
try:
    genai.configure(api_key=API_KEY)
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')
    print("Performing AI model pre-flight check...")
    GEMINI_MODEL.generate_content("hello", generation_config=genai.types.GenerationConfig(max_output_tokens=10))
    print("Gemini API (Multimodal) configured and verified successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to configure or verify Gemini API: {e}")
    sys.exit(1)

# --- Helper Functions ---
def sanitize_text(text: str) -> str:
    """
    Cleans the AI's output for better display.
    """
    return text.replace('*', '').replace('`', '')

def get_feedback_from_ai_analyst(juggle_count, duration, instability_event, visual_evidence):
    """
    Generates Guardiola-style feedback based on a detailed dossier.
    """
    prompt = f"""
    You are Pep Guardiola, a world-class football tactician. You are analyzing a player's single most unstable touch from a juggling sequence of {juggle_count} total touches.

    **EVENT DATA:**
    {instability_event}

    **VISUAL EVIDENCE:**
    Analyze this specific three-frame sequence provided:
    1.  The setup (the touch BEFORE the error).
    2.  The moment of instability itself.
    3.  The recovery (the touch AFTER the error).

    **YOUR TASK (3 lines, maximum precision, no conversational filler):**
    1.  **Flaw:** Based on the visual evidence, what is the single, precise technical error on the unstable touch? (e.g., "Contact was made too high on the instep, causing excess spin.")
    2.  **Consequence:** Briefly describe the result of this flaw. (e.g., "This forced a panicked, off-balance step to the left.")
    3.  **Instruction:** Give a single, clear command to fix this specific issue. (e.g., "Focus on a flat, cushioned contact with the center of your laces.")
    """
    prepared_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in visual_evidence]
    try:
        response = GEMINI_MODEL.generate_content([prompt] + prepared_images)
        return sanitize_text(response.text.strip())
    except Exception as e:
        print(f"CRITICAL ERROR during Gemini analysis: {e}")
        return "Analysis failed. See terminal for details."

def wrap_text(text, font, scale, thickness, max_width):
    """
    Wraps text to fit within a specified width, respecting newline characters.
    """
    lines = []
    raw_lines = text.split('\n')
    for raw_line in raw_lines:
        words = raw_line.split()
        line = ''
        while words:
            if not line:
                line = words.pop(0)
            else:
                new_word = words.pop(0)
                if cv2.getTextSize(line + ' ' + new_word, font, scale, thickness)[0][0] <= max_width:
                    line += ' ' + new_word
                else:
                    lines.append(line)
                    line = new_word
        if line:
            lines.append(line)
    return lines

def find_critical_moment(touch_events, all_frames):
    """
    Analyzes a sequence to find the single most unstable touch.
    Returns the analysis text and the three key frames for the AI.
    """
    if len(touch_events) < 3:
        return None, None # Not enough data to find a critical moment

    max_movement = 0
    worst_touch_index = -1

    for i in range(1, len(touch_events)):
        movement_dist = abs(touch_events[i]['player_x'] - touch_events[i-1]['player_x'])
        if movement_dist > max_movement:
            max_movement = movement_dist
            worst_touch_index = i

    if max_movement > CORRECTIVE_STEP_THRESHOLD:
        event_description = f"A moment of instability was detected at Juggle #{touch_events[worst_touch_index]['count']}. The player had to take a large corrective step of {max_movement:.0f} pixels."
        
        frame_index_before = touch_events[worst_touch_index - 1]['frame_number']
        frame_index_during = touch_events[worst_touch_index]['frame_number']
        frame_index_after = touch_events[worst_touch_index + 1]['frame_number'] if worst_touch_index + 1 < len(touch_events) else frame_index_during

        visual_evidence = [
            all_frames[frame_index_before],
            all_frames[frame_index_during],
            all_frames[frame_index_after]
        ]
        return event_description, visual_evidence
    
    return None, None # No moment exceeded the threshold

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return
    width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    if fps == 0: fps = 30
    GROUND_Y_THRESHOLD = height * (1 - GROUND_ZONE_PERCENT)
    JUGGLE_COOLDOWN_FRAMES = int(JUGGLE_COOLDOWN * fps)
    FEEDBACK_DISPLAY_FRAMES = int(FEEDBACK_DISPLAY_DURATION * fps)

    model = YOLO(MODEL_NAME)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    ball_trace = defaultdict(lambda: deque(maxlen=TRACE_LENGTH))
    juggle_count, last_event_frame = 0, 0
    is_airborne = False
    touch_events = []
    sequence_start_frame = 0
    feedback_text, feedback_start_frame = None, 0
    
    all_video_frames = []
    print("Reading video frames into memory...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        all_video_frames.append(frame)
    cap.release()
    print(f"Video loaded. Starting analysis of {len(all_video_frames)} frames...")

    # This list will store the final, annotated frames for saving.
    processed_frames_for_video = []

    for current_frame_number, frame in enumerate(all_video_frames):
        annotated_frame = frame.copy()
        
        kicking_zone, current_player_x = None, None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_estimator.process(rgb_frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            visible_hips_y = [landmarks[i.value].y for i in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP] if landmarks[i.value].visibility > 0.5]
            foot_indices = [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            visible_feet_y = [landmarks[i.value].y for i in foot_indices if landmarks[i.value].visibility > 0.5]
            if visible_hips_y and visible_feet_y:
                top_y, bottom_y = min(visible_hips_y) * height, max(visible_feet_y) * height
                kicking_zone = (top_y, bottom_y + 30)
        
        results = model.track(frame, persist=True, classes=[0, 32], tracker="bytetrack.yaml", verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        player_detections = detections[detections.class_id == 0]
        if len(player_detections) > 0:
            current_player_x = (player_detections.xyxy[0][0] + player_detections.xyxy[0][2]) / 2
        
        ball_detections = detections[detections.class_id == 32]
        if len(ball_detections) > 0:
            ball_box, tracker_id = ball_detections.xyxy[0], ball_detections.tracker_id[0] if ball_detections.tracker_id is not None else 1
            ball_center_y = (ball_box[1] + ball_box[3]) / 2
            ball_trace[tracker_id].append(ball_center_y)

            if len(ball_trace[tracker_id]) == TRACE_LENGTH:
                y_coords = list(ball_trace[tracker_id])
                is_apex = y_coords[-2] > y_coords[-3] and y_coords[-2] > y_coords[-1]
                if is_apex and (current_frame_number - last_event_frame > JUGGLE_COOLDOWN_FRAMES):
                    apex_y = y_coords[-2]
                    is_in_kicking_zone, is_in_ground_zone = kicking_zone and (kicking_zone[0] < apex_y < kicking_zone[1]), apex_y > GROUND_Y_THRESHOLD
                    if is_in_ground_zone:
                        if is_airborne:
                           print(f"Ground bounce detected. Analyzing sequence...")
                           instability_event, visual_evidence = find_critical_moment(touch_events, all_video_frames)
                           if instability_event:
                               duration_seconds = (current_frame_number - sequence_start_frame) / fps
                               feedback_text = get_feedback_from_ai_analyst(juggle_count, duration_seconds, instability_event, visual_evidence)
                               feedback_start_frame = current_frame_number
                           is_airborne, juggle_count, touch_events = False, 0, []
                    elif is_in_kicking_zone:
                        if not is_airborne:
                            is_airborne, juggle_count = True, 1
                            sequence_start_frame, last_event_frame = current_frame_number, current_frame_number
                            touch_events = []
                            if current_player_x is not None: touch_events.append({'count': 1, 'frame_number': current_frame_number, 'player_x': current_player_x})
                            print(f"LIFT-OFF DETECTED! JUGGLE #1 at frame {current_frame_number}")
                        else:
                            juggle_count += 1
                            last_event_frame = current_frame_number
                            if current_player_x is not None: touch_events.append({'count': juggle_count, 'frame_number': current_frame_number, 'player_x': current_player_x})
                            print(f"JUGGLE #{juggle_count} DETECTED at frame {current_frame_number}")

        # Drawing Phase
        labels = [f"ID:{tid}" for tid in detections.tracker_id] if detections.tracker_id is not None else []
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        if labels: annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.putText(annotated_frame, f"Juggles: {juggle_count}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Juggles: {juggle_count}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
        if feedback_text and (current_frame_number - feedback_start_frame < FEEDBACK_DISPLAY_FRAMES):
            max_width = int(width * 0.8)
            wrapped_lines = wrap_text(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3, max_width)
            total_height = len(wrapped_lines) * 45
            start_y = height - 60 - total_height
            for i, line in enumerate(wrapped_lines):
                text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                feedback_x = (width - text_size[0]) // 2
                feedback_y = start_y + (i * 45)
                cv2.putText(annotated_frame, line, (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 8, cv2.LINE_AA)
                cv2.putText(annotated_frame, line, (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        
        # --- FIX: Append the final, fully annotated frame to the list for saving ---
        processed_frames_for_video.append(annotated_frame)
        
        cv2.imshow("Guardiola Intelligent Analyst", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cv2.destroyAllWindows()
    
    # --- FIX: Save the correct list of annotated frames ---
    print("Saving final annotated video...")
    output_path = 'juggling_coach_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in processed_frames_for_video:
        out.write(frame)
    out.release()
    print(f"Processing complete. Final video saved to {output_path}")
    
if __name__ == "__main__":
    main()
