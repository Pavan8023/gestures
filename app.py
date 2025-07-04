import cv2
import mediapipe as mp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import os

# Remove previous authentication cache
if os.path.exists(".cache"):
    os.remove(".cache")

# Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="user-modify-playback-state user-read-playback-state"
))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Open webcam
cap = cv2.VideoCapture(0)

# State tracking
song_playing = True
paused_by_hand = False  # <-- New flag to control pause/resume state
gesture_cooldown = 1.0
last_gesture_time = time.time()

def safe_spotify_call(action, *args):
    try:
        action(*args)
    except spotipy.SpotifyException as e:
        print(f"⚠️ Spotify API Error: {e}")

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand_closed = False
    right_hand_closed = False
    num_hands_detected = 0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            handedness = results.multi_handedness[idx].classification[0].label
            is_left_hand = handedness == "Left"
            is_right_hand = handedness == "Right"

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

            hand_closed = (index_tip > index_mcp and
                           middle_tip > middle_mcp and
                           ring_tip > ring_mcp and
                           pinky_tip > pinky_mcp)

            if is_left_hand:
                left_hand_closed = hand_closed
            elif is_right_hand:
                right_hand_closed = hand_closed

            num_hands_detected += 1

    current_time = time.time()

    # ✅ Play/Pause Logic (With anti-glitch behavior)
    if num_hands_detected == 1:
        if song_playing:
            print("🔇 Pausing song (Only 1 hand visible)")
            safe_spotify_call(sp.pause_playback)
            song_playing = False
            paused_by_hand = True
            last_gesture_time = current_time

    elif (num_hands_detected == 0 or num_hands_detected == 2):
        if not song_playing and paused_by_hand:
            print("▶️ Resuming song (Both/No hands visible after pause)")
            safe_spotify_call(sp.start_playback)
            song_playing = True
            paused_by_hand = False
            last_gesture_time = current_time

    # ✅ Previous/Next Logic
    if num_hands_detected >= 1 and (current_time - last_gesture_time) > gesture_cooldown:
        if left_hand_closed and not right_hand_closed:
            print("⏮ Previous Track (Left hand closed)")
            safe_spotify_call(sp.previous_track)
            last_gesture_time = current_time
        elif right_hand_closed and not left_hand_closed:
            print("⏭ Next Track (Right hand closed)")
            safe_spotify_call(sp.next_track)
            last_gesture_time = current_time

    # Show webcam feed
    cv2.imshow("Spotify Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
