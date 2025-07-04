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
    client_id="ec3171edc134453bbabe60f73241ca90",
    client_secret="bcfedf92b0d14c2ca347c3aab33b0031",
    redirect_uri="http://127.0.0.1:8888/callback",
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
last_action_time = time.time()
gesture_cooldown = 1.5  # Cooldown between actions

def safe_spotify_call(action, *args):
    """Tries a Spotify API call, prevents crashes if Spotify is not responding."""
    try:
        action(*args)
    except spotipy.SpotifyException as e:
        print(f"⚠️ Spotify API Error: {e}")

while True:
    success, frame = cap.read()
    if not success:
        continue

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Hand states
    right_hand_detected = False
    left_hand_detected = False
    two_fingers_up = False

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect handedness
            handedness = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
            is_left_hand = handedness == "Left"
            is_right_hand = handedness == "Right"

            # Get finger tip & MCP positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

            # Detect if two fingers are up (Index and Middle fingers up, others down)
            two_fingers_up = (index_tip < index_mcp and middle_tip < middle_mcp and 
                              ring_tip > ring_mcp and pinky_tip > pinky_mcp)

            if is_right_hand:
                right_hand_detected = True
            elif is_left_hand:
                left_hand_detected = True

    # Get current time for cooldown check
    current_time = time.time()

    # Gesture Actions
    if (current_time - last_action_time) > gesture_cooldown:
        if right_hand_detected and not left_hand_detected:
            print("Next song ⏭️ (Right hand detected)")
            safe_spotify_call(sp.next_track)
            last_action_time = current_time

        elif left_hand_detected and not right_hand_detected:
            print("Previous song ⏮️ (Left hand detected)")
            safe_spotify_call(sp.previous_track)
            last_action_time = current_time

        elif two_fingers_up:
            if song_playing:
                print("Pausing song ⏸️ (Two fingers up)")
                safe_spotify_call(sp.pause_playback)
                song_playing = False
            else:
                print("Resuming song ▶️ (Two fingers up)")
                safe_spotify_call(sp.start_playback)
                song_playing = True
            last_action_time = current_time

    # Show webcam feed
    cv2.imshow("Spotify Hand Gesture Control", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()