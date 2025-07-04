import cv2
import mediapipe as mp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time

# Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="ec3171edc134453bbabe60f73241ca90",
    client_secret="bcfedf92b0d14c2ca347c3aab33b0031",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)

# State tracking
song_playing = True
prev_x = None
swipe_threshold = 0.15  # Sensitivity for left/right swipe
gesture_cooldown = 1.0  # Cooldown time (seconds)
last_gesture_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        continue

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hand_closed = False
    hand_x = None  # Track hand position

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

            # Get hand position (X-axis)
            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x

            # Detect closed fist (All fingertips near palm)
            if (index_tip > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
                middle_tip > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
                ring_tip > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
                pinky_tip > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
                hand_closed = True

    # Control Spotify Playback
    current_time = time.time()

    if hand_closed and song_playing:
        print("Pausing song...")
        sp.pause_playback()
        song_playing = False
        last_gesture_time = current_time

    elif not hand_closed and not song_playing:
        print("Resuming song...")
        sp.start_playback()
        song_playing = True
        last_gesture_time = current_time

    # Detect Swipe Left/Right
    if prev_x is not None and hand_x is not None and (current_time - last_gesture_time) > gesture_cooldown:
        if hand_x - prev_x > swipe_threshold:
            print("Next song...")
            sp.next_track()
            last_gesture_time = current_time
        elif prev_x - hand_x > swipe_threshold:
            print("Previous song...")
            sp.previous_track()
            last_gesture_time = current_time

    # Update previous hand position
    prev_x = hand_x
