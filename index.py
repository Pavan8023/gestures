import cv2
import mediapipe as mp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import os

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
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

# State tracking
song_playing = True
prev_x = None
swipe_threshold = 0.15  # Sensitivity for left/right swipe
gesture_cooldown = 1.0  # Cooldown time (seconds) to avoid multiple triggers
last_gesture_time = time.time()

while True:  # Runs indefinitely
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
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get finger tip positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

            # Get finger MCP (Metacarpophalangeal joint) positions
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y  # Corrected definition

            # Get hand position (X-axis)
            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x

            # Detect closed fist (All fingertips near palm)
            hand_closed = (index_tip > index_mcp and
                        middle_tip > middle_mcp and
                        ring_tip > ring_mcp and
                        pinky_tip > pinky_mcp)

        # Control Spotify Playback
        current_time = time.time()  # Ensure time module is imported

    if hand_closed and song_playing:
        print("Pausing song...")
        sp.pause_playback()
        song_playing = False
        last_gesture_time = current_time  # Reset cooldown timer

    elif not hand_closed and not song_playing:
        print("Resuming song...")
        sp.start_playback()
        song_playing = True
        last_gesture_time = current_time  # Reset cooldown timer

    # Ensure prev_x is initialized
    if 'prev_x' not in locals():
        prev_x = hand_x

    # Ensure prev_x is initialized once outside the loop
    if prev_x is None:
        prev_x = hand_x  # Initialize once

    # Detect Swipe Left (Previous Song) & Swipe Right (Next Song)
    if prev_x is not None and hand_x is not None and (current_time - last_gesture_time) > gesture_cooldown:
        swipe_threshold = 0.08  # Reduce threshold for better detection

        movement = hand_x - prev_x  # Calculate movement
        print(f"Prev X: {prev_x}, Current X: {hand_x}, Movement: {movement}")  # Debugging

        if movement > swipe_threshold:  # Right Swipe → Next Song
            print("Next song...")
            sp.next_track()
            last_gesture_time = current_time  # Reset cooldown timer
            prev_x = hand_x  # Update prev_x after valid swipe

        elif movement < -swipe_threshold:  # Left Swipe → Previous Song
            print("Previous song...")
            sp.previous_track()
            last_gesture_time = current_time  # Reset cooldown timer
            prev_x = hand_x  # Update prev_x after valid swipe


    # Update previous hand position
    prev_x = hand_x


    # Show webcam feed
    cv2.imshow("Spotify Hand Gesture Control", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
