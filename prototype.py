import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

# ================= CONFIG =================
SCREEN_W, SCREEN_H = pyautogui.size()
SMOOTHING = 0.25
CLICK_DISTANCE = 35
CLICK_DELAY = 0.4
# =========================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
last_click_time = 0

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

print("🖐️ Hand Mouse Started | Press Q to quit")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        index_tip = (int(lm[8].x * w), int(lm[8].y * h))
        thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))

        screen_x = np.interp(index_tip[0], (0, w), (0, SCREEN_W))
        screen_y = np.interp(index_tip[1], (0, h), (0, SCREEN_H))

        curr_x = prev_x + (screen_x - prev_x) * SMOOTHING
        curr_y = prev_y + (screen_y - prev_y) * SMOOTHING

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        dist = distance(index_tip, thumb_tip)
        current_time = time.time()

        if dist < CLICK_DISTANCE and current_time - last_click_time > CLICK_DELAY:
            pyautogui.click()
            last_click_time = current_time

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()