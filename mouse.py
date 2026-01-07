import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
SCREEN_W, SCREEN_H = pyautogui.size()
SMOOTHING = 3
CLICK_DISTANCE = 35
DOUBLE_CLICK_TIME = 0.4 
CLICK_COOLDOWN = 0.0
pyautogui.FAILSAFE = False
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0
last_pinch_time = 0
last_click_time = 0
print("🖐️ Hand Mouse Started | Q to quit")
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
lm = []
for i, point in enumerate(hand.landmark):
x, y = int(point.x * w), int(point.y * h)
lm.append((i, x, y))
ix, iy = lm[8][1], lm[8][2]
tx, ty = lm[4][1], lm[4][2]
screen_x = np.interp(ix, (0, w), (0, SCREEN_W))
screen_y = np.interp(iy, (0, h), (0, SCREEN_H))
curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
curr_y = prev_y + (screen_y - prev_y) / SMOOTHING
pyautogui.moveTo(curr_x, curr_y)
prev_x, prev_y = curr_x, curr_y
distance = np.hypot(tx - ix, ty - iy)
now = time.time()
if distance < CLICK_DISTANCE and now - last_click_time > CLICK_COOLDOWN:
if now - last_pinch_time < DOUBLE_CLICK_TIME:
 pyautogui.doubleClick()
print("Double Click")
last_pinch_time = 0
else:
pyautogui.click()
print("Single Click")
last_pinch_time = now
last_click_time = now
cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
cv2.circle(frame, (ix, iy), 10, (255, 0, 255), cv2.FILLED)
cv2.circle(frame, (tx, ty), 10, (0, 255, 255), cv2.FILLED)
cv2.imshow("Hand Mouse", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
cap.release()
cv2.destroyAllWindows()
