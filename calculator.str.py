import cv2
import numpy as np
import mediapipe as mp
import math
import time
import streamlit as st

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Button class
class Button:
    def __init__(self, pos, text, size=[80, 80]):
        self.pos = pos
        self.text = text
        self.size = size

    def draw(self, img):
        x, y = self.pos
        w, h = self.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        font_scale = 2 if self.text != "⌫" else 1.5
        offset = 20 if self.text != "⌫" else 10
        cv2.putText(img, self.text, (x + offset, y + 55),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 255), 2)

    def is_clicked(self, x, y):
        bx, by = self.pos
        bw, bh = self.size
        return bx < x < bx + bw and by < y < by + bh

# Calculator layout
keys = [
    ['7', '8', '9', '/'],
    ['4', '5', '6', '*'],
    ['1', '2', '3', '-'],
    ['C', '0', '=', '+'],
    ['⌫']  # Unicode for backspace icon
]

# Create buttons
button_list = []
for i, row in enumerate(keys):
    for j, key in enumerate(row):
        x = 100 + j * 100
        y = 200 + i * 100
        button_list.append(Button((x, y), key))

def run_calculator():
    expression = ""
    last_click_time = 0
    click_cooldown = 0.8  # seconds

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    placeholder = st.empty()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.write("Failed to open camera.")
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        lm_list = []

        # Draw buttons
        for button in button_list:
            button.draw(img)

        # Show current expression
        cv2.rectangle(img, (100, 100), (700, 180), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, expression, (110, 160),
                    cv2.FONT_HERSHEY_PLAIN, 3.5, (0, 0, 0), 3)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if lm_list:
                index_x, index_y = lm_list[8]  # Index tip
                thumb_x, thumb_y = lm_list[4]  # Thumb tip

                # Draw circle
                cv2.circle(img, (index_x, index_y), 12, (0, 255, 0), cv2.FILLED)

                # Click condition: index and thumb close
                distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
                current_time = time.time()

                if distance < 40 and (current_time - last_click_time) > click_cooldown:
                    for button in button_list:
                        if button.is_clicked(index_x, index_y):
                            value = button.text
                            if value == "=":
                                try:
                                    expression = str(eval(expression))
                                except:
                                    expression = "Error"
                            elif value == "C":
                                expression = ""
                            elif value == "⌫":
                                expression = expression[:-1]
                            else:
                                expression += value
                            last_click_time = current_time
                            break

        # Show frame in Streamlit
        placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

# Streamlit UI
st.title("🖐️ Hand Gesture Controlled Calculator")

if st.button("Start Calculator"):
    run_calculator()
