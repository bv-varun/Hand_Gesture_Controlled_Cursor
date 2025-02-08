import cv2
import mediapipe as mp
import win32api
import pyautogui
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

prev_click = 0
smoothing_factor = 5
cursor_positions = np.zeros((smoothing_factor, 2), dtype=np.float32)

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                         mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                indexfingertip_x, indexfingertip_y = 0, 0
                thumbfingertip_x, thumbfingertip_y = 0, 0
                middlefingertip_x, middlefingertip_y = 0, 0
                pinkyfingertip_x, pinkyfingertip_y = 0, 0
                ringfingertip_x, ringfingertip_y = 0, 0

                for point in mp_hands.HandLandmark:
                    landmark = hand_landmarks.landmark[point]
                    normalized_x, normalized_y = int(landmark.x * image_width), int(landmark.y * image_height)

                    if point == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        indexfingertip_x, indexfingertip_y = normalized_x, normalized_y

                    elif point == mp_hands.HandLandmark.THUMB_TIP:
                        thumbfingertip_x, thumbfingertip_y = normalized_x, normalized_y

                    elif point == mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
                        middlefingertip_x, middlefingertip_y = normalized_x, normalized_y
                    elif point == mp_hands.HandLandmark.PINKY_TIP:
                        pinkyfingertip_x, pinkyfingertip_y = normalized_x, normalized_y
                    elif point == mp_hands.HandLandmark.RING_FINGER_TIP:
                        ringfingertip_x, ringfingertip_y = normalized_x, normalized_y

                try:
                    cursor_positions = np.roll(cursor_positions, -1, axis=0)
                    cursor_positions[-1] = (indexfingertip_x * 4, indexfingertip_y * 3)
                    smooth_cursor_position = np.mean(cursor_positions, axis=0).astype(int)

                    win32api.SetCursorPos(tuple(smooth_cursor_position))

                    distance_index_thumb = calculate_distance(indexfingertip_x, indexfingertip_y, thumbfingertip_x, thumbfingertip_y)
                    distance_middle_thumb = calculate_distance(middlefingertip_x, middlefingertip_y, thumbfingertip_x, thumbfingertip_y)
                    distance_index_middle = calculate_distance(indexfingertip_x, indexfingertip_y, middlefingertip_x, middlefingertip_y)
                    distance_ring_pinky = calculate_distance(ringfingertip_x, ringfingertip_y, pinkyfingertip_x, pinkyfingertip_y)
                    if distance_index_thumb < 30:
                        click = 1
                        if click != prev_click:
                            prev_click = click
                            print("Left Click Performed")
                            pyautogui.click()
                    elif distance_middle_thumb < 30:
                        click = 2
                        if click != prev_click:
                            prev_click = click
                            print("Right Click Performed")
                            pyautogui.rightClick()
                    elif distance_index_middle < 10:
                        click = 3
                        while(distance_index_middle < 10):  
                         if click != prev_click:
                             print("Scroll DOWN")
                             pyautogui.vscroll(clicks=20)
                             max_click=max_click+1
                             if(max_click>5):
                                 break
                    elif distance_ring_pinky < 30:
                        click = 4
                        while(distance_ring_pinky < 30):  
                         if click != prev_click:
                             print("Scroll UP")
                             pyautogui.vscroll(clicks=-20)
                             max_click=max_click+1
                             if(max_click>5):
                                 break              
                    else:
                        click = 0
                        if click != prev_click:
                            prev_click = click
                            print("No Click")
                except Exception as e:
                    print(f"An error occurred: {e}")

        cv2.imshow('HTC [HAND TRACKING CURSOR]', image)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

video.release()
cv2.destroyAllWindows()