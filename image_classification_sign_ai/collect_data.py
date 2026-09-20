import cv2
import mediapipe as mp
import csv
import os

# MediaPipe setup 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


# Create dataset file if not exist
file_name = "dataset.csv"
if not os.path.exists(file_name):
        with open(file_name, "w") as f:
                pass

cap = cv2.VideoCapture(0)

# Type sign name here while running 
label = input("Type sign name , (Hala , yes , No ...)")

print("Start Now Using Sign Language : ", label)

while True:
        success, frame= cap.read()
        if not success: 
                break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                data = []
                        
                for lm in hand_landmarks.landmark:
                       data.extend([lm.x , lm.y , lm.z])

                # save label at end
                data.append(label)

                with open(file_name, "a") as f:
                       writer = csv.writer(f)
                       writer.writerow(data)

        cv2.imshow("Collecting Data - Sign Language AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
               break
cap.release()
cv2.destroyAllWindows()
