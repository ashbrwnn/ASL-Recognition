import cv2
import mediapipe as mp
import csv
import time

# ==== SETTINGS ====
import string
LABELS = list(string.ascii_uppercase)   # ['A','B','C', ... 'Z']
SAMPLES_PER_LETTER = 300                 # how many frames to collect per letter
CSV_FILENAME = "asl_datasetfinal.csv"       # output file

# ==== MEDIAPIPE ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# ==== OPEN CAMERA ====
cap = cv2.VideoCapture(0)

# ==== OPEN CSV FILE ====
csv_file = open(CSV_FILENAME, "a", newline="")
csv_writer = csv.writer(csv_file)

print("\n📌 ASL Dataset Collection Started")
print("🖐 Show one sign at a time when asked.")
print("🎯 Collecting letters:", LABELS)
print("➡ Press 'Q' at any time to quit.\n")

# ==== LOOP THROUGH EACH LETTER ====
for label in LABELS:
    print(f"\n➡ Get ready to record letter '{label}'")
    time.sleep(2)
    print(f"✋ Hold up sign '{label}'... START!")
    count = 0

    while count < SAMPLES_PER_LETTER:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Build row: label + 21 * (x,y,z)
                row = [label]
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                csv_writer.writerow(row)
                count += 1
                print(f"Saved {count}/{SAMPLES_PER_LETTER} for '{label}'")

                # Draw dots on screen
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Letter: {label}  ({count}/{SAMPLES_PER_LETTER})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("ASL Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⛔ Quit early.")
            cap.release()
            csv_file.close()
            cv2.destroyAllWindows()
            exit()

print("\n🎉 DONE COLLECTING ALL LETTERS!")
print(f"📁 Saved in: {CSV_FILENAME}")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
