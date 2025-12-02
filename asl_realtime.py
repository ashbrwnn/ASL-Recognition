import cv2
import numpy as np
import mediapipe as mp
import tflite_runtime.interpreter as tflite

# ==== LOAD MODEL ====
interpreter = tflite.Interpreter(model_path="asl_landmarks.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== LOAD CLASS LABELS ====
label_classes = np.load("label_classes.npy")

# ==== MEDIAPIPE HANDS ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ==== CAMERA ====
cap = cv2.VideoCapture(0)

# ==== SMOOTHING VARIABLES ====
last_label = None
stable_count = 0
CONF_THRESH = 0.70        # Ignore low-confidence guesses
STABLE_THRESH = 3         # Require 3 frames of agreement

print("\n🤖 ASL Detection Started! Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    display_text = "No Hand"

    if result.multi_hand_landmarks:
        # Draw landmarks
        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # Extract 21 * (x,y,z)
        hand = result.multi_hand_landmarks[0]
        features = []
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])

        x_in = np.array(features, dtype=np.float32).reshape(1, -1)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], x_in)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        class_idx = int(np.argmax(output))
        conf = float(np.max(output))
        current_label = label_classes[class_idx]

        # ==== CONFIDENCE FILTER ====
        if conf < CONF_THRESH:
            display_text = "..."
        else:
            # ==== SMOOTHING ====
            if current_label == last_label:
                stable_count += 1
            else:
                stable_count = 0
            last_label = current_label

            if stable_count >= STABLE_THRESH:
                display_text = f"{current_label} ({conf:.2f})"
            else:
                display_text = "..."

    # ==== SHOW ON SCREEN ====
    cv2.putText(frame, display_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
