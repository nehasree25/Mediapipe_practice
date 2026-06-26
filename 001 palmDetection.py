import cv2
import mediapipe as mp
import time

pTime = time.time()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()
    if not success:
        break

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,0,255), 2)
    
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    h, w, c = img.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            cv2.rectangle(
                img,
                (xmin - 20, ymin - 20),
                (xmax + 20, ymax + 20),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                "Palm Detected",
                (xmin, ymin ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Palm Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()