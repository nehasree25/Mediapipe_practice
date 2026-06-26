import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture("videos/video3.mp4")

while True:
    success, img = cap.read()

    if not success:
        break
    
    img = cv2.resize(img, (1000, 980))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_img)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(
            img,
            "Pose Detected",
            (100, 200 ),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            4
        )
    
    cv2.imshow("Pose Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()