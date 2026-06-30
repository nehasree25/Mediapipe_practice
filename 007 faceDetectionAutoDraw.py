import cv2
import mediapipe as mp
import time

pTime = 0
cap =  cv2.VideoCapture("videos/video1.mp4")
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    if not success:
        break
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2)
    pTime = cTime

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            cv2.putText(img, f'Face Detected', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows
