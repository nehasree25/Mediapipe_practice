import cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.mp_draw = mp.solutions.drawing_utils
    def detectPose(self, img, draw=True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(rgb_img)
        if self.results.pose_landmarks and draw:
            cv2.putText(
                img,
                "Pose Detected",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 255),
                4
            )
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def landmarkPositions(self, img, draw=True):
        landmark_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return landmark_list
def main():
    cap = cv2.VideoCapture("videos/video3.mp4")
    detector = PoseDetector()
    pTime = time.time()
    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        
        if not success:
            break
        cv2.putText(img, f'FPS: {int(fps)}', (800,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255,0,255), 4)
        img = cv2.resize(img, (1000, 980))
        img = detector.detectPose(img)
        landmark_list = detector.landmarkPositions(img)
    
        
        cv2.imshow("Pose Detection", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()