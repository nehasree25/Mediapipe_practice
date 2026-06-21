import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
class HandDetector:
    def __init__(self,mode = False, max_hands = 2, detectionCon= 0.5, trackCon = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.hands = mp_hands.Hands(
                        static_image_mode=self.mode,
                        max_num_hands=self.max_hands,
                        min_detection_confidence=self.detectionCon,
                        min_tracking_confidence=self.trackCon
                    )
        self.mp_draw = mp.solutions.drawing_utils


    def detect_hands(self, img, draw = True):
        h, w, c = img.shape

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_img)

        if  self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                cv2.putText(
                    img,
                    "Landmark Detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        return img

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()

        if not success:
            break
        img = detector.detect_hands(img)       
        cv2.imshow("Landmark Detection", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()