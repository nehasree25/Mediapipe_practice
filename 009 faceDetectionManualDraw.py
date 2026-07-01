import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=self.min_detection_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def fancy_draw(self, img,l = 20, draw=True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_img)

        if results.detections:
            for id, detection in enumerate(results.detections):
                if draw:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(img, bbox, (255, 0, 255), 1)
                    cv2.putText(
                        img,
                        f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 0, 255),
                        3
                    )
                    xmin,ymin,w,h = bbox
                    x1, y1 = xmin+w, ymin+h
                    cv2.line(img, (xmin, ymin), (xmin+l, ymin), (255, 255, 255), 3)
                    cv2.line(img, (xmin, ymin), (xmin, ymin+l), (255, 255, 255), 3)
                    cv2.line(img, (xmin, y1), (xmin, y1-l), (255, 255, 255), 3)
                    cv2.line(img, (xmin, y1), (xmin+l, y1), (255, 255, 255), 3)
                    cv2.line(img, (x1, ymin), (x1-l, ymin), (255, 255, 255), 3)
                    cv2.line(img, (x1, ymin), (x1, ymin+l), (255, 255, 255), 3)
                    cv2.line(img, (x1, y1), (x1, y1-l), (255, 255, 255), 3)
                    cv2.line(img, (x1, y1), (x1-l, y1), (255, 255, 255), 3)

        return img
def main():
    cap = cv2.VideoCapture("videos/video1.mp4")
    detector = FaceDetector()
    pTime = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

        img = detector.fancy_draw(img)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()