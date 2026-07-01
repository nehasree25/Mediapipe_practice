import cv2
import mediapipe as mp
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh()
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    key = cv2.waitKey(1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    green_spec = mp_draw.DrawingSpec(
        color=(0, 255, 0),  # Green in BGR format
        thickness=1,
        circle_radius=1
    )

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=green_spec
            )
        # print("Face Detected")
    cv2.imshow("image", img)
    if key == ord('q'):
        break
    
    