from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-pose.pt")  # Modelo de pose

cap = cv2.VideoCapture(0)  # Cambia por el nombre real del archivo
cv2.namedWindow("Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose", 640, 360)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    for kps in results[0].keypoints.xy:  # Lista de personas (shape: [N, 17, 2])
        if kps.shape[0] < 7:
            continue  # Saltar si hay pocos keypoints

        nose = kps[0]         # Nariz
        left_shoulder = kps[5]
        right_shoulder = kps[6]

        # Inferir orientación simple
        if nose[0] < left_shoulder[0]:
            orientation = "derecha"
        elif nose[0] > right_shoulder[0]:
            orientation = "izquierda"
        else:
            orientation = "frente"

        print("Orientación:", orientation)

    cv2.imshow("Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
