import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("cardumen_yolo/video/walking.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 160))

visioneye = solutions.VisionEye(
    show=False,  # ahora no muestra automáticamente
    model="yolo11n.pt",
    classes=[0, 2],
    vision_point=(50, 50),
)

# Procesar video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = visioneye(im0)

    print(results)

    # Redimensionar para mostrar en ventana más chica
    resized = cv2.resize(results.plot_im, (640, 480))
    cv2.imshow("VisionEye Small", resized)

    video_writer.write(results.plot_im)

    # Cierra con 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    results = visioneye(im0)

    print(results)  # access the output

    video_writer.write(results.plot_im)  # write the video file

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows