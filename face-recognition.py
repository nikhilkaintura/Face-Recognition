import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

face_match = False

reference_img = cv2.imread("reference.jpg")


def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']: 
            face_match = True
            
        else:
            face_match = False
    except ValueError:
        face_match = False
 

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 ==0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()


# Here's an overview of what the code does:

# 1. It imports the necessary libraries, including `threading` for parallel processing, `cv2` for OpenCV, and `DeepFace` for face recognition.

# 2. It sets up a video capture using OpenCV to capture video from the default camera (usually the webcam).

# 3. It configures the video capture to set the frame width and height.

# 4. The program defines a `counter` to keep track of the number of frames processed and a `face_match` variable to determine if there's a match between the captured face and a reference face.

# 5. It loads a reference image from "reference.jpg" to use for face comparison.

# 6. The `check_face` function is defined to perform face verification using the DeepFace library. If a match is found, it sets `face_match` to `True`.

# 7. Inside the main loop, the program continuously captures frames from the camera. Every 30 frames, a new thread is started to check for a face match using the `check_face` function.

# 8. If a face match is found, it displays "MATCH!" on the video frame in green. If no match is found, it displays "NO MATCH!" in red.

# 9. The program uses OpenCV to display the video feed with the recognition results.

# 10. The program continues running until you press the "q" key, at which point it cleans up and closes the video windows.

# Please note that the success of face recognition may vary based on the quality of the reference image and the lighting conditions.
# Make sure to have a high-quality reference image for better results.
# Additionally, the code uses multi-threading to improve performance, but be cautious when using threading in real applications as it can introduce complexity and potential issues with shared resources.