# facial_data.py

import cv2
import os

face_classifier = cv2.CascadeClassifier("data.xml")

name = input("Enter the name of person : ")

path = "datasets/" + name

if not os.path.exists(path):
    os.makedirs(path)

camera = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        count += 1

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (200, 200))

        file_name = path + "/" + str(count) + ".jpg"

        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(
            frame,
            f"Captured : {count}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Face Dataset Creator", frame)

    if cv2.waitKey(1) == 13 or count == 100:
        break

camera.release()
cv2.destroyAllWindows()

print("Dataset collection completed successfully")