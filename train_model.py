# train_model.py

import cv2
import numpy as np
from PIL import Image
import os

dataset_path = "datasets"

faces = []
labels = []
label_map = {}

current_id = 0

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        img = Image.open(image_path).convert('L')

        imageNp = np.array(img, 'uint8')

        faces.append(imageNp)
        labels.append(current_id)

    current_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, np.array(labels))

recognizer.save("trained_model.yml")

print("Training completed successfully")