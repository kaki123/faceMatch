import math
from sklearn import neighbors
import os
import os.path
import pickle
import json
import codecs
import numpy as np
import json
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def saveface(X_img_path):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    # Load image file and find face locations
    # NEED TO WRITE IN SERVER
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)
    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []
    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    print(faces_encodings)


def main():
    saveface('mz.jpg')

if __name__ == "__main__":
    main()