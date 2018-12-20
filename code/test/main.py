import json
import os
import io

# Incorporated logging features. est logging to GCP here
import logging
print('Hello, stdout!')
logging.warn('Hello, logging handler!')
logging.warn(os.getcwd())
logging.warn(os.listdir(os.getcwd()))

# Downloaded dlib on GCP shell 
import imp
# import dlib by tracing to its downloaded location on GCP shell
dlib = imp.load_source('dlib', 'lib/dlib.so')
dlib.MyClass()
import face_recognition
import numpy as np
import cv2 as cv
import logging

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import url_for
from PIL import Image
from google.cloud import storage
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from google.appengine.api import app_identity

# Getting the credentials to make change to the project/app
credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials)

# locate project name
project = 'projects/{}'.format('face-match-219722')

# choose machine learning model. This should be directed by the frontend
# category information if backend and frontend and connected
model_name = os.getenv('MODEL_NAME', 'celeb_model')

# locate the storage bucket that stores instance images
storage_client = storage.Client()
bucket = storage_client.get_bucket("face-match-219722-images")

app = Flask(__name__)

# get iamge from Cloud Storage and preprocess it using the face_recognition module
def read_file():
  gcs_file = bucket.get_blob("josh.jpg")
  img_string = gcs_file.download_as_string()
  img = Image.open(io.BytesIO(img_string)).convert('BGR') 

  # convert image to numpy arrays using openCV
  open_cv_image = np.array(img)

  # locate face locations with the face_recognition module
  X_face_locations = face_recognition.face_locations(open_cv_image)
  if len(X_face_locations) == 0:
      return []

  # get face encodings from the face_recognition module
  faces_encodings = face_recognition.face_encodings(open_cv_image, known_face_locations=X_face_locations)
  print()
  encoding_list = np.asarray(faces_encodings).reshape(1,-1).tolist()
  json_dict = {img_crop}
  json_dict["instances"] = encoding_list
  return encoding_list

# function to get match prediction on face features
def get_prediction(features):
  # get data - preprocessed image from Google Cloud Storage
  input_data = {"instances": [
        read_file()
    ]}
  input_data = {"instances": read_file()}
  parent = '%s/models/%s' % (project, model_name)
  prediction = api.projects().predict(body=input_data, name=parent).execute()
  print(prediction["predictions"][0])
  
  return prediction["predictions"][0]

# display the login page
@app.route('/')
def index():
  return render_template('index.html')

# display the homepage
@app.route('/form')
def input_form():
  return render_template('form.html')

@app.route('/api/predict', methods=['POST'])
def predict():
  # parse frontend info including person and category
  def person2file(val):
    people = {'1': 'huize.json', '2': 'kaitlyn.json'}
    return people[val]

  def categ2model(val):
    models = {'prof': 'prof_model', 'student': 'student_model', 'celeb': 'celeb_model'}
    return models[val]

  data = json.loads(request.data.decode())
  mandatory_items = ['person', 'category']
  for item in mandatory_items:
    if item not in data.keys():
      return jsonify({'result': 'Set all items.'})

  features = {}
  features['person'] = person2file(data['person'])
  features['category'] = categ2model(data['category'])

  prediction = get_prediction(features)
  return jsonify({'result': prediction})