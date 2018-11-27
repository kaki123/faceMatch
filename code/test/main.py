import json
import os
import math
import os.path
import pickle
import json
import codecs
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import url_for
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from google.appengine.api import app_identity

credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials)
project = 'projects/{}'.format('face-match-219722')
model_name = os.getenv('MODEL_NAME', 'prof_model')

app = Flask(__name__)

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
    return faces_encodings

def remove_linebreak(content):
    n = 0
    while n <= len(content):
        if content[n] == '\n':
            content = content[:n] + content[n+1:]
        n += 1
    return content

def getData(uri):
    head, data = uri.split(',')
    decoded =data.decode('base64','strict')
    # decoded base64.b64decode(data)
    return decoded


def get_prediction(features):
  input_data = {"instances": [
          [-0.06597716, 0.08036334, 0.04592936, -0.02924938, -0.11216526, -0.00579459, -0.04527251, -0.10679842, 0.12171349, -0.15069072, 0.17678921, -0.03977152, -0.20733738, -0.07680656, -0.05653973, 0.2357996, -0.20273849, -0.15965953, -0.07236928, -0.01950016, 0.03847497, 0.00682649, 0.03156324, 0.00964098, -0.12973678, -0.34650919, -0.08619533, -0.05112593, -0.00816575, -0.07232255, -0.0190074, 0.08992251, -0.11135629, 0.04615228, 0.01190836, 0.09891944, -0.01582302, -0.11361597, 0.19355386, 0.00127007, -0.25896999, -0.00400663, 0.0544384, 0.24666744, 0.19696322, -0.01460781, 0.04038575, -0.11172455, 0.14222731, -0.17496657, -0.03431916, 0.13554302, 0.00554927, 0.06239656, 0.01828778, -0.09732553, 0.0398411, 0.08447129, -0.06741522, -0.07012016, 0.06162431, -0.09661818, -0.02836204, -0.09529686, 0.22595046, 0.04746725, -0.12231571, -0.17037532, 0.13914472, -0.13573512, -0.10326248, 0.01658992, -0.16176306, -0.12445088, -0.29002661, -0.01911377, 0.40853998, 0.0787245, -0.13884187, 0.07340954, -0.03091032, -0.01635932, 0.10761777, 0.20756008, -0.0319608, 0.02040027, -0.10154387, 0.02662699, 0.23967889, -0.09834416, -0.01078151, 0.16885833, 0.00322786, 0.07769771, 0.00247599, 0.00662124, -0.07490808, 0.09112778, -0.08935837, 0.01372287, 0.09497557, -0.04095287, 0.00843203, 0.1231631, -0.10844653, 0.13131899, 0.00160929, 0.07023587, 0.08300769, 0.02726312, -0.11413629, -0.07554265, 0.1213301, -0.19464579, 0.14731106, 0.18305425, 0.02018389, 0.08201285, 0.15310611, 0.1480303, -0.03532432, 0.02664389, -0.22279146, -0.01430921, 0.09179895, -0.01170638, 0.03429175, 0.00770477]
      ]}
  parent = '%s/models/%s' % (project, model_name)
  prediction = api.projects().predict(body=input_data, name=parent).execute()
  print(prediction["predictions"][0])
  
  return prediction["predictions"][0]


@app.route('/')
def index():
  return render_template('index.html')

@app.route('/form')
def input_form():
  return render_template('form.html')

@app.route('/api/predict', methods=['POST'])
def predict():
  def convert_file(val):
    people = saveface(val)
    return remove_linebreak(people)

  def person2file(val):
    people = {'1': 'huize.json', '2': 'kaitlyn.json'}
    return people[val]

  def categ2model(val):
    models = {'prof': 'prof', 'student': 'student', 'celeb': 'celeb'}
    return models[val]

  data = json.loads(request.data.decode())
  mandatory_items = ['person', 'category']
  for item in mandatory_items:
    if item not in data.keys():
      return jsonify({'result': 'Set all items.'})

  features = {}
  person = getData(data['person'])
  features['person'] = convert_file(person)
  features['category'] = categ2model(data['category'])

  prediction = get_prediction(features)
  return jsonify({'result': prediction})
  #return prediction