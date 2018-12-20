# faceMatch

The folders faceMatch directory is split up like this

```bash
faceMatch
├── code (code for training and saving the machine learning model)
|
├──── test (code for gcp app using flask, angularjs, and the ML engine)
|
├── helperScripts (scrips that help with data preprocessing and other purposes)
|
├── savedModels (trained models that are uploaded to GCP)
|
├── testInput (JSON file that are used as an input to make online predictions in GCP)
|
├── knn_examples (training and test file for the KNN model)
|
├── grs (training photos for golden retrievers)
|
├── pugs (training photos for pugs)
|
├── FrontEnd (frontend code for web pages)
|
└── README.md
```
# Instruction to view Frontend Deliverables :
  1) To view the web page, go to https://face-match-upload.appspot.com, which is the url for the following file: code/test/templates/index.html
  2) Sign in with your Google creditials to access the form.html page, which is the main web page for Face Match.

# Instruction to view Backend Deliverables :
  1) GCP Account: mzheng20@students.claremontmckenna.edu
  2) Project Name: Face Match
  3) Project ID: face-match-219722
  4) BUCKET_NAME: face-match-219722celeb, face-match-219722student, and face-match-219722prof
  5) MODEL_NAME: celeb_model, student_model, and prof_model
  6) VERSION_NAME: celeb1, student1, and prof1
  7) INPUT_FILE: from root of this github repository: /testInputs/obama3.json  (stored is a list of pixel info of an obama pic)
  8) set up environment variables and Run Command: gcloud ml-engine predict --model $MODEL_NAME --version $VERSION_NAME --json-instances $INPUT_FILE
  9) expected output: [u'obama']

# Two Versions of the final app code package :
There are two versions of the final app code package inside the "code" directory. 
  1) The "test_copy1" directory contains the first working version of our app, which has harded coded input for ML predictions. That version has less functionality than our final version, but it does not contain any bugs and can be deployed successfully on GCP
  2) The "test" directory contains more features than the first working version. It incorporated fetching image from Cloud Storage and converting it to string. It also does the image preprocessing using the "face recognition" module. However, because we couldn't resolve bugs caused by the required dlib library of the face recognition module, this version cannot be run successfully on GCP yet
  
# Two Versions of functionality code that can be test locally :
There are two versions of the final app code package inside the home directory. 
  1) The "predict.py" file contains the same functionality as the app version in the "test" directory mentioned above. However, this file can be run successfully on one's local machine because it is not subject to the limitations on library used on GCP. It fetches an image from Cloud Storage, preprocesses it, and successfully makes a face match prediction on that image input.
  2) The "predict_changed.py" contains the latest functionality that we were able to incorporate locally, including using the Google Cloud Vision API for image preprocessing. However, that file only prints the face feature locations information form the Cloud Vision API but cannot successfully make predictions yet because we haven't converted the face locations to a 1D array ML model input yet.
