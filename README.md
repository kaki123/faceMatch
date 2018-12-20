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

# Phase 5
# Task to be done in frontend :
  1) Upload the imported photo to the Google Cloud Bucket
  2) Send the filename of the photo to the ML engine

# Task to be done in backend :
  1) Make a prediction function to take in input from the front end
  2) Send the prediction result to the front end  
