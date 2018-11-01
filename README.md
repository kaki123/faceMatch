# faceMatch

The folders faceMatch directory is split up like this

```bash
faceMatch
├── code (code for training and saving the machine learning model)
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

# Instruction to view Backend Deliverables :
1) GCP Account: mzheng20@students.claremontmckenna.edu
2) Project Name: Face Match
3) Project ID: face-match-219722
4) BUCKET_NAME: face-match-219722knn
5) MODEL_NAME: knn_1dlist
6) VERSION_NAME: first_working
7) INPUT_FILE: from root of this github repository: /testInputs/obama3.json  (stored is a list of pixel info of an obama pic)
9) set up environment variables and Run Command: gcloud ml-engine predict --model $MODEL_NAME --version $VERSION_NAME --json-instances $INPUT_FILE
10) expected output: [u'obama']

# Phase 2
# # faceMatch
# Task to be done in frontend :
  1) Create category selection menu 
  2) Give the import photo button functionality on the home page
  3) Give the download button functionality on the results page

# Task to be done in backend :
  1) save the trained model and export to a Pickle File
  2) BackEnd - upload Pickle File to GCP
  3) BackEnd - write a Python Script that invokes the trained model on test data
  4) BackEnd - setup GCP account and project
  5) Fix bug in python script. 

# Task to be done in frontend and backend connection:
  1) Receive uploaded photo from frontend and save to GCP server
  2) Get the results from GCP and return the result photo back to the frontend
  
