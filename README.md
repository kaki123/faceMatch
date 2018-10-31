# faceMatch

The faceMatch directory is split up like this

```bash
faceMatch
├── code (code for training and saving the machine learning model)
├── helperScripts (scrips that help with data preprocessing and other purposes)
├── savedModels (trained models that are uploaded to GCP)
|
└── README.md
```

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
  
