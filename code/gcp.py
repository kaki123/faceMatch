from googleapiclient import discovery
from googleapiclient import errors

#export GOOGLE_APPLICATION_CREDENTIALS="/Users/kakifung/Documents/faceMatch/code/Face Match-3e013a9cd558.json"

# Store your full project ID in a variable in the format the API needs.
projectID = 'projects/{}'.format('face-match-219722')

# Build a representation of the Cloud ML API.
ml = discovery.build('ml', 'v1')

# Create a dictionary with the fields from the request body.
requestDict = {'name': 'test_model_2',
               'description': 'your_model_description'}

# Create a request to call projects.models.create.
request = ml.projects().models().create(
              parent=projectID, body=requestDict)

# Make the call.
try:
    response = request.execute()
    print(response)
except errors.HttpError as err:
    # Something went wrong, print out some information.
    print('There was an error creating the model. Check the details:')
    print(err._get_reason())
