import os
import json
import numpy as np
import predicting
from flask import Flask, request, jsonify
from json import JSONEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize web app
app = Flask(__name__)

# Get and Post routes
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Function for handling main GET and POST (Numpy ndarray with mfcc features) requests
    """
    if request.method == 'POST':
        # Get request data
        encodedMfcc = request.get_json()

        # Decode list of lists to numpy ndarray
        decodedMfcc = np.array(json.loads(encodedMfcc["data"]))

        if decodedMfcc is None:
            return jsonify({"error": "Not any mfcc features received with the request!"})
        
        try:
            # Create 2 instances of the LungCondition
            lungConditionInstance1 = predicting.lungConditionPredicting()
            lungConditionInstance2 = predicting.lungConditionPredicting()

            # Check that different instances of the LungCondition point back to the same object (singleton)
            assert lungConditionInstance1 is lungConditionInstance2

            # Make a prediction
            condition = lungConditionInstance1.startPredicting(decodedMfcc)
            
            # Construct output
            data = {
                    "success": True,
                    "condition": condition
                   }

            return jsonify(data)

        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({"message": "Request successfully processed!"})

# Execute the app
if __name__ == "__main__":
    app.run(host="0.0.0.0")

