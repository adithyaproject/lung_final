# Uncomment below 3 lines if you are dealing with Raspberry pi & OLED display
# import sys
# sys.path.insert(0,"/home/Your_project_directory/Adafruit_Python_SSD1306/examples")
# import text
import recording
import mfcc
import requests
import json

# Start recording
if(recording.record_sound()):
    # Extract mfcc features
    mfccFeatures = mfcc.startPreProcessing("filtered.wav")

    # Encode numpy ndarray to list  of lists
    mfccList = mfccFeatures.tolist()
    serializedMfcc = json.dumps(mfccList)

    # Send post request for tha backend by passing encoded list
    res = requests.post("http://192.168.8.101:5000/", json = {"data":serializedMfcc})
    print(res.json()["condition"])

    # Uncomment below line if you are dealing with Raspberry pi & OLED display
    # text.displayText(res.json()["condition"])
