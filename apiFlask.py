from flask import *
import json
import cv2
import pickle
import cvzone
import numpy as np
import base64
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# video feed
cap = cv2.VideoCapture(0)

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 46


def checkParkingSpace(imgPro, img):
    SpaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        if count < 900:
            color = (0, 255, 0)
            SpaceCounter += 1
        else:
            color = (0, 0, 255)
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, 2)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0,
                           colorR=color)

    cvzone.putTextRect(img, f'Free Spaces : {str(SpaceCounter)}', (100, 50), scale=3, thickness=2,
                       offset=20, colorR=(0, 200, 0))


def captureVideoFrame():
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernal = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernal, iterations=1)

    checkParkingSpace(imgDilate, img)

    return img


@app.route('/', methods=['GET'])
def send_video_as_json():
    # Capture video frame
    img = captureVideoFrame()

    # Convert image to Base64
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Prepare JSON response
    response = {
        'video': img_base64
    }

    # Send API response
    return jsonify(response)


if __name__ == '__main__':
    app.run()
