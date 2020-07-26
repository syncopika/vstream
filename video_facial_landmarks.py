"""
	The code here is pretty much copied (although I hand-typed)
	from Adrian Rosebrock's amazing tutorial here:
	https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/

	For the landmark predictor, I used shape_predictor_68_face_landmarks.dat that you can download here:
	https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat

	to get dlib,
	I did:
	pip install cmake
	pip install dlib 
	
	I'm using conda as well on Windows 10.


	idea:
	have a local server that gets those facial landmark coords (flask/socketio)
	send them off to a client browser
	use those coords to draw the face on a canvas 
	
	https://flask-socketio.readthedocs.io/en/latest/
	
	then next step:
	apply to a virtual avatar
	profit?

	then next next step:
	how about hand detection?

"""

from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("setting up camera sensor...")
vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray, 0)
	
	# get the facial landmarks of the faces
	for face in faces:
		shape = predictor(gray, face) # can we separate the shapes in the face?
		shape = face_utils.shape_to_np(shape)
		
		for (x,y) in shape:
			cv2.circle(frame, (x, y), 5, (255, 0, 0), -1) #BGR format
			
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break
		
cv2.destroyAllWindows()
vs.stop()