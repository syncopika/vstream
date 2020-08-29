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
import math

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("setting up camera sensor...")
#vs = VideoStream(usePiCamera=False).start()
#time.sleep(2.0)

# https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
# helpful! https://www.learnopencv.com/blob-detection-using-opencv-python-c/
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector_params.filterByConvexity = False
detector_params.filterByInertia = False

eye_detector = cv2.SimpleBlobDetector_create(detector_params)

def func(x):
	pass


vs = cv2.VideoCapture(0)
cv2.namedWindow('Frame')	
cv2.createTrackbar('threshold', 'Frame', 0, 255, func)

while True:
	#frame = vs.read()
	ret, frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray, 0)
	
	# get the facial landmarks of the faces
	for face in faces:
		shape = predictor(gray, face) # can we separate the shapes in the face?
		shape = face_utils.shape_to_np(shape)
		
		left_eye = {'minX': 0, 'maxX': 0, 'maxY': 0, 'minY': 0} # top/bottom y values are the vertical boundaries, left/right x values are the horizontal bounds
		right_eye = {'minX': 0, 'maxX': 0, 'maxY': 0, 'minY': 0} 
		index = 0
		
		for (x,y) in shape:
		
			# get info on eyes
			if index in [36, 39, 37, 41]:
				# left eye coords 
				xCoord = int(x)
				yCoord = int(y)
				
				if left_eye['minX'] == 0:
					left_eye['minX'] = xCoord
				else:
					left_eye['minX'] = min(xCoord, left_eye['minX'])
				
				if left_eye['minY'] == 0:
					left_eye['minY'] = yCoord
				else:
					left_eye['minY'] = min(yCoord, left_eye['minY'])
				
				left_eye['maxX'] = max(xCoord, left_eye['maxX'])
				left_eye['maxY'] = max(yCoord, left_eye['maxY'])
				
			elif index in [42, 45, 43, 47]:
				# right eye coords
				xCoord = int(x)
				yCoord = int(y)
				
				if right_eye['minX'] == 0:
					right_eye['minX'] = xCoord
				else:
					right_eye['minX'] = min(xCoord, right_eye['minX'])
				
				if right_eye['minY'] == 0:
					right_eye['minY'] = yCoord
				else:
					right_eye['minY'] = min(yCoord, right_eye['minY'])
				
				right_eye['maxX'] = max(xCoord, right_eye['maxX'])
				right_eye['maxY'] = max(yCoord, right_eye['maxY'])

			index += 1
			

			# make a dot for the landmark coord
			#cv2.circle(frame, (x, y), 3, (255, 0, 0), -1) #BGR format
		
		# draw in pupils with eye info
		offset = 10
		for eye in [left_eye, right_eye]:
			# get the region-of-interest (ROI) based on eye info
			width = eye['maxX'] - eye['minX'] + offset
			height = eye['maxY'] - eye['minY'] + offset
			
			y = eye['minY'] - int(offset/2)
			x = eye['minX']
			
			#print(eye)
			#print(f"y: {y}")
			#print(f"x: {x}")
			
			# https://stackoverflow.com/questions/9084609/how-to-copy-a-image-region-using-opencv-in-python
			roi = gray[y:(y+height), x:(x+width)]
			cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 1);
			
			threshold = cv2.getTrackbarPos('threshold', 'Frame')
			
			img = cv2.blur(roi, (4,4))
			#img = cv2.medianBlur(roi, 7)
			ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
			
			keypoints = eye_detector.detect(img)
			
			# if multiple results, just get the first one
			for point in keypoints[:1]:
				kx = int(point.pt[0])
				ky = int(point.pt[1])
				#print(point.pt[0])
				#print(point.pt[1])
				cv2.circle(frame, (x+kx, y+ky), 3, (255, 0, 0), -1) #BGR format
			
			original = frame[y:y+height, x:x+width]
			#cv2.drawKeypoints(original, keypoints, original, (255,180,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break
		
cv2.destroyAllWindows()
vs.stop()