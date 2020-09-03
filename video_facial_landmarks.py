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


def estimate_vertical_loc(x_coord, y_top, y_bottom, shape, gray):
	
	curr_intensity = gray[x_coord, y_top]
	max_start_y = 0
	max_end_y = 0
	
	for i in range(y_top, y_bottom):
		intensity = gray[x_coord, i]
		if intensity > curr_intensity:
			curr_intensity = intensity
			max_start_y = i
		elif intensity < curr_intensity:
			max_end_y = i
			
	new_y = y_top + int((max_end_y - max_start_y) / 2)
	return new_y

def get_pupil_coords(frame, gray, shape, threshold):

	# gray - grayscaled image 
	# shape - the facial landmarks 
	# coords - the dict to store the estimated pupil coords
	
	# hmmm, currently not handling vertical movement. maybe need to test multiple rows?
	# also, lateral movement doesn't seem to be getting picked up. need to check the grayscaled images?
	# but presentation/placement of the pupils look fine (they're pretty centered at least lol)

	ret, gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY) # use a binary threshold to make it easier to find peak intensity?

	left_eye_l = (int(shape[36][0]), int(shape[36][1]))
	left_eye_r = (int(shape[39][0]), int(shape[39][1]))
	left_slope = (left_eye_r[1] - left_eye_l[1]) / (left_eye_r[0] - left_eye_l[0])
	left_b = left_eye_l[1] - (left_slope * left_eye_l[0]) # y-intercept from y=mx+b
	
	right_eye_l = (int(shape[42][0]), int(shape[42][1]))
	right_eye_r = (int(shape[45][0]), int(shape[45][1]))
	right_slope = (right_eye_r[1] - right_eye_l[1]) / (right_eye_r[0] - right_eye_l[0])
	right_b = right_eye_l[1] - (right_slope * right_eye_l[0])
	
	# estimate pupil location (given by 1 coord)
	# as as a separate field in the json output? one field for landmarks, one field for pupils
	left_start = left_eye_l[0] + 2
	left_end = left_eye_r[0] - 2
	max = 0 # we're looking for the peak of intensity given 
	max_coord_start = (left_start, (left_slope * left_start) + left_b)
	max_coord_end = (left_start, (left_slope * left_start) + left_b)
	
	for i in range(left_start, left_end+1):
		y = (left_slope * i) + left_b
		intensity = gray[i, int(y)]
		
		# note that you run the risk here of miscounting the right place if 
		# the point you start at happens to be a dark spot (i.e. the left corner of the eye (and right for that matter) might be dark
		# might require more investigating
		if max == 0:
			max = intensity
			max_coord_start = (i, int(y))
		elif max_coord_start and intensity > max:
			max = intensity
			max_coord_start = (i, int(y))
		elif max_coord_start and intensity < max:
			# find where the peak intensity ends
			max_coord_end = (i, int(y))
			break
		else:
			max_coord_end = (i, int(y))
	
	left_x = int((max_coord_end[0] + max_coord_start[0]) / 2)
	left_y = int((max_coord_end[1] + max_coord_start[1]) / 2) 
	
	# using the right x and y coords, get the vertical line that crosses through this point 
	# to also estimate the approx. y value of the pupil in case it's moved vertically significantly
	left_y_top = int(shape[38][1])
	left_y_bottom = int(shape[40][1]) # this is a larger num than left_y_top because going down == increasing
	
	new_left_y = estimate_vertical_loc(left_x, left_y_top + 2, left_y_bottom - 2, shape, gray)
	
	cv2.circle(frame, (left_x, new_left_y), 3, (255, 0, 0), -1)
	#cv2.circle(frame, (left_x, left_y), 3, (255, 0, 0), -1)
	"""
	coords['pupil_coords'].append({ 
		'x': max_coord_start[0] + int((max_coord_end[0] - max_coord_start[0]) / 2), 
		'y': max_coord_start[1] + int((max_coord_end[1] - max_coord_start[1]) / 2) 
	})
	"""
	
	max = 0 #reset
	
	
	right_start = right_eye_l[0] + 2
	right_end = right_eye_r[0] - 2
	for i in range(right_start, right_end+1):
		y = (right_slope * i) + right_b
		intensity = gray[i, int(y)]
		
		if max == 0:
			max = intensity
			max_coord_start = (i, int(y))
		elif max_coord_start and intensity > max:
			max = intensity
			max_coord_start = (i, int(y))
		elif max_coord_start and intensity < max:
			max_coord_end = (i, int(y))
			break
		else:
			max_coord_end = (i, int(y))

	right_x = int((max_coord_end[0] + max_coord_start[0]) / 2)
	right_y = int((max_coord_end[1] + max_coord_start[1]) / 2) 
	
	
	# using the right x and y coords, get the vertical line that crosses through this point 
	# to also estimate the approx. y value of the pupil in case it's moved vertically significantly
	right_y_top = int(shape[44][1])
	right_y_bottom = int(shape[46][1])
	
	new_right_y = estimate_vertical_loc(right_x, right_y_top + 2, right_y_bottom - 2, shape, gray)
	cv2.circle(frame, (right_x, new_right_y), 3, (255, 0, 0), -1)
	
	"""
	coords['pupil_coords'].append({ 
		'x': max_coord_start[0] + int((max_coord_end[0] - max_coord_start[0]) / 2), 
		'y': max_coord_start[1] + int((max_coord_end[1] - max_coord_start[1]) / 2) 
	})
	"""


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
		
			threshold = cv2.getTrackbarPos('threshold', 'Frame')
			get_pupil_coords(frame, gray, shape, threshold)
			
			# make a dot for the landmark coord
			#cv2.circle(frame, (x, y), 1, (0, 255, 0), -1) #BGR format
		
			"""
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
			
		"""

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break
		
cv2.destroyAllWindows()
vs.stop()