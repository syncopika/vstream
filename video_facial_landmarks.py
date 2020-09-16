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
	max_start_y = y_top
	max_end_y = y_bottom
	
	for i in range(y_top, y_bottom):
		intensity = gray[x_coord, i]
		if intensity > curr_intensity:
			# hit the pupil area
			curr_intensity = intensity
			max_start_y = i
		elif intensity < curr_intensity:
			# hit the sclera, end of pupil
			max_end_y = i
			break
		else:
			max_end_y = i
			
	new_y = int((max_end_y + max_start_y) / 2)
	return new_y
	
	
	
def	get_pupil_coord(landmark1, landmark2, gray):
	
	point1 = (int(landmark1[0]), int(landmark1[1]))
	point2 = (int(landmark2[0]), int(landmark2[1]))
	slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
	intercept = point1[1] - (slope * point1[0]) # y-intercept (i.e. b) from y=mx+b

	start = point1[0] # use the x-coord to go from start to end 
	end = point2[0]
	max = 0 # we're looking for the peak of intensity given. since we're working with binary colors, 0 == peak (black) and 255 is everything else (white) 
	max_coord_start = None #(start, (slope * start) + intercept)
	max_coord_end = (end, (slope * end) + intercept)
	
	for i in range(start, end+1):
		y = (slope * i) + intercept
		intensity = gray[i, int(y)]

		# note that you run the risk here of miscounting the right place if 
		# the point you start at happens to be a dark spot (i.e. the left corner of the eye (and right for that matter) might be dark
		# might require more investigating
		if max_coord_start is None:
			max = intensity
			max_coord_start = (i, int(y))
		elif max == 255 and intensity == 0:
			max = intensity
			max_coord_start = (i, int(y))
		elif max == 0 and intensity == 255:
			# find where the peak intensity ends
			max_coord_end = (i, int(y))
			break
		else:
			max_coord_end = (i, int(y))

	new_x = int((max_coord_end[0] + max_coord_start[0]) / 2)
	new_y = int((max_coord_end[1] + max_coord_start[1]) / 2) 
	
	max_coord_start = (int(max_coord_start[0]), int(max_coord_start[1]))
	max_coord_end = (int(max_coord_end[0]), int(max_coord_end[1]))
	
	return new_x, new_y, max_coord_start, max_coord_end
	

def get_pupil_coords(frame, gray, shape, threshold):

	# gray - grayscaled image 
	# shape - the facial landmarks 
	# coords - the dict to store the estimated pupil coords
	
	# hmmm, currently not handling vertical movement. maybe need to test multiple rows?
	# also, lateral movement doesn't seem to be getting picked up. need to check the grayscaled images?
	# but presentation/placement of the pupils look fine (they're pretty centered at least lol)

	ret, grayscale = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY) # use a binary threshold to make it easier to find peak intensity?

	left_x, left_y, left_start, left_end = get_pupil_coord(shape[36], shape[39], grayscale)
	
	# using the right x and y coords, get the vertical line that crosses through this point 
	# to also estimate the approx. y value of the pupil in case it's moved vertically significantly
	left_y_top = int(shape[38][1])
	left_y_bottom = int(shape[40][1]) # this is a larger num than left_y_top because going down == increasing
	
	new_left_y = estimate_vertical_loc(left_x, left_y_top + 2, left_y_bottom - 2, shape, grayscale)
	
	cv2.circle(frame, (left_x, new_left_y), 2, (255, 0, 0), -1)
	cv2.circle(frame, (left_start[0], left_start[1]), 1, (0, 0, 255), -1)
	cv2.circle(frame, (left_end[0], left_end[1]), 1, (0, 255, 255), -1)
	#cv2.circle(frame, (left_x, left_y), 3, (255, 0, 0), -1)

	right_x, right_y, right_start, right_end = get_pupil_coord(shape[42], shape[45], grayscale)
		
	# using the right x and y coords, get the vertical line that crosses through this point 
	# to also estimate the approx. y value of the pupil in case it's moved vertically significantly
	right_y_top = int(shape[44][1])
	right_y_bottom = int(shape[46][1])
	
	new_right_y = estimate_vertical_loc(right_x, right_y_top + 2, right_y_bottom - 2, shape, grayscale)
	cv2.circle(frame, (right_x, new_right_y), 2, (255, 0, 0), -1)
	cv2.circle(frame, (right_start[0], right_start[1]), 1, (0, 0, 255), -1)
	cv2.circle(frame, (right_end[0], right_end[1]), 1, (0, 255, 255), -1)


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
		
		threshold = cv2.getTrackbarPos('threshold', 'Frame')
		ret, grayscale = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
		
		for (x,y) in shape:
		
			threshold = cv2.getTrackbarPos('threshold', 'Frame')
			get_pupil_coords(gray, gray, shape, threshold)
			
			# make a dot for the landmark coord
			cv2.circle(gray, (x, y), 1, (0, 255, 0), -1) #BGR format

	cv2.imshow("Frame", gray) #frame
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break
		
cv2.destroyAllWindows()
vs.stop()