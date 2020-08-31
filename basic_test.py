from imutils.video import VideoStream
from imutils import face_utils
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import datetime
import imutils
import time
import dlib
import cv2
import logging
import atexit
import json
import threading
import math

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)

SEND_DATA = True
VIDEO_STREAM = None
BACKGROUND_THREAD = None

# has some info that could be updated in the main and child thread
# I think this should be fine here since we really only should have 
# one child thread to run the facial landmark coordinate collection job
STATE = {
	"send_data": SEND_DATA,
	"video_stream": VIDEO_STREAM,
	"detector": None,
	"predictor": None,
	"threshold": 54,
	"socket": socketio
}

# get the facial landmark coordinates
# this should be running continuously in another thread (and there should only be one other thread)
class VStream(threading.Thread):
	def __init__(self, state):
		super().__init__()
		self.state = state
		#self.SEND_DATA = state['send_data']
		self.video_stream = state['video_stream']
		self.socket = state['socket']
		self.detector = state['detector']
		self.predictor = state['predictor']
		
		detector_params = cv2.SimpleBlobDetector_Params()
		detector_params.filterByArea = True
		detector_params.maxArea = 1500
		detector_params.filterByConvexity = False
		detector_params.filterByInertia = False
		self.eye_detector = cv2.SimpleBlobDetector_create(detector_params)

		
		
	#def get_pupil_coords(self, gray, shape, coords):
		# https://subscription.packtpub.com/book/application_development/9781785283932/4/ch04lvl1sec44/detecting-pupils
		# https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
	#	pass
		
	def get_pupil_coords(self, gray, shape, coords):
	
		# gray - grayscaled image 
		# shape - the facial landmarks 
		# coords - the dict to store the estimated pupil coords
		
		# hmmm, currently not handling vertical movement. maybe need to test multiple rows?
		# also, lateral movement doesn't seem to be getting picked up. need to check the grayscaled images?
		# but presentation/placement of the pupils look fine (they're pretty centered at least lol)
	
		global STATE
	
		threshold = STATE['threshold'] # 45
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
		max_coord_start = None
		max_coord_end = None
		
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
			
		
		coords['pupil_coords'].append({ 
			'x': max_coord_start[0] + int((max_coord_end[0] - max_coord_start[0]) / 2), 
			'y': max_coord_start[1] + int((max_coord_end[1] - max_coord_start[1]) / 2) 
		})
		#coords['pupil_coords'].append({'x': max_coord[0], 'y': max_coord[1]})
		
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

		coords['pupil_coords'].append({ 
			'x': max_coord_start[0] + int((max_coord_end[0] - max_coord_start[0]) / 2), 
			'y': max_coord_start[1] + int((max_coord_end[1] - max_coord_start[1]) / 2) 
		})
		#coords['pupil_coords'].append({'x': max_coord[0], 'y': max_coord[1]})
	
	
	def get_landmark_coords(self, gray, shape, coords):
	
		global STATE
		
		# for pupil tracking?
		# https://answers.opencv.org/question/173862/how-to-check-in-python-if-a-pixel-in-an-image-is-of-a-specific-color-other-than-black-and-white-for-grey-image/
		# https://stackoverflow.com/questions/51781843/color-intensity-of-pixelx-y-on-image-opencv-python
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
			
			coords['landmark_coords'].append({'x': int(x), 'y': int(y)})
			#cv2.circle(frame, (x, y), 1, (255, 0, 0), -1) # BGR format
			

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
			
			#threshold = cv2.getTrackbarPos('threshold', 'Frame')
			threshold = STATE['threshold'] # 54
			
			img = cv2.blur(roi, (3,3))
			#img = cv2.medianBlur(img, 7)
			ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

			
			keypoints = self.eye_detector.detect(img)
			
			for point in keypoints:
				kx = int(point.pt[0])
				ky = int(point.pt[1])
				#cv2.circle(frame, (x+kx, y+ky), 3, (255, 0, 0), -1) #BGR format
				coords['pupil_coords'].append({'x': x+kx, 'y': y+ky})
	
	
	def run(self):
	
		while True:
			if self.state['send_data']:
				frame = self.video_stream.read()
				frame = imutils.resize(frame, width=400)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faces = self.detector(gray, 0)
				
				# collect all the landmark coordinates
				# should be a list of dictionaries like [{'x': a, 'y': b},...]
				coords = {'landmark_coords': [], 'pupil_coords': []}
				
				# get the facial landmarks of the faces
				for face in faces:
				
					# can we separate the shapes in the face?
					# https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
					
					# might be helpful for pupil tracking: 
					# https://stackoverflow.com/questions/45789549/track-eye-pupil-position-with-webcam-opencv-and-python
					
					shape = self.predictor(gray, face)
					shape = face_utils.shape_to_np(shape)

					# for pupil tracking?
					# https://answers.opencv.org/question/173862/how-to-check-in-python-if-a-pixel-in-an-image-is-of-a-specific-color-other-than-black-and-white-for-grey-image/
					# https://stackoverflow.com/questions/51781843/color-intensity-of-pixelx-y-on-image-opencv-python
					left_eye = {'minX': 0, 'maxX': 0, 'maxY': 0, 'minY': 0} # top/bottom y values are the vertical boundaries, left/right x values are the horizontal bounds
					right_eye = {'minX': 0, 'maxX': 0, 'maxY': 0, 'minY': 0} 
					index = 0

					for (x,y) in shape: 
						xCoord = int(x)
						yCoord = int(y)
						coords['landmark_coords'].append({'x': int(x), 'y': int(y)})
						#cv2.circle(frame, (x, y), 1, (255, 0, 0), -1) # BGR format
						
					# get pupil info
					self.get_pupil_coords(gray, shape, coords)


				#cv2.imshow("Frame", frame)
				#cv2.waitKey(1)
				
				# send the coords off to the client browser!
				self.socket.emit('landmarkCoordinates', json.dumps(coords))
				
				#key = cv2.waitKey(1) & 0xFF
				
				#if key == ord("q"):
				#	break
		
		# cleanup		
		cv2.destroyAllWindows()
		self.video_stream.stop()

@app.route('/')
def vstream_display():

	global BACKGROUND_THREAD # global bg thread variable 
	global STATE
	
	logging.info("loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	logging.info("setting up camera sensor...")
	VIDEO_STREAM = VideoStream(usePiCamera=False).start()
	time.sleep(2.0)
	
	STATE['video_stream'] = VIDEO_STREAM
	STATE['detector'] = detector
	STATE['predictor'] = predictor

	# start thread to collect coords (only if one doesn't already exist)
	if BACKGROUND_THREAD is None:
		BACKGROUND_THREAD = VStream(STATE)
		BACKGROUND_THREAD.start()
		logging.info("started background thread...")

	return render_template("landmark_testing.html")


# allow the client to pause/continue the stream 
@socketio.on('toggleStream')
def toggle_stream(data):
	global STATE
	logging.info("--------------- SEND_DATA HAS BEEN TOGGLED ---------------")
	STATE['send_data'] = (not STATE['send_data'])
	
	
@socketio.on('updateThreshold')
def update_threshold(data):
	global STATE
	STATE['threshold'] = data
	

"""
handle disconnect? 

def cleanup():
	cv2.destroyAllWindows()
	video_stream.stop()

atexit.register(cleanup)
"""


# start the server 
if __name__ == "__main__":
	socketio.run(app)