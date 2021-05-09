"""
	The facial landmark code here is pretty much copied (although I hand-typed)
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
	how about hand detection? uh no

"""

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
		
	def estimate_vertical_loc(self, x_coord, y_top, y_bottom, shape, gray):
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
	
	def	get_pupil_coord(self, landmark1, landmark2, gray):
		point1 = (int(landmark1[0]), int(landmark1[1]))
		point2 = (int(landmark2[0]), int(landmark2[1]))
		slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
		intercept = point1[1] - (slope * point1[0]) # y-intercept from y=mx+b
		
		start = point1[0] + 2
		end = point2[0] - 2
		max = 0 # we're looking for the peak of intensity given 
		max_coord_start = (start, (slope * start) + intercept)
		max_coord_end = (start, (slope * start) + intercept)
		
		for i in range(start, end+1):
			y = (slope * i) + intercept
			intensity = gray[i, int(y)] # access the pixel's intensity, which is a single value, at coordinate (i, int(y))
			
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

		new_x = int((max_coord_end[0] + max_coord_start[0]) / 2)
		new_y = int((max_coord_end[1] + max_coord_start[1]) / 2) 
		
		return new_x, new_y
		
	def get_pupil_coords(self, gray, shape, coords):
		# gray - grayscaled image 
		# shape - the facial landmarks 
		# coords - the dict to store the estimated pupil coords
		
		# hmmm, currently not handling vertical movement. maybe need to test multiple rows?
		# also, lateral movement doesn't seem to be getting picked up. need to check the grayscaled images?
		# but presentation/placement of the pupils look fine (they're pretty centered at least lol)
	
		global STATE
	
		threshold = 0
		ret, gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY) # use a binary threshold to make it easier to find peak intensity?

		left_x, left_y = self.get_pupil_coord(shape[36], shape[39], gray)
		
		# using the right x and y coords, get the vertical line that crosses through this point 
		# to also estimate the approx. y value of the pupil in case it's moved vertically significantly
		left_y_top = int(shape[38][1])
		left_y_bottom = int(shape[40][1]) # this is a larger num than left_y_top because going down == increasing
		new_left_y = self.estimate_vertical_loc(left_x, left_y_top + 2, left_y_bottom - 2, shape, gray)
		coords['pupil_coords'].append({'x': left_x, 'y': new_left_y})

		right_x, right_y = self.get_pupil_coord(shape[42], shape[45], gray)
		
		# using the right x and y coords, get the vertical line that crosses through this point 
		# to also estimate the approx. y value of the pupil in case it's moved vertically significantly
		right_y_top = int(shape[44][1])
		right_y_bottom = int(shape[46][1])
		new_right_y = self.estimate_vertical_loc(right_x, right_y_top + 2, right_y_bottom - 2, shape, gray)
		coords['pupil_coords'].append({'x': right_x, 'y': new_right_y})
		
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
					shape = self.predictor(gray, face)
					shape = face_utils.shape_to_np(shape)
					
					try:
						self.get_pupil_coords(gray, shape, coords)
					except Error as err:
						print("an issue occurred getting the pupil coordinates")
					
					for (x,y) in shape:
						coords['landmark_coords'].append({'x': int(x), 'y': int(y)})
						cv2.circle(frame, (x, y), 1, (255, 0, 0), -1) # BGR format
				
				# send the coords off to the client browser!
				self.socket.emit('landmarkCoordinates', json.dumps(coords))

		# cleanup		
		cv2.destroyAllWindows()
		self.video_stream.stop()

@app.route('/')
@app.route('/vstream_display')
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

	return render_template("vstream_display.html")


# allow the client to pause/continue the stream 
@socketio.on('toggleStream')
def toggle_stream(data):
	global STATE
	logging.info("--------------- SEND_DATA HAS BEEN TOGGLED ---------------")
	STATE['send_data'] = (not STATE['send_data'])
	

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