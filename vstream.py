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
	how about hand detection?

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
		
	def get_pupil_coords(self, gray, shape, coords):
		# gray - grayscaled image 
		# shape - the facial landmarks 
		# coords - the dict to store the estimated pupil coords

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
		left_start = left_eye_l[0] + 1
		left_end = left_eye_r[0] - 1
		max = 0 # we're looking for the peak of intensity given 
		max_coord = []
		
		for i in range(left_start, left_end+1):
			y = (left_slope * i) + left_b
			intensity = gray[i, int(y)]
			
			if max == 0:
				max = intensity
				max_coord = [i, int(y)]
			else:
				if intensity > max:
					max = intensity
					max_coord = [i, int(y)]
		
		max = 0
		coords['pupil_coords'].append({'x': max_coord[0], 'y': max_coord[1]})
		
		right_start = right_eye_l[0] + 1
		right_end = right_eye_r[0] - 1
		for i in range(right_start, right_end+1):
			y = (right_slope * i) + right_b
			intensity = gray[i, int(y)]
			
			if max == 0:
				max = intensity
				max_coord = [i, int(y)]
			else:
				if intensity > max:
					max = intensity
					max_coord = [i, int(y)]

		coords['pupil_coords'].append({'x': max_coord[0], 'y': max_coord[1]})
	
		
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
					
					# for pupil tracking:
					# we know the landmark coord indices that match the eyes 
					# so we can create a box approximation around where we thing the pupil should be? (just go along the middle of the eye section)
					# then evaluate that area for where there's a contrast (left and right bound?)
					# then return the middle of the left and right bound (of the section that's darker)
					# so go from left of the box to the right side
					# as you keep going note any significant change in color (that's something you have to decide)
					# once you hit that point where there's a significant contrast, record it as the left bound
					# and keep going until you either hit a lighter area again (mark as right bound) or hit the right end of the box
					# take the middle of the left and right to estimate where the pupil is
					# https://answers.opencv.org/question/173862/how-to-check-in-python-if-a-pixel-in-an-image-is-of-a-specific-color-other-than-black-and-white-for-grey-image/
					
					shape = self.predictor(gray, face)
					shape = face_utils.shape_to_np(shape)
					
					# we're assuming that the head is leveled here (i.e. we can just go across via x-axis but what if the head is at an angle?)
					# we need a left coord and a width for left and right eye 
					# can we index shape?
					# https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
					self.get_pupil_coords(gray, shape, coords)
					
					# estimate pupil location (given by 1 coord)
					# as as a separate field in the json output? one field for landmarks, one field for pupils
					
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