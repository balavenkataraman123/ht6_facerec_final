# Video streaming backend hosted on device with camera

from picamera2 import Picamera2
import io
import time

picam2 = Picamera2()
picam2.start()
from websockets.sync.client import connect
import time
from time import time_ns
import base64


import RPi.GPIO as GPIO
from time import sleep

spin = 8

GPIO.setmode(GPIO.BOARD)
GPIO.setup(spin, GPIO.OUT)

pwm=GPIO.PWM(spin, 50)
pwm.start(0)

def setAngle(angle):
    duty = angle / 18 + 2
    GPIO.output(spin, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(spin, False)
    pwm.ChangeDutyCycle(duty)



NS_TO_MS: int = 1000000 # defines configurations
FPS = 15
SERVER_ADDRESS = "ws://192.168.137.217:8765"
FACEUID = "BALA" + "A" * 34

numframes = 0 # keeps count of frames transmitted

setAngle(90)

while(True):
    with connect(SERVER_ADDRESS) as websocket: # connects to websocket server
        numframes += 1
        start_time = time_ns() // NS_TO_MS # keeps time for constant transmission frame rate
        data = io.BytesIO()
        picam2.capture_file(data, format='jpeg')
        print(data)
        header = FACEUID
        imagetext = base64.b64encode(data.getvalue())
        websocket.send(header + "data:image/jpeg;base64," + str(imagetext)[1:]) # sends captured image with transmission header

        message = websocket.recv()
        if message == "FACE MATCH SUCCESS":
            print("face matched")
            setAngle(0)
            # do the servo thingies
        elif message == "FACE MATCH FAILURE":
            print("match not found")
            # do other servo thingies


        end_time = time_ns() // NS_TO_MS
        print(f"{str(min(FPS, 1000/(end_time - start_time)))} FPS") # displays frame rate
        next_tick = (1000/FPS) - (end_time - start_time)
        if next_tick < 0:
            next_tick = 0
        time.sleep(next_tick / 1000)    # keeps frame rate constant