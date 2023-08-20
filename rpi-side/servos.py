import RPi.GPIO as GPIO
from time import sleep

spin = int(input("servo pin:"))

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
while True:
    angle = int(input("angle:"))
    setAngle(angle)
