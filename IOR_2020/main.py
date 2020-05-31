import pymurapi as mur
import time
import cv2 as cv
import math

class PD(object):
    _kp = 0.0
    _kd = 0.0
    _prev_error = 0.0
    _timestamp = 0

    def __init__(self, kp, kd):
        self._kp = kp
        self._kd = kd

    def set_kp(self, value):
        self._kp = value

    def set_kd(self, value):
        self._kd = value

    def procces(self, error):
        timestamp = time.time()
        output = self._kp * error + self._kd / (timestamp - self._timestamp) * (error - self._prev_error)
        self._timestamp = timestamp
        self._prev_error = error
        return output

def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value

def clamp_motor_speed(value):
    return clamp(value, -100, 100)

def clamp_angle(angle):
    if angle > 180:
        return angle - 360
    if angle < -180:
        return angle + 360
    return angle

class Camera:
    def __init__(self):
        self.curr_image = None

    def show(self):
        if self.curr_image is not None:
            cv.imshow('img', self.curr_image)
            cv.waitKey(1)

    def update_img(self , img):
        self.curr_image = img

class CameraFront(Camera):
    def __init__(self):
        super(CameraFront, self).__init__()

class CameraBottom(Camera):
    def __init__(self):
        super(CameraBottom, self).__init__()

    def detect_line(self):
        # hsv_mask = {'h_min' : 15, 'h_max' : 30, 's_min' : 50, 's_max' : 255, 'v_min' : 50, 'v_max' : 255}
        hsv_mask_min = (15, 50, 50)
        hsv_mask_max = (30, 255, 255)
        hsv_image = cv.cvtColor(self.curr_image, cv.COLOR_BGR2HSV)

        mask_image = cv.inRange(hsv_image, hsv_mask_min, hsv_mask_max)

        center_x = 0
        center_y = 100

        center_x2 = 0
        center_y2 = 140

        sum = 0
        sum2 = 0

        for x in range(0, 320):
            center_x += mask_image[center_y, x] * x
            sum += mask_image[center_y, x]

            center_x2 += mask_image[center_y2, x] * x
            sum2 += mask_image[center_y2, x]
            
                
        if sum:
            center_x = int(center_x/sum)
            cv.circle(self.curr_image, (center_x, center_y), 10,  (255, 0, 0))
        
        if sum2:
            center_x2 = int(center_x2/sum2)
            cv.circle(self.curr_image, (center_x2, center_y2), 10, (255, 0, 0))

        if center_x2 - center_x == 0:
            return 0
        return math.atan((center_y2 - center_y)/(center_x-center_x2))

    def detect_rect(self):
        hsv_mask_min = (20, 50, 50)
        hsv_mask_max = (30, 255, 255)

        image_hsv = cv.cvtColor(self.curr_image, cv.COLOR_BGR2HSV)
        image_mask = cv.inRange(image_hsv, hsv_mask_min, hsv_mask_max)

        



class Robot(object):

    def __init__(self, yaw_p, yaw_d, depth_p, depth_d):
        self.auv = mur.mur_init()
        self.yaw_pd = PD(yaw_p, yaw_d)
        self.depth_pd = PD(depth_p, depth_d)
        self.bottom_cam = CameraBottom()
        self.state = 0
        self.yaw = 0
        self.speed = 0
    def is_depth_stable(self, val):
        return abs(self.auv.get_depth()-val) < 1e-1

    def is_yaw_stable(self, val):
        return abs(self.auv.get_yaw()-val) < 1e-1

    def keep_yaw(self, yaw, speed):
            error = self.auv.get_yaw() - yaw
            error = clamp_angle(error)

            output = self.yaw_pd.procces(error)
            output = clamp_motor_speed(output)

            self.auv.set_motor_power(0, clamp_motor_speed(speed - output))
            self.auv.set_motor_power(1, clamp_motor_speed(speed + output))

    def keep_depth(self, depth):
        error = self.auv.get_depth() - depth
        output = self.depth_pd.procces(error)
        output = clamp_motor_speed(output)
        self.auv.set_motor_power(2, output)
        self.auv.set_motor_power(3, output)
    
    def logic(self):
        self.bottom_cam.update_img(self.auv.get_image_bottom())

        # if self.state == 0:
        #     error = self.bottom_cam.detect_line()/math.pi*180
        #     print(error)
        #     self.state += 1
        #     self.yaw = error + self.auv.get_yaw()
        # elif self.state == 1:
        #     if self.is_yaw_stable(self.yaw):
        #         self.speed = 50
        # self.bottom_cam.show()
        # self.keep_depth(2)    
        # self.keep_yaw(self.yaw, self.speed)

KP_YAW = 0.8
KD_YAW = 0.5

KP_DEPTH = 70
KD_DEPTH = 5

robot = Robot(KP_YAW, KD_YAW, KP_DEPTH, KD_DEPTH)
# time.sleep(2)
while not robot.is_depth_stable(2):
    robot.keep_depth(2)
    time.sleep(0.03)
while True:
    # robot.keep_depth(2)
    # robot.keep_yaw(50, 0)
    robot.logic()
    time.sleep(0.03)
