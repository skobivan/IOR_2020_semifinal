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

        cnt, _ = cv.findContours(mask_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if cnt:
            for c in cnt:
                area = cv.contourArea(c)
                if abs(area) < 500:
                    continue
                hull = cv.convexHull(c)
                approx = cv.approxPolyDP(hull, cv.arcLength(c, True) * 0.02, True)
                if 4 <= len(approx) <= 5:
                    ((x, y), (w, h), angle) = cv.minAreaRect(approx)
                    # print(angle, w/float(h))
                    if w/float(h) < 1.:
                        return True, -angle
                    return True, -90-angle
        return False, 0

    def detect_rect(self):
        hsv_mask_min = (20, 50, 50)
        hsv_mask_max = (30, 255, 255)

        image_hsv = cv.cvtColor(self.curr_image, cv.COLOR_BGR2HSV)
        image_mask = cv.inRange(image_hsv, hsv_mask_min, hsv_mask_max)

        cnt, _ = cv.findContours(image_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        copy_curr_img = self.curr_image.copy()
        if cnt:
            for c in cnt:
                area = cv.contourArea(c)
                if abs(area) < 500:
                    continue
                hull = cv.convexHull(c)
                approx = cv.approxPolyDP(hull, cv.arcLength(c, True) * 0.02, True)
                # print(len(approx))
                if 4 <= len(approx) <= 5:
                    ((x, y), (w, h), angle) = cv.minAreaRect(approx)
                    aspect_ratio = w / float(h)
                    # print(aspect_ratio)
                    if not (0.7  <= aspect_ratio <= 1.7):
                        cv.drawContours(copy_curr_img, [c], 0, (0, 255, 0), 3)
                        cv.circle(copy_curr_img, (int(x), int(y)), 5, (0, 255, 0))
                        cv.imshow('img', copy_curr_img)
                        cv.imshow('mask', image_mask)

                        cv.waitKey(1)
                        return True, x, y

        cv.imshow('img', copy_curr_img)
        cv.imshow('mask', image_mask)

        cv.waitKey(1)
        
        return False, 0, 0

        

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
        return abs(self.auv.get_depth()-val) <= 1e-2

    def is_yaw_stable(self, val):
        return abs(self.auv.get_yaw()-val) <= 1e-2

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
    
    def stop_yaw(self):
        self.auv.set_motor_power(0, -10)
        self.auv.set_motor_power(1, -10)
        time.sleep(0.3)
        self.auv.set_motor_power(0, 0)
        self.auv.set_motor_power(1, 0)

    def logic(self):
        self.bottom_cam.update_img(self.auv.get_image_bottom())
        # self.bottom_cam.detect_rect()
        if self.state == 0:
            (check, error) = self.bottom_cam.detect_line()
            if (check):
                # print(error)
                self.yaw = -error + self.auv.get_yaw()
                if self.bottom_cam.detect_line()[0] and abs(self.bottom_cam.detect_line()[1]) < 1e-3:
                    self.state += 1
        elif self.state == 1:
            self.speed = 50
            (check, rect_x, rect_y) = self.bottom_cam.detect_rect()
            # print(self.speed)
            if check:
                # self.stop_yaw()
                self.speed = 0
                if 150 <= rect_x <= 170 and 110 <= rect_y <= 130:
                    self.stop_yaw()
                    self.speed = 0
                    self.state += 1
                else:
                    try:
                        error = math.atan((rect_y-120)/(160-rect_x))/math.pi*180
                    except:
                        error = -90
                    error -= 90
                    if error < -90:
                        error = error + 180
                    print(error)
                    if abs(error) < 1e-2:
                        self.speed = 50
                    self.yaw = -error + self.auv.get_yaw()
        # self.bottom_cam.show()
        self.keep_depth(2)    
        self.keep_yaw(self.yaw, self.speed)

KP_YAW = 0.8
KD_YAW = 0.0

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
