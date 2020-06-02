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
        output = self._kp * error + self._kd * (timestamp - self._timestamp) * (error - self._prev_error)
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
    
    def plate_area(self, min_masks, max_masks):
        copy_curr_img = self.curr_image.copy()

        image_hsv = cv.cvtColor(self.curr_image, cv.COLOR_BGR2HSV)

        image_mask = cv.inRange(image_hsv, min_masks[0], max_masks[0])
        image_mask += cv.inRange(image_hsv, min_masks[1], max_masks[1])
        image_mask += cv.inRange(image_hsv, min_masks[2], max_masks[2])

        cnt, _ = cv.findContours(image_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        total_area = 0

        if cnt:
            for c in cnt:
                area = cv.contourArea(c)
                if abs(area) < 300:
                    continue
                total_area += area
                cv.drawContours(copy_curr_img, [c], 0, (255, 255, 255), 3)

        cv.imshow('mask', image_mask)
        cv.imshow('mask', copy_curr_img)
        cv.waitKey(1)

        return total_area
    
    def find_circle(self, min_masks, max_masks):
        img_hsv = cv.cvtColor(self.curr_image, cv.COLOR_BGR2HSV)
        copy_img = self.curr_image.copy()
        mask = cv.inRange(img_hsv, min_masks[0], max_masks[0])
        mask += cv.inRange(img_hsv, min_masks[1], max_masks[1])
        mask += cv.inRange(img_hsv, min_masks[2], max_masks[2])

        cnt, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        for c in cnt:
            area = cv.contourArea(c)
            if area < 500:
                continue
            hull = cv.convexHull(c)
            approx = cv.approxPolyDP(hull, cv.arcLength(c, True) * 0.02, True)
            if len(approx) >= 7:
                cv.drawContours(copy_img, [c], 0, (0, 0, 255), 3)
                mm = cv.moments(approx)
                x = mm['m10']/mm['m00']
                y = mm['m01']/mm['m00'] 
                # print(x, y)
                cv.imshow("img", copy_img)
                cv.imshow('mask', mask)
                cv.waitKey(1)
                return True, x, y, area
            cv.drawContours(copy_img, [c], 0, (255, 255, 255), 3)
        cv.imshow("img", copy_img)
        cv.imshow('mask', mask)

        cv.waitKey(1)
        return False, 0, 0, 0


class CameraBottom(Camera):
    def __init__(self):
        super(CameraBottom, self).__init__()

    def detect_line(self, hsv_mask_min, hsv_mask_max):
        # hsv_mask = {'h_min' : 15, 'h_max' : 30, 's_min' : 50, 's_max' : 255, 'v_min' : 50, 'v_max' : 255}
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
                if abs(area) < 1500:
                    continue
                hull = cv.convexHull(c)
                approx = cv.approxPolyDP(hull, cv.arcLength(c, True) * 0.02, True)
                # print(len(approx))
                if len(approx) == 4:
                    ((x, y), (w, h), angle) = cv.minAreaRect(approx)
                    aspect_ratio = w / float(h)
                    # print(aspect_ratio)
                    if not (0.7  <= aspect_ratio <= 1.7):
                        print(area)
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
        self.end = False
        self.final = False
        self.auv = mur.mur_init()
        self.yaw_pd = PD(yaw_p, yaw_d)
        self.depth_pd = PD(depth_p, depth_d)

        self.bottom_cam = CameraBottom()
        self.front_cam = CameraFront()
        
        self.state = 0
        self.yaw = 0
        self.depth = 2
        self.speed = 0

        self.hsv_mask_min_orange = (15, 50, 50)
        self.hsv_mask_max_orange = (30, 255, 255)

        self.hsv_mask_min_yellow = (20, 50, 50)
        self.hsv_mask_max_yellow = (30, 255, 255)
        
        self.hsv_mask_min_green = (40, 40, 20)
        self.hsv_mask_max_green = (80, 255, 255)

        self.hsv_mask_min_blue = (130, 40, 20)
        self.hsv_mask_max_blue = (150, 255, 255)

        self.hsv_mask_min_red = (0, 40, 0)
        self.hsv_mask_max_red = (15, 255, 255)

    def is_depth_stable(self, val):
        return abs(self.auv.get_depth()-val) <= 1e-2

    def is_yaw_stable(self, val):
        return abs(self.auv.get_yaw()-val) <= 1.

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
        self.auv.set_motor_power(0, -15)
        self.auv.set_motor_power(1, -15)
        time.sleep(0.5)
        self.auv.set_motor_power(0, 0)
        self.auv.set_motor_power(1, 0)

    def logic(self):
        self.bottom_cam.update_img(self.auv.get_image_bottom())
        self.front_cam.update_img(self.auv.get_image_front())

        if self.state == 0:
            self.depth = 2
            (check, error) = self.bottom_cam.detect_line(self.hsv_mask_min_orange, self.hsv_mask_max_orange)
            if (check):
                # print(error)
                self.yaw = -error + self.auv.get_yaw()
                if self.bottom_cam.detect_line(self.hsv_mask_min_orange, self.hsv_mask_max_orange)[0] and \
                abs(self.bottom_cam.detect_line(self.hsv_mask_min_orange, self.hsv_mask_max_orange)[1]) < 1e-3:
                    self.state += 1
        elif self.state == 1:
            self.depth = 2
            self.speed = 30
            (check, rect_x, rect_y) = self.bottom_cam.detect_rect()
            # print(self.speed)
            if check:
                # self.stop_yaw()
                self.speed = 0
                if 155 <= rect_x <= 165 and 115 <= rect_y <= 125:
                    self.stop_yaw()
                    self.speed = 0
                    if self.final:
                        self.state = 8
                    else:
                        self.state += 1
                else:
                    try:
                        error = math.atan((rect_y-120)/(160-rect_x))/math.pi*180
                    except:
                        error = -90
                    error -= 90
                    if error < -90:
                        error = error + 180
                    # print(error)
                    if abs(error) < 2.0:
                        self.speed = 30
                    self.yaw = -error + self.auv.get_yaw()
        elif self.state == 2:
            self.depth = 2
            (check, error) = self.bottom_cam.detect_line(self.hsv_mask_min_yellow, self.hsv_mask_max_yellow)
            if (check):
                # print(error)
                self.yaw = -error + self.auv.get_yaw()
                if self.bottom_cam.detect_line(self.hsv_mask_min_yellow, self.hsv_mask_max_yellow)[0] and \
                abs(self.bottom_cam.detect_line(self.hsv_mask_min_yellow, self.hsv_mask_max_yellow)[1]) < 1e-3:
                    self.state += 1
        elif self.state == 3:
            self.depth = 3
            if self.is_depth_stable(3):
                self.yaw = self.auv.get_yaw() - 90
                self.state += 1
        elif self.state == 4:
            min_area = 1e6
            min_area_index = 0
            for i in range(3):
                while not self.is_yaw_stable(self.yaw):
                    self.keep_yaw(self.yaw, 0)
                    self.keep_depth(self.depth)
                    time.sleep(0.03)
                self.front_cam.update_img(self.auv.get_image_front())
                area = self.front_cam.plate_area([self.hsv_mask_min_green, self.hsv_mask_min_red, self.hsv_mask_min_blue], \
                [self.hsv_mask_max_green, self.hsv_mask_max_red, self.hsv_mask_max_blue])
                print(area)
                if area < min_area:
                    min_area = area
                    min_area_index = i
                self.yaw = self.auv.get_yaw() + 90
            print(min_area, min_area_index)
            self.yaw = self.auv.get_yaw() - (180-90*min_area_index)
            # self.auv.shoot()
            self.state += 1
        elif self.state == 5:
            min_masks = [self.hsv_mask_min_green, self.hsv_mask_min_blue, self.hsv_mask_min_red]
            max_masks = [self.hsv_mask_max_green, self.hsv_mask_max_blue, self.hsv_mask_max_red]
            (check, circle_x, circle_y, area) = self.front_cam.find_circle(min_masks, max_masks)
            if check:
                self.yaw = self.auv.get_yaw() - (160 - circle_x)*0.2
                self.depth = self.auv.get_depth() - (120 - circle_y)*1e-2
                if abs(circle_y - 120)*0.2 < 1. and abs(circle_x - 160)*1e-2 < 1.:
                    self.state += 1
            else:
                print('Where is the circle!?')
        elif self.state == 6:
            min_masks = [self.hsv_mask_min_green, self.hsv_mask_min_blue, self.hsv_mask_min_red]
            max_masks = [self.hsv_mask_max_green, self.hsv_mask_max_blue, self.hsv_mask_max_red]
            (check, x, y, area) = self.front_cam.find_circle(min_masks, max_masks)
            while area < 8000:
                print(area)
                self.front_cam.update_img(self.auv.get_image_front())
                (check, x, y, area) = self.front_cam.find_circle(min_masks, max_masks)
                if not check:
                    print('Where is the circle!?')
                self.keep_yaw(self.yaw, 30)
                self.keep_depth(self.depth)
                time.sleep(0.03)
            self.stop_yaw()
            t0 = time.time()
            while time.time() - t0 < 1:
                self.auv.shoot()
                self.keep_depth(self.depth)
                time.sleep(0.03)
            self.state += 1
        elif self.state == 7:
            self.yaw = self.auv.get_yaw() - 180
            while not self.is_yaw_stable(self.yaw):
                self.keep_depth(self.depth)
                self.keep_yaw(self.yaw, 0)
                time.sleep(0.03)
            self.speed = 30
            self.final = True
            self.state = 1
        elif self.state == 8:
            self.depth = 0
            self.speed = 5
            self.state += 1
        elif self.state == 9:
            if self.auv.get_depth() < 0.4:
                self.end = True
                self.auv.set_motor_power(0, 0)
                self.auv.set_motor_power(1, 0)
                self.auv.set_motor_power(2, 0)
                self.auv.set_motor_power(3, 0)
                return
        # self.bottom_cam.show()
        self.keep_depth(self.depth)
        self.keep_yaw(self.yaw, self.speed)

KP_YAW = 0.5
KD_YAW = 40

KP_DEPTH = 40
KD_DEPTH = 100

robot = Robot(KP_YAW, KD_YAW, KP_DEPTH, KD_DEPTH)

while not robot.is_depth_stable(2):
    robot.keep_depth(2)
    time.sleep(0.03)
while not robot.end:
    # robot.keep_depth(2)
    # robot.keep_yaw(50, 0)
    robot.logic()
    time.sleep(0.03)
