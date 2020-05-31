
import math

import cv2


class Shape_detection:
    def __init__(self, coords):
        self.coords = coords
        ang_num = len(self.coords)
        if ang_num == 3:
            self.name = "Triangle"
        elif ang_num == 4:
            self.name = "Rectangle"
        elif ang_num == 5:
            self.name = "Pentagon"
        else:
            self.name = "Circle"
        self.get_s()
        self.get_center()
        self.get_convex()

    def get_s(self):
        sum1 = 0
        sum2 = 0
        if len(self.coords) >= 3:
            for i in range(len(self.coords)):
                sum1 += self.coords[i - 1][0][0] * self.coords[i][0][1]
                sum2 += self.coords[i - 1][0][1] * self.coords[i][0][0]
            self.s = 0.5 * abs(sum2 - sum1)
        else:
            self.s = 0

    def get_convex(self):
        i, j, k = 0, 0, 0
        flag = 0
        z = 0
        n = len(self.coords)
        if n < 3:
            return 0

        for i in range(n):
            j = (i + 1) % n
            k = (i + 2) % n
            z = (self.coords[j][0][0] - self.coords[i][0][0]) * (self.coords[k][0][1] - self.coords[j][0][1])
            z -= (self.coords[j][0][1] - self.coords[i][0][1]) * (self.coords[k][0][0] - self.coords[j][0][0])
            if z < 0:
                flag |= 1
            elif z > 0:
                flag |= 2
            if flag == 3:
                self.convex = False
                return
        if flag != 0:
            self.convex = True
        else:
            self.convex = False

    def get_center(self):
        sum_x = 0
        sum_y = 0
        for i in range(len(self.coords)):
            sum_x += self.coords[i][0][0]
            sum_y += self.coords[i][0][1]
        self.center = (sum_x // len(self.coords), sum_y // len(self.coords))

    def draw(self, frame):
        cv2.drawContours(frame, [self.coords], 0, (0, 255, 0), 5)
        cv2.putText(frame, self.name, (self.center[0], self.center[1]), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0))
        cv2.circle(frame, (self.center[0], self.center[1]), 5, (0, 255, 0))

    def get_radius(self):
        max_len = 0
        for i in range(len(self.coords)):
            length = (self.center[0] - self.coords[i][0][0]) ** 2 + (self.center[1] - self.coords[i][0][1]) ** 2
            if length > max_len:
                max_len = length
                self.longest_rad = math.sqrt(max_len)

        min_len = (self.center[0] - self.coords[0][0][0]) ** 2 + (self.center[1] - self.coords[0][0][1]) ** 2
        self.shirtest_rad = min_len
        for i in range(len(self.coords)):
            length = (self.center[0] - self.coords[i][0][0]) ** 2 + (self.center[1] - self.coords[i][0][1]) ** 2
            if length < min_len:
                min_len = length
                self.shirtest_rad = math.sqrt(min_len)

    def get_long_side(self):
        max_len = 0
        for i in range(len(self.coords)):
            length = (self.coords[i][0][0] - self.coords[i - 1][0][0]) ** 2 + \
                     (self.coords[i][0][1] - self.coords[i - 1][0][1]) ** 2
            if length > max_len:
                max_len = length
                self.longest = [[self.coords[i][0][0], self.coords[i][0][1]],
                                [self.coords[i - 1][0][0], self.coords[i - 1][0][1]]]
        return self.longest
