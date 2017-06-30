from collections import deque
import math
import numpy as np
from scipy.ndimage.measurements import label

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap

def convert_to_bboxes(labels):
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    return bboxes

def make_heatmap(img_shape, bboxes, probs):
    heatmap = np.zeros(img_shape)
    for bbox, prob in zip(bboxes, probs):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        heatmap[ymin : ymax, xmin : xmax] += prob

    labels = label(heatmap)
    label_boxes = convert_to_bboxes(labels)

    return heatmap, label_boxes

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
     # Make a copy of the image
     imcopy = np.copy(img)
     # Iterate through the bounding boxes
     for bbox in bboxes:
         # Draw a rectangle given bbox coordinates
         cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
     # Return the image copy with boxes drawn
     return imcopy

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_distance(self, other):
        return math.sqrt((self.x - other.x) ** 2
                       + (self.y - other.y) ** 2)

def get_center(a, b):
    return Point((a.x + b.x) // 2, (a.y + b.y) // 2)

class Box:
    def __init__(self, tuple):
        self.top_left = Point(tuple[0][0], tuple[0][1])
        self.bottom_right = Point(tuple[1][0], tuple[1][1])
        self.center = get_center(self.top_left, self.bottom_right)

    def get_area(self):
        width = self.bottom_right.x - self.top_left.x
        height = self.bottom_right.y - self.top_left.y
        return width * height

    def get_overlap_area(self, other):
        x_overlap = max(0, min(self.bottom_right.x, other.bottom_right.x) - max(self.top_left.x, other.top_left.x))
        y_overlap = max(0, min(self.bottom_right.y, other.bottom_right.y) - max(self.top_left.y, other.top_left.y))
        return x_overlap * y_overlap

    def as_tuple(self):
        return ((self.top_left.x, self.top_left.y), (self.bottom_right.x, self.bottom_right.y))

    def get_center_distance(self, other):
        return self.center.get_distance(other.center)

class Vehicle:
    def __init__(self, box, window_size):
        self.box = box
        self.window_size = window_size

        self.boxes = deque()
        self.boxes.appendleft(box)

        self.frames_since_detected = 0

        self.frames_detected = 1

    def check_ownership(self, boxes):
        claimed = []
        for box in boxes:
            claimed.append(self.check_ownership_single(box))
        for c in claimed:
            if c:
                self.frames_since_detected = 0
                self.frames_detected += 1
                return claimed
        self.frames_since_detected += 1
        if self.frames_since_detected > self.window_size:
            self.box = None
            self.frames_detected = 0
        return claimed

    def check_ownership_single(self, other):
        if (self.box.get_overlap_area(other) > 0 or self.box.get_center_distance(other) < 20):
            self.boxes.appendleft(other)
            if len(self.boxes) > self.window_size:
                self.boxes.pop()
            self.box = other
            # self.update_box()
            return True
        else:
            return False

    def update_box(self):
        avg_box = [[0, 0], [0, 0]]
        for box in self.boxes:
            b = box.as_tuple()
            for i in [0, 1]:
                for j in [0, 1]:
                    avg_box[i][j] += b[i][j]
        for i in [0, 1]:
            for j in [0, 1]:
                avg_box[i][j] //= len(self.boxes)

        self.box = Box(((avg_box[0][0], avg_box[0][1]), (avg_box[1][0], avg_box[1][1])))

class Frame:
    def __init__(self, heatmap, label_boxes):
        self.heatmap = heatmap
        self.label_boxes = label_boxes

# For the vehicle detection project, I used these parameters:
#  - Heatmap window size: 12
#  - Vehicle window size: 7
#  - Heatmap threshold per frame: 1.7
#  - Threshold for boosting heatmap around a vehicle: 24 detected frames
#  - Multiplier for boosting heatmap: 3
#  - Threshold of frames indicating genuine box count change (for resetting vehicle-tracking): 3
class Tracker:
    def __init__(self,
                 img_shape,
                 heatmap_window_size,
                 heatmap_threshold_per_frame,
                 vehicle_window_size):
        self.img_shape = img_shape
        self.heatmap_window_size = heatmap_window_size
        self.heatmap_threshold_per_frame = heatmap_threshold_per_frame
        self.vehicle_window_size = vehicle_window_size

        self.frames = deque()
        self.smoothed_frames = deque()
        self.vehicles = []
        self.heatmap_boxes_count = 0

    def track(self, bboxes, probs):
        self.add_frame(bboxes, probs)

        heatmap, boxes = self.smooth_heatmaps()
        self.add_smoothed_frame(heatmap, boxes)

        self.check_box_change(boxes)

        self.update_vehicles(boxes)
        self.remove_vehicles()

    def draw_vehicle_boxes(self, img):
        boxes = []
        for vehicle in self.vehicles:
            if vehicle.box is not None:
                boxes.append(vehicle.box.as_tuple())

        return draw_boxes(img, boxes)

    def check_box_change(self, boxes):
        if len(boxes) != self.heatmap_boxes_count:
            i = 0
            for f in self.smoothed_frames:
                if i >= 3:
                    # Box change is genuine, so reset vehicles.
                    break
                if len(boxes) != len(f.label_boxes):
                    # Do not reset vehicles because count of heatmap boxes has not stabilized.
                    return
                i+=1
            self.reset_vehicles()
            self.heatmap_boxes_count = len(boxes)

    def reset_vehicles(self):
        self.vehicles[:] = []

    def update_vehicles(self, box_tuples):
        boxes = []
        for tuple in box_tuples:
            boxes.append(Box(tuple))

        claimed = [False] * len(boxes)

        for vehicle in self.vehicles:
            box_claimed = vehicle.check_ownership(boxes)
            for i in range(len(claimed)):
                claimed[i] = claimed[i] or box_claimed[i]

        for claimed, box in zip(claimed, boxes):
            if not claimed:
                self.vehicles.append(Vehicle(box, self.vehicle_window_size))

    def remove_vehicles(self):
        removal_list = []
        for vehicle in self.vehicles:
            if vehicle.box is None:
                removal_list.append(vehicle)
        for vehicle in removal_list:
            self.vehicles.remove(vehicle)

    def add_frame(self, bboxes, probs):
        heatmap, label_boxes = make_heatmap(self.img_shape, bboxes, probs)
        self.frames.appendleft(Frame(heatmap, label_boxes))

        if len(self.frames) > self.heatmap_window_size:
            discard_frame = self.frames.pop()

    def add_smoothed_frame(self, heatmap, boxes):
        self.smoothed_frames.appendleft(Frame(heatmap, boxes))

        if len(self.smoothed_frames) > self.heatmap_window_size:
            discard_frame = self.smoothed_frames.pop()

    def smooth_heatmaps(self):
        heatmaps = []
        for f in self.frames:
            heatmaps.append(f.heatmap)
        heatmap = np.sum(heatmaps, axis = 0)
        heatmap = self.boost_heatmap(heatmap)
        heatmap = apply_threshold(heatmap, int(self.heatmap_threshold_per_frame * len(self.frames)))
        labels = label(heatmap)
        boxes = convert_to_bboxes(labels)
        return heatmap, boxes

    def boost_heatmap(self, heatmap):
        for vehicle in self.vehicles:
            if vehicle.frames_detected > 24:
                b = vehicle.box
                heatmap[b.top_left.y : b.bottom_right.y, b.top_left.x : b.bottom_right.x] *= 3
        return heatmap

