from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import requests

# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Load a pretrained YOLOv8n segmentation model
model = YOLO('./trained_models/v2__YOLOv8s_Time2hGPU_train630val19_640x640/weights/best.pt')

# Open the video file
video_path = "./datasets/test_trash_v1/videos/truck_mockup_v1_6load_32seconds.mp4"
cap = cv2.VideoCapture(video_path)

# Skip to frame 100
# cap.set(cv2.CAP_PROP_POS_FRAMES, 200)

# Store the track history
track_history = defaultdict(lambda: [])

# Define the font and text properties for OpenCV Text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
color = (0, 0, 255)  # color in BGR

# the framerate of the video
frames_per_second = 24

number_of_tracked_bags = 0

# list of strings to render on the screen each frame
information_to_render = []

# a list of frame_results
frame_history = [
    # [{...}] # previous frame result
    # [{...}] # previous frame result
    # [{...}] # previous frame result
]

frame_results = [
    # {...}, # 1 detected object
    # {
    #     'id': 3,
    #     'counted_this_frame': False,
    #     'box': {
    #         'xywh': {'x': x.item(), 'y': y.item(), 'w': w.item(), 'h': h.item()
    #         'xyxy': {'x1': x1.item(), 'y1': y1.item(), 'x2': x2.item(), 'y2': y2.item()},
    #         'counted_this_frame': False,
    #      },
    # }
]

'''
# GLOBAL 
- only needed for modifying global variables 
- by default, when you assign a value to a variable inside a function, Python treats it as a new local variable unless explicitly told otherwise.
'''

'''
Add to information_to_render: 
Current frame objects
    id, total pixels

frame history: size

previous frame inference time
previous frame total time
'''



def process_next_frame(results, current_frame):


    if results[0].masks == None:
        return


    # ------- PROCESS FRAME ------- #

    # add frame_results to frame_history
    store_current_frame_results()

    # create a 'frame_results' for this frame
    create_frame_results(results)

    # Run the counting algorithm on the frame and frame_history
    check_for_crossed_bags(results, current_frame)


    # ------- ON SCREEN VARIABLES ------- #
    calculate_pixels_per_instance(results)


    # ------- RENDER FRAME ------- #

    # Use Yolo to plot the bounding boxes, labels, and segmentation onto a CV frame
    YOLO_CV_frame = results[0].plot()

    # COUNTING AREA
    render_counting_area(YOLO_CV_frame, current_frame)

    # PLOT THE TRACKING LINES
    render_tracking_line(results, YOLO_CV_frame)

    # render_history_lines(YOLO_CV_frame)

    render_information(YOLO_CV_frame)

    render(YOLO_CV_frame)


def store_current_frame_results():
    # add frame_results to frame_history
    global frame_history
    global frame_results

    frame_history.append(frame_results)

    # only keep the last 24 frames (1 second)
    if len(frame_history) > 5:  # Check if there are more than 10 seconds of frames
        # frame_history = frame_history[0:4]  # only keep the last 24 frames (1 second)
        frame_history = []

def create_frame_results(results):

    # create a 'frame_results' for this frame
    global frame_results
    detected_boxes = results[0].boxes.xywh.cpu()
    detected_boxes_2 = results[0].boxes.xyxy.cpu()
    detected_tracking_ids = results[0].boxes.id.int().cpu().tolist()
    for box, box2, track_id in zip(detected_boxes, detected_boxes_2, detected_tracking_ids):
        x, y, w, h = box
        x1, y1, x2, y2 = box2
        frame = {
            'id': track_id,
            'xywh': {'x':   x.item(), 'y':  y.item(), 'w':  w.item(), 'h':  h.item()},
            'xyxy': {'x1':   x1.item(), 'y1': y1.item(), 'x2': x2.item(), 'y2': y2.item()},
            'counted_this_frame': False,
        }
        frame_results.append(frame)


def check_for_crossed_bags(results, current_frame):
    # of all bags this frame, which ones are below the line?
    objects_below_the_line = get_objects_below_the_line(current_frame)

    # of the bags below the line, which ones where above the line in the last second?
    recently_crossed_boxes = get_objects_above_the_line_in_the_last_second(objects_below_the_line, current_frame)

    # of those "recent crossed" boxes, where they counted in the last X seconds (X frames)?
    newly_crossed_boxes = get_objects_that_have_not_been_counted(recently_crossed_boxes)

    # NO: 1) get their masks
    ''' # MASKS 
    - Yolo inferences the boxes and masks separately. 
    - Therefor, when a box has crossed the threshold line, 
    - only then we must find the mask of that box and calculate the total number of pixels in that mask.'''
    boxes_with_masks = attached_segmentation_masks_to_boxes(newly_crossed_boxes, results)

    # 2) estimate their size
    boxes_with_estimated_size = estimate_bag_size(boxes_with_masks)

    # 3) count them
    send_counted_bag_to_api(boxes_with_estimated_size)

    # YES: do nothing



def get_objects_below_the_line(current_frame):

    boxes_below_the_line = []

    for box in frame_results:

        frame_height, width = current_frame.shape[:2]
        screen_half_threshold_line = frame_height // 2  # Start from the left-middle point of the frame

        # 0 = the top/left of the screen
        # 640 = the middle of the frame (the threshold line)
        # 1280 = the bottom of the screen

        if box['xywh']['y'] > screen_half_threshold_line: # e.g: 700 < 640 (just below the line)
            boxes_below_the_line.append(box)

    return boxes_below_the_line

def get_objects_above_the_line_in_the_last_second(objects_below_the_line, current_frame):
    boxes_recently_above_the_line = [] # of the boxes below the line, which ones where above the line in the last X seconds?

    for box in objects_below_the_line:

        frame_height, width = current_frame.shape[:2]
        screen_half_threshold_line = frame_height // 2  # Start from the left-middle point of the frame

        # 0 = the top/left of the screen
        # 640 = the middle of the frame (the threshold line)
        # 1280 = the bottom of the screen

        for previous_frame in frame_history:    # frame = [{...}, {...}, {...}]  |  frame_history = [[...], [...], [...]]
            for old_box in previous_frame:      # old_box = {...}
                if old_box['id'] == box['id']:  # is this previously detected box(id) the same as this current box(id)?

                    if old_box['xywh']['y'] < screen_half_threshold_line: # e.g: 600 < 640 (just above the line)
                        boxes_recently_above_the_line.append(box)

    return boxes_recently_above_the_line

def get_objects_that_have_not_been_counted(recently_crossed_boxes):
    # go through the frame_history, find this bag id, and check any of them were marked as counted_this_frame
    uncounted_boxes = []

    for new_box in recently_crossed_boxes:  # for each box that has "recently crossed the line"
        box_has_been_counted = False        # Has this box been previously counted in the last X seconds (X frames)?
        for frame in frame_history:         # frame = [{...}, {...}, {...}]  |  frame_history = [[...], [...], [...]]
            for old_box in frame:           # old_box = {...}
                if old_box['id'] == new_box['id']:  # is this previously detected box(id) the same as this current box(id)?
                    if old_box['counted_this_frame'] == True:
                        box_has_been_counted = True
        if box_has_been_counted == False:
            uncounted_boxes.append(new_box)
            new_box['counted_this_frame'] = True

    return uncounted_boxes

def attached_segmentation_masks_to_boxes(newly_crossed_boxes, results):
    # of the boxes that have not been counted, get their masks
    boxes_with_masks = []

    for box in newly_crossed_boxes:
        mask = find_segmentation_mask_for_box_id(results, box)
        box['mask'] = mask
        boxes_with_masks.append(box)

    return boxes_with_masks

def find_segmentation_mask_for_box_id(results, box):
    print('find_segmentation_mask_for_box_id')
    detected_masks = results[0].masks.data.cpu().numpy()

    highest_iou = 0
    highest_iou_mask = None

    for detected_mask in detected_masks:
        iou = compute_iou(detected_mask, box)
        if iou > highest_iou:
            highest_iou = iou
            highest_iou_mask = detected_mask

    return highest_iou_mask

def compute_iou(mask, box):
    """
    Computes the intersection over union (IOU) between a bounding box and a mask.

    Returns:
        The IOU between the bounding box and the mask
    """
    x1 = int( box['xyxy']['x1'] )
    y1 = int( box['xyxy']['y1'] )
    x2 = int( box['xyxy']['x2'] )
    y2 = int( box['xyxy']['y2'] )

    '''
        mask: 
            the mask ndarray is a 2D array of 0s and 1s, where each element is a pixel 
            1 represents a pixel that the object is in 
            and 0 represents a pixel that the object is not in
            YOLO downscales the masks. use the retina_masks flag in the .track(...) to get original resolution masks
        
        mask[col:col, row:row]
        
        mask[y1:y2, x1:x2], mask[y1:y2, x1:x2]
            of the e.g 1280x1080 mask, we only want the pixels that are inside the box
            because, then we count the pixels inside that subregion box
            to get the overlap between the Mask and the Box
    
    '''


    # Get the intersection area of the box and the mask
    intersection = cv2.bitwise_and(
        mask[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    ).sum()


    # Get the area of the box and the mask
    box_area = (x2 - x1) * (y2 - y1)
    mask_area = np.sum(mask)

    # Compute the union area
    union = box_area + mask_area - intersection

    iou = intersection / (union + 1e-10)
    return iou

def estimate_bag_size(boxes_with_masks):
    bag_sizes = [
        ('1LT', 900),
        ('2LT', 1500),
        ('5LT', 2500),
        ('10LT', 3500),
        ('25LT', 5000),
        ('50LT', 7000),
        ('75LT', 9000),
        ('100LT', 13000),
    ]

    for box in boxes_with_masks:
        total_pixels = box['mask'].sum()
        box['total_pixels'] = total_pixels

        # Iterate through bag size thresholds
        # Start at the smallest bag size (900), and work up to the largest bag size (13000)
        # If the bag(1200) is bigger than the size(900), then set it to that size
        # Stop when the size(1500) is bigger than the bag(1200)
        for bag_size, min_size in bag_sizes:
            if total_pixels >= min_size:
                box['estimated_size'] = bag_size
                break  # Stop searching once the bag is larger than the next threshold

    return boxes_with_masks

def send_counted_bag_to_api(boxes_with_estimated_size):

    for bag in boxes_with_estimated_size:
        print('------------------------ NEW BAG ------------------------')
        print(bag)
        # Create a JSON payload
        payload = {
            "size": bag['estimated_size'],
        }

        # Send the POST request
        response = requests.post("http://localhost:5000", json=payload)
        print(response)



# -------------- RENDERING FUNCTIONS -------------- #
def calculate_pixels_per_instance(results):
    global information_to_render

    total_pixels_per_instance = results[0].masks.data.cpu().numpy().sum(axis=(1, 2))
    for i, count in enumerate(total_pixels_per_instance):
        text = f"Object {i + 1}: {count} pixels"
        information_to_render.append(text)


def render_counting_area(next_annotated_frame, frame):
    # THRESHOLD LINE
    # Define start and end coordinates of the line
    height, width = frame.shape[:2]
    color = (0, 255, 0)  # Green color in BGR
    thickness = 3  # Line thickness

    start_point = (0, height // 2)  # Start from the left-middle point of the frame
    end_point = (width, height // 2)  # End at the right-middle point of the frame
    cv2.line(next_annotated_frame, start_point, end_point, color, thickness)


def render_information(next_annotated_frame):
    global information_to_render

    for i, string in enumerate(information_to_render):
        position = (10, 60 + (i * 30))
        cv2.putText(next_annotated_frame, string, position, font, font_scale, color, font_thickness)


def render_tracking_line(results, next_annotated_frame):
    # Get the boxes and track IDs
    detected_boxes = results[0].boxes.xywh.cpu()
    detected_tracking_ids = results[0].boxes.id.int().cpu().tolist()

    # Plot the tracks
    for box, track_id in zip(detected_boxes, detected_tracking_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(next_annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        # cv2.putText(next_annotated_frame, 'asdf', [points[-1]], font, font_scale, color, font_thickness)

def render_history_lines(next_annotated_frame):

    # render a circle for each box, in each frame, in the frame_history

    for frame in frame_history:
        for box in frame:
            x, y, w, h = box['xywh']['x'], box['xywh']['y'], box['xywh']['w'], box['xywh']['h']
            cv2.circle(next_annotated_frame, (int(x), int(y)), radius=5, color=(255, 255, 255), thickness=2)

def render(next_annotated_frame):
    # Display the final frame
    cv2.imshow("Woosong Vision", next_annotated_frame)
    global information_to_render
    information_to_render = []


# TODO: move the above loose functions into the below class
class WoosongVisionAIPipeline:

    def __init__(self):
        pass

    def process_next_frame(self, results, current_frame):
        pass


# -------------- MAIN -------------- #

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame,              # The frame to analyse (from OpenCV)
            persist=True,       # Persist/Track/ID objects between inferences/frames
            retina_masks=True,  # use high-resolution segmentation masks
            # device=0,       # Use GPU - CUDA GPU device id
            # vid_stride=False # what is this?
        )

        # Only process the frame if there is at least 1 detection
        if len(results) > 0:
            process_next_frame(results, frame)

    else:
        # Break the loop when the end of the video finishes playing
        break

    # Break the loop on 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
