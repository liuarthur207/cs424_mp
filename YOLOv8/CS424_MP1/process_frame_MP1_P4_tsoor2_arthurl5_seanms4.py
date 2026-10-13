# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

from IoTObjectDetection.IoTObjectDetectionModuleHelperFunctions import *
from IoTObjectDetection.TaskEntity import *


# read the input bounding box data from file
box_info = read_json_file('../../dataset/waymo_ground_truth_flat.json')

# array of processed boxes where [0] = bounding and [1] = frame_count when processed
processed_boxes = []
frame_count = 0

def process_frame(frame):
    global processed_boxes
    global frame_count

    """Process frame for scheduling.

    Process a image frame to obtain cluster boxes and corresponding scheduling parameters
    for scheduling. 

    Student's code here.

    Args:
        param1: The image frame to be processed. 

    Returns:
        A list of tasks with each task containing image_path and other necessary information. 
    """

    cluster_boxes_data = get_bbox_info(frame, box_info)
    
    # student's code here
    task_list = []
    
    for bounding in cluster_boxes_data:
        duplicate = False

        for processed_box in processed_boxes:
            processed_bounding = processed_box[0]

            position_change = (
                abs(bounding[0] - processed_bounding[0]) +
                abs(bounding[1] - processed_bounding[1]) +
                abs(bounding[2] - processed_bounding[2]) +
                abs(bounding[3] - processed_bounding[3])
            )

            depth_change = abs(bounding[4] - processed_bounding[4])

            if position_change < 100 and depth_change < 5:
                duplicate = True
                break

        if duplicate:
            continue

        task = TaskEntity(
            frame.path,
            coord=bounding[:4],
            priority=int(bounding[4] * 1000),
            depth=bounding[4]
        )
        task_list.append(task)

        processed_boxes.append([bounding, frame_count])

    frame_count = frame_count + 1

    # clears processed boxes from more then 10 frames ago from global processed_boxes using the recent_processed_boxes temp
    recent_processed_boxes = []

    for processed_box in processed_boxes:
        if frame_count - processed_box[1] < 10:
            recent_processed_boxes.append(processed_box)

    processed_boxes = recent_processed_boxes

    return task_list