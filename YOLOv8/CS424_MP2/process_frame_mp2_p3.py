# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

from IoTObjectDetection.IoTObjectDetectionModuleHelperFunctions import *
from IoTObjectDetection.TaskEntity import *

import math


# read the input bounding box data from file
box_info = read_json_file('../../dataset/depth_clustering_detection_flat.json')


def should_merge(bb1, bb2):
    bb1_coords = bb1[:4]
    bb2_coords = bb2[:4]
    bb1_depth = bb1[4]
    bb2_depth = bb2[4]

    if abs(bb1_depth - bb2_depth) > 2.0:
        return False

    center_x1 = (bb1_coords[0] + bb1_coords[2]) / 2
    center_y1 = (bb1_coords[1] + bb1_coords[3]) / 2

    center_x2 = (bb2_coords[0] + bb2_coords[2]) / 2
    center_y2 = (bb2_coords[1] + bb2_coords[3]) / 2

    distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5

    area_1 = (bb1_coords[2] - bb1_coords[0]) * (bb1_coords[3] - bb1_coords[1])
    area_2 = (bb2_coords[2] - bb2_coords[0]) * (bb2_coords[3] - bb2_coords[1])

    merge_threshold = 0.5 * (area_1 ** 0.5 + area_2 ** 0.5) #Adjust parameters as needed

    if distance <= merge_threshold:
        return True
    
    return False


def find(u, parent):
    if parent[u] != u:
        parent[u] = find(parent[u], parent)
    return parent[u]


def union(u, v, parent, size):
    urep = find(u, parent)
    vrep = find(v, parent)

    if urep == vrep:
        return False

    if size[urep] > size[vrep]:
        size[urep] += size[vrep]
        parent[vrep] = urep
    else:
        size[vrep] += size[urep]
        parent[urep] = vrep

    return True


def merge_cluster_boxes(cluster_boxes_data):
    """Merge cluster boxes using Union-Find.
    """

    n = len(cluster_boxes_data)
    parent = list(range(n))
    size = [1] * n

    merged_boxes_data = []

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(cluster_boxes_data[i], cluster_boxes_data[j]):
                union(i, j, parent, size)

    grouped_boxes = {}

    for i in range(n):
        root = find(i, parent)
        if root not in grouped_boxes:
            grouped_boxes[root] = []

        grouped_boxes[root].append(cluster_boxes_data[i])

    for group in grouped_boxes.values():
        if len(group) == 1:
            merged_boxes_data.append(group[0])
        else:
            merged_box = [float('inf'), float('inf'), float('-inf'), float('-inf'), 0]
            
            for box in group:
                merged_box[0] = min(merged_box[0], box[0])
                merged_box[1] = min(merged_box[1], box[1])
                merged_box[2] = max(merged_box[2], box[2])
                merged_box[3] = max(merged_box[3], box[3])
                merged_box[4] += box[4]

            merged_box[4] /= len(group) #Average depths

            #Add padding
            padding = 0.05

            width = merged_box[2] - merged_box[0]
            height = merged_box[3] - merged_box[1]

            merged_box[0] = max(0, math.floor(merged_box[0] - width * padding))
            merged_box[1] = max(0, math.floor(merged_box[1] - height * padding))
            merged_box[2] = min(1920, math.ceil(merged_box[2] + width * padding))
            merged_box[3] = min(1280, math.ceil(merged_box[3] + height * padding))

            merged_boxes_data.append(merged_box)

    return merged_boxes_data


def process_frame(frame):
    """Process frame for scheduling.

    Process a image frame to obtain cluster boxes and corresponding scheduling parameters
    for scheduling. 

    Args:
        param1: The image frame to be processed. 

    Returns:
        A list of tasks with each task containing image_path and other necessary information. 
    """

    cluster_boxes_data = get_cluster_box_info(frame, box_info)

    task_batches = []

    # student's code here
    merged_boxes_data = merge_cluster_boxes(cluster_boxes_data)

    for bb in merged_boxes_data:
        temp = TaskEntity(
            frame.path, 
            coord=bb[:4], #this is the coords and depth
            depth=bb[4]
        )
        
        hxw = temp.img_height * temp.img_width

        depth_group = max(0, int(bb[4] // 10))
        priority = depth_group * (1920 * 1280 + 1) + hxw
        
        btemp = TaskBatch(
            [temp],
            temp.img_height,
            temp.img_width,
            priority= priority
        )
        
        task_batches.append(btemp)
    
    


    return task_batches