# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import os
import argparse
import json
import numpy as np
import cv2

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Read a file line by line.')
    parser.add_argument('file_path', type=str, help='Path to the file to read')
    parser.add_argument('output_path', type=str, help='Path to where output should go')
    args = parser.parse_args()
    file_path = args.file_path
    output_path = args.output_path

    #Read json file
    edex = None
    with open(file_path, 'r') as file:
        edex = json.load(file)

    downscale_factor = 1

    size = edex[0]["cameras"][0]["intrinsics"]["size"]
    focal_length = abs(edex[0]["cameras"][0]["intrinsics"]["focal"][0])
    width, height = int(size[0]/downscale_factor), int(size[1]/downscale_factor)

    min_value = -80
    max_value = 80
    # for key, value in edex[1]["points3d"].items():
    #     min_value = min(min_value, value[0])
    #     min_value = min(min_value, value[1])
    #     min_value = min(min_value, value[2])
    #     max_value = max(max_value, value[0])
    #     max_value = max(max_value, value[1])
    #     max_value = max(max_value, value[2])

    image_size = 1000
    projection_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    second_dim = 2
    points3d = {}
    for key, value in edex[1]["points3d"].items():
        x = (value[0] - min_value) / (max_value - min_value) * image_size
        y = (value[second_dim] - min_value) / (max_value - min_value) * image_size
        points3d[int(key)] = (x, y)
        projection_image = cv2.circle(projection_image, (int(x), int(y)), 1, (0, 0, 255), -1)

    for frame_id in range(300):
    #for key, value in edex[1]["rig_positions"].items():
        value = edex[1]["rig_positions"][str(frame_id)]
        x = (value["translation"][0] - min_value) / (max_value - min_value) * image_size
        y = (value["translation"][second_dim] - min_value) / (max_value - min_value) * image_size

        projection_image = cv2.circle(projection_image, (int(x), int(y)), 1, (255, 0, 0), -1)

    # Start creating a video
    cv2.imwrite(f'{output_path}/projection.png', projection_image)

    features_video = cv2.VideoWriter(f'{output_path}/edex_analysis/features.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, (image_size,image_size))
    obs_video = cv2.VideoWriter(f'{output_path}/edex_analysis/obs.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, (width,height))
    for frame_id in range(300):
    #for frame_id, measurements in edex[1]["points2d"].items():
        # Convert key to integer
        #frame_id = int(frame_id)
        measurements = edex[1]["points2d"][str(frame_id)]
        measurements = measurements[0]

        # Create image to show point associations
        rig_u = (edex[1]["rig_positions"][str(frame_id)]["translation"][0] - min_value) / (max_value - min_value) * image_size
        rig_v = (edex[1]["rig_positions"][str(frame_id)]["translation"][2] - min_value) / (max_value - min_value) * image_size

        image = projection_image.copy()
        for measurement_id, _ in measurements.items():
            if int(measurement_id) in points3d:
                point_u, point_v = points3d[int(measurement_id)]
                #cv2.circle(image, (int(point_u), int(point_v)), 1, (0, 255, 0), -1)
                cv2.line(image, (int(point_u), int(point_v)), (int(rig_u), int(rig_v)), (0, 255, 0), 1)
            else:
                pass
                #print("Missing measurement id ", measurement_id)

        cv2.imwrite(f'{output_path}/edex_analysis/features_{frame_id}.png', image)
        features_video.write(image)

        # Create an image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for measurement_id, measurement in measurements.items():
            measurement_id = int(measurement_id)

            # Draw each measurement on the image
            x, y = -(measurement[0] * focal_length) + width/2, (measurement[1] * focal_length) + height/2
            cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)

        # Save image
        cv2.imwrite(f'{output_path}/edex_analysis/obs_{frame_id}.png', image)
        obs_video.write(image)

    features_video.release()
    obs_video.release()

main()
