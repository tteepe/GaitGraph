import os
import csv

import cv2
import numpy as np

from visualization.utils import chunhua_style as color_style
from visualization.utils import map_joint_dict

# xxx: subject id, from 001 to 124.
# mm: walking status, can be 'nm' (normal), 'cl' (in a coat) or 'bg' (with a bag).
# nn: sequence number.
# ttt: view angle, can be '000', '018', ..., '180'.
subject_id = '004'
walking_status = 'bg'
sequence_number = '01'
sequence = f"{subject_id}-{walking_status}-{sequence_number}"
base_path = '../../../datasets/CASIA/casia-b/all_frames'

angels = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']

frame_data = dict()
max_frame_num = {a: 0 for a in angels}
with open('../../data/casia-b_pose_coco.csv') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        _, sequence_id, frame = row[0].split('/')
        frame_num = int(frame[:-4])
        if sequence_id.startswith(sequence):
            angle = sequence_id.split('-')[-1]
            data = np.array(row[1:], dtype=np.float32).reshape((-1,3))
            frame_data[f"{angle}-{frame_num}"] = data

            if frame_num > max_frame_num[angle]:
                max_frame_num[angle] = frame_num

output_size = (720, 1280)
fps = 5
# filename = f"output/{sequence}.mp4"
# out = cv2.VideoWriter(f"output/{sequence}.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 25, output_size)
out = cv2.VideoWriter('../../output/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1280,720))

for i in range(1, max(max_frame_num.values())-1):
    frame = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)

    for j in range(len(angels)):
        angle = angels[j]
        if i > max_frame_num[angle]:
            continue

        data = None
        key = f"{angle}-{i}"
        if key in frame_data:
            data = frame_data[key]

        img = cv2.imread(os.path.join(base_path, f"{sequence}-{angle}", f"{i:06d}.jpg"))
        height, width, _ = img.shape

        if data is not None:
            joints_dict = map_joint_dict(data)
            for k, link_pair in enumerate(color_style.link_pairs):
                cv2.line(
                    img,
                    (joints_dict[link_pair[0]][0], joints_dict[link_pair[0]][1]),
                    (joints_dict[link_pair[1]][0], joints_dict[link_pair[1]][1]),
                    color=np.array(link_pair[2]) * 255
                )

        pos_x = j % 4
        pos_y = j // 4
        frame[pos_y*height:(pos_y+1)*height, pos_x*width:(pos_x+1)*width] = img

    cv2.imshow('frame', frame)
    out.write(frame)
    k = cv2.waitKey(10)
    if k == 27:
        break

out.release()
cv2.destroyAllWindows()
