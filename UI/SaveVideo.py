import cv2
import numpy as np
import scipy.io
import os.path
from heatmappy import Heatmapper
from PIL import Image
'''
This is use for save tracking result video
And save in 
video/camera1_result.avi
video/camera2_result.avi
...
video/camera8_result.avi
'''

start_time = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
NumFrames = [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220]
PartFrames = [[38370, 38370, 38400, 38670, 38370, 38400, 38790, 38370], [
    38370, 38370, 38370, 38670, 38370, 38370, 38640, 38370
], [38370, 38370, 38370, 38670, 38370, 38370, 38460,
    38370], [38370, 38370, 38370, 38670, 38370, 38370, 38610, 38370], [
        38370, 38370, 38370, 38670, 38370, 38400, 38760, 38370
    ], [38370, 38370, 38370, 38700, 38370, 38400, 38760,
        38370], [38370, 38370, 38370, 38670, 38370, 38370, 38790, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38490, 38370],
              [38370, 38370, 38370, 38670, 38370, 37350, 28380,
               38370], [14250, 15390, 10020, 26790, 21060, 0, 0, 7890]]


def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 10):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def load_mat():
    trajectory = scipy.io.loadmat(
        'D:/Code/DeepCC/DeepCC/experiments/demo/L3-identities/L3Final_trajectories.mat'
    )
    data = trajectory['trackerOutputL3']
    return data


def simple_data(all_data):
    # store the np array foam data
    # [:, 0]: id / [:, 1]: start time in camera1 / ... / [:, 8]: start time in carera8
    # because it will cost many time
    # so i save the nparray to id_data.npy
    if os.path.isfile('data/id_data.npy'):
        id_data = np.load('data/id_data.npy')
        return id_data
    else:
        total_id = np.unique(all_data[:, 2])
        id_data = np.zeros((len(total_id), 9))
        for id_num in total_id:
            print(id_num)
            id_data[int(id_num) - 1, 0] = id_num
            data_new = [
                i for i in range(len(data)) if int(data[i, 2]) == id_num
            ]
            icam_check = np.unique(data[data_new, 0])
            for icam in range(1, 9):
                if icam in icam_check:
                    id_data[int(id_num) - 1, int(icam)] = np.min(
                        [data[i, 1] for i in data_new if data[i, 0] == icam])
        np.save('/data/id_data.npy', id_data)
        return id_data


def random_color(number_people):
    color = np.zeros((number_people + 1, 3))
    for i in range(0, number_people + 1):
        color[i] = list(np.random.choice(range(256), size=3))
    return color


data = load_mat()
global data_part
color = random_color(len(set(data[:, 2])))
id_data = simple_data(data)


def show_video(icam, startFrame, endFrame):
    #   only show video
    part_cam, part_frame = calucate_part(icam, startFrame)
    filename = 'D:/Code/DukeMTMC/videos/camera' + str(icam) + '/0000' + str(
        part_cam) + '.MTS'
    cap = cv2.VideoCapture(filename)
    part_cam_previous = part_cam
    cap.set(1, part_frame)
    for frame_num in range(startFrame, endFrame):
        part_cam, part_frame = calucate_part(icam, frame_num)
        if part_cam != part_cam_previous:
            filename = 'D:/Code/DukeMTMC/videos/camera' + str(
                icam) + '/0000' + str(part_cam) + '.MTS'
            cap = cv2.VideoCapture(filename)
            part_cam_previous = part_cam
        ret, frame_img = cap.read()
        frame_img = draw_bb(icam, frame_num, frame_img, startFrame)
        frame_img = cv2.resize(frame_img, (640, 360))
        cv2.imshow("video", frame_img)
        cv2.waitKey(1)
        print(str(frame_num) + 'no')
    cap.release()
    cv2.destroyAllWindows()


def find_index(icam, frame, startFrame):
    window_size_bb = 80
    window_size_heatmap = 8000
    find_ind_heat = [
        i for i in range(len(data_part))
        if data_part[i][1] <= frame and data_part[i][1] >= frame -
        window_size_heatmap and data_part[i][1] >= startFrame
    ]
    find_ind_bb = [
        i for i in find_ind_heat if data_part[i][1] >= frame - window_size_bb
    ]
    find_ind = [i for i in find_ind_bb if data_part[i][1] == frame]
    return find_ind_bb, find_ind_heat, find_ind, len(find_ind)


def draw_bb(icam, frame, img, startFrame, find_ind):
    # draw the bounding box
    for i in find_ind:
        color_id = tuple(color[int(data_part[i][2])])
        left_x = int(data_part[i][3])
        left_y = int(data_part[i][4])
        right_x = int(data_part[i][3] + data_part[i][5])
        right_y = int(data_part[i][4] + data_part[i][6])
        if data_part[i][1] == frame:
            id_num = int(data_part[i][2])
            duaration_s = int(
                int(data_part[i][1] - id_data[id_num - 1, icam]) / 60)
            label_text = str(int(data_part[i][2]))
            duaration_text = str(duaration_s)
            cv2.rectangle(img, (left_x, left_y), (right_x, right_y), color_id,
                          3)
            cv2.rectangle(img, (left_x - 3, left_y - 60),
                          (right_x + 3, left_y), color_id, -1)
            cv2.putText(img, label_text, (left_x, left_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, duaration_text, (left_x, left_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.circle(img, (int(data_part[i][3] + data_part[i][5] / 2), right_y),
                   7, color_id, -1)
    return img


def worldTomap(point_x, point_y):
    # get the map point
    image_points = np.array([[307.4323, 469.2366], [485.2483, 708.9507]])
    world_points = np.array([[0, 0], [24.955, 32.85]])
    diff = image_points[1] - image_points[0]
    scale = diff / world_points[1]
    trans = image_points[0]
    map_x = int(point_x * scale[0] + trans[0])
    map_y = int(point_y * scale[1] + trans[1])
    return map_x, map_y


def draw_traj(icam, frame, find_ind):
    # draw the 2d location in the map
    img = cv2.imread('D:/Code/DeepCC/DeepCC/src/visualization/data/map.jpg')
    for i in find_ind:
        color_id = tuple(color[int(data_part[i][2])])
        px, py = worldTomap(int(data_part[i][7]), int(data_part[i][8]))
        cv2.circle(img, (px, py), 7, color_id, -1)
    return img


def cal_heatmap(icam, frame, startFrame, find_ind):
    # draw the image for heatmap
    heatmap_value = []
    path = 'D:/Code/DeepCC/DeepCC/src/visualization/data/background' + str(
        icam) + '.jpg'
    background_img = Image.open(path)
    for i in find_ind:
        center_x = int(data_part[i][3] + (data_part[i][5] / 2))
        center_y = int(data_part[i][4] + (data_part[i][6] / 2))
        heatmap_value.append((center_x, center_y))
    heatmapper = Heatmapper()
    heatmap = heatmapper.heatmap_on_img(heatmap_value, background_img)
    img = cv2.cvtColor(np.asarray(heatmap), cv2.COLOR_RGB2BGR)
    return img


def cal_localtime(icam, frame_num):
    # get the real locat time
    start_sequence = 127720
    return start_sequence + frame_num - start_time[icam - 1] + 1


def main():
    startFrame_global = 0
    endFrame_global = 59820

    for icam in range(1, 9):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_filename = 'video/camera' + str(icam) + '_result.avi'
        out = cv2.VideoWriter(out_filename, fourcc, 60, (1018, 750))

        part_cam_previous = -1
        startFrame = cal_localtime(icam, startFrame_global)
        endFrame = cal_localtime(icam, endFrame_global)

        global data_part
        data_part = [
            data[i, :] for i in range(len(data)) if data[i, 0] == icam
            and data[i, 1] >= startFrame and data[i, 1] <= endFrame
        ]

        for current_frame in range(startFrame, endFrame):
            part_cam, part_frame = calucate_part(icam, current_frame)
            if current_frame == startFrame:
                filename = 'D:/Code/DukeMTMC/videos/camera' + str(
                    icam) + '/0000' + str(part_cam) + '.MTS'
                cap = cv2.VideoCapture(filename)
                cap.set(1, part_frame)
                part_cam_previous = part_cam
            if part_cam != part_cam_previous:
                filename = 'D:/Code/DukeMTMC/videos/camera' + str(
                    icam) + '/0000' + str(part_cam) + '.MTS'
                part_cam_previous = part_cam
                cap = cv2.VideoCapture(filename)
            ret, frame_img = cap.read()

            find_ind_bb, find_ind_heat, find_ind, num_visitor = find_index(
                icam, current_frame, startFrame)

            # get the bounding box and put to image box 1
            frame_img = draw_bb(icam, current_frame, frame_img, startFrame,
                                find_ind_bb)
            frame_img = cv2.resize(frame_img, (640, 360))
            frame_img = cv2.copyMakeBorder(
                frame_img,
                10,
                10,
                10,
                10,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255])

            img_traj = draw_traj(icam, current_frame, find_ind)
            img_traj = cv2.resize(img_traj, (348, 730))
            img_traj = cv2.copyMakeBorder(
                img_traj,
                10,
                10,
                0,
                10,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255])

            img_heat = cal_heatmap(icam, current_frame, startFrame,
                                   find_ind_heat)
            img_heat = cv2.resize(img_heat, (640, 360))
            img_heat = cv2.copyMakeBorder(
                img_heat,
                0,
                10,
                10,
                10,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255])

            img_left = np.concatenate((frame_img, img_heat), axis=0)
            img = np.concatenate((img_left, img_traj), axis=1)

            cv2.imshow("video", img)
            cv2.waitKey(1)
            print('icam = ' + str(icam))
            print('frame = ' + str(current_frame) + ' / ' + str(endFrame))
            out.write(img)

        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
