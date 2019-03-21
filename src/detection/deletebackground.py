import numpy as np
import cv2
import scipy.io
import h5py

start_time = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
NumFrames = [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220]
PartFrames = [[38370, 38370, 38400, 38670, 38370, 38400, 38790, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38640, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38460, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38610, 38370],
              [38370, 38370, 38370, 38670, 38370, 38400, 38760, 38370],
              [38370, 38370, 38370, 38700, 38370, 38400, 38760, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38790, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38490, 38370],
              [38370, 38370, 38370, 38670, 38370, 37350, 28380, 38370],
              [14250, 15390, 10020, 26790, 21060, 0, 0, 7890]]
start_sequence = 127720
end_sequence = 187540

openpose_root = 'D:/Code/TrackingSystem/dataset/detections/openpose_bb/camera'
background_root = 'D:/Code/TrackingSystem/dataset/background/bg_cam'

def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 10):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def cal_localtime(icam, frame_num):
    # get the real locat time
    return frame_num - start_time[icam - 1] + 1


def load_mat(icam):
    load_file = openpose_root + str(icam) + '.mat'
    detection = scipy.io.loadmat(load_file)
    data = detection['detections']
    return data


def get_bg(icam):
    img = cv2.imread(background_root + str(icam) + '.jpg')
    return img


def get_detection(icam, frame, left, top, right, bottom):
    part_cam, part_frame = calucate_part(icam, frame-1)
    filename = 'D:/Code/DukeMTMC/videos/camera' + str(icam) + '/0000' + str(part_cam) + '.MTS'
    cap = cv2.VideoCapture(filename)
    cap.set(1, part_frame)
    ret, img = cap.read()
    img = img[top:bottom, left:right]
    return img


def main():
    icam = 1
    detections = load_mat(icam)

    bg = get_bg(icam)
    for k in range(0, len(detections)):
        frame = int(detections[k, 1])
        left = int(detections[k, 2])
        top = int(detections[k, 3])
        right = int(detections[k, 2] + detections[k, 4])
        bottom = int(detections[k, 3] + detections[k, 5])
        detection_bg = bg[top:bottom, left:right]
        detection_img = get_detection(icam, frame, left, top, right, bottom)

        cv2.imshow("video", detection_bg)
        cv2.waitKey(1)


cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
