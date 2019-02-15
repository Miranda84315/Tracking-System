import numpy as np
import cv2
import scipy.io
import h5py

detection_root = ['D:/Code/TrackingSystem/dataset/detections/faster_rcnn/', 'D:/Code/TrackingSystem/dataset/detections/openpose/']
start_time = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
NumFrames = [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220]
PartFrames = [
    [38370, 38370, 38400, 38670, 38370, 38400, 38790, 38370], 
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

def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 10):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def load_detection(detection_path):
    detection = scipy.io.loadmat(detection_path)
    data = detection['detections']
    return data


def load_detection_h5py(detection_path):
    detection = h5py.File(detection_path)
    data = np.transpose(detection['detections'])
    return data


def draw_pose(pose, img):



def show_detections(detection, iCam, frame):
    part_cam, part_frame = calucate_part(iCam, frame)
    filename = 'D:/Code/DukeMTMC/videos/camera' + str(iCam) + '/0000' + str(part_cam) + '.MTS'
    cap = cv2.VideoCapture(filename)
    cap.set(1, part_frame)
    _, img = cap.read()
    # cv2.imshow("video", img)
    # cv2.waitKey(1)
    detections = [detection[i] for i in range(0, len(detection)) if detection[i][1] == frame]
    img_pose = draw_pose(detections, img)

    return img_pose


# def adaptive_detector(detection1, detection2):
def main():
    iCam = 1
    detection1_path = detection_root[0] + 'camera' + str(iCam) + '.mat'
    detection2_path = detection_root[1] + 'camera' + str(iCam) + '.mat'

    detection1 = load_detection(detection1_path)
    detection2 = load_detection_h5py(detection2_path)

    start = start_sequence - start_time[iCam - 1] + 1
    end = end_sequence - start_time[iCam - 1] + 1

    # because in faster_rcnn I only use global range(127720, 187540) 
    # but openpose is use all range(1, 359580)
    # so in there, i delete openpose's detections which is out of range
    detection2 = [detection2[i] for i in range(0, len(detection2)) if detection2[i, 1] >= start and detection2[i, 1] <= end]

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
