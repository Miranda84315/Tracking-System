import numpy as np
import dlib
import math
import os
import cv2


detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor()

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
start_sequence = 127720
end_sequence = 187540
height = 1080
width = 1920


def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 10):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def get_img(iCam, frame):
    part_cam, part_frame = calucate_part(iCam, frame)
    filename = 'D:/Code/DukeMTMC/videos/camera' + str(iCam) + '/0000' + str(part_cam) + '.MTS'
    cap = cv2.VideoCapture(filename)
    cap.set(1, part_frame)
    _, img = cap.read()
    return img


def get_facepoint(img):
    img2 = img
    faces = detector(img, 1)

    if len(faces) > 0:
        for k, d in enumerate(faces):
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
    return img2


def main():
    icam = 1
    startFrame = 122178
    endFrame = 122228

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
        img = get_facepoint(frame_img)
        cv2.imshow("video", img)
        cv2.waitKey(1)
        print(frame_num)
    cap.release()
    cv2.destroyAllWindows()
    #img = get_img(iCam, frame)


if __name__ == '__main__':
    main()
