import cv2
import numpy as np

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


def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 10):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def get_frame(icam, frame_num):
    #   only show video
    part_cam, part_frame = calucate_part(icam, frame_num)
    filename = 'D:/Code/DukeMTMC/videos/camera' + str(icam) + '/0000' + str(
        part_cam) + '.MTS'
    cap = cv2.VideoCapture(filename)
    cap.set(1, part_frame)
    ret, frame_img = cap.read()
    cv2.imshow("video", frame_img)
    cv2.waitKey(1)
    path_name = 'D:/Code/DeepCC/DeepCC/src/visualization/data/background' + str(icam) + '.jpg'
    cv2.imwrite(path_name, frame_img)
    cap.release()


def main():
    frame = [179420, 144635, 114510, 126070, 151775, 133435, 124980, 105910]
    for i in range(1, 9):
        get_frame(i, frame[i - 1])
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


