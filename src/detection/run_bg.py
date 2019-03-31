import numpy as np
import cv2
import scipy.io
from matplotlib import pyplot as plt

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

#background_root = 'background/bg_cam'
#save_root = 'openpose_bg/camera'
background_root = 'D:/Code/TrackingSystem/dataset/background/bg_cam'
save_root = 'D:/Code/TrackingSystem/dataset/detections/openpose_bg/camera'


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
    load_file = save_root + str(icam) + '.mat'
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


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    print(err)
    return err


def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_good_match(des1, des2):
    if des1 is None or des2 is None:
        return []
    else:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        if(len(matches) % 2 is 1 or len(matches) <= 2):
            return []
        else:
            try:
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                return good
            except ValueError:
                return []


def siftImageAlignment(img1, img2):
    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1, des2)
    imgOut = 0
    flag = False
    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        if H is not None:
            imgOut = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            flag = True
    return imgOut, flag


def main():
    icam = 7
    detections = load_mat(icam)
    bg = get_bg(icam)
    for k in range(100, 200):
        print('----- detection = ', k, ' / ', len(detections))
        frame = int(detections[k, 1])
        left = int(detections[k, 2])
        top = int(detections[k, 3])
        right = int(detections[k, 2] + detections[k, 4])
        bottom = int(detections[k, 3] + detections[k, 5])
        if(detections[k, 4] < 20 or detections[k, 5] < 20 or detections[k, 5] > 450):
            detections[k, 6] = 0
        else:
            if(bottom < 1080 and right < 1920 and left >= 30 and top >= 30):
                detection_bg = bg[top:bottom, left:right]
                detection_img = get_detection(icam, frame, left, top, right, bottom)

                result, flag = siftImageAlignment(detection_img, detection_bg)
                if flag is True:
                    height = detection_img.shape[0]
                    width = detection_img.shape[1]

                    #detection_img = detection_img[10:height-20, 10:width-20]
                    #result = result[10:height-20, 10:width-20]
                    score = int(mse(detection_img, result))
                    detection_bg2 = cv2.copyMakeBorder(detection_bg, 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    detection_img2= cv2.copyMakeBorder(detection_img, 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    allImg = np.concatenate((detection_bg2, detection_img2, result), axis=1)
                    cv2.namedWindow('Result2', cv2.WINDOW_NORMAL)
                    cv2.imshow('Result2', allImg)
                    cv2.waitKey(0)
                    cv2.imwrite('result/' + str(k) + '_' + str(score) + '.jpg', allImg)

                    detection_bg = detection_bg[10:height-20, 10:width-20]
                    detection_img = detection_img[10:height-20, 10:width-20]
                    result = result[10:height-20, 10:width-20]

                    score = int(mse(detection_img, result))
                    detection_bg = cv2.copyMakeBorder(detection_bg, 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    detection_img= cv2.copyMakeBorder(detection_img, 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    allImg = np.concatenate((detection_bg, detection_img, result), axis=1)
                    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
                    cv2.imshow('Result', allImg)
                    cv2.waitKey(0)
                    cv2.imwrite('result/' + str(k) + '_after' + str(score) + '.jpg', allImg)




cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
