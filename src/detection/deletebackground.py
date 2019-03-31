import numpy as np
import cv2
import scipy.io
import h5py
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

openpose_root = 'D:/Code/TrackingSystem/dataset/detections/openpose_bb/camera'
background_root = 'D:/Code/TrackingSystem/dataset/background/bg_cam'
save_root = 'D:/Code/TrackingSystem/dataset/detections/openpose_bg/camera'
score_save = 'D:/Code/TrackingSystem/dataset/detections/openpose_bg/score'


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


def similarity_img(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return np.sum(np.absolute(img1 - img2) < 30) / (len(img1[0])*len(img1)) / 255


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang*(180/np.pi/2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2', bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    cv2.imshow('remap', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return res


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
    icam = 5
    detections = load_mat(icam)
    bg = get_bg(icam)
    #detections = np.hstack((detections, np.ones((len(detections), 2))))
    for k in range(1570000, len(detections)):
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
                    detection_bg = detection_bg[10:height-20, 10:width-20]
                    detection_img = detection_img[10:height-20, 10:width-20]
                    result = result[10:height-20, 10:width-20]
                    #allImg = np.concatenate((detection_bg, detection_img, result), axis=1)
                    #cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
                    #cv2.imshow('Result', allImg)
                    #cv2.waitKey(0)
                    detections[k, 7] = mse(detection_img, result)
                    if mse(detection_img, result) < 5000:
                        detections[k, 6] = 0
        if k % 1000 == 0:
            scipy.io.savemat(save_root + str(icam) + '.mat', mdict={'detections': detections})
    scipy.io.savemat(save_root + str(icam) + '.mat', mdict={'detections': detections})


cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

'''
            #img1 = cv2.cvtColor(detection_bg, cv2.COLOR_BGR2GRAY)
            #img2 = cv2.cvtColor(detection_img, cv2.COLOR_BGR2GRAY)
            #flow = cv2.calcOpticalFlowFarneback(img2, img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #hsv = draw_hsv(flow)
            #result = warp_flow(img2, flow)
            #score.append(mse(img1, img2))
            #img_diff = cv2.absdiff(detection_bg, detection_img)
            #cv2.imshow('frame1', img1)
            #cv2.imshow('frame2', img2)
            #cv2.imshow('diff', img_diff)
            #cv2.waitKey(0)
'''