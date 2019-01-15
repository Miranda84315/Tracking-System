import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
import PyQt5.QtGui as QtGui
from vis import Ui_Form
import scipy.io
import cv2
import numpy as np
from heatmappy import Heatmapper
from PIL import Image
import resource_rc
from PyQt5 import QtCore, QtWidgets
import os.path

# pyuic5 vis.ui > vis.py
# Pyrcc5 resource.qrc -o resource_rc.py
# python showui.py
tail_colors = [(0, 0, 1), (0, 1, 0), (0, 0, 0)]
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


def load_mat():
    trajectory = scipy.io.loadmat(
        'D:/Code/baseline/L3-identities/L3Final_trajectories.mat'
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
global data_part_id
color = random_color(len(set(data[:, 2])))
id_data = simple_data(data)


def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 10):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


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


def nparrayToQimage(frame_img):
    # let the nparray To Qimage type
    height, width, channel = frame_img.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(frame_img.data, width, height, bytesPerLine,
                        QtGui.QImage.Format_RGB888).rgbSwapped()
    pixmap = QPixmap(qImg)
    return pixmap


def cal_localtime(icam, frame_num):
    # get the real locat time
    return start_sequence + frame_num - start_time[icam - 1] + 1


def show_cam(icam):
    # show the background for id choose
    path = 'D:/Code/DeepCC/DeepCC/src/visualization/data/background' + str(
        icam) + '.jpg'
    background_img = Image.open(path)
    img = cv2.cvtColor(np.asarray(background_img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (320, 180))
    return img


def draw_bb_id(icam, frame, img):
    # draw the bounding box
    img = cv2.resize(img, (320, 180))
    ind = [i for i in range(len(data_part_id)) if data_part_id[i][1] == frame]

    if len(ind) == 0:
        return False, img
    for i in ind:
        left_x = int(data_part_id[i][3] / 6)
        left_y = int(data_part_id[i][4] / 6)
        right_x = int((data_part_id[i][3] + data_part_id[i][5]) / 6)
        right_y = int((data_part_id[i][4] + data_part_id[i][6]) / 6)
        color_id = tuple(color[int(data_part_id[i][2])])
        cv2.rectangle(img, (left_x, left_y), (right_x, right_y), color_id, 2)

    return True, img


class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # set current frame and the part of camera
        self.current_frame = None
        self.part_cam_previous = None
        self.total_visitor = None
        self.id = None
        self.cam_check = None
        self.turn_icam = None
        self.cam_check_start = None

        # initial UI
        self.setUI()
        # let the window can be move
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)

        # link th ok, start, stop, exit, min button
        self.ui.okButton.clicked.connect(self.pushButton_Click)
        self.ui.startButton.clicked.connect(self.start_video)
        self.ui.stopButton.clicked.connect(self.stop_video)
        self.ui.exit_Button.clicked.connect(self.exit_window)
        self.ui.min_Button.clicked.connect(self.minimize_window)
        self.show()

    def pushButton_Click(self):
        # print info about cam/start/end frame
        # and show the initial image to the box
        if self.ui.choose_id.text() != '':
            self.show_initial_id()
        else:
            self.show_initial_video()

    def nextFrameVideo(self):
        # because we can choose play video or select id to play
        # so if the choose_id value is not type, then go to play video
        # or play the id video
        if self.ui.choose_id.text() != '':
            self.nextFrame_id()
        else:
            self.nextFrame_video()

    def show_initial_video(self):
        # this is to initial the startFrame video
        self.icam = int(self.ui.cam_num.currentText())
        self.start_frame = int(self.ui.startFrame.currentText())
        self.end_frame = int(self.ui.endFrame.currentText())
        self.current_frame = self.start_frame

        # print information
        info_string = ' camera: ' + str(self.icam) + '\n start: ' + str(
            self.start_frame) + '\n end: ' + str(self.end_frame)
        self.ui.label_4.setText(info_string)

        # calucate the current part of camera
        filename = 'D:/Code/DukeMTMC/result_video/camera' + str(
            self.icam) + '_result.avi'
        self.cap = cv2.VideoCapture(filename)
        self.cap.set(1, self.start_frame)
        ret, frame_img = self.cap.read()

        # get the bounding box and put to image box 1
        pixmap = nparrayToQimage(frame_img)
        show_img = self.ui.videoBox
        show_img.setPixmap(pixmap)

    def show_initial_id(self):
        self.id = int(self.ui.choose_id.text())
        global data_part_id
        data_part_id = [
            data[i, :] for i in range(len(data)) if data[i, 2] == self.id
        ]
        name = [
            self.ui.cam1_box, self.ui.cam2_box, self.ui.cam3_box,
            self.ui.cam4_box, self.ui.cam5_box, self.ui.cam6_box,
            self.ui.cam7_box, self.ui.cam8_box
        ]
        self.cam_check = [
            k for k in range(1, 9) if id_data[self.id - 1, k] != 0
        ]
        for i in range(1, 9):
            if i in self.cam_check:
                start_frame = int(id_data[self.id - 1, i])
                img_temp = self.show_start_frame(i, start_frame)
            else:
                img_temp = show_cam(i)
            img = nparrayToQimage(img_temp)
            show_cam_temp = name[i - 1]
            show_cam_temp.setPixmap(img)

        self.cam_check_start = [
            id_data[self.id - 1, i] for i in self.cam_check
        ]

        self.turn_icam = self.cam_check[np.argmin(self.cam_check_start)]
        self.current_frame = int(id_data[self.id - 1, self.turn_icam])
        label_text = 'ID: ' + str(self.id) + '\nIn: ' + str(
            self.cam_check) + '\nNow: ' + str(
                self.turn_icam) + '\nFrame: ' + str(self.current_frame)
        self.ui.label_4.setText(label_text)

        part_cam, part_frame = calucate_part(self.turn_icam,
                                             self.current_frame)
        self.part_cam_previous = part_cam

        filename = 'D:/Code/DukeMTMC/videos/camera' + str(
            self.turn_icam) + '/0000' + str(part_cam) + '.MTS'
        self.cap = cv2.VideoCapture(filename)
        self.cap.set(1, part_frame)

    def nextFrame_video(self):
        # calucate the part of current frame
        # if change, then change the capture video
        # if not change, just read next frame
        ret, frame_img = self.cap.read()

        # get the bounding box and put to image box 1
        pixmap = nparrayToQimage(frame_img)
        show_img = self.ui.videoBox
        show_img.setPixmap(pixmap)

        # print information
        info_string = ' camera: ' + str(self.icam) + '\n frame: ' + str(
            self.current_frame)
        self.ui.label_4.setText(info_string)

        # change current frame, and chack if is out of frame
        self.current_frame += 1
        if self.current_frame >= self.end_frame:
            self.timer.timeout.connect(self.stop_video)

    def nextFrame_id(self):
        name = [
            self.ui.cam1_box, self.ui.cam2_box, self.ui.cam3_box,
            self.ui.cam4_box, self.ui.cam5_box, self.ui.cam6_box,
            self.ui.cam7_box, self.ui.cam8_box
        ]
        for icam in range(1, 9):
            if icam == self.turn_icam:
                img_temp = self.show_id_frame(icam, self.current_frame)
                img = nparrayToQimage(img_temp)
                show_cam_temp = name[icam - 1]
                show_cam_temp.setPixmap(img)

        self.current_frame += 1

        label_text = 'ID: ' + str(self.id) + '\nIn: ' + str(
            self.cam_check) + '\nNow: ' + str(
                self.turn_icam) + '\nFrame: ' + str(self.current_frame)
        self.ui.label_4.setText(label_text)

    def show_start_frame(self, icam, start_frame):
        # if we have choose the id, and the ok button is click
        # we need to show the id will appear in which frame of each camera
        part_cam, part_frame = calucate_part(icam, start_frame)
        filename = 'D:/Code/DukeMTMC/videos/camera' + str(
            icam) + '/0000' + str(part_cam) + '.MTS'
        cap = cv2.VideoCapture(filename)
        cap.set(1, part_frame)
        ret, frame_img = cap.read()
        check, img = draw_bb_id(icam, start_frame, frame_img)
        return img

    def show_id_frame(self, icam, current_frame):
        self.icam = icam
        part_cam, part_frame = calucate_part(self.icam, self.current_frame)
        if part_cam != self.part_cam_previous:
            filename = 'D:/Code/DukeMTMC/videos/camera' + str(
                self.icam) + '/0000' + str(part_cam) + '.MTS'
            self.part_cam_previous = part_cam
            self.cap = cv2.VideoCapture(filename)
        ret, frame_img = self.cap.read()
        check, img = draw_bb_id(self.icam, current_frame, frame_img)
        if check is False:
            print('change cam')
            temp_cam_check = np.asarray(self.cam_check)
            index = np.argwhere(temp_cam_check == self.turn_icam)
            self.cam_check = np.delete(self.cam_check, index)
            self.cam_check_start = np.delete(self.cam_check_start, index)
            if len(self.cam_check) > 0:
                self.turn_icam = self.cam_check[np.argmin(
                    self.cam_check_start)]
                print('index = ' + str(index))
                print('new cam_check:' + str(self.cam_check))
                print('new cam_check_start:' + str(self.cam_check_start))
                print('new camera:' + str(self.turn_icam))
                self.current_frame = int(id_data[self.id - 1, self.turn_icam])
                self.icam = self.turn_icam
                part_cam, part_frame = calucate_part(self.icam,
                                                     self.current_frame)
                filename = 'D:/Code/DukeMTMC/videos/camera' + str(
                    self.icam) + '/0000' + str(part_cam) + '.MTS'
                self.part_cam_previous = part_cam
                self.cap = cv2.VideoCapture(filename)
                self.cap.set(1, part_frame)
            else:
                self.timer.timeout.connect(self.stop_video)

        return img

    # -- play the video
    def start_video(self):
        self.timer = QTimer()
        self.vid_rate = 10
        self.timer.timeout.connect(self.nextFrameVideo)
        self.timer.start()

    # -- stop the video
    def stop_video(self):
        self.timer.stop()

    # -- move window when mousePress
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragPosition = event.globalPos() - self.frameGeometry(
            ).topLeft()
            QApplication.postEvent(self, QtCore.QEvent(174))
            event.accept()

    # -- move window when mouseMove
    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            self.move(event.globalPos() - self.dragPosition)
            event.accept()

    def setUI(self):
        # total widget
        self.ui.lefttop_widget.setLayout(self.ui.lefttop_Layout)
        self.ui.leftdown_widget.setLayout(self.ui.leftdown_Layout)

        # 1. left top: choose the camera, frame value
        self.ui.lefttop_Layout.addWidget(self.ui.label1, 0, 0)
        self.ui.lefttop_Layout.addWidget(self.ui.cam_num, 1, 0)

        self.ui.lefttop_Layout.addWidget(self.ui.label2, 2, 0)
        self.ui.lefttop_Layout.addWidget(self.ui.startFrame, 3, 0)

        self.ui.lefttop_Layout.addWidget(self.ui.label3, 4, 0)
        self.ui.lefttop_Layout.addWidget(self.ui.endFrame, 5, 0)

        self.ui.lefttop_Layout.addWidget(self.ui.label5, 6, 0)
        self.ui.lefttop_Layout.addWidget(self.ui.choose_id, 7, 0)

        # 2. left down: 3 button (ok, start, stop)
        self.ui.leftdown_Layout.addWidget(self.ui.okButton, 1, 0)
        self.ui.leftdown_Layout.addWidget(self.ui.startButton, 2, 0)
        self.ui.leftdown_Layout.addWidget(self.ui.stopButton, 3, 0)

        # 3. right: contain 3 image box
        self.ui.right_Layout.setSpacing(1)
        self.ui.right_widget.setLayout(self.ui.right_Layout)

        # right tab 1 contain 3 image box
        self.ui.subtab_widget.setLayout(self.ui.subtab_Layout)
        self.ui.subtab_Layout.addWidget(self.ui.videoBox, 0, 0, 1018, 750)

        # right tab 2 contain 8 image box
        self.ui.subtab2_widget.setLayout(self.ui.subtab2_Layout)
        self.ui.subtab2_Layout.addWidget(self.ui.cam1_box, 0, 0)
        self.ui.subtab2_Layout.addWidget(self.ui.cam2_box, 1, 0)
        self.ui.subtab2_Layout.addWidget(self.ui.cam3_box, 2, 0)
        self.ui.subtab2_Layout.addWidget(self.ui.cam4_box, 3, 0)
        self.ui.subtab2_Layout.addWidget(self.ui.cam5_box, 0, 1)
        self.ui.subtab2_Layout.addWidget(self.ui.cam6_box, 1, 1)
        self.ui.subtab2_Layout.addWidget(self.ui.cam7_box, 2, 1)
        self.ui.subtab2_Layout.addWidget(self.ui.cam8_box, 3, 1)

        # combine 2 tab to right_layout
        self.ui.tabWidget.addTab(self.ui.subtab_widget, 'Tracking Video')
        self.ui.tabWidget.addTab(self.ui.subtab2_widget, 'Specify ID')
        self.ui.right_Layout.addWidget(self.ui.tabWidget)
        self.ui.tabWidget.removeTab(0)
        self.ui.tabWidget.removeTab(0)

        self.ui.videoBox.setObjectName('button')

        # 4. window: contain close, minize the window
        self.ui.window_widget.setLayout(self.ui.window_Layout)
        self.ui.window_Layout.addWidget(self.ui.exit_Button, 0, 0)
        self.ui.window_Layout.addWidget(self.ui.min_Button, 0, 2)

        # 5. info: print the information
        self.ui.info_widget.setLayout(self.ui.info_Layout)
        self.ui.info_Layout.addWidget(self.ui.label_4)

        # combine total widget
        self.ui.main_widget.setLayout(self.ui.main_Layout)
        self.ui.main_Layout.addChildWidget(self.ui.lefttop_widget)
        self.ui.main_Layout.addChildWidget(self.ui.leftdown_widget)
        self.ui.main_Layout.addChildWidget(self.ui.window_widget)

        self.ui.main_widget.setStyleSheet('''
        QWidget{background:	#e0e0e0;
        border-top-left-radius:10px;}
        ''')
        self.ui.info_widget.setStyleSheet('''
        QWidget{background:	#e0e0e0;
        border-bottom-left-radius:10px;}
        ''')

        self.ui.tabWidget.setStyleSheet('''
        QTabWidget{
        color:#232C51;
        background:white;
        border:none;
        border-top-right-radius:10px;
        border-bottom-right-radius:10px;}
        QTabWidget::tab-bar{alignment: center;}
        QTabWidget::pane{border:none;}
        QTabBar::tab {font:16px Consolas;color: white;max-width: 300px; min-width:300px; min-height:20px;padding: 5px;}
        QTabBar::tab:selected, QTabBar::tab:hover {background:#4c6666; border-radius:5px;}
        QTabBar::tab:!selected {background:#89b7b7; border-radius:5px; margin-top: 2px;}
        ''')

        self.ui.lefttop_widget.setStyleSheet('''
        QLabel{border:none;color:#4c6666;text-align:left;}
        QComboBox{border: 2px solid #4c6666;border-radius: 5px;min-width: 6em;}
        QLineEdit{border: 2px solid #4c6666;border-radius: 8px;min-width: 6em;
        padding:2px 4px;}
        QComboBox:drop-down {border: 2px solid #CCCCCC;border-radius: 5px;}
        ''')
        self.ui.leftdown_widget.setStyleSheet('''
        QPushButton{border:none;color:#4c6666;text-align:left;}
        QPushButton:hover{color:#666666;}
        ''')
        self.ui.label_4.setStyleSheet('''
        QLabel{border:none;color:#4c6666;text-align:left;}
        ''')
        self.ui.right_widget.setStyleSheet('''
        QWidget#right_widget{
        color:#232C51;
        background:white;
        border-top:2px solid #f0f0f0;
        border-bottom:2px solid #f0f0f0;
        border-right:2px solid #f0f0f0;
        border-top-right-radius:10px;
        border-bottom-right-radius:10px;}
        ''')

        self.ui.exit_Button.setStyleSheet('''
        QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}
        ''')
        self.ui.min_Button.setStyleSheet('''
        QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}
        ''')
        # hide the background
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # window hint close
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

    def exit_window(self):
        sys.exit(app.exec_())

    def minimize_window(self):
        self.showMinimized()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec_())
