#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolov4 import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from deep_sort.videocaptureasync import VideoCaptureAsync

warnings.filterwarnings('ignore')

def main(yolo):

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    max_iou_distance=0.8
    max_age = 30 # 消失后保留帧数
    min_confirm = 1 # 最少确认帧数

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric=metric,
                      max_age=max_age,
                      max_iou_distance=max_iou_distance,
                      min_confirm = min_confirm,
                      classNum=len(yolo.class_names))

    #输入的视频
    input_video_name='1.mp4'
    input_video_path = './input_videos/'+input_video_name
    video_capture = cv2.VideoCapture(input_video_path)
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    print('video shpae:',w,h)

    #输出的视频
    new_w=960
    new_h=540
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./output_videos/'+input_video_name.split('.')[0]+'_yolov4.avi', fourcc, 30, (new_w, new_h))

    t1 = time.time()
    frameIndex = 0
    frameBlank = 3

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
             break
        frame=cv2.resize(frame, (new_w,new_h))

        #yolo预测
        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)[0]
        confidence = yolo.detect_image(image)[1]
        classIndexList = yolo.detect_image(image)[2]

        features = encoder(frame,boxs)

        detections = []

        for bbox, confidence, feature, classIndex in zip(boxs, confidence, features, classIndexList):
            detections.append(Detection(bbox, confidence, feature, classIndex))
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        #视频输出帧
        outFrame=np.array(frame)
        for track in tracker.tracks:
            # 已经确认帧数
            confirm_num = tracker.confirmed_time_list[track.classIndex][track.objectIndex]

            if not track.is_confirmed() or track.time_since_update > 1 or confirm_num < min_confirm:
                continue
            bbox = track.to_tlbr()
            # 显示索引从1开始
            index = tracker.confirmed_id_list[track.classIndex].index(track.track_id)
            text = str(yolo.class_names[track.classIndex]) + '_' + str(index)

            # 分割图片
            if confirm_num == min_confirm:
                # 横坐标
                x_min = min(int(bbox[0]), int(bbox[2]))
                x_max = max(int(bbox[0]), int(bbox[2]))
                # 纵坐标
                y_min = min(int(bbox[1]), int(bbox[3]))
                y_max = max(int(bbox[1]), int(bbox[3]))
                tempFrame=np.array(frame)
                cv2.rectangle(tempFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(tempFrame, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 120, (14, 241, 255), 2)
                cv2.imwrite('./output_pictures/'+text + '.png', tempFrame)


            # 在视频里绘制track框与对应编号
            cv2.rectangle(outFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(outFrame, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 120, (14, 241, 255), 2)


        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2)
            # 在视频里绘制detection框与对应得分
            cv2.rectangle(outFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(outFrame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (255, 255, 0), 2)

        out.write(outFrame)

        if frameIndex % frameBlank == frameBlank - 1:
            t2 = time.time()
            print("frame Index = %f" % (frameIndex), "FPS = %f" %(frameBlank / (t2 - t1)))
            t1 = time.time()
        frameIndex += 1

    video_capture.release()
    out.release()

    #写入每种数量
    result=tracker.confirmed_time_list
    detailFile = open('./detail.txt', 'w')
    countFile = open('./count.txt', 'w')
    for i in range(0,len(result)):
        detailData=[]
        sum=0
        for j in range(0,len(result[i])):
            if result[i][j] >= min_confirm:
                detailData.append(result[i][j])
                sum+=1
        if sum>0:
            countFile.write(yolo.class_names[i]+': '+str(sum)+'\n')
            detailFile.write(yolo.class_names[i] + ': ' + str(detailData) + '\n')
    countFile.close()
    detailFile.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startTime = time.time()
    main(YOLO())
    endTime = time.time()
    print('using time:',endTime-startTime)
    print('over')