import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
from random import randint
import argparse
import logging
import time
import operator
import imutils
import cv2
import numpy as np
import math

from antares_http import antares
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from itertools import chain, count
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

CAMERA = [1]
ROTATE = [0]
IMAGE = [1024,576]
SYS_OPOSE = True
            
SKX = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
SKY = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]

class mainhuman_activity:

    # Pre-processing for every image
    def preprocess(raws):
        imgs = cv2.resize(raws, (1024, 576), interpolation=cv2.INTER_CUBIC)
        return imgs
    
    def __init__(self):
        cam = cv2.VideoCapture(0)
        
        antares.setDebug(False)
        antares.setAccessKey('4129c7daf5d02430:9ef42d1eb685dd2c')
        latestData = antares.get('Gusti_WorkshopAntares', 'OMG-Mirror')
        # resets = 4
        # set_kiri = 0
        # set_kanan = 0
        # reptoset = 12
        # rep_kiri = 0
        # rep_kanan = 0
        resets = latestData["content"]["reset"]
        set_kiri = latestData["content"]["set_kiri"]
        set_kanan = latestData["content"]["set_kanan"]
        reptoset = latestData["content"]["reptoset"]
        rep_kiri = latestData["content"]["rep_kiri"]
        rep_kanan = latestData["content"]["rep_kanan"]
        condition_kiri = "DOWN"
        condition_kanan = "DOWN"
        kiri_overtraining_thinker = "OVER"
        kiri_overtraining_count = 0
        kanan_overtraining_thinker = "OVER"
        kanan_overtraining_count = 0

        _, imgs = cam.read()
        
        image = mainhuman_activity.preprocess(imgs)
        
        # Face camera, not rendered on main image
        
        self.im_h, self.im_w = image.shape[:2]
        
        ###print("\n######################## Openpose")
        if SYS_OPOSE:
            opose = openpose_human(image)
        
        # Main loop
        while True:
            _, imgs = cam.read()
            # # TEST, 4 camera simulation
            # for i in range(3):
                # imgs.append(img)
                
            image = mainhuman_activity.preprocess(imgs)
            
            if SYS_OPOSE:
                human_keypoints, human_ids, humans = opose.runopenpose(image)
                # print(humans, human_keypoints)
            else:
                human_keypoints = {0: [np.zeros(36)]}
                human_ids = {0: 0}
                humans = []
            for key, human_keypoint in human_keypoints.items():
                if(len(human_keypoint)==5 and key == 0):
                    kanan_x1 = human_keypoint[0][4]
                    kanan_y1 = human_keypoint[0][5]
                    kanan_x2 = human_keypoint[0][6]
                    kanan_y2 = human_keypoint[0][7]
                    kanan_x3 = human_keypoint[0][8]
                    kanan_y3 = human_keypoint[0][9]
                    kiri_x1 = human_keypoint[0][10]
                    kiri_y1 = human_keypoint[0][11]
                    kiri_x2 = human_keypoint[0][12]
                    kiri_y2 = human_keypoint[0][13]
                    kiri_x3 = human_keypoint[0][14]
                    kiri_y3 = human_keypoint[0][15]
                    if(kiri_x1!=0.0 and kiri_x2!=0.0 and kiri_x3!=0.0 and kiri_y1!=0.0 and kiri_y2!=0.0 and kiri_y3!=0.0):
                        angle = self.getAngle((kiri_x1, kiri_y1), (kiri_x2, kiri_y2), (kiri_x3, kiri_y3))
                        r12 = self.resultan((kiri_x1, kiri_y1), (kiri_x2, kiri_y2))
                        r23 = self.resultan((kiri_x2, kiri_y2),(kiri_x3, kiri_y3))
                        if(angle < 80.0 and condition_kiri=="DOWN" and (r12*0.5 < r23 or 1.5*r12 > r23) ):
                            condition_kiri="UP"
                            if(kiri_overtraining_thinker=="OVER"):
                                kiri_overtraining_thinker="NORMAL"
                            else:
                                kiri_overtraining_thinker="NORMAL"
                                kiri_overtraining_count=0
                        elif(angle > 120.0 and condition_kiri=="UP" and (r12*0.5 < r23 or 1.5*r12 > r23) ):
                            condition_kiri="DOWN"
                            rep_kiri+=1
                            if rep_kiri == reptoset:
                                rep_kiri = 0
                                set_kiri +=1
                                if set_kiri == resets:
                                    set_kiri = 0
                            mydata= {}
                            mydata["set_kiri"]=set_kiri
                            mydata["set_kanan"]=set_kanan
                            mydata["reset"]=resets
                            mydata["reptoset"]=reptoset
                            mydata["rep_kiri"]=rep_kiri
                            mydata["rep_kanan"]=rep_kanan
                            antares.send(mydata, 'Gusti_WorkshopAntares', 'OMG-Mirror')
                            if(kiri_overtraining_thinker=="OVER"):
                                kiri_overtraining_thinker="NORMAL"
                            else:
                                kiri_overtraining_thinker="NORMAL"
                                kiri_overtraining_count=0
                        if((angle < 40.0 or angle > 160.0 ) and kiri_overtraining_thinker=="NORMAL" and (r12*0.5 < r23 or 1.5*r12 > r23)):
                            kiri_overtraining_thinker = "OVER"
                            kiri_overtraining_count += 1

                    if(kanan_x1!=0.0 and kanan_x2!=0.0 and kanan_x3!=0.0 and kanan_y1!=0.0 and kanan_y2!=0.0 and kanan_y3!=0.0):
                        angle = self.getAngle((kanan_x1, kanan_y1), (kanan_x2, kanan_y2), (kanan_x3, kanan_y3))
                        r12 = self.resultan((kanan_x1, kanan_y1), (kanan_x2, kanan_y2))
                        r23 = self.resultan((kanan_x2, kanan_y2),(kanan_x3, kanan_y3))

                        if(angle < 80.0 and condition_kanan=="DOWN" and (r12*0.5 < r23 or 1.5*r12 > r23) ):
                            condition_kanan="UP"
                            if(kanan_overtraining_thinker=="OVER"):
                                kanan_overtraining_thinker="NORMAL"
                            else:
                                kanan_overtraining_thinker="NORMAL"
                                kanan_overtraining_count=0
                        elif(angle > 120.0 and condition_kanan=="UP" and (r12*0.5 < r23 or 1.5*r12 > r23) ):
                            condition_kanan="DOWN"
                            rep_kanan+=1
                            if rep_kanan == reptoset:
                                rep_kanan = 0
                                set_kanan +=1
                                if set_kanan == resets:
                                    set_kanan = 0
                            mydata= {}
                            mydata["set_kiri"]=set_kiri
                            mydata["set_kanan"]=set_kanan
                            mydata["reset"]=resets
                            mydata["reptoset"]=reptoset
                            mydata["rep_kiri"]=rep_kiri
                            mydata["rep_kanan"]=rep_kanan
                            antares.send(mydata, 'Gusti_WorkshopAntares', 'OMG-Mirror')
                            if(kanan_overtraining_thinker=="OVER"):
                                kanan_overtraining_thinker="NORMAL"
                            else:
                                kanan_overtraining_thinker="NORMAL"
                                kanan_overtraining_count=0

                        if((angle < 40.0 or angle > 160.0 ) and kanan_overtraining_thinker=="NORMAL" and (r12*0.5 < r23 or 1.5*r12 > r23)):
                            kanan_overtraining_thinker = "OVER"
                            kanan_overtraining_count += 1

            self.display_all(image, humans,kanan_overtraining_count,kiri_overtraining_count,rep_kiri,rep_kanan)
            
            if cv2.waitKey(1) == 27:
                break
        
        cv2.destroyAllWindows()
    
    def display_all(self, image, humans,kanan_overtraining_count,kiri_overtraining_count,rep_kiri,rep_kanan):
        blank_image = np.zeros((576,1024,3),np.uint8)
        blank_image = TfPoseEstimator.draw_humans(blank_image, humans, imgcopy=False)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        if(kanan_overtraining_count > 4 or kiri_overtraining_count > 4):
            cv2.putText(blank_image,
                "YOU HAVE OVERTRAIN YOUR MUSCLE",
                (10, 110),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
        cv2.putText(blank_image,
            "REP_KANAN: %.2f" % rep_kanan,
            (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)
        cv2.putText(blank_image,
            "REP_KIRI: %.2f" % rep_kiri,
            (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)
        cv2.imshow('OMG color mirror',blank_image)
        cv2.imshow('OMG Mirror',image)
    
    def getAngle(self, a, b, c):
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return abs(ang) if ang < 0 else ang

    def resultan(self,a,b):
        r = math.sqrt((a[1]-b[1])**2+(a[0]-b[0])**2)
        return r

class openpose_human:
    # def __init__(self, camera=0,resize='0x0',resize_out_ratio=4.0,model='mobilenet_thin',show_process=False):
    def __init__(self, image, resize='1024x576',model='mobilenet_v2_large'):
        self.logger = logging.getLogger('TfPoseEstimator-WebCam')
        self.logger.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        ##self.logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
        self.w, self.h = model_wh(resize)
        if self.w > 0 and self.h > 0:
            self.e = TfPoseEstimator(get_graph_path(model), target_size=(self.w, self.h))
        else:
            self.e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
        ##self.logger.debug('cam read+')
        # cam = cv2.VideoCapture(camera)
        # ret_val, image = cam.read()
        self.im_h, self.im_w = image.shape[:2]
        # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
        self.videostep = 0
        self.human_keypoint = {0: [np.zeros(36)]}
        self.human_ids = {0: 0}
        
    def runopenpose(self, image, resize_out_ratio=4.0):
        # ret_val, image = cam.read()
        ##self.logger.debug('image process+')
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=resize_out_ratio)
        skeletoncount = 0
        skels = np.array([np.zeros(36)])
        
        for human in humans:
            if skeletoncount == 0:  # Initialize by adding N_STEPS of empty skeletons
                skels = np.array([openpose_human.write_coco_json(human, self.im_w,self.im_h)])
            else:                   # Append the rest
                skels = np.vstack([skels, np.array(openpose_human.write_coco_json(human, self.im_w,self.im_h))])
            skeletoncount = skeletoncount + 1
            
        # if skeletoncount == 1:  # Just assume it's the same prson if there's only one
            # self.human_keypoint[0].append(skels)
        
        if skeletoncount > 0:
            self.human_keypoint, self.human_ids = openpose_human.push(self.human_keypoint, self.human_ids, skels)
        else:
            # No human actually detected (humans is empty, thus skcount = 0)
            self.human_keypoint = {0: [np.zeros(36)]}
            self.human_ids = {0: 0}
        
        tf.reset_default_graph() # Reset the graph
        # self.logger.debug('finished+')
        
        return (self.human_keypoint, self.human_ids, humans)
        # Basically, human_keypoint store a string of poses, length N_STEPS, and tracked.
        # Humans is the result of a single inference, formatting still raw.
    
    def draw_box(image, coord_type, bounds, text='', conf=1, loc=0, thickness=3):
        # Based on the input detection coordinate
        if coord_type == 0:
            # Input (x, y) describes the top-left corner of detection
            x = int(bounds[0])
            y = int(bounds[1])
        else: # Input (x, y) describes the center of detection
            # Move it to the top-left corner
            x = int(bounds[0] - bounds[2]/2)
            y = int(bounds[1] - bounds[3]/2)
            
        w = int(bounds[2])
        h = int(bounds[3])
        
        color = (int(255 * (1 - (conf ** 2))), int(255 * (conf ** 2)), 0)
        
        # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        
        # Object text
        if loc == 0:
            cv2.putText(image, "%s %.2f" % (text, conf), (x, y-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif loc == 1:
            cv2.putText(image, "%s %.2f" % (text, conf), (x, y+h+15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image, color
    
    def write_coco_json(human, image_w, image_h):
        keypoints = []
        coco_ids = coco_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        for coco_id in coco_ids:
            if coco_id not in human.body_parts.keys():
                keypoints.extend([0, 0])
                continue
            body_part = human.body_parts[coco_id]
            keypoints.extend([round(body_part.x * image_w, 3), round(body_part.y * image_h, 3)])
        return keypoints

    def push(traces, ids, new_skels, THRESHOLD = 100, TRACE_SIZE = 5):
    
        ###print("##### Multi-human")
        
        """Add the keypoints from a new frame into the buffer."""
        # dists, neighbors = openpose_human.nearest_neighbors(traces, new_skels)
        dists, neighbors = openpose_human.point(traces, new_skels)
        keygen = []
        # New skeletons which aren't close to a previously observed skeleton:
        unslotted = []
        # Previously observed skeletons which aren't close to a new one:
        for each in traces.keys():
            keygen.append(each)
        unseen = set(keygen)
        for skel, dist, neighbor in zip(new_skels, dists, neighbors):
            ###print(dist, neighbor)
            if dist <= THRESHOLD:
                if neighbor in traces:
                    traces[neighbor].append(skel)
                else:
                    id = randint(0,100)     # Only used for naming
                    traces[neighbor] = []
                    traces[neighbor].append(skel)
                    ids[neighbor] = id
                if len(traces[neighbor]) > TRACE_SIZE:
                    traces[neighbor].pop(0)
                unseen.discard(neighbor)
            else:
                unslotted.append(skel)

        for i in unseen:
            del traces[i]
            del ids[i]

        # Indices we didn't match, and the rest of the numbers are fair game
        availible_slots = chain(sorted(unseen), count(len(traces)))
        for slot, skel in zip(availible_slots, unslotted):
            id = randint(0,100)     # Only used for naming
            if slot in traces:
                traces[slot].append(skel)
            else:
                traces[slot] = []
                traces[slot].append(skel)
                ids[slot] = id
                
        return traces, ids
    
    def point(traces, skels, TRACE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]):
        if not traces:  # First pass
            return np.zeros(len(skels)), np.arange(len(skels))
            
        prev = np.array([  # Pull the most recent location of each skeleton, [-1] means get 1 data from behind
            coords[-1][TRACE_IDX] for _, coords in sorted(traces.items())])
            
        curr = skels[:, TRACE_IDX]
        # Determine representative point, may use various method such as median, average, etc
        prev_point = openpose_human.average(prev)
        curr_point = openpose_human.average(curr)
        
        # N is typically small (< 40) so brute force is fast
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='brute')
        nn_model.fit(prev_point)
        dist, nn = nn_model.kneighbors(curr_point, return_distance=True)
        
        return dist.flatten(), nn.flatten()
    def average(skels):
        avg_skels = np.empty((0, 2))
        for skel in skels:
            # Remember that a point might not be detected, giving zero. Count the non-zero.
            # Below line is equivalent to COUNTIF(not-zero).
            
            # Count non-zeros
            nzero_x = sum(1 if (x != 0) else 0 for x in skel[SKX])
            nzero_y = sum(1 if (x != 0) else 0 for x in skel[SKY])
            
            if (nzero_x == 0):
                nzero_x = 1
            if (nzero_y == 0):
                nzero_y = 1
                
            x = sum(skel[SKX]) / nzero_x
            y = sum(skel[SKY]) / nzero_y
            avg_skels = np.vstack((avg_skels, np.array([x, y])))
            
        return avg_skels

    def nearest_neighbors(traces, skels, TRACE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]):
    
        if not traces:  # First pass
            return np.zeros(len(skels)), np.arange(len(skels))
        prev = np.array([  # Pull the most recent location of each skeleton
            coords[-1][TRACE_IDX] for _, coords in sorted(traces.items())])
        curr = skels[:, TRACE_IDX]
        # N is typically small (< 40) so brute force is fast
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='brute')
        nn_model.fit(prev)
        dist, nn = nn_model.kneighbors(curr, return_distance=True)
        return dist.flatten(), nn.flatten()
     
if __name__ == '__main__':
    mainhuman_activity()

