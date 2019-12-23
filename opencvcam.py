from basecamera import BaseCamera
import cv2
import tensorflow as tf
import os
from object_detection.utils import visualization_utils, label_map_util, ops
import numpy as np
# import face_recognizer as fr
from mainhuman_activity import openpose_human

import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import random
from random import randint
import argparse
import logging
import time
import operator
import imutils
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

# class ObjectDetector(object):
#     def __init__(self,model_name):
#         self.model_name = model_name
#         self.graph = tf.Graph()
#         self.num_class = 1
#         self.initialize_graph()
#         self.initialize_labels()
#         self.session = None
    
#     def __del__(self):
#         if self.session is not None:
#             self.session.close()

#     def run_inference_for_single_image(self,image,session):
#         ops = tf.get_default_graph().get_operations()
#         all_tensor_names = {output.name for op in ops for output in op.outputs}
#         tensor_dict = {}
#         for key in [
#             'num_detections', 'detection_boxes', 'detection_scores',
#             'detection_classes', 'detection_masks'
#         ]:
#             tensor_name = key + ':0'
#             if tensor_name in all_tensor_names:
#                 tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
#                     tensor_name)
#         if 'detection_masks' in tensor_dict:
#             # The following processing is only for single image
#             detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
#             detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
#             # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
#             real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
#             detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
#             detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
#             detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
#                 detection_masks, detection_boxes, image.shape[0], image.shape[1])
#             detection_masks_reframed = tf.cast(
#                 tf.greater(detection_masks_reframed, 0.5), tf.uint8)
#             # Follow the convention by adding back the batch dimension
#             tensor_dict['detection_masks'] = tf.expand_dims(
#                 detection_masks_reframed, 0)
#         image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

#         # Run inference
#         output_dict = session.run(tensor_dict,
#                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

#         # all outputs are float32 numpy arrays, so convert types as appropriate
#         output_dict['num_detections'] = int(output_dict['num_detections'][0])
#         output_dict['detection_classes'] = output_dict[
#             'detection_classes'][0].astype(np.uint8)
#         output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#         output_dict['detection_scores'] = output_dict['detection_scores'][0]
#         if 'detection_masks' in output_dict:
#             output_dict['detection_masks'] = output_dict['detection_masks'][0]
#         visualization_utils.visualize_boxes_and_labels_on_image_array(
#             image,
#             output_dict['detection_boxes'],
#             output_dict['detection_classes'],
#             output_dict['detection_scores'],
#             self.category_index,
#             use_normalized_coordinates=True,
#             line_thickness=8)
#         return output_dict, image

#     def initialize_graph(self):
#         model_path = os.path.join(self.model_name,'frozen_inference_graph.pb')
#         with self.graph.as_default():
#             temp_graph_def = tf.GraphDef()
#             with tf.gfile.GFile(model_path,'rb') as f:
#                 ser_graph = f.read()
#                 temp_graph_def.ParseFromString(ser_graph)
#                 tf.import_graph_def(temp_graph_def,name='')
    
#     def initialize_labels(self):
#         path_to_label = os.path.join(self.model_name,'label.pbtxt')        
#         label_map = label_map_util.load_labelmap(path=path_to_label)
#         categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_class, use_display_name=True)
#         self.category_index = label_map_util.create_category_index(categories)

    
# detector = ObjectDetector("facedetector")

class Camera(BaseCamera):
    def __init__(self):
        # self.detector = ObjectDetector("facedetector")
        return super().__init__()
    @staticmethod
    def frames(self):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise RuntimeError('Camera not found')
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
        # facer = fr.face_recognizer(face_dir="face/")
        _, imgs = cam.read()
        
        image = self.preprocess(self,imgs) # Face camera, not rendered on main image
        
        self.im_h, self.im_w = image.shape[:2]
        
        ###print("\n######################## Openpose")
        if SYS_OPOSE:
            opose = openpose_human(image)
        
        # Main loop
        while True:
            _, img = cam.read()
            # face_locs, face_names = facer.runinference(img, tolerance=0.6, prescale=0.25, upsample=2)
            # # TEST, 4 camera simulation
            # for i in range(3):
                # imgs.append(img)
                
            image = self.preprocess(self,imgs)
            
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
                        angle = self.getAngle(self,(kiri_x1, kiri_y1), (kiri_x2, kiri_y2), (kiri_x3, kiri_y3))
                        r12 = self.resultan(self,(kiri_x1, kiri_y1), (kiri_x2, kiri_y2))
                        r23 = self.resultan(self,(kiri_x2, kiri_y2),(kiri_x3, kiri_y3))
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
                        angle = self.getAngle(self,(kanan_x1, kanan_y1), (kanan_x2, kanan_y2), (kanan_x3, kanan_y3))
                        r12 = self.resultan(self,(kanan_x1, kanan_y1), (kanan_x2, kanan_y2))
                        r23 = self.resultan(self,(kanan_x2, kanan_y2),(kanan_x3, kanan_y3))

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

            displayimage = self.display_all(self,image, humans,kanan_overtraining_count,kiri_overtraining_count)
            
            # img = facer.display(img, face_locs, face_names, 0.25)
            yield cv2.imencode('.jpg',displayimage)[1].tobytes()
    
    # Pre-processing for every image
    def preprocess(self, raws):
        imgs = cv2.resize(raws, (1024, 576), interpolation=cv2.INTER_CUBIC)
        return imgs
    
    def display_all(self, image, humans,kanan_overtraining_count,kiri_overtraining_count):
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        if(kanan_overtraining_count > 4 or kiri_overtraining_count > 4):
            cv2.putText(image,
                "YOU HAVE OVERTRAIN YOUR MUSCLE",
                (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
        return image
    
    def getAngle(self, a, b, c):
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return abs(ang) if ang < 0 else ang

    def resultan(self,a,b):
        r = math.sqrt((a[1]-b[1])**2+(a[0]-b[0])**2)
        return r
    