import os
from flask import Flask,render_template,request,redirect,url_for,send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename
import numpy as np
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as visualization_utils
MODEL_NAME='ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT=MODEL_NAME+'/frozen_inference_graph.pb'
PATH_TO_LABELS=os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES=90

#Detection
detection_graph=tf.Graph()
with detection_graph.as_default():
    od_graph_def=tf.GraphDef()
    with tf.gfile.Gfile(PATH_TO_CKPT,'rb') as fid:
        serialized_graph=fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def,name="")
label_map=label_map_util.load_labelmap(PATH_TO_LABELS)
categories=label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
category_index=label_map_util.create_category_index(categories)

