import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import coco
import utils
import model as modellib


import cv2
import numpy as np
from models import *


from keras.models import model_from_json
import numpy as np

img_rows = 32
img_cols = 32
frames = 20


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def get_activity_prediction(model,lines,frames,width,height,depth):
    framearray = []
    if(len(frames)!=20):
        return 0,0,0
    for i in range(20):
        try:
            frame = cv2.resize(frames[i], (height, width))
        except:
            return 0,0,0
        framearray.append(frame)
    x = np.array(framearray)
    X = []
    X.append(x)

    X = np.array(X).transpose((0, 2, 3, 4, 1))
    X = X.reshape((X.shape[0], 32, 32, 20, 3))
    y = model.predict(X)
    classid = np.argmax(y);
    return classid,lines[classid][:-1],y[0][classid]*100



def display_instances(image, boxes, masks, ids, names, scores, frames, cnn3d,lines):
  
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        #caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        

        if (ids[i]==1):
            print("found human ")
            activity_frames = []
            for i in range(20):
                if( frames[i] is not None ):
                    crop_img = frames[i][y1:y2, x1:x2]
                    activity_frames.append(crop_img)
                        
            activity_class_id,activity,percentage = get_activity_prediction(cnn3d,lines,activity_frames,32,32,20)
            print("action prediction " + activity+";confidence:"+str(percentage.item()))
            #image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            caption = activity +";"+ str(percentage.item())
            image = cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
            )

    return image

def get_activity_model():
    base_path = "saved_model_20frames"
    modelJson = base_path + "/ucf101_3dcnnmodel.json"
    weights_path = base_path + "/ucf101_3dcnnmodel-gpu.hd5"
    text_file = open(base_path + "/classes.txt", "r")
    lines = text_file.readlines()
    
    with open(modelJson, 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(weights_path)
    return model,lines


def get_activity_prediction(model,lines,frames,width,height,depth):
    framearray = []
    if(len(frames)!=20):
        return 0,0,0
    for i in range(20):
        try:
            frame = cv2.resize(frames[i], (height, width))
        except:
            return 0,0,0
        framearray.append(frame)
    x = np.array(framearray)
    X = []
    X.append(x)

    X = np.array(X).transpose((0, 2, 3, 4, 1))
    X = X.reshape((X.shape[0], 32, 32, 20, 3))
    y = model.predict(X)
    classid = np.argmax(y);
    return classid,lines[classid][:-1],y[0][classid]*100

# We use a K80 GPU with 24GB memory, which can fit 3 images.
batch_size = 20

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = batch_size
    DETECTION_MIN_CONFIDENCE = 0.1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]
activity_model,lines = get_activity_model()

capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, 'worker_all_yolo.avi'))
try:
    if not os.path.exists(VIDEO_SAVE_DIR):
        os.makedirs(VIDEO_SAVE_DIR)
except OSError:
    print ('Error: Creating directory of data')
frames = []
frame_count = 0
# these 2 lines can be removed if you dont have a 1080p camera.
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = capture.read()
    # Bail out when the video file ends
    if not ret:
        break
    
    # Save each frame of the video to a list
    frame_count += 1
    frames.append(frame)
    print('frame_count :{0}'.format(frame_count))
    if len(frames) == batch_size:
        results = model.detect(frames, verbose=0)
        print('Predicted')
        for i, item in enumerate(zip(frames, results)):
            frame = item[0]
            r = item[1]
            frame = display_instances(
                frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],frames,activity_model,lines
            )
            name = '{0}.jpg'.format(frame_count + i - batch_size)
            name = os.path.join(VIDEO_SAVE_DIR, name)
            cv2.imwrite(name, frame)
            print('writing to file:{0}'.format(name))
        # Clear the frames array to start the next batch
        frames = []

capture.release()