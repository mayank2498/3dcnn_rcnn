import cv2
import numpy as np
from models import *

from torchvision.utils import make_grid
from PIL import Image, ImageDraw

from torch.autograd import Variable
from torchvision.models import resnet152

from torch.utils.data import Dataset
import torchvision.transforms as transforms



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


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
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
          
            activity_class_id,activity,percentage = get_activity_prediction(activity_model,lines,image,32,32,20)
            print("action prediction " + activity+";confidence:"+str(percentage.item()))
            #image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            caption = activity +";"+ str(percentage.item())
            image = cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
            )

    return image
  
def get_activity_prediction(activity_model,lines,frame,width,height,depth):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (3,112,112)

    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    image_tensor = Variable(transform(image)).to(device)
    image_tensor = image_tensor.view(1, 1, *image_tensor.shape)

    with torch.no_grad():
        prediction = activity_model(image_tensor)
        classid=prediction.argmax(1).item()

    return classid,lines[classid][:-1],prediction[0][classid]*100
  
def get_activity_model():
    base_path = "saved_model_20frames"
    weights_path = base_path + "/ConvLSTM_5.pth"
    text_file = open(base_path + "/classes.txt", "r")
    lines = text_file.readlines()
    
    activity_model = ConvLSTM( num_classes=14, latent_dim=512)
    # Load weights into the new model
    activity_model.load_state_dict(torch.load(weights_path))
    activity_model.eval()
    activity_model.cuda()
    return activity_model,lines



if __name__ == '__main__':
    #activity_model,lines = get_activity_model()

    """
        test everything
    """
    import os
    import sys
    import coco
    import utils
    import model as modellib
    
    # We use a K80 GPU with 24GB memory, which can fit 3 images.
    batch_size = 3

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
        DETECTION_MIN_CONFIDENCE = 0.45

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
                #frame = display_instances(
                #    frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
                #)
                name = '{0}.jpg'.format(frame_count + i - batch_size)
                name = os.path.join(VIDEO_SAVE_DIR, name)
                cv2.imwrite(name, frame)
                print('writing to file:{0}'.format(name))
            # Clear the frames array to start the next batch
            frames = []

    capture.release()