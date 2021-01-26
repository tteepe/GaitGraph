import argparse
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from detector.models import *
from utils import *


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class DetectorYOLOv3:
    def __init__(self,
                 model_def='config/yolov3.cfg',
                 weights_path='../weights/yolov3.weights',
                 conf_thres=0.8,
                 nms_thres=0.4,
                 img_size=416):
        self.model_def = model_def
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(self.model_def, img_size=self.img_size).to(device)

        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weights_path))

        self.model.eval()  # Set in evaluation mode

    def detect_from_image(self, img):
        input_img = preprocess_image(img)

        # Configure input
        input_img = Variable(input_img.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)[0]
            if detections is None:
                return []
            else:
                detections = detections.data.cpu().numpy()

        # Draw bounding boxes and labels of detections
        human_candidates = []
        if detections is not None:
            # Rescale boxes to original img
            detections = rescale_boxes(detections, self.img_size, img.shape[:2])

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_w = x2 - x1
                box_h = y2 - y1

                if int(cls_pred) == 0:
                    human_candidate = [x1, y1, box_w, box_h]
                    human_candidates.append(human_candidate)
        return human_candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../../models/yolov3.weights",
                        help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    opt = parser.parse_args()

    detector = DetectorYOLOv3(**vars(opt))

    img = np.array(Image.open('../data/samples/messi.jpg'))
    human_candidates = detector.detect_from_image(img)

    # Create plot
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for x1, y1, box_w, box_h in human_candidates:
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=(1, 0, 0), facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)

    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()
