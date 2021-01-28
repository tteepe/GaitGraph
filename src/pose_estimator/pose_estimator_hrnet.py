import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse

from datasets.preparation import box_to_center_scale
from pose_estimator import model_hrnet
from pose_estimator.config import _C as config, update_config
from utils import *


class PoseEstimatorHRNet:
    def __init__(self,
                 config_path='inference-config.yaml',
                 weights_path='../../models/pose_hrnet_w32_384x288.pth'):
        self.config_path = config_path
        self.weights_path = weights_path

        args = argparse.Namespace()
        args.cfg = self.config_path
        # opt expected by supporting codebase
        args.opt = ''
        args.modelDir = ''
        args.logDir = ''
        args.dataDir = ''
        args.prevModelDir = ''

        update_config(config, args)
        self.config = config

        self.model = model_hrnet.get_pose_net(config, is_train=False)
        self.model.load_state_dict(torch.load(weights_path), strict=False)
        self.model = torch.nn.DataParallel(self.model).cuda()

        self.model.eval()  # Set in evaluation mode

    def estimate_pose_from_image(self, img, box):
        center, scale = box_to_center_scale(box, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])

        rotation = 0

        # pose estimation transformation
        trans = get_affine_transform(center, scale, rotation, config.MODEL.IMAGE_SIZE)
        model_input = cv2.warpAffine(
            img,
            trans,
            (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # pose estimation inference
        input_img = transform(model_input).unsqueeze(0)

        with torch.no_grad():
            # compute output heatmap
            output = self.model(input_img)
            preds, _ = get_final_preds(
                config,
                output.clone().cpu().numpy(),
                np.asarray([center]),
                np.asarray([scale]))

            return preds


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.ticker import NullLocator

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--cfg', type=str, default='inference-config.yaml')
    parser.add_argument('opt', help='Modify config options using the command-line', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    pose_estimator = PoseEstimatorHRNet()

    img = np.array(Image.open('../data/samples/messi.jpg'))
    boxes = [
             [17.860302, 26.873545, 824.93115, 694.90466],
             [1202.5271, 475.52982, 88.31201, 215.9581],
             [648.0603, 104.8192, 492.93066, 621.0242]
             ]

    # Create plot
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box in boxes:
        pose_predictions = pose_estimator.estimate_pose_from_image(img, box)
        for _, mat in enumerate(pose_predictions[0]):
            x, y = int(mat[0]), int(mat[1])
            circle = patches.Circle((x, y), radius=5, linewidth=2, edgecolor=(1, 0, 0), facecolor="none")
            # Add the pose points to the plot
            ax.add_patch(circle)

            x1, y1, box_w, box_h = box
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=(0, 1, 0), facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)

    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()
