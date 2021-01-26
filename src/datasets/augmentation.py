import numpy as np
import cv2
import torch

from pose_estimator.utils import get_affine_transform


class ToTensor(object):
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float)


class MultiInput(object):
    def __init__(self, connect_joint, enabled=False):
        self.connect_joint = connect_joint
        self.enabled = enabled

    def __call__(self, data):
        # (C, T, V) -> (I, C * 2, T, V)
        data = np.transpose(data, (2, 0, 1))

        if not self.enabled:
            return data[np.newaxis, ...]

        C, T, V = data.shape
        data_new = np.zeros((3, C * 2, T, V))
        # Joints
        data_new[0, :C, :, :] = data
        for i in range(V):
            data_new[0, C:, :, i] = data[:, :, i] - data[:, :, 1]
        # Velocity
        for i in range(T - 2):
            data_new[1, :C, i, :] = data[:, i + 1, :] - data[:, i, :]
            data_new[1, C:, i, :] = data[:, i + 2, :] - data[:, i, :]
        # Bones
        for i in range(len(self.connect_joint)):
            data_new[2, :C, :, i] = data[:, :, i] - data[:, :, self.connect_joint[i]]
        bone_length = 0
        for i in range(C - 1):
            bone_length += np.power(data_new[2, i, :, :], 2)
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            data_new[2, C, :, :] = np.arccos(data_new[2, i, :, :] / bone_length)

        return data_new


class FlipSequence(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            return np.flip(data, axis=0).copy()
        return data


class MirrorPoses(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            center = np.mean(data[:, :, 0], axis=1, keepdims=True)
            data[:, :, 0] = center - data[:, :, 0] + center

        return data


class RandomSelectSequence(object):
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = np.random.randint(0, data.shape[0] - self.sequence_length)
        except ValueError:
            print(data.shape[0])
            raise ValueError
        end = start + self.sequence_length
        return data[start:end]


class SelectSequenceCenter(object):
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = int((data.shape[0]/2) - (self.sequence_length / 2))
        except ValueError:
            print(data.shape[0])
            raise ValueError
        end = start + self.sequence_length
        return data[start:end]


class ShuffleSequence(object):
    def __init__(self, enabled=False):
        self.enabled = enabled

    def __call__(self, data):
        if self.enabled:
            np.random.shuffle(data)
        return data


class TwoNoiseTransform(object):
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class PointNoise(object):
    """
    Add Gaussian noise to pose points
    std: standard deviation
    """

    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
        return data + noise


class JointNoise(object):
    """
    Add Gaussian noise to joint
    std: standard deviation
    """

    def __init__(self, std=0.5):
        self.std = std

    def __call__(self, data):
        # T, V, C
        noise = np.hstack((
            np.random.normal(0, 0.25, (data.shape[1], 2)),
            np.zeros((data.shape[1], 1))
        )).astype(np.float32)

        return data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)


class DropOutFrames(object):
    """
    Type of data augmentation. Dropout frames randomly from a sequence.
    Properties:
    dropout_rate_range: Defines the range from which dropout rate is picked
    prob: Probability that this technique is applied on a sample.
    """

    def __init__(self, probability=0.1, sequence_length=60):
        self.probability = probability
        self.sequence_length = sequence_length

    def __call__(self, data):
        T, V, C = data.shape

        new_data = []
        dropped = 0
        for i in range(T):
            if np.random.random() <= self.probability:
                new_data.append(data[i])
            else:
                dropped += 1
            if T - dropped <= self.sequence_length:
                break

        for j in range(i, T):
            new_data.append(data[j])

        return np.array(new_data)


class DropOutJoints(object):
    """
    Type of data augmentation. Zero joints randomly from a pose.
    Properties:
    dropout_rate_range:
    prob: Probability that this technique is applied on a sample.
    """

    def __init__(
        self, prob=1, dropout_rate_range=0.1,
    ):
        self.dropout_rate_range = dropout_rate_range
        self.prob = prob

    def __call__(self, data):
        if np.random.binomial(1, self.prob, 1) != 1:
            return data

        T, V, C = data.shape
        data = data.reshape(T * V, C)
        # Choose the dropout_rate randomly for every sample from 0 - dropout range
        dropout_rate = np.random.uniform(0, self.dropout_rate_range, 1)
        zero_indices = 1 - np.random.binomial(1, dropout_rate, T * V)
        for i in range(3):
            data[:, i] = zero_indices * data[:, i]
        data = data.reshape(T, V, C)
        return data


class InterpolateFrames(object):
    """
    Type of data augmentation. Create more frames between adjacent frames by interpolation
    """

    def __init__(self, probability=0.1):
        """
        :param probability: The probability with which this augmentation technique will be applied
        """
        self.probability = probability

    def __call__(self, data):
        # data shape is T,V,C = Frames, Joints, Channels (X,Y,conf)
        T, V, C = data.shape

        # interpolated_data = np.zeros((T + T - 1, V, C), dtype=np.float32)
        interpolated_data = []
        for i in range(T):
            # Add original frame
            interpolated_data.append(data[i])

            # Skip last
            if i == T - 1:
                break

            if np.random.random() <= self.probability:
                continue

            # Calculate difference between x and y points of each joint of current frame and current frame plus 1
            x_difference = data[i + 1, :, 0] - data[i, :, 0]
            y_difference = data[i + 1, :, 1] - data[i, :, 1]

            new_frame_x = (
                data[i, :, 0] + (x_difference * np.random.normal(0.5, 1))
            )
            new_frame_y = (
                data[i, :, 1] + (y_difference * np.random.normal(0.5, 1))
            )
            # Take average of conf of current and next frame to find the conf of the interpolated frame
            new_frame_conf = (data[i + 1, :, 2] + data[i, :, 2]) / 2
            interpolated_frame = np.array(
                [new_frame_x, new_frame_y, new_frame_conf]
            ).transpose()

            interpolated_data.append(interpolated_frame)

        return np.array(interpolated_data)


class CropToBox(object):
    """Crop image to detection box
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, img, center, scale):
        rotation = 0
        # pose estimation transformation
        trans = get_affine_transform(
            center, scale, rotation, self.config.MODEL.IMAGE_SIZE
        )
        model_input = cv2.warpAffine(
            np.array(img),
            trans,
            (
                int(self.config.MODEL.IMAGE_SIZE[0]),
                int(self.config.MODEL.IMAGE_SIZE[1]),
            ),
            flags=cv2.INTER_LINEAR,
        )

        return model_input
