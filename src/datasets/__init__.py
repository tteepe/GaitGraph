from .preparation import DatasetSimple, DatasetDetections
from .gait import (
    CasiaBPose,
)


def dataset_factory(name):
    if name == "casia-b":
        return CasiaBPose

    raise ValueError()
