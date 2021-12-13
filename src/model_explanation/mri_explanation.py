from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import DeepLiftShap
from captum.attr import NoiseTunnel
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz

import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pandas as pd
import numpy as np


def explain_mri(image_id,orientation,slice):
    
def explain_image(img,label,prediction,threshold=0.5,explanation_algorithm=DeepLift):
    pass