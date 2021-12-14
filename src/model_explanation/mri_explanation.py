import sys

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import DeepLiftShap
from captum.attr import NoiseTunnel
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz

sys.path.append("./../models")
from neural_network import load_model,load_trained_model,device

class MRIExplainer:
    
    '''
    Explain an MRI CNN classification.

    Parameters
    ------------

    image_id: IMAGE_DATA_ID ou IMAGEUID to be explained.

    classes: tuple or list containing the classes of the problem. Options are (AD,CN) or (MCI,CN)

    prediction_reference: Dataframe reference file containing the predictions, true labels and model location for the classification problem.

    device: Device to load the model and images. Options are 'cuda' or 'cpu'.
    '''
    
    def __init__(self,image_id,classes=['AD','CN'],prediction_reference='',device=device):
        self.image_id = image_id
        self.classes = classes
        self.device = device
        
        if isinstance(prediction_reference,pd.DataFrame):
            self.df_prediction_ref = prediction_reference
        else:
            self.df_prediction_ref = pd.read_csv(prediction_reference)

        self.images = {}
        self.models = {}

    def explain(self,orientation=None,algorithm=DeepLift):

        if orientation is None:
            orientations = ['coronal','sagittal','axial']
            for current_orientation in orientations:
                self.explain(orientation = current_orientation,algorithm=algorithm)
        
        df = self.df_prediction_ref.loc[self.df_prediction_ref['ORIENTATION'] == orientation]
        model_path = df.iloc[0]['MODEL_PATH']
        model_name = df.iloc[0]['MODEL_NAME']
        slice = df.iloc[0]['SLICE']
        
        net = self.load_model(orientation, model_name, model_path)
        
        image = self.load_image(orientation,slice)
        image =image.view(-1,1, 100,100)
        image.requires_grad = True
        image = image.to(self.device)

        _, = DeepLift(net,image)


    def load_image(self,orientation,slice):
        if self.images.get(orientation) is None:
            path = '' #pegar path do dataframe de referencia com slice e orientation
            # Acho q vou ter q gerar um novo dataframe de referencia contendo path da imagem, path do modelo, nome da imagem e nome do modelo
            X = np.load(path)['arr_0']
            self.images[orientation] = torch.from_numpy(X/X.max())

        return self.images[orientation]
        # TODO: tirar logica daqui dessa funcao de baixo. 
    def get_images_and_labels(df):
        '''
        Load images into memory as PyTorch tensors
        '''

        image_paths = df['IMAGE_PATH']
        imgs = []
        start_total = datetime.now()
        for path in image_paths:
            start = datetime.now()
            print(f"\nProcessing {path}")
            X = np.load(path)['arr_0']
            X = torch.from_numpy(X/X.max())
            # X = X.to(device)
            # print(f"Image normalized")
            imgs.append(X)
            print("Loop took: % 2.2f seconds" % (datetime.now() - start).seconds)
        labels = df['MACRO_GROUP'].tolist()
        print("Total processing time: % 2.2f" % (datetime.now() - start_total).seconds)
        return imgs,labels




    def load_model(self,orientation,model_name,model_path):
        if self.models.get(orientation) is None:
            self.models[orientation] = load_trained_model(model=model_name,model_path=model_path,device=self.device)
        return self.models[orientation]

prediction_reference = ''
explainer = MRIExplainer(image_id='I689023',
                                        classes=['AD','CN'],
                                        prediction_reference=prediction_reference,
                                        model_reference='vgg19_bn')

explainer.explain(orientation='coronal',algorithm=DeepLift)
explainer.explain(orientation='axial',algorithm=DeepLift)
explainer.explain(orientation='sagittal',algorithm=DeepLift)

explainer.explain(algorithm=DeepLift)


