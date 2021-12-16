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

    prediction_reference: Dataframe reference containing the predictions, true labels and 
    model location for the classification problem. Can be either a path or dataframe object.

    device: Device to load the model and images. Options are 'cuda' or 'cpu'.
    '''

    def __init__(self,image_id,prediction_reference='',device=device):
        self.image_id = image_id
        self.device = device
        
        self.images = {}
        self.models = {}
        self.explanations = {}
        
        df_ref = prediction_reference
        if isinstance(prediction_reference,str):
            df_ref = pd.read_csv(prediction_reference)
        self.df_reference = self._get_image_reference(df_ref,image_id)
    
    @staticmethod
    def _get_image_reference(df,image_id):
        df = df.query("IMAGE_DATA_ID == @image_id")
        df.columns = df.columns.str.upper()
        return df

    def explain(self,orientation=None,algorithm='IntegratedGradients'):

        if orientation is None:
            orientation = ['coronal','sagittal','axial']
        
        if isinstance(orientation,list):
            for current_orientation in orientation:
                self.explain(orientation = current_orientation,algorithm = algorithm)
        
        img_info = self.df_reference.query("ORIENTATION == @orientation").iloc[0]

        predict_label = 1 if img_info['CNN_SCORE'] >= 0.5 else 0
        true_label = img_info['MACRO_GROUP']
        print('Target:',true_label, 
        'Predict Label (0.5 threshold):',predict_label,
        'Predicted Probability:', img_info['CNN_SCORE'])
        

        net = self._get_model(orientation, model_name=img_info['MODEL'], model_path=img_info['MODEL_PATH'])
        image = self._get_image(orientation,image_path = img_info['IMAGE_PATH'])
        self._explain_image(orientation,net,image,algorithm)

    def _get_image(self,orientation,image_path):
        if self.images.get(orientation) is None:
            
            X = np.load(image_path)['arr_0']
            X = torch.from_numpy((X/X.max()).copy())
            X = X.view(-1,1, 100,100) #Transforms 100x100 in 1x100x100. Image with one channel.
            X.requires_grad = True
            X = X.to(self.device)
            self.images[orientation] = X
            del [X]
        return self.images[orientation]
    
    def _get_model(self,orientation,model_name,model_path):
        if self.models.get(orientation) is None:
            self.models[orientation] = load_trained_model(model=model_name,model_path=model_path,device=self.device)
            self.models[orientation].zero_grad()
        return self.models[orientation]
    
    def _explain_image(self,orientation,net, image, algorithm):
        
        transpose_array = (2,1,0)
        net.zero_grad()
        explain_input = image.unsqueeze(0)
        original_image = np.transpose(explain_input.squeeze(0).cpu().detach().numpy(), transpose_array)
        original_image = np.rot90(original_image)
        
        if self.explanations.get(orientation + '_' + algorithm) is not None:
            explanation_image = self.explanations[orientation + '_' + algorithm]
            _ = viz.visualize_image_attr(explanation_image, original_image, method="blended_heat_map", sign="absolute_value", 
                                    outlier_perc=10, show_colorbar=True, 
                                    title=f"Overlayed {algorithm} explanation for {self.orientation} orientation")
            return 
            
        if algorithm == 'IntegratedGradients':
            ig = IntegratedGradients(net)
            nt = NoiseTunnel(ig)
            explanation_image = self._attribute_image_features(nt, explain_input,labels=0, baselines=explain_input * 0, nt_type='smoothgrad_sq',
                                                nt_samples=100, stdevs=0.2,internal_batch_size=10)
        elif algorithm == 'DeepLift':
            dl = DeepLift(net)
            explanation_image = self._attribute_image_features(dl, explain_input,labels=0, baselines=explain_input * 0)
        else:
            raise("Explanation options available are DeepLift or IntegratedGradients")
        
        #TODO: Remove these 90º rotations after fixing input image rotation issue
        explanation_image = np.transpose(explanation_image.squeeze(0).cpu().detach().numpy(), transpose_array)
        explanation_image = np.rot90(explanation_image)

        _ = viz.visualize_image_attr(explanation_image, original_image, method="blended_heat_map", sign="absolute_value", 
                                    outlier_perc=10, show_colorbar=True, 
                                    title=f"Overlayed {algorithm} explanation for {orientation} orientation")
        
        self.explanations[orientation + '_' + algorithm] = explanation_image
        
    def _attribute_image_features(self,algorithm, input, labels, **kwargs):
        
        tensor_attributions = algorithm.attribute(input,
                                                target=labels,
                                                **kwargs
                                                )
        return tensor_attributions

# prediction_reference = ''
# explainer = MRIExplainer(image_id='I689023',
#                         prediction_reference=prediction_reference,
#                         device=device)

# explainer.explain(orientation='coronal',algorithm=DeepLift)
# explainer.explain(orientation='axial',algorithm=DeepLift)
# explainer.explain(orientation='sagittal',algorithm=DeepLift)
# explainer.explain(algorithm=DeepLift)

def show_images(dataloader):

    # get some random training images
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % labels[j] for j in range(batch_size)))

def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
   
