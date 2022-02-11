import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz

sys.path.append("./../models")
from neural_network import load_model,load_trained_model,device


class MRIExplainer:
    
    '''
    Explain an MRI CNN classification.

    Parameters
    ----------

    image_id: IMAGE_DATA_ID ou IMAGEUID to be explained.

    prediction_reference: Dataframe reference containing the predictions, true labels and 
    model location for the classification problem. Can be either a path or dataframe object.

    device: Device to load the model and images. Options are 'cuda' or 'cpu'.
    '''

    def __init__(self,image_id,prediction_reference,device=device):
        self.image_id = image_id
        self.device = device
        self.prediction_reference = prediction_reference

        self.images = {}
        self.models = {}
        self.explanations = {}
        self.orientations = []
        self.transpose_array = (2,1,0)


    def explain_all_orientations(self,figsize=(15,6),original_image_overlay=0.7,outlier_scale=10,separate_negative_contributions = False):
        explain_kwargs={'figsize':figsize,'original_image_overlay':original_image_overlay,
                        'outlier_scale':outlier_scale,'separate_negative_contributions':separate_negative_contributions}
        self.explain_one_orientation(orientation='sagittal',**explain_kwargs)
        self.explain_one_orientation(orientation='coronal',**explain_kwargs)
        self.explain_one_orientation(orientation='axial',**explain_kwargs)


    def explain_one_orientation(self,orientation='coronal',figsize=(15,6),original_image_overlay=0.7,outlier_scale=10,separate_negative_contributions = False):
        
        '''
        Explain one orientation prediction with DeepLift and GradCAM algorithms. Plots 3 or 4 figures side by side.
        
        '''

        img_orientation = self.image_reference.query("ORIENTATION == @orientation").iloc[0]
        net = self._get_model(orientation, model_name=img_orientation['MODEL'], model_path=img_orientation['MODEL_PATH']);
        image = self._get_image(orientation,image_path = img_orientation['IMAGE_PATH'])

        attr_dl = self._make_deeplift_explanation(image,net)
        attr_gc = self._make_gradcam_explanation(image,net)

        original_image = np.transpose(image.squeeze(0).cpu().detach().numpy(), self.transpose_array)
        original_image = np.rot90(original_image)

        orientation_label = orientation +'_' +str(img_orientation['SLICE'])
        
        self._show_explanations(original_image,attr_gc,attr_dl,self.image_id,orientation_label,
                                true_label = img_orientation['MACRO_GROUP'],
                                score=img_orientation['CNN_SCORE'],
                                prediction=img_orientation['CNN_PREDICTION'],
                                figsize=figsize,
                                original_image_overlay=original_image_overlay,
                                outlier_scale=outlier_scale,
                                separate_negative_contributions = separate_negative_contributions)
    
    @property
    def image_reference(self):
        df_ref = self.prediction_reference
        if isinstance(self.prediction_reference,str):
            df_ref = pd.read_csv(self.prediction_reference)
        df_ref = df_ref.loc[df_ref['IMAGE_DATA_ID'] == self.image_id,:]
        df_ref.columns = df_ref.columns.str.upper()
        return df_ref

    def _get_image(self,orientation,image_path):
        if self.images.get(orientation) is None:
            
            X = np.load(image_path)['arr_0']
            X = torch.from_numpy((X/X.max()).copy())
            X = X.view(-1,1, 100,100) #Transforms 100x100 in 1x100x100. Image with one channel.
            X.requires_grad = True
            X = X.to(self.device)
        return X
    
    def _get_model(self,orientation,model_name,model_path):
        if self.models.get(orientation) is None:
            self.models[orientation] = load_trained_model(model=model_name,model_path=model_path,device=self.device);
            self.models[orientation].zero_grad();
        return self.models[orientation]
    
    def _make_gradcam_explanation(self,image,net):
        gc = GuidedGradCam(net,layer=net.features[-1])
        attr_gc = gc.attribute(image,interpolate_mode='area')
        attr_gc = np.transpose(attr_gc.squeeze(0).cpu().detach().numpy(), self.transpose_array)
        attr_gc = np.rot90(attr_gc)
        return attr_gc

    def _make_deeplift_explanation(self,image,net,smoothing_samples=100):
        dl = DeepLift(net)
        nt = NoiseTunnel(dl)

        blur = GaussianBlur((5,5), sigma=1)
        ref_image = blur(image.detach().clone().to(self.device))

        attr_dl = nt.attribute(image, baselines=ref_image,  nt_type='smoothgrad',
                                            nt_samples=smoothing_samples, stdevs=0.1)
        attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), self.transpose_array)
        attr_dl = np.rot90(attr_dl)
        return attr_dl
    
    def _show_explanations(self,plot_image,gradcam_explanation,deeplift_explanation,sample_id,orientation,
                           true_label,score,prediction,figsize=(17,8),original_image_overlay=0.7,outlier_scale=10,
                           separate_negative_contributions=True):
        '''
        Shows overlayed local explanations side by side with the original image.
        '''

        n_plots=3
        title_y_position = 0.91
        if separate_negative_contributions: 
          n_plots=4
          title_y_position=0.82
        
        fig, ax = plt.subplots(1,n_plots)
        fig.set_size_inches(figsize)

        fig.suptitle(f"Local {orientation.upper()} explanations - True label: {true_label} - Predicted label: {prediction} - Predicted score: {score:.4f}",y=title_y_position,fontsize=18,horizontalalignment='center')

        fig,ax0 = viz.visualize_image_attr(None, plot_image, method="original_image",
                                    plt_fig_axis =  (fig, ax[0]),
                                    show_colorbar=True,
                                    use_pyplot=False)
        ax0.set_title("Original Image",fontdict={'size':18})


        gradcam_method="blended_heat_map"
        if gradcam_explanation.max() == 0: 
            gradcam_explanation = None
            gradcam_method = 'original_image'
        fig,ax1 = viz.visualize_image_attr(gradcam_explanation, plot_image, method=gradcam_method,
                                    show_colorbar=True,
                                    plt_fig_axis =  (fig, ax[1]),
                                    outlier_perc=outlier_scale,
                                    alpha_overlay=original_image_overlay,
                                    sign='positive',
                                    use_pyplot=False)
        ax1.set_title("Overlayed Guided GradCam",fontdict={'size':18})

        if separate_negative_contributions:
          fig,ax2 = viz.visualize_image_attr(deeplift_explanation, plot_image, method="blended_heat_map",
                                      show_colorbar=True,
                                      plt_fig_axis =  (fig, ax[2]),
                                      outlier_perc=outlier_scale,
                                      alpha_overlay=original_image_overlay,
                                      sign='positive',
                                      use_pyplot=False)
          ax2.set_title("Overlayed DeepLift (Positive)",fontdict={'size':18})

          fig,ax3 = viz.visualize_image_attr(deeplift_explanation, plot_image, method="blended_heat_map",
                                      show_colorbar=True,
                                      plt_fig_axis =  (fig, ax[3]),
                                      outlier_perc=outlier_scale,
                                      alpha_overlay=original_image_overlay,
                                      sign='negative',
                                      use_pyplot=False)
          ax3.set_title("Overlayed DeepLift (Negative)",fontdict={'size':18})
        else:
          fig,ax2 = viz.visualize_image_attr(deeplift_explanation, plot_image, method="blended_heat_map",
                                      show_colorbar=True,
                                      plt_fig_axis =  (fig, ax[2]),
                                      outlier_perc=outlier_scale,
                                      alpha_overlay=original_image_overlay,
                                      sign='all',
                                      use_pyplot=False)
          ax2.set_title("Overlayed DeepLift",fontdict={'size':18})

        plt.show()

    def _get_score_and_prediction(self,net,image,threshold):
        y_pred_proba = torch.sigmoid(net(image)).cpu().detach().numpy()
        y_pred_label = y_pred_proba.copy()
        y_pred_label[y_pred_proba >= threshold] = 1
        y_pred_label[y_pred_proba < threshold] = 0
        
        return y_pred_proba[0], y_pred_label[0]

class MRIDiagnosisExplainer(MRIExplainer):
    '''
    Explain the MRI diagnosis given by all three CNNs.

    Parameters
    ----------

    image_id: IMAGE_DATA_ID ou IMAGEUID to be explained.

    prediction_reference: Dataframe reference containing the predictions, true labels and 
    model location for the classification problem. Can be either a path or dataframe object.

    device: Device to load the model and images. Options are 'cuda' or 'cpu'.
    '''

    def __init__(self,image_id,prediction_reference='',device=device) -> None:
        MRIExplainer.__init__(self,image_id,prediction_reference=prediction_reference,device=device)
    
    def explain_diagnosis(self,algorithm='DeepLift',figsize=(15,6),original_image_overlay=0.7,outlier_scale=10):
        
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(figsize)
        fig.suptitle(f"MRI explanations",y=0.91,fontsize=22,horizontalalignment='center')
        
        for ii,orientation in enumerate(['sagittal','axial','coronal']):
            img_info = self.image_reference.query("ORIENTATION == @orientation").iloc[0]
            net = self._get_model(orientation, model_name=img_info['MODEL'], model_path=img_info['MODEL_PATH']);
            image = self._get_image(orientation,image_path = img_info['IMAGE_PATH'])
            
            method="blended_heat_map"
            if algorithm == 'DeepLift':
              
                explanation = self._make_deeplift_explanation(image,net,smoothing_samples=20)
                sign = 'all'
            else:
                explanation = self._make_gradcam_explanation(image,net)
                sign = 'positive'
                if explanation.max() == 0: 
                    explanation = None
                    method = 'original_image'

            plot_image = np.transpose(image.squeeze(0).cpu().detach().numpy(), self.transpose_array)
            plot_image = np.rot90(plot_image)

            fig,ax[ii] = viz.visualize_image_attr(explanation, plot_image, method=method,
                                        show_colorbar=True,
                                        plt_fig_axis =  (fig, ax[ii]),
                                        outlier_perc=outlier_scale,
                                        alpha_overlay=original_image_overlay,
                                        sign=sign,
                                        use_pyplot=False)
            ax[ii].set_title(f"{orientation.upper()} Slice",fontdict={'size':18})

        plt.show()