import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



class EnsembleExplainer:
    '''
    Explain the Ensemble CNN+Demographics classification.

    Parameters
    ----------

    model: Trained model (Explainable Boosting machine)

    ensemble_data: Dataframe containing features and target about the ensemble.

    patients_data: Dataframe containing information about the patients to enhance explanation views.

    '''
    
    def __init__(self,model,ensemble_data,patients_data,
                axial_label='AXIAL_23',coronal_label='CORONAL_43',sagittal_label='SAGITTAL_26'):
        self.ensemble_data = ensemble_data
        self.patients_data = patients_data
        self.model = model
        self.axial_label=axial_label
        self.coronal_label=coronal_label
        self.sagittal_label=sagittal_label


    def explain(self,sample_id,top_features=10,figsize=(3.5,5),show_true_diagnosis=True):
        '''
        Shows a local explanation plot for the EBM along with some relevant patient information.
        
        sample_id: IMAGE_DATA_ID ou IMAGEUID to be explained. This is the ID on the Ensemble dataset.
        
        top_features: Number to show only N top features.

        figsize: tuple to set size of the plot (width,height).
        '''
        df_ensemble = self.ensemble_data.query("IMAGE_DATA_ID == @sample_id")
        df_patient = self.patients_data.query("IMAGE_DATA_ID == @sample_id")
        local_explanation = self.model.explain_local(df_ensemble.drop('DIAGNOSIS',axis=1),df_ensemble['DIAGNOSIS'])._internal_obj['specific'][0]
        
        feature_names = local_explanation['names']
        feature_importances = local_explanation['scores']
        
        df_weights_ebm = pd.DataFrame(index = feature_names,columns = ['Weights'])
        df_weights_ebm['Weights'] = feature_importances
        df_weights_ebm['abs_Weights'] = np.abs(df_weights_ebm['Weights'])
        df_weights_ebm = df_weights_ebm.sort_values(by=['abs_Weights'],ascending = True, inplace = False)
            
        if top_features is not None:
            df_weights_ebm = df_weights_ebm.iloc[-top_features:]

        fig,ax = plt.subplots(1,figsize=figsize)
        
        color = ('r','g')
        color_mask = df_weights_ebm['Weights'] > 0
        bar_colors = [color[0] if c else color[1] for c in color_mask]
        legend_elements = [
            Patch(facecolor = color[1], edgecolor='k', label='Negative Contribution'),
            Patch(facecolor = color[0], edgecolor='k', label='Positive Contribution')]

        df_weights_ebm['Weights'].plot(kind = 'barh', grid = True,color = bar_colors, edgecolor='k', alpha = 0.6,ax = ax)
        ax.set_yticklabels(labels=df_weights_ebm.index.tolist(),fontdict={'size':12,'weight':'heavy'})

        plt.grid(True)

        ax.legend(handles=legend_elements, bbox_to_anchor=(1, -0.05), borderaxespad=0.,fancybox=True, shadow=True,ncol = 2)

        plt.subplots_adjust(top=0.9, bottom=0, left=0.10, right=0.95, hspace=0.40, wspace=0.85)

        features = df_patient.iloc[0].to_dict()
        age,gender,years_education = features['AGE'],features['GENDER'],features['YEARS_EDUCATION']
        hispanic,race,widowed = features['HISPANIC'],features['RACE'],features['WIDOWED']
        true_diagnosis,predicted_score,predicted_diagnosis = features['DIAGNOSIS'],features['FINAL_PREDICTED_SCORE'],features['FINAL_PREDICTION']
        axial,coronal,sagittal = features[self.axial_label],features[self.coronal_label],features[self.sagittal_label]
        
        if figsize[0] >= figsize[1]:
            fig.suptitle(f'Diagnose Explanations - ID: {sample_id}',fontsize = 20,x=0.35)
            plt.figtext(-0.2, -0.2, f'Is Hispanic:{hispanic}    Race:{race}   Is Widowed:{widowed}   Age:{age}    Gender:{gender}   Years of Education:{years_education}', horizontalalignment='left',fontdict={'size':16})
            plt.figtext(-0.2, -0.3, f'{self.axial_label} score:{axial:.4f}    {self.coronal_label} score:{coronal:.4f}   {self.sagittal_label} score:{sagittal:.4f}', horizontalalignment='left',fontdict={'size':16})
            
            diagnosis_text = f'Diagnosis: {predicted_diagnosis}'
            if show_true_diagnosis:
                diagnosis_text = f'Diagnosis: {predicted_diagnosis}   True Class:{true_diagnosis}'
            plt.figtext(-0.2, -0.4, diagnosis_text, horizontalalignment='left',fontdict={'size':16,'weight':'heavy'})
        else:
            fig.suptitle(f'Diagnose Explanations - ID: {sample_id}',fontsize = 20,x=0.15)
            hor_align='right'
            hor_margin=0.9
            
            plt.figtext(hor_margin, -0.2, f'Is Hispanic:{hispanic}   Race:{race}', horizontalalignment=hor_align,fontdict={'size':16})
            plt.figtext(hor_margin, -0.3, f'Is Widowed:{widowed}   Age:{age}', horizontalalignment=hor_align,fontdict={'size':16})
            plt.figtext(hor_margin, -0.4, f'Gender:{gender}   Years of Education:{years_education}', horizontalalignment=hor_align,fontdict={'size':16})
            plt.figtext(hor_margin, -0.5, f'{self.sagittal_label} score:{sagittal:.4f}', horizontalalignment=hor_align,fontdict={'size':16})
            plt.figtext(hor_margin, -0.6, f'{self.coronal_label} score:{coronal:.4f}', horizontalalignment=hor_align,fontdict={'size':16})
            plt.figtext(hor_margin, -0.7, f'{self.axial_label} score:{axial:.4f}', horizontalalignment=hor_align,fontdict={'size':16})

            diagnosis_text = f'Diagnosis: {predicted_diagnosis}'
            if show_true_diagnosis:
                diagnosis_text = f'Diagnosis: {predicted_diagnosis}   True Class:{true_diagnosis}'

            plt.figtext(hor_margin, -0.8, diagnosis_text, horizontalalignment=hor_align,fontdict={'size':16,'weight':'heavy'})
        return fig

    def compare_patients_explanations(self,sample_ids,top_features=10,figsize=(9,5)):
        '''
        Compares local explanations between two patients, along with some relevant demographics information.
        
        sample_ids: tuple or list of IMAGE_DATA_ID/IMAGEUID to be explained. This is the ID on the Ensemble dataset.
        
        top_features: Number to show only N top features.

        figsize: tuple to set size of the plot (width,height).
        '''

        fig,ax = plt.subplots(1,2,figsize=figsize)
        fig.suptitle(f'Local Explanations - EBM',fontsize = 20,x=0.5)

        for ii,sample_id in enumerate(sample_ids):
            df_ensemble = self.ensemble_data.query("IMAGE_DATA_ID == @sample_id")
            df_patient = self.patients_data.query("IMAGE_DATA_ID == @sample_id")
            local_explanation = self.model.explain_local(df_ensemble.drop('DIAGNOSIS',axis=1),df_ensemble['DIAGNOSIS'])._internal_obj['specific'][0]
            
            feature_names = local_explanation['names']
            feature_importances = local_explanation['scores']
            
            df_weights_ebm = pd.DataFrame(index = feature_names,columns = ['Weights'])
            df_weights_ebm['Weights'] = feature_importances
            df_weights_ebm['abs_Weights'] = np.abs(df_weights_ebm['Weights'])
            df_weights_ebm = df_weights_ebm.sort_values(by=['abs_Weights'],ascending = True, inplace = False)
                
            if top_features is not None:
                df_weights_ebm = df_weights_ebm.iloc[-top_features:]
        
            color = ('r','g')
            color_mask = df_weights_ebm['Weights'] > 0
            bar_colors = [color[0] if c else color[1] for c in color_mask]
            legend_elements = [
                Patch(facecolor = color[1], edgecolor='k', label='Negative Contribution'),
                Patch(facecolor = color[0], edgecolor='k', label='Positive Contribution')]
            df_weights_ebm['Weights'].plot(kind = 'barh', grid = True,color = bar_colors, edgecolor='k', alpha = 0.6,ax = ax[ii])

            ax[ii].legend(handles=legend_elements, bbox_to_anchor=(1, -0.05), borderaxespad=0.,fancybox=True, shadow=True,ncol = 2)
            
            ax[ii].set_yticklabels(labels=df_weights_ebm.index.tolist(),fontdict={'size':12,'weight':'heavy'})
            ax[ii].set_title(f'sample_ID: {sample_id}',fontdict={'size':16})
            # plt.grid(True)

            features = df_patient.iloc[0].to_dict()
            age,gender,years_education = features['AGE'],features['GENDER'],features['YEARS_EDUCATION']
            hispanic,race,widowed = features['HISPANIC'],features['RACE'],features['WIDOWED']
            true_diagnosis,predicted_score,predicted_diagnosis = features['DIAGNOSIS'],features['FINAL_PREDICTED_SCORE'],features['FINAL_PREDICTION']
            axial,coronal,sagittal = features[self.axial_label],features[self.coronal_label],features[self.sagittal_label]
            
            hor_align = -0.1 if ii == 0 else 0.5
            plt.figtext(hor_align, -0.2, f'Is Hispanic:{hispanic}   Race:{race}', horizontalalignment='left',fontdict={'size':16})
            plt.figtext(hor_align, -0.3, f'Is Widowed:{widowed}   Age:{age}', horizontalalignment='left',fontdict={'size':16})
            plt.figtext(hor_align, -0.4, f'Gender:{gender}   Years of Education:{years_education}', horizontalalignment='left',fontdict={'size':16})
            plt.figtext(hor_align, -0.5, f'{self.sagittal_label} score:{sagittal:.4f}', horizontalalignment='left',fontdict={'size':16})
            plt.figtext(hor_align, -0.6, f'{self.coronal_label} score:{coronal:.4f}', horizontalalignment='left',fontdict={'size':16})
            plt.figtext(hor_align, -0.7, f'{self.axial_label} score:{axial:.4f}', horizontalalignment='left',fontdict={'size':16})
            plt.figtext(hor_align, -0.8, f'Diagnosis: {predicted_diagnosis}   True Class:{true_diagnosis}', horizontalalignment='left',fontdict={'size':16,'weight':'heavy'})

        plt.subplots_adjust(top=0.9, bottom=0, left=0.10, right=0.95, hspace=0.40, wspace=0.85)
        return fig

    def _get_patient_reference(self,sample_id):
        return self.ensemble_data.query("IMAGE_DATA_ID == @sample_id")

def prepare_patient_data_for_explanations(patient_data_path,df_ensemble,ebm,cutoff,
                                        positive_case = 'AD',axial_label='AXIAL_23',
                                        coronal_label='CORONAL_43',sagittal_label='SAGITTAL_26',
                                        label='DIAGNOSIS'):

    predicted_probas = ebm.predict_proba(df_ensemble.drop(label,axis=1))[:,-1]

    df_essential_data = df_ensemble[[axial_label,coronal_label,sagittal_label,label]]
    df_essential_data.loc[:,'FINAL_PREDICTED_SCORE'] = np.round(predicted_probas * 10000000) / 10000000
    df_essential_data.loc[:,'FINAL_PREDICTION'] = [1  if x>=cutoff else 0 for x in predicted_probas]


    df_patient_data = pd.read_csv(patient_data_path)
    imageid = df_patient_data['IMAGEUID']
    df_patient_data['IMAGE_DATA_ID'] = ['I'+str(x) for x in imageid]
    df_patient_data.set_index("IMAGE_DATA_ID",inplace=True)
    df_patient_data = df_patient_data[['AGE','MALE','YEARS_EDUCATION','HISPANIC', 'RACE','WIDOWED']]

    df_patient_data.rename(columns={'MALE':'GENDER'},inplace=True)
    df_patient_data['RACE'] = df_patient_data['RACE'].str.upper()

    df_patient_data.loc[df_patient_data['HISPANIC'] == 1,'HISPANIC'] = 'YES'
    df_patient_data.loc[df_patient_data['HISPANIC'] == 0,'HISPANIC'] = 'NO'

    df_patient_data.loc[df_patient_data['WIDOWED'] == 1,'WIDOWED'] = 'YES'
    df_patient_data.loc[df_patient_data['WIDOWED'] == 0,'WIDOWED'] = 'NO'

    df_patient_data.loc[df_patient_data['GENDER'] == 1,'GENDER'] = 'MALE'
    df_patient_data.loc[df_patient_data['GENDER'] == 0,'GENDER'] = 'FEMALE'

    df_patient_data = df_patient_data.merge(df_essential_data,right_index=True,left_index=True)

    df_patient_data.loc[df_patient_data[label] == 1,label] = positive_case
    df_patient_data.loc[df_patient_data[label] == 0,label] = 'CN'

    df_patient_data.loc[df_patient_data['FINAL_PREDICTION'] == 1,'FINAL_PREDICTION'] = positive_case
    df_patient_data.loc[df_patient_data['FINAL_PREDICTION'] == 0,'FINAL_PREDICTION'] = 'CN'

    df_patient_data[axial_label] = [np.round(x*10000000)/10000000 for x in df_patient_data[axial_label]]
    df_patient_data[coronal_label] = [np.round(x*10000000)/10000000 for x in df_patient_data[coronal_label]]
    df_patient_data[sagittal_label] = [np.round(x*10000000)/10000000 for x in df_patient_data[sagittal_label]]
    return df_patient_data


def plot_global_explanations(features_lr,coefficients_lr,features_ebm,coefficients_ebm,title,
                            normalized=True,figsize=(9,4),top_features=10,vertical_space=0.85,horizontal_space=0.85):
    df_weights_lr = prepare_feature_importance(features_lr,coefficients_lr)
    df_weights_ebm = prepare_feature_importance(features_ebm,coefficients_ebm)
    
    if normalized:
        column = 'normalized'
    else:
        column = 'Weights'
        
    if top_features is not None:
        df_weights_lr = df_weights_lr.iloc[-top_features:]
        df_weights_ebm = df_weights_ebm.iloc[-top_features:]

    fig,ax = plt.subplots(1,2,figsize=figsize)
    color = ('r','g')
    colors = (0.2,0.4,0.8)

    if color is not None:
        color_mask = df_weights_lr['normalized'] > 0
        colors = [color[0] if c else color[1] for c in color_mask]
        legend_elements = [
            Patch(facecolor = color[1], edgecolor='k', label='Negative Contribution'),
            Patch(facecolor = color[0], edgecolor='k', label='Positive Contribution')]

    ax[0].tick_params(axis = 'both',labelsize = 'large')
    ax[1].tick_params(axis = 'both',labelsize = 'large')
    df_weights_lr[column].plot(kind = 'barh',title='LR', grid = True, color = colors,edgecolor='k', alpha = 0.6,ax = ax[0])
    df_weights_ebm[column].plot(kind = 'barh',title='EBM', grid = True,edgecolor='k', alpha = 0.6,ax = ax[1])

    plt.grid(True)
    fig.suptitle('Global Explanations - '+title,fontsize = 20,x=0.45)

    if color:
        ax[0].legend(handles=legend_elements, bbox_to_anchor=(1, -0.05), borderaxespad=0.,fancybox=True, shadow=True,ncol = 2)

    plt.subplots_adjust(top=vertical_space, bottom=0, left=0.10, right=0.95, hspace=0.40, wspace=horizontal_space)
    plt.show()
    return fig

def prepare_feature_importance(features,coefficients):
    df_weights = create_normalized_by_feature_weight(features,coefficients);
    
    df_weights = df_weights.sort_values(by=['abs_Weights'],ascending = True,inplace = False)
    return df_weights



def show_feature_weights(features, coefficients,model_title, color = None, absolute_values = False, normalized = False,figsize=(8,8),top=None):

    '''
    Show a feature importance bar plot by feature weights.
    

    Parameters:
    -----------

    features: vector with the feature names

    coefficients: 1-D array values of weights given to features.

    model_title: Name of the model

    color: Tuple to give different colors to positive and negative weights. Example: color = ('red','green')

    absolute_values: Flag to analyse just absolute values of weights

    normalized: Flag to normalize the feature weights 


    top: show only top most important features.

    '''
    
    df_weights = create_normalized_by_feature_weight(features,coefficients);
    column = 'Weights'
    if absolute_values:
        column = 'abs_Weights'
    if normalized:
        column = column.replace('Weights','normalized')

    df_weights = df_weights.sort_values(by=[column],ascending = True,inplace = False)
    
    if top is not None:
        df_weights = df_weights.iloc[:top]
    
    fig,ax = plt.subplots(1,figsize=figsize)
    colors = (0.2,0.4,0.8)
    if color is not None:
        color_mask = df_weights['normalized'] > 0
        colors = [color[0] if c else color[1] for c in color_mask]
        legend_elements = [
            Patch(facecolor = color[1], edgecolor='k', label='Negative Contribution'),
            Patch(facecolor = color[0], edgecolor='k', label='Positive Contribution')]
    
    ax.tick_params(axis = 'both',labelsize = 'large')
    df_weights[column].plot(kind = 'barh', grid = True, color = colors,edgecolor='k', alpha = 0.6,ax = ax)
    fig.suptitle(f'Feature Weights - {model_title}',x = 0.3,fontsize = 20)

    if color:
        ax.legend(handles=legend_elements, bbox_to_anchor=(1, -0.05), borderaxespad=0.,fancybox=True, shadow=True,ncol = 2)

    if normalized:
        plt.figtext(0.91, 0.03, '*All values add up to one', horizontalalignment='right')
    if absolute_values:
        plt.figtext(0.91, 0.01, '*All values are absolute', horizontalalignment='right')

    plt.subplots_adjust(top=0.93, bottom=0, left=0.10, right=0.95, hspace=0.40, wspace=0.35)
    plt.show()

def create_normalized_by_feature_weight(features,coefficients):
    
    '''
        Create a Dataframe based on the Weights given to each feature by linear models, with the following columns:
          Weights: Weights of each feature
          abs_Weights: Absolute value of the weights
          normalized: Normalized weights (all values of this column add up to one)
          abs_normalized: Absolute value of the feature importance
      
        Parameters:
          features: 1-D array with the feature names
          coefficients: 1-D array values of weights given to features.
    '''

    df_weights = pd.DataFrame(index = features,columns = ['Weights'])
    df_weights['Weights'] = coefficients
    df_weights['abs_Weights'] = np.abs(df_weights['Weights'])
    total_weights = df_weights['abs_Weights'].values.ravel().sum()
    df_weights['normalized'] = df_weights['Weights'].values / total_weights
    df_weights['abs_normalized'] = df_weights['abs_Weights'].values / total_weights
    df_weights.sort_values(by=['abs_normalized'],ascending = False,inplace = True)
    return df_weights


