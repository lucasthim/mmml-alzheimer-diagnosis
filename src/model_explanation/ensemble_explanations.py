import math

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.special import erfcinv
from scipy.interpolate import interp1d
from statsmodels.stats import proportion


def show_feature_weights(features, coefficients,model_title, color = None, absolute_values = False, normalized = False,figsize=(8,8),top=None):

    '''
      Show a feature importance bar plot by feature weights.
      
      Parameters:
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


def calculate_sensibility_at_level(tpr, fpr, level):
    level_fpr = 1 - level  # fpr is (1-specificity)
    f_sens = interp1d(fpr, tpr)  # interpolate sensibility (tpr = sensibility)
    return (f_sens(level_fpr))


def find_optimal_cutoff(fpr, tpr, thresholds):
    """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        fpr: False positive rate

        tpr : True positive rate

        Returns
        -------
        cutoff value

        """
    #optimal_idx = np.argmax(tpr - fpr)
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))  # Minimum distance to the upper left corner (By Pathagoras' theorem)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    return optimal_sensitivity, optimal_specificity, optimal_threshold


def se_auc(auc, cls):
    """
    Standard error of area
    :param auc: area under the curve
    :param cls: the column of the tag: unhealthy (1) and healthy (0)
    :return: standard error
    """
    auc2 = auc ** 2
    q1 = auc / (2 - auc)
    q2 = 2 * auc2 / (1 + auc)
    lu = sum(cls == 1)  # Number of unhealthy subjects (class == 1)
    lh = sum(cls == 0)  # Number of healthy subjects (class == 0)
    V = (auc * (1 - auc) + (lu - 1) * (q1 - auc2) + (lh - 1) * (q2 - auc2)) / (lu * lh)
    se = math.sqrt(V)
    return se


def ci_auc(auc, se, alpha=0.05):
    """
    Confidence interval of AUC
    :param auc: area under the curve
    :param se: standard error
    :param alpha: significance level (default = 0.05)
    :return: confidence interval
    """
    ci_lo = auc + (-1 * math.sqrt(2) * erfcinv(alpha) * se)
    ci_up = auc + (math.sqrt(2) * erfcinv(alpha) * se)
    return ci_lo, ci_up


def ci_sen(optimal_sensitivity, cls):
    """
    Confidence interval of Sensitivity using Simple Asymptotic
    :param optimal_sensitivity: optimal cutoff point
    :param cls: the column of the tag: unhealthy (1) and healthy (0)
    :return: confidence interval - array[(low, high)]
    """
    num_u = sum(cls == 1)  # Number of unhealthy subjects (class == 1)
    sa = 1.96 * math.sqrt(optimal_sensitivity * (1 - optimal_sensitivity) / num_u)
    ci_sen = np.zeros(2)
    ci_sen = [optimal_sensitivity - sa, optimal_sensitivity + sa]
    return ci_sen


def ci_spe(optimal_specificity, cls):
    """
        Confidence interval of Specificity using Simple Asymptotic
        :param optimal_specificity: optimal cutoff point
        :param cls: the column of the tag: unhealthy (1) and healthy (0)
        :return: confidence interval - array[(low, high)]
    """
    num_h = sum(cls == 0)  # Number of healthy subjects (class == 0)
    sa = 1.96 * math.sqrt(optimal_specificity * (1 - optimal_specificity) / num_h)
    ci_spe = np.zeros(2)
    ci_spe = [optimal_specificity - sa, optimal_specificity + sa]
    return ci_spe


def plotroc(df, models, levels=[0.75,0.9],label='DIAGNOSIS'):

    roc_df = pd.DataFrame(columns=['SensLevel0', 'SensLevel1',
                                   'AUC', 'AucCI_lo', 'AucCI_hi',
                                   'SE',
                                   'OpSen', 'SenCI_lo', 'SenCI_hi',
                                   'OpSpe', 'SpeCI_lo', 'SpeCI_hi'])
    classes = df[label]

    fig =plt.figure(figsize=(8,8))

    for model in models:

        sens = [math.nan, math.nan]  # create a list to hold the sensibility
        # model = m['model']  # select the model
        y_proba = model.predict(df.drop(label,axis=1))

        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_proba, drop_intermediate=False)
        # Calculate Area under the curve to display on the plot
        # auc = metrics.roc_auc_score(y_test, model.predict(x_test))
        roc_df.loc[model, 'AUC'] = metrics.auc(fpr, tpr)

        # calculate the sensitivity at levels
        roc_df.loc[model, 'SensLevel0'] = calculate_sensibility_at_level(tpr, fpr, levels[0])
        roc_df.loc[model, 'SensLevel1'] = calculate_sensibility_at_level(tpr, fpr, levels[1])

        # Calculate the standard error of AUC
        roc_df.loc[model, 'SE'] = se_auc(roc_df.loc[model, 'AUC'], classes)

        # Calculate the confidence interval of AUC
        roc_df.loc[model, ['AucCI_lo', 'AucCI_hi']] = ci_auc(roc_df.loc[model, 'AUC'], roc_df.loc[model, 'SE'])

        # Calculate the optimal cutoff point, Sensitivity and specificity
        roc_df.loc[model, 'OpSen'], roc_df.loc[model, 'OpSpe'], optimal_threshold = find_optimal_cutoff(fpr, tpr, thresholds)

        # Calculate the confidence interval of Sensitivity
        roc_df.loc[model, ['SenCI_lo', 'SenCI_hi']] = ci_sen(roc_df.loc[model, 'OpSen'], classes)

        # Calculate the confidence interval of Specificity
        roc_df.loc[model, ['SpeCI_lo', 'SpeCI_hi']] = ci_spe(roc_df.loc[model, 'OpSpe'], classes)

        # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        # roc_auc = metrics.auc(fpr, tpr)
        # Now, plot the computed values
        model_name = type(model).__name__
        plt.plot(fpr, tpr, label=model_name)
    # Custom settings for the plot
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()  # Display
    #plt.savefig(filename, format='png', dpi=300)
    return roc_df, fig
