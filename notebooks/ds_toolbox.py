# Convenience functions

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from IPython import get_ipython

def pretty_notebook(disable_warnings=True):
    """
    Configures the output to display nicely on a retina MAC. Configures graphics to be displayed inlined,
    configures the context to seaborn's pallettes. Configures pandas warnings to a minimum.
    It hides FutureWarning to be displayed (especially useful for "deliverable" notebooks, however when developing
    we might want to see all warnings)
    """
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"') 
    # A hack to make the graphics look nicer on a MAC
    sns.set()

    # Pandas has an annyoing warning for chained assignments like df['Column_New'] = df['Column_Old']
    # The line below disables it
    pd.options.mode.chained_assignment = None
    
    if disable_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)

def pretty_heatmap(confusion_matrix, classes, cmap="Y1GnBu"):
    """
    
    Prints a pretty heatmap. Based on: 
    https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    
    """
    heatmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlGnBu",
                          xticklabels=classes,
                          yticklabels=classes)
    
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')


def pretty_feature_importance(feature_importances, feature_names, top=10, signs = []):
    """
    Bar plot of feature importance given feature names
    
    :param feature_importances: a 1-D list with the feature importances
    :param feature_names: list with names for the features (size has to match
                          with the one of feature_importances)
    :param top: integer > 0 - how many features to display
    :param signs: a list with the same size as feature importances containing [0, -1, 1]
                  for the sign of each importance
    """
    plt.figure(figsize=(8,8))
    idx_sorted = np.argsort(feature_importances)[::-1]
    x = np.array(feature_importances)[idx_sorted][:top]
    y = np.array(feature_names)[idx_sorted][:top]
    
    if len(signs) > 0:
        x = x*np.array(signs)[idx_sorted][:top]
    

    sns.barplot(y=y,x=x,orient='h');

def get_minutes(time):
    """
    Gets a string in datetime-ish format and calculate the 'minute of the day'
    
    :param time: a string  with time
    """
    
    parsed_time = parser.parse(time)
    return 60*parsed_time.hour + parsed_time.minute

def get_month(time):
    return parser.parse(time).month
