import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir


def load_tensorboard(path):
    '''Function to load tensorboard file from a folder.
    Assumes one file per folder!'''

    event_file = next(filter(lambda filename: filename[:6] == 'events', listdir(path)))
    summary_iterator = EventAccumulator(str(path + event_file)).Reload()

    tags = summary_iterator.Tags()['scalars']
    steps = [[event.step for event in summary_iterator.Scalars(tag)] for tag in tags]
    data = [[event.value for event in summary_iterator.Scalars(tag)] for tag in tags]
    
    # Creating dataframe: we have missing values due to coefficients being deleted, 
    #so we have to do this column by column
    
    df = pd.DataFrame()
    for idx, tag in enumerate(tags):
        df[tag] = pd.Series(index=steps[idx], data=data[idx]) 
    df[df.isna()] = 0.0 # if they're deleted, they're zero.

    return df
