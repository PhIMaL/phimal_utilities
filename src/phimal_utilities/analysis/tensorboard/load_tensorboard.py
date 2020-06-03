import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir


def load_tensorboard(path: str) -> pd.DataFrame:
    '''Function to load tensorboard file from a folder.
    Assumes one file per folder!'''

    assert len(listdir(path)) == 1, 'No or more than one event file found.'
    event_file = next(filter(lambda filename: filename[:6] == 'events', listdir(path)))
    summary_iterator = EventAccumulator(str(path + event_file)).Reload()

    tags = summary_iterator.Tags()['scalars']
    steps = np.array([event.step for event in summary_iterator.Scalars(tags[0])])
    data = np.array([[event.value for event in summary_iterator.Scalars(tag)] for tag in tags]).T
    df = pd.DataFrame(data=data, index=steps, columns=tags)

    return df
