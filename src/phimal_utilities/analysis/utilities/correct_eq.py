import numpy as np
import pandas as pd


def classify_error(found_coeffs: pd.DataFrame, true_coeffs: np.ndarray, simple: bool = False) -> np.ndarray:
    ''' Classifies error of found equation. Can either do correct/incorrect or advanced classification:
        0 - Correct
        1 - Found superset of correct terms (i.e. not sparse enough)
        2 - Found a subset of correct terms (i.e too sparse)
        3 - Found something else (i.e. completely off)

        found_coeffs: dataframe of found coefficients
        true_coeffs: array of true coefficients
        simple: whether to return full classificatin or just correct/incorrect
    '''

    ratio = true_coeffs / found_coeffs
    inactive_components = (true_coeffs == 0).squeeze()
    active_components = (true_coeffs != 0).squeeze()

    incorrect_active = np.count_nonzero(np.abs(ratio.loc[:, active_components]) == np.inf, axis=1).astype(bool)  # ratio is infinite is term which we should have isnt there
    incorrect_inactive = np.count_nonzero(~np.isnan(ratio.loc[:, inactive_components]), axis=1).astype(bool)  # ratio is nan if term which should be zero is, else it has a finite value

    if simple is True:
        classification = ~((incorrect_active + incorrect_inactive).astype(bool))
    else:
        classification = np.zeros(found_coeffs.shape[0], dtype=np.int)
        classification[incorrect_inactive] = 1
        classification[incorrect_active] = 2
        classification[(incorrect_inactive + incorrect_active)] = 3

    return pd.DataFrame(classification, columns=['Error type'])


def calculate_error(found_coeffs: pd.DataFrame, true_coeffs: np.ndarray) -> pd.DataFrame:
    ''' Calculates relative error of found coefficients. Returns nan as incorrect error result'''
    relative_error = np.mean(np.abs((found_coeffs - true_coeffs)/true_coeffs), axis=1)
    relative_error[relative_error == np.Inf] = np.NaN
    return pd.DataFrame(relative_error, columns=['Relative error'])
