from phimal_utilities.analysis import load_tensorboard
from phimal_utilities.analysis import check_correct, classify_error
import numpy as np
path = 'tests/data/runs/'

df = load_tensorboard(path)

coeff_keys = [key for key in df.keys() if key[:5] == 'coeff']
true_coeffs = np.zeros((1, 12))
true_coeffs[0, 2] = 0.1
true_coeffs[0, 5] = -1.0

correct = check_correct(df[coeff_keys], true_coeffs)
correct = classify_error(df[coeff_keys], true_coeffs)
print(correct)