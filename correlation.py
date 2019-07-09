import sys
sys.path.append('C://Users//tauseef.ur.rahman//Desktop//MyPythonfiles')

import classification_utils as cutils
import numpy as np

# 2-d classification pattern

X,y = cutils.generate_linear_synthetic_data_classification(n_samples=50,n_features=1,n_classes=2,weights=[0.5,0.5])
--np.mean()
np.cov(X,rowvar=False)
np.corrcoef(X, rowvar=False)
np.cov(y,rowvar=False)
