import os
import errno


for i in range(77):
    for j in range(5):
        try:
            os.makedirs('best_epoch_models_chicago/'+str(i)+'/'+str(j))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
