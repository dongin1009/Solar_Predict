# for private LB
# should be run in /Solar_Predict/private_LB

#######################################################################################
#### submissions/test_submission_for_code.csv should be deleted in real private LB ####
#######################################################################################

import pandas as pd
from privateLB_models import RfPredict, LgbmPredict, XgbPredict

# date of prediction
base_date = "20210606"
base_time = '2000'

# predictions from the models
submission_rf = RfPredict(base_date, base_time).get_submission()
submission_lgbm = LgbmPredict(base_date, base_time).get_submission()
#submission_xgb = XgbPredict(base_date, base_time).get_submission()

# add each predictions
submission_ensemble = submission_rf.add(submission_lgbm, fill_value = 0)
#submission_ensemble = submission_ensemble.add(submission_xgb, fill_value = 0)

submission_ensemble["ulsan"] = submission_ensemble["ulsan"] / 3
 
# write CSV
print(submission_ensemble)
submission_ensemble.to_csv(f"./submissions/{base_date}_{base_time}.csv", index = False)
