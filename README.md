# Crime-Prediction-with-Transfer-Learning

This research addresses inter region transfer learning based crime prediction for the city Chicago

How to run the model:
1. Run `python initial.py`. This will create folders for storing the generated models.
2. Set the variable in line 22 of train.py to any value from 0 to 76 (Represents regions of Chicago). Example: `chicago_region = 73`
3. Set the variable in line 23 of train.py to any value from 0 to 4 (Represents different crime categories: Theft, Robbery, Assault, Burglary, Dangerous Drugs). Example: `current_crime_category = 0`
4. Set line 92 to either `criterion = nn.L1Loss()` or `criterion = nn.MSELoss()`
5. Run `python train.py`
