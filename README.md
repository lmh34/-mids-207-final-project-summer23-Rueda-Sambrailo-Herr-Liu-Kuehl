# mids-207-final-project-summer23-Rueda-Sambrailo-Herr-Liu-Kuehl
MIDS 207 – Applied Machine Learning
Final project: "Predicting Pet Adoption Speeds"

## Contributors
- Erik Sambrailo
- Alberto Lopez Rueda
- Nicole Liu
- Lucy Herr
- Bailey Kuehl

## Original Datasets 
https://www.kaggle.com/competitions/petfinder-adoption-prediction/rules

## File Overview: 
- 1_Parsing_and_Mering_AllData -- this will take all of the raw data from the data folder, parse it into columns, and merge it on PetID. It will also split the data into test and train. By using "random_state = 1", all executions of this notebook will result in the same test/train split.
- 2_DataExploration_Charts -- Optionally, you can run this file to see data distributions, charts, etc. that were used to produce the cleaned features created in step 2.
- 3_Features_DataCleaning -- this will perform feature engineering (data cleaning, transformations) on all data.
- 4_ModelTraining -- this will train 3 different models, including: baseline, FF neural network, and transformers.
