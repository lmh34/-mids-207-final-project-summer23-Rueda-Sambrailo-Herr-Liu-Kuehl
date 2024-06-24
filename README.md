# mids-207-final-project-summer23-Rueda-Sambrailo-Herr-Liu-Kuehl
Final project for MIDS 207

## Contributors
- Erik Sambrailo
- Alberto Lopez Rueda
- Nicole Liu
- Lucy Herr
- Bailey Kuehl

## Dataset 
https://www.kaggle.com/competitions/petfinder-adoption-prediction/rules

## Setting up this repo on your local device
### 1. Go to the dataset website an download it to your local device. Rename the file "data" if it is not named that already.
Unfortunately, the files are gigantic, and GitHub doesn't allow you to upload files > 25MB.
Naming the file "data" will ensure that you can run the files in step 2 below without changing your path. If you prefer, you can change your path to access the data and name it whatever you like.

### 2. Clone this repo to your local device
```git clone https://github.com/UC-Berkeley-I-School/mids-207-final-project-summer23-Rueda-Sambrailo-Herr-Liu-Kuehl.git```

### 3. Move the "data" file containing the PetFinder dataset into the cloned repo.

### 4. Run the files in successive order.
1. 1_Parsing_and_Mering_AllData -- this will take all of the raw data from the data folder, parse it into columns, and merge it on PetID. It will also split the data into test and train. By using "random_state = 1", all executions of this notebook will result in the same test/train split.

2. 2_DataExploration_Charts -- Optionally, you can run this file to see data distributions, charts, etc. that were used to produce the cleaned features created in step 2.

3. 3_Features_DataCleaning -- this will perform feature engineering (data cleaning, transformations) on all data.

4. 4_ModelTraining -- this will train 3 different models, including: baseline, FF neural network, and transformers.
