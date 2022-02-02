

# ----------------------------------
#      GOOGLE CLOUD PARAMS
# ----------------------------------

# project id - replace with your GCP project id
PROJECT_ID='le-wagon-745'

# bucket name - replace with your GCP bucket name
BUCKET_NAME='dantas-lrmd'

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME='TaxiFareEstimator'
FILENAME='trainer'

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'
# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

STORAGE_LOCATION = 'models/taxifareestimator/model.joblib'

# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
LOCAL_PATH="/home/louisedantas/code/LouiseDantas/TaxiFareEstimator/raw_data/train_1k.csv"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER='data'

##### Training  - - - - - - - - - - - - - - - - - - - - - -
# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'
