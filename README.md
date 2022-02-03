# Introduction
### The project has the objective to create a public API of a taxi fare prediction in the city of New York.
> **a.** Create the Model and Prediction - based on available data\
**b.** Make it into a package\
**c.** Push API in container running locally\
**d.** Push API in Cloud Run


# Model

- A Linear Regression was trained in 10,000 data points based on 7 features.
- The features are:
>>>a.pick up location(lat and long)\
>>>b.drop off location(lat and long)\
>>>c.number of passengers\
>>>d.time and day of the week

The model is already trained and available in gcp and is being used to predict the taxi fare in api.
However it can be trained again with new data if necessary.

# Package

The package can be installed by:
```bash
pip install git+ssh://git@github.com/LouiseDantas/TaxiFareEstimator
```
Can use the trainer class to set the pipeline, run the model, evaluate it and upload into gcp or save locally (save_model method).

# API

The API was build using FastAPI and pushed as a Docker container into Google Cloud Run. This way the api is available at:

https:// taxifareapi-v75w2fyqhq-ew.a.run.app/127.0.0.1:8000/predict
