import pickle
from basic_preprocessing import normalizer, transform_in_hu

with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)




## Preprocess a new image to retrieve predictions on it

## check if image endswith .dcom
## check if image endswith .png
## resize to 150x150




loaded_model.predict()
