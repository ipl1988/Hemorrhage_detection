import tensorflow as tf
import pickle

from basic_preprocessing import

# Load the model from the SavedModel format
model = tf.keras.models.load_model('model_V1_test')

# Save the model as a pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
