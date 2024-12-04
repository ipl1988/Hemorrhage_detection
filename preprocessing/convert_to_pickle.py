import tensorflow as tf
import pickle

# Load the model from the SavedModel format
model = tf.keras.models.load_model('model_V3_windowed')

# Save the model as a pickle
with open('model_V3_windowed.pkl', 'wb') as f:
    pickle.dump(model, f)
