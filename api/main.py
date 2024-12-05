
#uvicorn api.main:app --reload >> Dont forget to run the code in makefile! this command needs to be executed in Terminal to run the server

#import model
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import os

#import model
#from pydantic import BaseModel
#import fastapi files-import module
#import aiofiles


#import fastapi in order to build API
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

#define http route get in root endpoint /
@app.get("/")
def root():
    return {"message": "¡Hola, FastAPI!"}

#model prediction to be returned in the API'''
import pickle
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
file = open(model_path, "rb")
app.state.model = pickle.load(file)

#endpoint model prediction
@app.post("/prediction/")
async def predictimage(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes))
    img = img.resize((150,150))
    img = img_to_array(img)
    img = img.reshape((-1, 150, 150, 1))
    res = app.state.model.predict(img)
    print(res)

    return {"injury" : float(res)}
