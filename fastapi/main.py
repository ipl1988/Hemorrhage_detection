

#import model
from pydantic import BaseModel

#import fastapi files-import module
import aiofiles

#import fastapi in order to build API
from fastapi import FastAPI
app = FastAPI()

#define http route get in root endpoint /
@app.get("/")
def root():
    return {"message": "¡Hola, FastAPI!"}

#class Picture(BaseModel):
    #photo: str

@app.post("/uploadfile/")
async def cache_favicon(file: UploadFile = File(...)):
    img = await file.read()

    async with aiofiles.open("destination.jpg" , "wb") as f:
        await f.write(img)

@app.post("/predict/")
def predict_injury(item: Picture):
    #load_model
    #model.predict
    return {"message": "There is injury", "item": item}
