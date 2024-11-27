
#import model
from pydantic import BaseModel

#import fastapi files-import module
import aiofiles

#import fastapi in order to build API
from fastapi import FastAPI, UploadFile, File
app = FastAPI()

#define http route get in root endpoint /
@app.get("/")
def root():
    return {"message": "¡Hola, FastAPI!"}


@app.post("/uploadfile/")
async def upload_file (file: UploadFile = File(...)):
    img = await file.read()

    async with aiofiles.open("destination.png" , "wb") as f:
        await f.write(img)

    return { 'prediction' : [0.23,0.34,0.56,0.54,0.67,0.45]}
#uvicorn api.main:app --reload this command needs to be executed in Terminal
