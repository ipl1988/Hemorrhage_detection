
#uvicorn api.main:app --reload >> Dont forget to run the code in makefile! this command needs to be executed in Terminal to run the server

#import model
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

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


@app.post("/testuploadfile/")
async def test_upload_file (file: UploadFile = File(...)):
    img = await file.read()

    async with aiofiles.open("destination.png" , "wb") as f:
        await f.write(img)

    return {'description': 'just to test if image upload works',
            'harcoded prediction' : [0.23,0.34,0.56,0.54,0.67,0.45]}

# load model to system memory
app.state.model = load_model("/Users/ines/code/ipl1988/Hemorrhage_detection/base_cat_dog")

@app.post("/prediction/")
async def predictimage(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes))
    img = img.resize((150,150))
    img = img_to_array(img)
    img = img.reshape((-1, 150, 150, 3))
    res = app.state.model.predict(img)[0][0]
    if(res < 0.5):
        injury = "present"
        prob = 1-res
    if(res >= 0.5):
        injury = "not present"
        prob = res

    print("injury: ", injury)
    print("probability = ",prob)

    return {"injury" : injury,
            "probability" : prob}
# model_new = load_model("/Users/ines/code/ipl1988/Hemorrhage_detection/base_cat_dog")


# #make any image we upload compatible with the architecture of the dummy model
# img=Image.open('/Users/ines/Desktop/sherry-christian-8Myh76_3M2U-unsplash.jpg')
# img=img.resize((150,150))
# img = img_to_array(img)
# img = img.reshape((-1, 150, 150, 3))
# res = model_new.predict(img)[0][0]
# print(res)
# THIS CODE MIGHT BE USEFUL ONLY IN CASE WE NEED TO LOAD THE IMAGE FROM AN API (I DONT THINK SO)
# import requests
# from io import BytesIO||/
#def getImage(url):
    #response = requests.get(url)
    #img = Image.open(BytesIO(response.content))
    #plt.imshow(img)
    #img = img.resize((150, 150))
    #return img



#model prediction to be returned in the API'''

########## TEST LOCAL ###########
path='/Users/ines/Desktop/sherry-christian-8Myh76_3M2U-unsplash.jpg'
model_new = load_model("/Users/ines/code/ipl1988/Hemorrhage_detection/base_cat_dog")

def predictimage(img_path, model):

    img = Image.open(img_path)
    img=img.resize((150,150))
    img = img_to_array(img)
    img = img.reshape((-1, 150, 150, 3))
    res = model.predict(img)[0][0]
    if(res < 0.5):
        injury = "present"
        prob = 1-res
    if(res >= 0.5):
        injury = "not present"
        prob = res

    print("injury: ", injury)
    print("probability = ",prob)


    return injury, prob

if __name__ == '__main__':
    injury, prob = predictimage(path, model_new)
    print(injury, ' ', prob)
