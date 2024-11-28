########## TEST LOCAL ###########

#endpoint model
app.state.model = load_model('/Users/ines/code/ipl1988/Hemorrhage_detection')


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



#endpoint upload file
@app.post("/testuploadfile/")
async def test_upload_file (file: UploadFile = File(...)):
    img = await file.read()

    async with aiofiles.open("destination.png" , "wb") as f:
        await f.write(img)

    return {'description': 'just to test if image upload works',
            'harcoded prediction' : [0.23,0.34,0.56,0.54,0.67,0.45]}

#endpoint model prediction
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
