import requests

image_path = "/home/sebastian/code/ipl1988/test_data_dont_touch/images/ID_4fff99ddd.png"
url = "https://hmdetect-530280335142.europe-west1.run.app"

files = {
    'file': (open(image_path, 'rb'))
}


data = open(image_path,'rb').read()
r = requests.post(url,data=data)

#response = requests.post(url, files=files)
print(r)
