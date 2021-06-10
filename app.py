from flask import Flask, request, jsonify, send_file, send_from_directory
from datauri import DataURI
import urllib
import numpy as np
from PIL import Image
from pickle import dump, load
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
import base64
from gtts import gTTS
from io import BytesIO, StringIO

#ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required=True, help="Image Path")
#args = vars(ap.parse_args())

#img_path = "72218201_e0e9c7d65b.jpg"

def extract_features(image, model):
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        print(image.size)
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
      if index == integer:
          return word
  return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

app = Flask(__name__)

@app.route('/Predict', methods=['POST'])
def predict():
    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")
    
    print("----------")
    #request_data = request.get_json()
    print('----------------')
    URL = request.form.get('image')
    #print(URL)
    #URL = request_data['image']

##    with urllib.request.urlopen(URL) as url:
##        f = BytesIO(url.read())

    #img = Image.open(request.files.get('image', ''))

    #img = Image.open(f)
    img = Image.open(BytesIO(base64.b64decode(URL)))

    print("------------")
    print(request.form.get('image'))
    print("-------------")
    img = Image.open(BytesIO(base64.b64decode(request.form.get('image'))))
    #request_data = request.get_json()
    #img = Image.open(BytesIO(base64.b64decode(request_data['image'])))

    photo = extract_features(img, xception_model)
    description = generate_desc(model, tokenizer, photo, max_length)
    language = 'en'
    tts = gTTS(text = description, lang = language, slow = False)
    audio = BytesIO()
     
    tts.write_to_fp(audio)
    

##    return {
##            "description": description,
##            "audio": base64.b64encode(audio.getvalue()).decode()
##    }

    tts.save("audio.wav")
    
    return send_from_directory(app.root_path, "audio.wav")

if __name__ == '__main__':
    app.run(debug=True)
