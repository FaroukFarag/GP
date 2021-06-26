from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications import InceptionV3
import dill


def load_image(img, size=None):

    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    img = np.array(img)

    img = img / 255.0

    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img

image_model = InceptionV3(include_top=True, weights='imagenet')

transfer_layer = image_model.get_layer('avg_pool')

image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)

img_size = K.int_shape(image_model.input)[1:3]
transfer_values_size = K.int_shape(transfer_layer.output)[1]

mark_start = 'start '
mark_end = ' end'

class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, num_words=None):

        Tokenizer.__init__(self, num_words=num_words)

        self.fit_on_texts(texts)
        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        

        text = " ".join(words)
        return text
    
    def captions_to_tokens(self, captions_listlist):
       
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        return tokens



with open('tokenizer.pkl', 'rb') as input:
    tokenizer = dill.load(input)

num_words = 15000
token_start = tokenizer.word_index[mark_start.strip()]
token_end = tokenizer.word_index[mark_end.strip()]

def sparse_cross_entropy(y_true, y_pred):

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

from tensorflow.python.keras.layers import Dropout,CuDNNGRU

state_size = 512
embedding_size = 128

transfer_values_input = Input(shape=(transfer_values_size,),
                              name='transfer_values_input')

decoder_transfer_map = Dense(state_size,
                             activation='tanh',
                             name='decoder_transfer_map')

decoder_input = Input(shape=(None, ), name='decoder_input')

decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

decoder_gru1 = CuDNNGRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = CuDNNGRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = CuDNNGRU(state_size, name='decoder_gru3',
                   return_sequences=True)
decoder_gru4 = CuDNNGRU(state_size, name='decoder_gru4',
                   return_sequences=True)

decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')

def connect_decoder(transfer_values):

    initial_state = decoder_transfer_map(transfer_values)

    net = decoder_input    
    net = decoder_embedding(net)
    # Connect all the GRU layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = Dropout(0.5)(net)
    net = decoder_gru3(net, initial_state=initial_state)
    net = Dropout(0.5)(net)
    net = decoder_gru4(net, initial_state=initial_state)
    
    decoder_output = decoder_dense(net)    
    return decoder_output

  
decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])

optimizer = RMSprop(lr=1e-3)

decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

decoder_model.compile(optimizer=optimizer,
                      loss=sparse_cross_entropy,
                      target_tensors=[decoder_target])

try:
    decoder_model.load_weights("IC_checkpoint.keras")
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

def generate_caption(image_path, max_tokens=30):

    image = load_image(image_path, size=img_size)

    image_batch = np.expand_dims(image, axis=0)

    transfer_values = image_model_transfer.predict(image_batch)

    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }


        decoder_output = decoder_model.predict(x_data)
        token_onehot = decoder_output[0, count_tokens, :]
        token_int = np.argmax(token_onehot)
        sampled_word = tokenizer.token_to_word(token_int)
        output_text += " " + sampled_word
        count_tokens += 1
    output_tokens = decoder_input_data[0]
    
    
    predicted_caption=output_text.split()
    del (predicted_caption[-1])
    
    print("Predicted caption:")
    print(output_text)
    print()
    return output_text


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET'])
@cross_origin()
def home():
        return "Welcome to Home Page."

@app.route('/Predict', methods=['POST'])
@cross_origin()
def predict():
    print("Test")
    print(request.files.get('image', ''))
    print("Test")
    img = Image.open(request.files.get('image', ''))
    description = generate_caption(img)

    return {
        "description": description
    }
