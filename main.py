import os
import io
from keras.models import load_model
from flask import Flask, request, jsonify
from keras.preprocessing.image import image_utils
import numpy as np
import PIL.Image as Image
import time
import base64

app = Flask(__name__)
model = load_model('latest_model.h5')
labels = ['katuk-bp',
 'katuk-ea_dg_palisade',
 'katuk-ea_dg_stomata',
 'katuk-eb',
 'katuk-parenkim_d_kko_b_roset',
 'keji_beling-bp',
 'keji_beling-ea',
 'keji_beling-ea_dg_litosit_d_stomata',
 'keji_beling-rp',
 'keji_beling-sistolit',
 'kelor-bp_t_tangga',
 'kelor-eb_dg_stomata',
 'kelor-kko_b_roset',
 'kelor-m_bp_dg_pt_tangga_d_kko_b_roset',
 'kelor-m_dg_selsekresi',
 'pegagan-bp',
 'pegagan-ea',
 'pegagan-eb_dg_stomata',
 'pegagan-mesofil',
 'pegagan-uratdaun_dg_kko_b_roset',
 'salam-ea',
 'salam-eb_dg_stomata',
 'salam-kko_b_prisma',
 'salam-sklerenkim',
 'salam-unsurxilem_dg_noktah',
 'sereh-e_dg_parenkim',
 'sereh-ea_d_bp_dg_p_t_tangga',
 'sereh-ea_dg_selpalisade_d_rp',
 'sereh-ea_dg_stomata_b_halter',
 'sereh-sklerenkim']

@app.route('/', methods=['POST'])
def rating():
    resImage=request.stream.read()
    decodedImage=base64.b64decode(resImage)
    imageFile=Image.open(io.BytesIO(decodedImage))

    imagePath='./images/' + time.strftime("%Y%m%d-%H%M%S") + '.png'
    imageFile.save(imagePath)

    image = preprocess_image(imagePath)

    prediction=model.predict(image, batch_size=10)
    predLabel=jsonify({'label':labels[np.argmax(prediction)], 'certainty':str(np.max(prediction))})

    os.remove(imagePath)

    return predLabel


def preprocess_image(input):
    image = image_utils.load_img(input, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.vstack([image])

    return image

if __name__ == '__main__':
    app.run(port='80', host='0.0.0.0')