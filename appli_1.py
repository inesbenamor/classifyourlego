# Import des librairies:

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf


###  En-tête de notre application:  ###

st.header("Classificateur de Lego")
st.subheader('A quelle boite appartient cette pièce?')

Classifier = st.sidebar.selectbox("Selectionner le Réseau de neuronnes",("VGG16","InceptionV3","Xception"))

param1 = st.sidebar.slider("X",0,50,3)

# Nous allons avoir besoin de 2 fonctions: L'une pour le téléchargement de la photo du Lego à classifier, 
# l'autre pour la prédiction (à quelle set appartient ce Lego?)

   ###  Première fonction:  ###
def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_image(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [224, 224])

def giveset(dataframe, pred):
    y = dataframe["Set"][dataframe["Ref"] == int(pred)].values[0]
    return y

def predict_class(model, img):
                                            
    test_image = np.expand_dims(img, axis = 0)                          
    class_names = ["300223","403221","4220631","6151578","6172421","6210741"]  
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    
    return image_class
def main():
    classifier_model = tf.keras.models.load_model("Xception224_model.h5")
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    file_uploaded = st.file_uploader(" Lego à identifier", type = ["jpg", "jpeg", "png"])
    
    if file_uploaded is not None :                      # Si le fichier est != de None, on le charge.
        # temp_file.write(file_uploaded.getvalue())
        # file_bytes = bytearray(file_uploaded.read())
        img = decode_img(file_uploaded.read())
        img = img/255

        predicted_class = predict_class(classifier_model,img)
        match_set = giveset(pd.read_csv("./bdd/Dataset_ref.csv", delimiter = ";"),predicted_class)

        st.text(" Prédiction:" + str(match_set))
        print(match_set)
        
      
if __name__ == "__main__":
    main()




