import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# Charger le modèle
# =========================
model = load_model("model/model_cnn.keras")

# =========================
# Liste des classes (à modifier selon ton dataset)
# =========================
CLASSES = ['glioma','meningioma','notumor','pituitary'] 


# =========================
# Fonction de prédiction
# =========================
def predict_image(imge):
    # Redimensionner à 224x224 si nécessaire
    if imge.shape != (224, 224, 3):
        imge = cv2.resize(imge, (224, 224))

    # Normalisation entre 0 et 1
    imge = imge / 255.0
    imge = imge.astype("float32")

    # Ajouter une dimension pour batch (1, 224, 224, 3)
    imge = np.expand_dims(imge, axis=0)

    # Prédiction
    prediction = model.predict(imge)
    pred_index = np.argmax(prediction)

    # Classe correspondante
    pred_class = CLASSES[pred_index]

    return pred_class, prediction[0],pred_index


# =========================
# Interface Streamlit
# =========================
st.set_page_config(page_title="Classification d'images CNN", page_icon="🧠", layout="centered")

st.title("🧠 Application de Classification d’Images CNN")
st.write("Téléverse une image et laisse le modèle prédire sa classe.")

# Upload image
uploaded_file = st.file_uploader("Choisis une image :", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire et afficher l’image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="🖼️ Image importée", use_container_width=True)

    # Bouton de prédiction
    if st.button("🔍 Prédire la classe"):
        pred_class, probas,predIndex = predict_image(image_np)

        # Résultat principal
        st.success(f"✅ Classe prédite : **{pred_class,predIndex}**")

        # Graphique des probabilités
        fig, ax = plt.subplots()
        ax.barh(CLASSES, probas, color='skyblue')
        ax.set_xlabel("Probabilité")
        ax.set_title("Distribution des prédictions")
        st.pyplot(fig)
