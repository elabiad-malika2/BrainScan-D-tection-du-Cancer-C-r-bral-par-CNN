import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import joblib
import time
import cv2
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Utilitaire robuste pour retrouver les fichiers médias (images)
def get_asset_path(filename: str) -> str | None:
    base = Path(__file__).parent
    candidates = [
        base / filename,
        base / "assets" / filename,
        base.parent / filename,
        base.parent / "notebooks" / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

# Configuration de la page
st.set_page_config(
    page_title="Rapport d'analyse CNN – Détection de tumeurs cérébrales", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# TITRE ET INTRO
# ======================
st.title("🧠 Rapport d'analyse du modèle CNN – Détection de tumeurs cérébrales")
st.markdown("""
Ce rapport présente l'analyse complète du modèle CNN avec les vraies courbes d'entraînement.
**Analyse de l'overfitting AVANT et APRÈS les corrections.**
""")

# ======================
# 1️⃣ DISTRIBUTION DES CLASSES
# ======================
st.header("1️⃣ Distribution des classes")

# Affichage de la vraie distribution
dist_img = get_asset_path("DustributionClass.png")
if dist_img:
    st.image(dist_img, caption="Distribution des classes dans le dataset", use_container_width=True)
else:
    st.warning("Image introuvable: DustributionClass.png")

st.markdown("""
**📊 Analyse de la distribution :**
- **Déséquilibre initial** : La classe `notumor` est sur-représentée (2000 images)
- **Classes sous-représentées** : `glioma` (1621) et `meningioma` (1645) 
- **Impact** : Ce déséquilibre peut biaiser le modèle vers la classe majoritaire
- **Solution** : Data augmentation pour équilibrer à 2000 images par classe
""")

# ======================
# 2️⃣ PROBLÈME D'OVERFITTING - AVANT CORRECTIONS
# ======================
st.header("2️⃣ Problème d'Overfitting - AVANT les corrections")

st.subheader("🚨 Courbes d'entraînement avec overfitting sévère")

# Affichage des courbes d'overfitting
overfit_img = get_asset_path("LossAccOverfiting.png")
if overfit_img:
    st.image(overfit_img, caption="Courbes d'Accuracy et Loss - AVANT corrections (avec overfitting)", use_container_width=True)
else:
    st.warning("Image introuvable: LossAccOverfiting.png")

st.markdown("""
**🔍 Analyse de l'overfitting (AVANT les corrections) - Basée sur vos vraies courbes :**

**Courbe d'Accuracy :**
- **Train Accuracy** : Commence à ~67% et monte de manière constante jusqu'à ~99% (surapprentissage)
- **Validation Accuracy** : Commence à ~80%, atteint un pic à ~90-93% vers l'époque 7-8, puis **stagne ou diminue légèrement** jusqu'à ~90%
- **Écart croissant** : Divergence claire entre train (99%) et validation (90%) - **gap de 9%**

**Courbe de Loss :**
- **Train Loss** : Commence à ~0.6, diminue rapidement jusqu'à ~0.0 (quasi nulle)
- **Validation Loss** : Commence à ~0.6, diminue jusqu'à ~0.3 vers l'époque 7-8, puis **augmente fortement** jusqu'à ~0.45
- **Signal d'alarme** : L'augmentation de la validation loss après l'époque 7-8 indique un surapprentissage sévère

**🚨 Diagnostic :** Le modèle mémorise parfaitement les données d'entraînement (99% accuracy, loss quasi nulle) mais perd sa capacité de généralisation sur les données de validation (90% accuracy, loss qui augmente). C'est un cas classique d'overfitting sévère.
""")

# ======================
# 3️⃣ SOLUTIONS IMPLÉMENTÉES
# ======================
st.header("3️⃣ Solutions implémentées pour réduire l'overfitting")

st.markdown("""
**🛠️ Techniques de régularisation appliquées :**

**1️⃣ Dropout Layers :**
- **Conv2D layers** : Dropout(0.25) après chaque bloc convolutionnel
- **Dense layers** : Dropout(0.5) après chaque couche dense
- **Effet** : Désactive aléatoirement 25-50% des neurones pendant l'entraînement

**2️⃣ Early Stopping :**
- **Monitor** : val_loss
- **Patience** : 5 époques
- **Effet** : Arrête l'entraînement quand la validation loss n'améliore plus

**3️⃣ ModelCheckpoint :**
- **Monitor** : val_accuracy
- **Save best only** : True
- **Effet** : Sauvegarde le meilleur modèle (pas le dernier)

""")

# ======================
# 4️⃣ RÉSULTATS APRÈS CORRECTIONS
# ======================
st.header("4️⃣ Résultats après les corrections")

st.subheader("✅ Courbes d'entraînement APRÈS corrections")

# Affichage des courbes après corrections
fixed_img = get_asset_path("LossAcc.png")
if fixed_img:
    st.image(fixed_img, caption="Courbes d'Accuracy et Loss - APRÈS corrections (avec régularisation)", use_container_width=True)
else:
    st.warning("Image introuvable: LossAcc.png")

st.markdown("""
**✅ Améliorations observées (APRÈS les corrections) :**

**Courbe d'Accuracy :**
- **Train Accuracy** : Augmente de manière plus contrôlée jusqu'à ~95%
- **Validation Accuracy** : Suit de près l'accuracy d'entraînement (~92%)
- **Écart réduit** : Divergence minimale entre train et validation (gap de 3% au lieu de 8%)

**Courbe de Loss :**
- **Train Loss** : Diminue de manière stable jusqu'à ~0.15
- **Validation Loss** : Diminue parallèlement sans augmentation après l'époque 7-8
- **Convergence stable** : Pas de divergence entre les courbes

**🎯 Résultat :** Le modèle généralise mieux et évite le surapprentissage grâce au dropout et early stopping.
""")

# ======================
# 5️⃣ ÉVALUATION FINALE
# ======================
st.header("5️⃣ Évaluation finale du modèle")

st.subheader("📈 Performances sur le jeu de test")

# Résultats du notebook
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy sur test", "90.38%", "0%")
with col2:
    st.metric("Loss sur test", "0.2669", "0%")
with col3:
    st.metric("Durée d'entraînement", "~58s/époque", "0%")

# Matrice de confusion
st.subheader("🧩 Matrice de confusion")

# Affichage de la vraie matrice de confusion
cm_img = get_asset_path("confusion.png")
if cm_img:
    st.image(cm_img, caption="Matrice de confusion réelle du modèle", use_container_width=True)
else:
    st.warning("Image introuvable: confusion.png")

st.markdown("""
**📊 Analyse de la matrice de confusion :**

**Classes (0=glioma, 1=meningioma, 2=notumor, 3=pituitary) :**
- **Classe 0 (glioma)** : 357 corrects, 38 confusions avec classe 1, 0 avec classe 2, 5 avec classe 3
- **Classe 1 (meningioma)** : 330 corrects, 39 confusions avec classe 0, 12 avec classe 2, 19 avec classe 3  
- **Classe 2 (notumor)** : 379 corrects, 3 confusions avec classe 0, 13 avec classe 1, 5 avec classe 3
- **Classe 3 (pituitary)** : 380 corrects, 6 confusions avec classe 0, 14 avec classe 1, 0 avec classe 2

**🎯 Performance par classe :**
- **Meilleure classe** : Pituitary (380/400 = 95% de précision)
- **Classe la plus confuse** : Glioma et Meningioma se confondent souvent (38+39 erreurs)
- **Classe la plus stable** : Notumor (379/400 = 94.75% de précision)
- **Total corrects** : 357+330+379+380 = 1446/1600 = 90.38% d'accuracy globale
""")

# Rapport de classification
st.subheader("📊 Rapport de classification")
report_data = {
    'Classe': ['glioma', 'meningioma', 'notumor', 'pituitary'],
    'Precision': [0.88, 0.84, 0.97, 0.93],
    'Recall': [0.89, 0.82, 0.95, 0.95],
    'F1-score': [0.89, 0.83, 0.96, 0.94],
    'Support': [400, 400, 400, 400]
}

df_report = pd.DataFrame(report_data)
df_report = df_report.set_index('Classe')
st.dataframe(df_report.style.background_gradient(cmap='Greens', axis=1))

st.markdown("""
**📊 Métriques globales :**
- **Accuracy globale** : 90.38%
- **F1-score moyen** : 90%
- **Macro avg** : 90%
- **Weighted avg** : 90%
""")

# ======================
# 6️⃣ ARCHITECTURE DU CNN
# ======================
st.header("6️⃣ Architecture détaillée du CNN")

st.subheader("🏗️ Structure complète du modèle")

st.code("""
# Architecture CNN implémentée
model = Sequential()

# 1ère couche convolutionnelle + pooling + dropout
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 2ème couche convolutionnelle + pooling + dropout
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 3ème couche convolutionnelle + pooling + dropout
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Couches denses avec dropout
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
""", language="python")

st.markdown("""
**🔧 Analyse de l'architecture :**

**1️⃣ Blocs convolutionnels (3 blocs) :**
- **Bloc 1** : 32 filtres 3×3 → MaxPooling 2×2 → Dropout 0.25
- **Bloc 2** : 64 filtres 3×3 → MaxPooling 2×2 → Dropout 0.25  
- **Bloc 3** : 128 filtres 3×3 → MaxPooling 2×2 → Dropout 0.4

**2️⃣ Couches denses (3 couches) :**
- **Dense 1** : 128 neurones + ReLU + Dropout 0.5
- **Dense 2** : 64 neurones + ReLU + Dropout 0.5
- **Dense 3** : 4 neurones + Softmax (sortie)

**3️⃣ Stratégie de régularisation :**
- **Dropout progressif** : 0.25 → 0.25 → 0.4 → 0.5 → 0.5
- **Augmentation des filtres** : 32 → 64 → 128 (extraction de features complexes)
- **Réduction spatiale** : MaxPooling 2×2 après chaque bloc conv
- **Fonctions d'activation** : ReLU pour les couches cachées, Softmax pour la sortie

**4️⃣ Paramètres totaux :**
- **Input shape** : (224, 224, 3) - Images RGB 224×224
- **Output** : 4 classes (glioma, meningioma, notumor, pituitary)
- **Optimiseur** : Adam avec learning rate 0.001
- **Fonction de perte** : categorical_crossentropy
""")

# ======================
# 7️⃣ CONCLUSION
# ======================
st.header("7️⃣ Conclusion et recommandations")

st.markdown("""
### 🎯 **Résumé des améliorations**

Le modèle CNN a été **considérablement amélioré** grâce à l'implémentation de techniques de régularisation :

**🔧 Techniques appliquées :**
1. **Dropout progressif** dans toutes les couches (0.25 → 0.5)
2. **Early Stopping** avec patience de 5 époques
3. **ModelCheckpoint** pour sauvegarder le meilleur modèle

**📈 Résultats obtenus :**
- **Accuracy de test** : 90.38%
- **Overfitting contrôlé** : Gap train-validation réduit de 8% à 3%
- **Généralisation améliorée** : Performance stable sur données non vues
- **Entraînement optimisé** : Arrêt automatique au meilleur moment

""")

# ======================
# SIDEBAR - INFORMATIONS TECHNIQUES
# ======================
with st.sidebar:
    st.header("🔧 Informations techniques")
    
    st.subheader("📊 Dataset")
    st.write("- **Total images** : 7,023")
    st.write("- **Classes** : 4 (glioma, meningioma, notumor, pituitary)")
    st.write("- **Taille des images** : 224×224×3")
    st.write("- **Train/Test split** : 80/20")
    
    st.subheader("🏗️ Architecture")
    st.write("- **Type** : CNN séquentiel")
    st.write("- **Couches conv** : 3 blocs")
    st.write("- **Filtres** : 32→64→128")
    st.write("- **Dropout** : 0.25→0.5")
    
    st.subheader("⚙️ Hyperparamètres")
    st.write("- **Optimiseur** : Adam")
    st.write("- **Learning rate** : 0.001")
    st.write("- **Batch size** : 32")
    st.write("- **Époques** : 30 (avec early stopping)")
    
    st.subheader("📈 Performances")
    st.write("- **Test Accuracy** : 90.38%")
    st.write("- **Test Loss** : 0.2669")
    st.write("- **F1-score moyen** : 90%")
    