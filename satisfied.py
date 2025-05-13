import streamlit as st
import joblib
import os
import zipfile
import numpy as np


st.set_page_config(page_title="Satisfaction en Vol", page_icon="✈️", layout="centered")

# === PARAMÈTRES ===
ZIP_PATH = "Model_rf.zip"
EXTRACT_PATH = "models/"
MODEL_FILENAME = "Model_rf.pkl"  
MODEL_PATH = os.path.join(EXTRACT_PATH, MODEL_FILENAME)


os.makedirs(EXTRACT_PATH, exist_ok=True)

# Décompresser le fichier zip s'il n'a pas encore été extrait
if not os.path.exists(MODEL_PATH):
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            st.info("Décompression du modèle en cours...")
            zip_ref.extractall(EXTRACT_PATH)
        st.success("Fichiers décompressés avec succès !")
    except Exception as e:
        st.error(f"Erreur lors de la décompression : {e}")
        st.stop()
else:
    st.info("Le modèle est déjà décompressé.")

# === Chargement du modèle ===
@st.cache_resource
def charger_modele():
    try:
        modele = joblib.load(MODEL_PATH)
        st.success("✅ Modèle chargé avec succès.")
        return modele
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        st.stop()

modele = charger_modele()


# === 3. Interface utilisateur ===
st.title("✈️ Satisfaction des passagers en vol")
st.write("*Ce site recueille et analyse les avis des passagers pour améliorer la qualité des vols.*")

# Formulaire
gender_options = ["Male", "Female"]
selected_gender = st.selectbox("Genre :", gender_options)

age = st.number_input("Âge", min_value=0, max_value=120)

customer_types = ["First-time", "Returning"]
selected_customer_type = st.selectbox("Type de client :", customer_types)

travel_types = ["Business", "Personal"]
selected_travel_type = st.selectbox("Type de voyage :", travel_types)

class_options = ["Economy", "Economy Plus", "Business"]
selected_class = st.selectbox("Classe :", class_options)

flight_distance = st.number_input("Distance du vol (en km) :", min_value=31, max_value=4983)
departure_delay = st.number_input("Retard au départ (en min) :", min_value=0, max_value=1592)
arrival_delay = st.number_input("Retard à l'arrivée (en min) :", min_value=0, max_value=1584)

# Services notés de 0 à 5
onboard_service = st.number_input("Service à bord :", min_value=0, max_value=5)
seat_comfort = st.number_input("Confort du siège :", min_value=0, max_value=5)
leg_room_service = st.number_input("Espace pour les jambes :", min_value=0, max_value=5)
cleanliness = st.number_input("Propreté :", min_value=0, max_value=5)
food_and_drink = st.number_input("Nourriture et boissons :", min_value=0, max_value=5)
inflight_service = st.number_input("Service en vol :", min_value=0, max_value=5)
wifi_service = st.number_input("Wifi en vol :", min_value=0, max_value=5)
entertainment = st.number_input("Divertissement en vol :", min_value=0, max_value=5)
baggage_handling = st.number_input("Gestion des bagages :", min_value=0, max_value=5)
online_booking = st.number_input("Facilité de réservation en ligne :", min_value=0, max_value=5)
checkin_service = st.number_input("Service d'enregistrement :", min_value=0, max_value=5)
online_boarding = st.number_input("Embarquement en ligne :", min_value=0, max_value=5)

# Encodage des variables
gender_female = 1 if selected_gender == "Female" else 0
gender_male = 1 if selected_gender == "Male" else 0
first_time = 1 if selected_customer_type == "First-time" else 0
returning = 1 if selected_customer_type == "Returning" else 0
travel_business = 1 if selected_travel_type == "Business" else 0
travel_personal = 1 if selected_travel_type == "Personal" else 0
class_economy = 1 if selected_class == "Economy" else 0
class_economy_plus = 1 if selected_class == "Economy Plus" else 0
class_business = 1 if selected_class == "Business" else 0

# Création du tableau de caractéristiques
features = np.array([[float(age),
                      first_time,
                      returning,
                      travel_business,
                      travel_personal,
                      class_economy,
                      class_economy_plus,
                      class_business,
                      float(flight_distance),
                      float(departure_delay),
                      float(arrival_delay),
                      float(onboard_service),
                      float(seat_comfort),
                      float(leg_room_service),
                      float(cleanliness),
                      float(food_and_drink),
                      float(inflight_service),
                      float(wifi_service),
                      float(entertainment),
                      float(baggage_handling),
                      float(online_booking),
                      float(checkin_service),
                      float(online_boarding),
                      gender_female,
                      gender_male]])

# === 4. Prédiction ===
if st.button("Soumettre"):
    if modele:
        prediction = modele.predict(features)[0]
        if prediction == 1:
            st.success("✅ Client **satisfait**.")
        else:
            st.error("❌ Client **non satisfait**.")
    else:
        st.error("❌ Le modèle n’a pas pu être chargé.")
