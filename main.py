import streamlit as st
import pandas as pd
import numpy as np
import joblib







#Chargement des models

#model du conversion ou preparation pour le degré 2

poly=joblib.load("poly.pkl")

# model pour la conversion ou la predition

model_LR_n2=joblib.load("model_LR_n2.pkl")

# Chargement des scalers

scaler_age=joblib.load("scaler_age.pkl")
scaler_weight=joblib.load("scaler_weight.pkl")
scaler_height=joblib.load("scaler_height.pkl")
scaler_max_bpm=joblib.load("scaler_max_bpm.pkl")
scaler_avg_bpm=joblib.load("scaler_avg_bpm.pkl")
scaler_rest_bpm=joblib.load("scaler_rest_bpm.pkl")
scaler_tpsession=joblib.load("scaler_tpsession.pkl")
scaler_calories=joblib.load("scaler_calories.pkl")
scaler_fat=joblib.load("scaler_fat.pkl")
scaler_water=joblib.load("scaler_water.pkl")
scaler_freq=joblib.load("scaler_freq.pkl")
scaler_exp=joblib.load("scaler_exp.pkl")
scaler_bmi=joblib.load("scaler_bmi.pkl")


# Define function to make predictions

def predict(input_features,features):

    # Perform any necessary preprocessing on the input_features

    # Make predictions using the loaded model

    input_poly=poly.transform(input_features)
    cal=model_LR_n2.predict(input_poly)
    
    output_pred=features.copy()
    output_pred["Calories Brulé"]=cal
    
    vcal=scaler_calories.inverse_transform(output_pred[["Calories Brulé"]])
    output_pred["Calories Brulé"]=vcal
    
    #Valeurs de sortie 

    return output_pred
    #fig, ax = plt.subplots(figsize=(4, 4))
    #ax.bar(["Math","Lecture","Ecriture"],prediction[0] , color=['red','blue','black'], label=[f"Math = {prediction[0][0]:.2f}",f"Lecture = {prediction[0][1]:.2f}",f"Ecriture = {prediction[0][2]:.2f}"])
    #ax.set_ylim(0, 100)
    #ax.set_title('Les trois notes')
    #ax.legend()
    #ax.set_ylabel('Notes')
    #plot=st.pyplot(fig)


 #fonction pour le calcul de Pc:


def main():

    st.title('')

    st.write("****")

    st.write("")

    


    # création des champs pour le remplissage

    age=st.number_input("Votre Age", min_value=6)
    encodedage=scaler_age.transform([[age]])[0][0]

    poids=st.number_input("Votre poids", min_value=20.0,format="%.1f") 
    encodedpoids=scaler_weight.transform([[poids]])[0][0]

    height=st.number_input("votre mesure 'en Metre' ",min_value=0.5, format="%.2f")
    encodedheight=scaler_height.transform([[height]])[0][0]

    max_bpm=st.number_input("Vos batements de coeur maximal durant la séance", min_value=40)
    encodedmax_bpm=scaler_max_bpm.transform([[max_bpm]])[0][0]

   


    avg_bpm=st.number_input("Batement de votre coeur moyen durant la séance", min_value=40)
    encodedavg_bpm = scaler_avg_bpm.transform([[avg_bpm]])[0][0]

    rest_bpm=st.number_input("BPM pendant la pause", min_value=0)
    encodedrest_bpm=scaler_rest_bpm.transform([[rest_bpm]])[0][0]

    tpsession=st.number_input("Durée de la séance en 'heure'", min_value=0.5,format="%.1f")
    encodedtpsession=scaler_tpsession.transform([[tpsession]])[0][0]

    hanches=st.number_input("Tour des hanches en 'cm'(voir le schéma)", min_value=10.0,format="%.2f")
    taille=st.number_input("Tour de taille en 'cm'(voir le schéma)", min_value=8.0,format="%.2f")
    cou=st.number_input("Tour de cou en 'cm'(voir le schéma)", min_value=05.0,format="%.2f")

    water=st.number_input("Quantité d'eau bu par jour en 'Litre'", min_value=0.5,format="%.2f")
    encodedwater=scaler_water.transform([[water]])[0][0] 

    xp=st.selectbox("Ton experience dans la salle",["débutant","moyen","expert"])
    if xp=="débutant":
      encodedxp=scaler_exp.transform([[1]])[0][0]
    elif xp=="moyen":
      encodedxp=scaler_exp.transform([[2]])[0][0]
    elif xp=="expert":
      encodedxp=scaler_exp.transform([[3]])[0][0]
    
    freq=st.selectbox("Combien de fois je vais à la salle par semaine",["1","2","3","4","5","6","7"])
    if freq=="1":
      encodedfreq=scaler_freq.transform([[1]])[0][0]
    elif freq=="2":
      encodedfreq=scaler_freq.transform([[2]])[0][0]
    elif freq=="3":
      encodedfreq=scaler_freq.transform([[3]])[0][0]
    elif freq=="4":
      encodedfreq=scaler_freq.transform([[4]])[0][0]
    elif freq=="5":
      encodedfreq=scaler_freq.transform([[5]])[0][0]
    elif freq=="6":
      encodedfreq=scaler_freq.transform([[6]])[0][0]
    elif freq=="7":
      encodedfreq=scaler_freq.transform([[7]])[0][0]
    
    bmi=poids/((height)**2)
    encodedbmi=scaler_bmi.transform([[bmi]])[0][0]
    
    genre=st.selectbox("votre genre",["homme","femme"])
    if genre=="homme":
      encodedgenre=1
      x=taille-cou
      fat=86.010*np.log10(x)-70.041*np.log10(taille)+36.76
      encodedfat=scaler_fat.transform([[fat]])[0][0]
    elif genre=="femme":
      encodedgenre=0
      x=taille+hanches-cou
      fat=163.205*np.log10(x)-97.684*np.log10(taille)-78.387
      encodedfat=scaler_fat.transform([[fat]])[0][0]
      
    type_entrainnement=st.selectbox("Type d'entrainnement : ",["Yoga","Crossfit","Cardio","Muscu"])
    
    
    # Combine input features into a DataFrame

    features = {

    'age': [age],
    "genre": [genre],
    'poids': [poids],
    'mesure': [height],
    'max_bpm': [max_bpm],
    'avg_bpm': [avg_bpm],
    'rest_bpm': [rest_bpm],
    'fat_pourcentage': [fat],
    'frequence': [freq],
    'experience': [xp],
    "temps d'entrainnement":[tpsession],
    "quantité_H2O":[water],
    "IMC" :[bmi],
    "type d'entrainnement": [type_entrainnement],
    "Xp":[xp]
}
    input_data = {

    'Age': [encodedage],
    'Weight (kg)':[encodedpoids],
    'Height (m)': [encodedheight],
    'Max_BPM': [encodedmax_bpm],
    'Avg_BPM': [encodedavg_bpm],
    'Resting_BPM': [encodedrest_bpm],
    'Session_Duration (hours)': [encodedtpsession],
    'Fat_Percentage': [encodedfat],
    'Water_Intake (liters)': [encodedwater],
    'Workout_Frequency (days/week)': [encodedfreq],
    'Experience_Level': [encodedxp],
    'BMI': [encodedbmi],
    'Gender_Male': [encodedgenre],
    'Workout_Type_HIIT': [0],
    'Workout_Type_Strength': [0],
    'Workout_Type_Yoga': [0]
}


    # Création du DataFrame
    input_data = pd.DataFrame(input_data)
    features = pd.DataFrame(features)
    #encodage du type d'entrainnement
    

    if type_entrainnement=="Cardio":
      pass
    elif type_entrainnement=="Yoga":
      input_data["Workout_Type_Yoga"]=1
    elif type_entrainnement=="Crossfit":
      input_data["Workout_Type_HIIT"]=1
    elif type_entrainnement=="Muscu":
      input_data["Workout_Type_Strength"]=1

    
    



    if st.button('Combien je brule...'):

        prediction = predict(input_data,features)
        
        

        st.write("Copyrights tidjaha 2025 (hamza.tidjani@yahoo.fr) \n\n Link Linkedin : https://www.linkedin.com/in/hamza-tidjani-539b78237 \n\n",prediction  )

        # URL de Google Drive (assurez-vous que c'est un lien de téléchargement direct)
        url = "https://drive.google.com/uc?export=download&id=1mdMdvXYGiowfy3UwNtMBCAllv7wt1DUT"  # Exemple d'ID
        




if __name__ == '__main__':

    main()
