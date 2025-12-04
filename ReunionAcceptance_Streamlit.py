
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("reunion_xgboost_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Streamlit inputs
st.title("Reunion Acceptance Prediction")

Reunion_Years_Out = st.slider("Years Since Graduation", 0, 50, 20)
Greek = st.checkbox("Greek?")
Sorority = st.checkbox("Sorority?")
Frat = st.checkbox("Fraternity?")
Parent = st.checkbox("Parent?")
Grandparent = st.checkbox("Grandparent?")
Alumni = st.checkbox("Alumni?")
Board_of_Trustees = st.checkbox("Board of Trustees?")
Donor = st.checkbox("Donor?")
Bucknell_Staff = st.checkbox("Bucknell Staff?")
Fundraiser = st.checkbox("Fundraiser?")

# Prediction
if st.button("Predict Acceptance"):
    new_invitee = pd.DataFrame({
        "Reunion_Years_Out": [Reunion_Years_Out],
        "Greek": [Greek],
        "Sorority": [Sorority],
        "Frat": [Frat],
        "Parent": [Parent],
        "Grandparent": [Grandparent],
        "Alumni": [Alumni],
        "Board_of_Trustees": [Board_of_Trustees],
        "Donor": [Donor],
        "Bucknell_Staff": [Bucknell_Staff],
        "Fundraiser": [Fundraiser]
    })

    predicted_prob = loaded_model.predict_proba(new_invitee)[:, 1]
    predicted_class = loaded_model.predict(new_invitee)
    formatted_prob = f"{predicted_prob[0]:.2f}"

    st.write(f"Predicted Probability of Acceptance: **{formatted_prob}**")
    if predicted_class[0] == 1:
        st.success("Predicted Class: **Will Accept**")
    else:
        st.success("Predicted Class: **Will Not Accept**")

    probabilities = [predicted_prob[0], 1 - predicted_prob[0]]
    labels = ['Will Accept', 'Will Not Accept']
    chart_data = pd.DataFrame({'Probability': probabilities}, index=labels)
    st.write("Prediction Breakdown:")
    st.bar_chart(chart_data)

st.markdown("---")
st.markdown("**Developed by Your Team** | Powered by Streamlit")
