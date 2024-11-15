import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go

# Define the Streamlit app
def app():
    st.title("Patient Data Input Form")

    # Form for patient information input
    with st.form(key="patient_form"):
        st.header("Enter Patient Information")

        patient_id = st.text_input("Patient ID", value="")
        hospital_id = st.text_input("Hospital ID", value="")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        hgb = st.number_input("Hemoglobin Level", min_value=0.0, step=0.1, value=12.0)
        clinstg = st.number_input("Clinical Stage", min_value=1, value=1)
        predict_at = st.number_input("Prediction Time Horizon (Days)", min_value=0, value=30)
        ch = st.selectbox("CH (Y/N)", options=["Y", "N"])
        rt = st.selectbox("RT (Y/N)", options=["Y", "N"])

        # Submit button
        submit_button = st.form_submit_button(label="Submit")

    # Handle form submission
    if submit_button:
        # Prepare the input data as a dictionary
        form_data = {
            "patient_id": patient_id,
            "hospital_id": hospital_id,
            "age": age,
            "hgb": hgb,
            "clinstg": clinstg,
            "predict_at": predict_at,
            "ch": ch,
            "rt": rt
        }

        try:
            # Send the POST request to FastAPI backend
            response = requests.post("http://localhost:8000/predict", json=form_data)
            response.raise_for_status()
            prediction_result = response.json()

            st.success("Prediction Successful!")
            st.json(prediction_result)  # Display the raw prediction result

            # Plotting the Cumulative Incidence Function (CIF) and Cumulative Hazard Function (CHF)
            cif_series = prediction_result.get("cif_series", [])
            chf_series = prediction_result.get("chf_series", [])

            if cif_series and chf_series:
                st.subheader("Cumulative Charts")

                # Create CIF Chart
                cif_chart = go.Figure()
                cif_chart.add_trace(go.Scatter(y=cif_series, mode="lines", name="CIF"))
                cif_chart.update_layout(
                    title="Cumulative Incidence Function (CIF)",
                    xaxis_title="Time Points",
                    yaxis_title="CIF Value",
                )
                st.plotly_chart(cif_chart)

                # Create CHF Chart
                chf_chart = go.Figure()
                chf_chart.add_trace(go.Scatter(y=chf_series, mode="lines", name="CHF"))
                chf_chart.update_layout(
                    title="Cumulative Hazard Function (CHF)",
                    xaxis_title="Time Points",
                    yaxis_title="CHF Value",
                )
                st.plotly_chart(chf_chart)

            # Display prediction table
            st.subheader("Prediction Results")
            result_table = pd.DataFrame({
                "Event": ["Relapse", "Death"],
                "CIF": [prediction_result.get("cif_event1", 0), prediction_result.get("cif_event2", 0)],
                "CHF": [prediction_result.get("chf_event1", 0), prediction_result.get("chf_event2", 0)]
            })

            st.table(result_table)

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    st.set_page_config(page_title="Patient Prediction App", layout="wide")
    app()
