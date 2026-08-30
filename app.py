import pickle as pkl 
import numpy as np
import streamlit as st

# Load Models 
scaler = pkl.load(open(r"Models/Scaler.pkl","rb"))
model = pkl.load(open(r"Models/Model.pkl","rb"))
encoder = pkl.load(open(r"Models/Encoder.pkl","rb"))




def Prediction_model(site_area,structure_type,water_consumption,recycling_rate,utilisation_rate,air_qality_index,resident_count):

    # Encode the data 
    structure_type =  encoder.transform([[structure_type]])[0]
    structure_type = structure_type[0]

    # Collect the features
    features = np.array([site_area,structure_type,water_consumption,recycling_rate,utilisation_rate,air_qality_index,resident_count])
    feature_scale = scaler.transform([features])
    model_pred = model.predict(feature_scale)
    return model_pred[0]



st.markdown(
    """
<style>

#electricity-cost-prediction-system{
    color: #4a89dc;

}
section , header {
    background: #ecf0f3 !important;
    font-family: sans-serif;
}
.stMainBlockContainer{
    max-width: 936px;
}
div[data-testid="stNumberInputContainer"] , div[data-testid="stTextInputRootElement"]{
    border: 1px solid;
}
input{
    background: #ffffff61 !important;
    }

div[data-testid="stElementContainer"]{
width: 80%;
}
div[direction="column"] {
    display: flex;
    align-items: center;
    gap: 30px;
}
.stForm{
    padding: 30px 0px;
    box-shadow:
		10px 10px 10px #d1d9e6,
		-10px -10px 10px #d1d9e6;
        padding-bottom: 40px; 
}
p {
    font-size: 15px !important;
}
h1 span {
    font-size: 35px;
}
span[data-testid="stHeaderActionElements"] {
    display: none;
}
div[data-testid="stHeadingWithActionElements"]{
text-align: center;
}

@media screen and (max-width: 600px){

h1 span {
    font-size: 30px;
}

}
</style>
""",
    unsafe_allow_html=True,
)


with st.form("Input_form"):
    st.title("Electricity Cost Prediction System")
    site_area = st.number_input("Site Area : ")
    structure_type = st.text_input("Structure Type : ")
    water_consumption = st.number_input("Water Consumption : ")
    recycling_rate = st.number_input("Recycling Rate : ")
    utilisation_rate = st.number_input("Utilisation Rate : ")
    air_qality_index = st.number_input("Air Qality Index : ")
    resident_count = st.number_input("Resident Count : ")
    submit_btn = st.form_submit_button("Model Prediction")

if submit_btn:
    print(site_area,structure_type,water_consumption,recycling_rate,utilisation_rate,air_qality_index,resident_count)
    model_prediction  = Prediction_model(site_area,structure_type,water_consumption,recycling_rate,utilisation_rate,air_qality_index,resident_count)
    st.success(f"According to the Linear Regression Model, The Electricity Bill will be Approximately {model_prediction}" )
