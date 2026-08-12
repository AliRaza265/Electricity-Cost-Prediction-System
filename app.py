import pickle as pkl 
import numpy as np

# Load Models 
scaler = pkl.load(open(r"Models\Scaler.pkl","rb"))
model = pkl.load(open(r"Models\Model.pkl","rb"))
encoder = pkl.load(open(r"Models\Encoder.pkl","rb"))
print(model)



def Prediction_model(site_area,structure_type,water_consumption,recycling_rate,utilisation_rate,air_qality_index,resident_count):

    # Encode the data 
    structure_type =  encoder.transform([[structure_type]])[0]
    structure_type = structure_type[0]

    # Collect the features
    features = np.array([site_area,structure_type,water_consumption,recycling_rate,utilisation_rate,air_qality_index,resident_count])
    feature_scale = scaler.transform([features])
    print(feature_scale)


structure_type = input("structure_type : ")
site_area = int(input("site_area : "))
water_consumption = float(input("water_consumption : "))
recycling_rate = int(input("recycling_rate : "))
utilisation_rate = int(input("utilisation_rate : "))
air_qality_index = int(input("air_qality_index : "))
resident_count = int(input("resident_count : "))
Prediction_model(site_area,structure_type,water_consumption,recycling_rate,utilisation_rate,air_qality_index,resident_count)