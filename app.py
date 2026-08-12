import pickle as pkl 

scaler = pkl.load(open(r"Models\Scaler.pkl","rb"))
model = pkl.load(open(r"Models\Model.pkl","rb"))
print(model)



def Prediction_model(site_area,structure_type,water_consumption,recycling_rate,utilisation_rate,air_qality_index,issue_reolution_time,resident_count):
    pass 


