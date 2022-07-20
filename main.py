import pickle
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from lena import Gleb
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictModel, Diagnosis
from max3 import three


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pkl_filename = 'pickle_model.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


class_bol = [
    'Здоров',
    'Острая',
    'Насморк'
]


@app.post('/upload')
async def predict(file_in: UploadFile = File(...)):
    contents = await file_in.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer, sep=';', decimal=',', dtype={'PatientKey': 'Int32'},
                     encoding='utf-8', parse_dates=['BirthDate', 'LaboratoryResultsDate', 'MinLaboratoryResultsDate'])
    data, patolog = Gleb(df)
    prediction = pickle_model.predict_proba(data)
    prediction_list = prediction.tolist()
    prediction_list = prediction_list[0]
    new_names, new_predict = three(class_bol, prediction_list)
    print(new_names)
    print(new_predict)
    return PredictModel(
        predict=[
            Diagnosis(
                title=new_names[i],
                value=round(d*100, 2),
                pathologies=patolog
            ) for i, d in enumerate(new_predict[0:3])
        ]
    )
