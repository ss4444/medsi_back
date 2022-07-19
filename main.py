import pickle
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from Lena import Gleb
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from Schemas import PredictModel, Diagnosis
import os


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
    os.remove('Lena.csv')
    return patolog
    # return PredictModel(
    #     predict=[
    #         Diagnosis(
    #             title='Здоров',
    #             value=round(prediction_list[0]*100, 2)
    #         ),
    #         Diagnosis(
    #             title='Острая',
    #             value=round(prediction_list[1]*100, 2)
    #         ),
    #         Diagnosis(
    #             title='Насморк',
    #             value=round(prediction_list[2]*100, 2)
    #         )
    #     ]
    # )
