import pickle
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from gleb import Gleb
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


pkl_filename = 'model_75.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


class_bol = [
    'Здоров',
    'Гипертензивная [гипертоническая] болезнь с преимущественным поражением сердца без (застойной) сердечной недостаточности',
    'Нарушение сердечного ритма неуточненное'
]


@app.post('/upload')
async def predict(file_in: UploadFile = File(...)):
    contents = await file_in.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer, sep=';', decimal=',', dtype={'PatientKey': 'Int32'},
                     encoding='utf-8', parse_dates=['BirthDate', 'LaboratoryResultsDate', 'MinLaboratoryResultsDate'])
    data, pathologies = Gleb(df)
    prediction = pickle_model.predict_proba(data)
    prediction_list = prediction.tolist()
    prediction_list = prediction_list[0]
    new_names, new_predict = three(class_bol, prediction_list)
    return PredictModel(
        predict=[
            Diagnosis(
                title=new_names[i],
                value=round(d*100, 2),
                pathologies=pathologies
            ) for i, d in enumerate(new_predict[0:3])
        ]
    )
