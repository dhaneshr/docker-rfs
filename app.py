import sqlite3
from fastapi import FastAPI, HTTPException, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel, Field
from typing import Optional, List
import rpy2.robjects as robjects
from rpy2.robjects.environments import Environment
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
import json
import logging
import traceback
import os
from enum import Enum

DATABASE_PATH = '/app/local_data/predictions.db'

# Activate automatic conversion of pandas DataFrames to R DataFrames
pandas2ri.activate()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_r_libraries():
    robjects.r('library(randomForestSRC)')
    robjects.r('library(ggplot2)')
    robjects.r('library(scales)')
    robjects.r('library(survival)')

def load_r_model():
    try:
        robjects.r('load("dummy_model_clean2.RData")')
        if 'model' in robjects.globalenv:
            logger.info("R model loaded successfully.")
        else:
            logger.error("R model object 'model' not found in global environment.")
            raise ValueError("R model object not found.")
    except Exception as e:
        logger.error("Error loading R model: %s", e)
        raise e

load_r_libraries()
load_r_model()


app = FastAPI()

# Serve static files from the "resources" directory
app.mount("/resources", StaticFiles(directory="resources"), name="resources")


templates = Jinja2Templates(directory="templates")

# Enable CORS to allow requests from the web page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development purposes; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def get_db():
    if not os.path.exists(DATABASE_PATH):
        logger.warning("Database not found. Initializing...")
        initialize_db()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable name-based access
    try:
        yield conn
    finally:
        conn.close()


def initialize_db():
    """Create the database and initialize the predictions table if not present."""
    try:
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        logger.info(f"Ensured that the directory {os.path.dirname(DATABASE_PATH)} exists.")

        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row  # Enable name-based access
        logger.info(f"Connected to SQLite database at {DATABASE_PATH}.")

        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER,
                sex_f INTEGER,
                elective_adm INTEGER,
                homelessness INTEGER,
                peripheral_AD INTEGER,
                coronary_AD INTEGER,
                stroke INTEGER,
                CHF INTEGER,
                hypertension INTEGER,
                COPD INTEGER,
                CKD INTEGER,
                malignancy INTEGER,
                mental_illness INTEGER,
                creatinine REAL,
                Hb_A1C REAL,
                albumin REAL,
                Hb_A1C_missing INTEGER,
                creatinine_missing INTEGER,
                albumin_missing INTEGER,
                predict_at INTEGER,
                prediction TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

        # Log table schema
        cursor.execute("PRAGMA table_info(predictions);")
        columns = cursor.fetchall()
        column_info = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
        logger.info(f"Table 'predictions' schema: {column_info}")

        conn.close()

        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error("Error initializing the database: %s", e)
        raise HTTPException(status_code=500, detail="Error initializing the database")
    except Exception as e:
        logger.error("Unexpected error during database initialization: %s", e)
        raise HTTPException(status_code=500, detail="Unexpected error during database initialization")

# Initialize the database once during startup
initialize_db()

class ModelInput(BaseModel):
    age: int
    sex_f: int
    elective_adm: int
    homelessness: int
    peripheral_AD: int
    coronary_AD: int
    stroke: int
    CHF: int
    hypertension: int
    COPD: int
    CKD: int
    malignancy: int
    mental_illness: int
    creatinine: Optional[float] = None
    Hb_A1C: Optional[float] = None
    albumin: Optional[float] = None
    Hb_A1C_missing: int
    creatinine_missing: int
    albumin_missing: int
    predict_at: int

# Helper functions
def prepare_input_data(input: ModelInput) -> pd.DataFrame:
    data = {
        'age': [input.age],
        'sex_f': [input.sex_f],
        'elective_adm': [input.elective_adm],
        'homelessness': [input.homelessness],
        'peripheral_AD': [input.peripheral_AD],
        'coronary_AD': [input.coronary_AD],
        'stroke': [input.stroke],
        'CHF': [input.CHF],
        'hypertension': [input.hypertension],
        'COPD': [input.COPD],
        'CKD': [input.CKD],
        'malignancy': [input.malignancy],
        'mental_illness': [input.mental_illness],
        'creatinine': [input.creatinine],
        'Hb_A1C': [input.Hb_A1C],
        'albumin': [input.albumin],
        'Hb_A1C_missing': [input.Hb_A1C_missing],
        'creatinine_missing': [input.creatinine_missing],
        'albumin_missing': [input.albumin_missing],
    }
    logger.debug(f"Prepared input data for model: {data}")
    df = pd.DataFrame(data)
    logger.debug(f"DataFrame created:\n{df}")
    return df

def make_prediction(model, new_data: pd.DataFrame) -> dict:
    logger.debug("Starting prediction with R model.")
    try:
        r_env = Environment()
        r_new_data = pandas2ri.py2rpy(new_data)
        r_env['r_new_data'] = r_new_data
        r_env['model'] = model

        # Make predictions
        predict_func = robjects.r('predict')
        predictions = predict_func(r_env['model'], newdata=r_env['r_new_data'])
        logger.debug("Prediction successful.")
        return predictions
    except Exception as e:
        logger.error("Error during R model prediction: %s", traceback.format_exc())
        raise e

def extract_prediction_details(predictions, time_point: int) -> dict:
    cif = np.array(predictions.rx2('cif')[0])
    chf = np.array(predictions.rx2('chf')[0])
    time_interest = np.array(predictions.rx2('time.interest'))

    # Find the closest time index to the requested time point
    time_index = np.argmin(np.abs(time_interest - time_point))

    # Extract individual event values
    cif_event1_at_time = cif[time_index, 0]
    cif_event2_at_time = cif[time_index, 1]
    chf_event1_at_time = chf[time_index, 0]
    chf_event2_at_time = chf[time_index, 1]

    logger.info(f"CIF event 1 at time {time_point}: {cif_event1_at_time}")
    logger.info(f"CIF event 2 at time {time_point}: {cif_event2_at_time}")
    logger.info(f"CHF event 1 at time {time_point}: {chf_event1_at_time}")
    logger.info(f"CHF event 2 at time {time_point}: {chf_event2_at_time}")

    # Prepare response with both the series and individual event values
    return {
        "cif_series": cif[:, 0].tolist(),
        "chf_series": chf[:, 0].tolist(),
        "cif_event1": cif_event1_at_time,
        "cif_event2": cif_event2_at_time,
        "chf_event1": chf_event1_at_time,
        "chf_event2": chf_event2_at_time
    }

def store_prediction(db: sqlite3.Connection, input: ModelInput, prediction_result: dict):
    try:
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                age, sex_f, elective_adm, homelessness, 
                peripheral_AD, coronary_AD, stroke, CHF, hypertension, COPD, 
                CKD, malignancy, mental_illness, creatinine, Hb_A1C, albumin, 
                Hb_A1C_missing, creatinine_missing, albumin_missing, predict_at, prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            input.age,
            input.sex_f,
            input.elective_adm,
            input.homelessness,
            input.peripheral_AD,
            input.coronary_AD,
            input.stroke,
            input.CHF,
            input.hypertension,
            input.COPD,
            input.CKD,
            input.malignancy,
            input.mental_illness,
            input.creatinine,
            input.Hb_A1C,
            input.albumin,
            input.Hb_A1C_missing,
            input.creatinine_missing,
            input.albumin_missing,
            input.predict_at,
            json.dumps(prediction_result)
        ))
        db.commit()
        logger.info("Prediction inserted into database.")
    except sqlite3.Error as e:
        logger.error("Database insertion error: %s", e)
        raise HTTPException(status_code=500, detail="Error inserting into the database")

# FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/predict")
def predict(input: ModelInput, db: sqlite3.Connection = Depends(get_db)):
    try:
        new_data = prepare_input_data(input)
        predictions = make_prediction(robjects.globalenv['model'], new_data)
        prediction_result = extract_prediction_details(predictions, input.predict_at)

        store_prediction(db, input, prediction_result)

        return prediction_result
    except Exception as e:
        logger.error("Error during prediction: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error during prediction.")

@app.get("/logs")
def get_logs(db: sqlite3.Connection = Depends(get_db)):

    query = '''
            SELECT id, age, sex_f, elective_adm, homelessness, peripheral_AD, coronary_AD, stroke, CHF, hypertension, 
            COPD, CKD, malignancy, mental_illness, creatinine, Hb_A1C, albumin, Hb_A1C_missing, creatinine_missing, 
            albumin_missing, predict_at, prediction, timestamp FROM predictions ORDER BY timestamp DESC
            '''
    try:
        cursor = db.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        logs = []
        for row in rows:
            prediction_raw = row['prediction']

            # Handle empty or NULL predictions
            if not prediction_raw:
                prediction_data = {}
                logger.warning(f"Row ID {row['id']} has empty 'prediction' field.")
            else:
                try:
                    prediction_data = json.loads(prediction_raw)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in 'prediction' for row ID {row['id']}: {prediction_raw}")
                    prediction_data = {"error": "Invalid prediction data"}

            log_entry = {
                "id": row['id'],
                "age": row['age'],
                "sex_f": row['sex_f'],
                "elective_adm": row['elective_adm'],
                "homelessness": row['homelessness'],
                "peripheral_AD": row['peripheral_AD'],
                "coronary_AD": row['coronary_AD'],
                "stroke": row['stroke'],
                "CHF": row['CHF'],
                "hypertension": row['hypertension'],
                "COPD": row['COPD'],
                "CKD": row['CKD'],
                "malignancy": row['malignancy'],
                "mental_illness": row['mental_illness'],
                "creatinine": row['creatinine'],
                "Hb_A1C": row['Hb_A1C'],
                "albumin": row['albumin'],
                "Hb_A1C_missing": row['Hb_A1C_missing'],
                "creatinine_missing": row['creatinine_missing'],
                "albumin_missing": row['albumin_missing'],
                "predict_at": row['predict_at'],
                "prediction": prediction_data,
                "timestamp": row['timestamp']
            }
            logs.append(log_entry)

        logger.info("Retrieved %d logs.", len(logs))
        return {"logs": logs}
    except sqlite3.Error as e:
        logger.error("Database retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Error retrieving logs from the database")

    except sqlite3.Error as e:
        logger.error("Database retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Error retrieving logs from the database")

@app.get("/health")
def health_check():
    """Health check to verify database availability."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("SELECT 1")  # Simple query to ensure DB is accessible
        conn.close()
        return {"status": "ok"}
    except sqlite3.Error as e:
        logger.error("Database health check failed: %s", e)
        raise HTTPException(status_code=500, detail="Database is not available")
