import sqlite3
from fastapi import FastAPI, HTTPException, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel, Field
import rpy2.robjects as robjects
from rpy2.robjects.environments import Environment
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
import json
import logging
import traceback
import os

DATABASE_PATH = '/app/local_data/predictions.db'


# Activate automatic conversion of pandas DataFrames to R DataFrames
pandas2ri.activate()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load R libraries and model
def load_r_libraries():
    robjects.r('library(randomForestSRC)')
    robjects.r('library(ggplot2)')
    robjects.r('library(scales)')
    robjects.r('library(survival)')


def load_r_model():
    try:
        robjects.r('load("follic_model.RData")')
        logger.info("R model loaded successfully.")
    except Exception as e:
        logger.error("Error loading R model: %s", e)
        raise e

load_r_libraries()
load_r_model()

# Initialize FastAPI
app = FastAPI()

# Serve static files from the "resources" directory
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

# Set up Jinja2 templates for HTML files
templates = Jinja2Templates(directory="templates")

# Enable CORS to allow requests from the web page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development purposes; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        yield conn
    finally:
        conn.close()


def initialize_db():
    try:
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER,
                hgb REAL,
                clinstg INTEGER,
                ch TEXT,
                rt TEXT,
                predict_at INTEGER,
                prediction TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error("Error initializing the database: %s", e)
        raise HTTPException(status_code=500, detail="Error initializing the database")

initialize_db()

# Define input schema with validation
class ModelInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    hgb: float = Field(..., ge=0)
    clinstg: int = Field(..., ge=1)
    ch: str = Field(..., pattern='^[YN]$')
    rt: str = Field(..., pattern='^[YN]$')
    predict_at: int = Field(..., ge=0, le=1000)

# Helper functions
def prepare_input_data(input: ModelInput) -> pd.DataFrame:
    return pd.DataFrame({
        'age': [input.age],
        'hgb': [input.hgb],
        'clinstg': [input.clinstg],
        'ch': [input.ch],
        'rt': [input.rt],
        'predict_at': [input.predict_at]
    })

def make_prediction(model, new_data: pd.DataFrame) -> dict:
    r_env = Environment()
    r_new_data = pandas2ri.py2rpy(new_data)
    r_env['r_new_data'] = r_new_data
    r_env['follic_obj'] = model

    # Make predictions
    predict_func = robjects.r('predict')
    predictions = predict_func(r_env['follic_obj'], newdata=r_env['r_new_data'])
    return predictions

def extract_prediction_details(predictions, time_point: int) -> dict:
    cif = predictions.rx2('cif')
    chf = predictions.rx2('chf')
    time_interest = np.array(predictions.rx2('time.interest'))
    time_index_python = np.argmin(np.abs(time_interest - time_point))

    cif_event1_at_time = cif[0, time_index_python, 0]
    cif_event2_at_time = cif[0, time_index_python, 1]
    chf_event1_at_time = chf[0, time_index_python, 0]
    chf_event2_at_time = chf[0, time_index_python, 1]

    logger.info(f"CIF event 1 at time {time_point}: {cif_event1_at_time}")
    logger.info(f"CIF event 2 at time {time_point}: {cif_event2_at_time}")
    logger.info(f"CHF event 1 at time {time_point}: {chf_event1_at_time}")
    logger.info(f"CHF event 2 at time {time_point}: {chf_event2_at_time}")

    predicted_array = np.array(predictions.rx2('predicted'))
    predicted_df = pd.DataFrame(predicted_array, columns=["event.1", "event.2"])
    return predicted_df.to_dict(orient='records')

def store_prediction(db: sqlite3.Connection, input: ModelInput, prediction_result: dict):
    try:
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO predictions (age, hgb, clinstg, ch, rt, predict_at, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            input.age,
            input.hgb,
            input.clinstg,
            input.ch,
            input.rt,
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
        logger.info("Input data:\n%s", new_data)

        predictions = make_prediction(robjects.globalenv['follic.obj'], new_data)
        prediction_result = extract_prediction_details(predictions, input.predict_at)

        store_prediction(db, input, prediction_result)

        return {"predictions": prediction_result}

    except Exception as e:
        logger.error("Error during prediction: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error during prediction.")

@app.get("/logs")
def get_logs(db: sqlite3.Connection = Depends(get_db)):
    try:
        cursor = db.cursor()
        cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
        rows = cursor.fetchall()

        logs = []
        for row in rows:
            prediction_data = json.loads(row[7])
            log_entry = {
                "id": row[0],
                "age": row[1],
                "hgb": row[2],
                "clinstg": row[3],
                "ch": row[4],
                "rt": row[5],
                "predict_at": row[6],
                "prediction": prediction_data,
                "timestamp": row[8]
            }
            logs.append(log_entry)

        logger.info("Retrieved %d logs.", len(logs))
        return {"logs": logs}

    except sqlite3.Error as e:
        logger.error("Database retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Error retrieving logs from the database")
