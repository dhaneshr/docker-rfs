import sqlite3
from fastapi import FastAPI, HTTPException, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import rpy2.robjects as robjects
from rpy2.robjects.environments import Environment
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
import json
import logging
import traceback

# Add this at the top of your file
DATABASE_PATH = '/data/predictions.db'

# Activate automatic conversion of pandas DataFrames to R DataFrames
pandas2ri.activate()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load R libraries
robjects.r('library(randomForestSRC)')
robjects.r('library(ggplot2)')
robjects.r('library(scales)')
robjects.r('library(survival)')

# Load the R model
try:
    robjects.r('load("follic_model.RData")')
    logger.info("R model loaded successfully.")
except Exception as e:
    logger.error("Error loading R model: %s", e)
    raise e

# Initialize FastAPI
app = FastAPI()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

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

# Ensure the table exists at startup
def initialize_db():
    try:
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

# Run the DB initialization
initialize_db()

# Define input schema with validation
class ModelInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    hgb: float = Field(..., ge=0)
    clinstg: int = Field(..., ge=1)
    ch: str = Field(..., pattern='^[YN]$')
    rt: str = Field(..., pattern='^[YN]$')

# Define the prediction endpoint
@app.post("/predict")
def predict(input: ModelInput, db: sqlite3.Connection = Depends(get_db)):
    try:
        # Prepare input data as a pandas DataFrame
        new_data = pd.DataFrame({
            'age': [input.age],
            'hgb': [input.hgb],
            'clinstg': [input.clinstg],
            'ch': [input.ch],
            'rt': [input.rt]
        })

        logger.info("Input data:\n%s", new_data)

        # Create a new R environment for thread safety
        r_env = Environment()
        r_new_data = pandas2ri.py2rpy(new_data)
        r_env['r_new_data'] = r_new_data
        r_env['follic_obj'] = robjects.globalenv['follic.obj']

        # Make predictions
        predict_func = robjects.r('predict')
        predictions = predict_func(r_env['follic_obj'], newdata=r_env['r_new_data'])
        predicted_values = predictions.rx2('predicted')
        predicted_array = np.array(predicted_values)
        predicted_df = pd.DataFrame(predicted_array, columns=["event.1", "event.2"])
        prediction_result = predicted_df.to_dict(orient='records')

        logger.info("Prediction result:\n%s", prediction_result)

        # Insert prediction into database
        try:
            cursor = db.cursor()
            cursor.execute('''
                INSERT INTO predictions (age, hgb, clinstg, ch, rt, prediction)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                input.age,
                input.hgb,
                input.clinstg,
                input.ch,
                input.rt,
                json.dumps(prediction_result)
            ))
            db.commit()
            logger.info("Prediction inserted into database.")
        except sqlite3.Error as e:
            logger.error("Database insertion error: %s", e)
            raise HTTPException(status_code=500, detail="Error inserting into the database")

        # Return the prediction result
        return {"predictions": prediction_result}

    except Exception as e:
        logger.error("Error during prediction: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error during prediction.")

# Endpoint to retrieve all logged predictions
@app.get("/logs")
def get_logs(db: sqlite3.Connection = Depends(get_db)):
    try:
        cursor = db.cursor()
        cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
        rows = cursor.fetchall()

        logs = []
        for row in rows:
            prediction_data = json.loads(row[6])  # Convert prediction string back to JSON
            log_entry = {
                "id": row[0],
                "age": row[1],
                "hgb": row[2],
                "clinstg": row[3],
                "ch": row[4],
                "rt": row[5],
                "prediction": prediction_data,
                "timestamp": row[7]
            }
            logs.append(log_entry)

        logger.info("Retrieved %d logs.", len(logs))
        return {"logs": logs}

    except sqlite3.Error as e:
        logger.error("Database retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Error retrieving logs from the database")
