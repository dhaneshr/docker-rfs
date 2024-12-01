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
from fastapi import File, UploadFile
from io import StringIO, BytesIO
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter


DATABASE_PATH = '/app/local_data/predictions.db'

# Global variable to hold the R model in memory
global_model = None

# Activate automatic conversion of pandas DataFrames to R DataFrames
pandas2ri.activate()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import R packages
rms = importr('rms')
risk_regression = importr('riskRegression')
prodlim = importr('prodlim')
Hmisc = importr('Hmisc')  # Ensure Hmisc is imported

def load_r_libraries():
    robjects.r('library(rms)')
    robjects.r('library(riskRegression)')
    robjects.r('library(prodlim)')
    robjects.r('library(Hmisc)')  


def load_r_model():
    global global_model
    try:
        robjects.r('load("final_FGR_clean.RData")')
        if 'model' in robjects.globalenv:
            global_model = robjects.globalenv['model']
            logger.info("R model loaded successfully into global environment.")
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

# Mount the static directory to serve JavaScript and CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

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


# to process csv as inputs
class ModelInputCSV(BaseModel):
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
        'creatinine': [np.nan if input.creatinine is None else input.creatinine],
        'Hb_A1C': [np.nan if input.Hb_A1C is None else input.Hb_A1C],
        'albumin': [np.nan if input.albumin is None else input.albumin],
        'Hb_A1C_missing': [input.Hb_A1C_missing],
        'creatinine_missing': [input.creatinine_missing],
        'albumin_missing': [input.albumin_missing],
    }
    df = pd.DataFrame(data)
    logger.debug(f"DataFrame created:\n{df}")

    try:
        # Convert pandas DataFrame to R data frame
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_df = robjects.conversion.py2rpy(df)

        # Extract knots from the R model
        model_params = global_model.rx2('params')
        age_knots = model_params.rx2('age_knots')
        creatinine_knots = model_params.rx2('creatinine_knots')
        hba1c_knots = model_params.rx2('hba1c_knots')
        albumin_knots = model_params.rx2('albumin_knots')

        # Generate spline basis functions using Hmisc.rcspline_eval
        age_splines = Hmisc.rcspline_eval(r_df.rx2('age'), knots=age_knots, inclx=True)
        creatinine_splines = Hmisc.rcspline_eval(r_df.rx2('creatinine'), knots=creatinine_knots, inclx=True)
        Hb_A1C_splines = Hmisc.rcspline_eval(r_df.rx2('Hb_A1C'), knots=hba1c_knots, inclx=True)
        albumin_splines = Hmisc.rcspline_eval(r_df.rx2('albumin'), knots=albumin_knots, inclx=True)

        # Convert spline results back to pandas DataFrames
        with localconverter(robjects.default_converter + pandas2ri.converter):
            age_splines_np = np.array(robjects.conversion.rpy2py(age_splines))
            creatinine_splines_np = np.array(robjects.conversion.rpy2py(creatinine_splines))
            Hb_A1C_splines_np = np.array(robjects.conversion.rpy2py(Hb_A1C_splines))
            albumin_splines_np = np.array(robjects.conversion.rpy2py(albumin_splines))

        # Add non-linear components to the pandas DataFrame using NumPy indexing
        df['age1'] = age_splines_np[:, 1]  # Second column
        df['age2'] = age_splines_np[:, 2]  # Third column
        df['creatinine1'] = creatinine_splines_np[:, 1]
        df['creatinine2'] = creatinine_splines_np[:, 2]
        df['Hb_A1C1'] = Hb_A1C_splines_np[:, 1]
        df['Hb_A1C2'] = Hb_A1C_splines_np[:, 2]
        df['albumin1'] = albumin_splines_np[:, 1]

    except Exception as e:
        logger.error("Error while generating splines: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error during spline generation")

    logger.debug(f"DataFrame with spline components:\n{df}")
    return df

def make_prediction(new_data: pd.DataFrame) -> dict:
    global global_model
    if global_model is None:
        raise RuntimeError("R model is not loaded.")
    logger.debug("Starting prediction with R model.")
    
    # Suppress R console output
    import rpy2.rinterface_lib.callbacks

    # Define a callback to suppress warnings/messages
    def console_write_disable(x):
        pass

    # Store the original console write function to restore later
    original_console_write = rpy2.rinterface_lib.callbacks.consolewrite_print

    # Suppress the console output
    rpy2.rinterface_lib.callbacks.consolewrite_print = console_write_disable

    try:
        # Convert pandas DataFrame to R data frame
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_new_data = robjects.conversion.py2rpy(new_data)

        predict_func = robjects.r['predict']

        # Predict at 1-year time point
        one_year_time = robjects.FloatVector([365.25])
        one_year_prediction = predict_func(global_model, newdata=r_new_data, times=one_year_time)

        # Convert to NumPy array
        one_year_prediction_np = np.array(one_year_prediction)
        logger.debug(f"one_year_prediction (NumPy array): {one_year_prediction_np}")

        # Extract the predicted value
        if len(one_year_prediction_np) > 0:
            one_year_risk = one_year_prediction_np[0]
        else:
            raise ValueError("Prediction returned an empty array for one-year risk.")

        logger.debug(f"Predicted risk at 1 year: {one_year_risk}")

        # Predict over multiple time points
        time_points = np.arange(1, 365 * 5 + 1, 5)  # Up to 5 years in steps of 5 days
        time_points_r = robjects.FloatVector(time_points)
        multiple_time_predictions = predict_func(global_model, newdata=r_new_data, times=time_points_r)

        # Convert the multiple time predictions to a NumPy array
        multiple_time_predictions_np = np.array(multiple_time_predictions[0])
        logger.debug(f"multiple_time_predictions (NumPy array, shape: {multiple_time_predictions_np.shape}): {multiple_time_predictions_np}")

        if multiple_time_predictions_np.size != len(time_points):
            raise ValueError("Mismatch between number of time points and predicted risks.")

        time_points_list = time_points.tolist()

        return {
            'one_year_risk': float(one_year_risk),
            'time_points': time_points_list,
            'predicted_risks': multiple_time_predictions_np.tolist()
        }

    except Exception as e:
        logger.error("Error during R model prediction: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error during prediction")

    finally:
        # Restore the original console write function
        rpy2.rinterface_lib.callbacks.consolewrite_print = original_console_write





def extract_prediction_details(predictions, time_point: int) -> dict:
    time_points = np.array(predictions['time_points'])
    predicted_risks = np.array(predictions['predicted_risks'])

    # Find the closest time index to the requested time point
    time_index = int(np.argmin(np.abs(time_points - time_point)))

    # Extract individual risk value at the requested time point
    risk_at_time = float(predicted_risks[time_index])

    '''
       # Prepare response with both the series and individual event values
    prediction_result = {
        "cif_series": cif_series,
        "chf_series": chf_series,
        "cif_event1": cif_event1_at_time,
        "cif_event2": cif_event2_at_time,
        "chf_event1": chf_event1_at_time,
        "chf_event2": chf_event2_at_time,
        "time_interest": time_interest.tolist()
    }
    '''
    # Prepare response with both the series and individual risk value
    prediction_result = {
        "cif_event1": predictions['one_year_risk'], # cif_event_1 = prediction of foot complication at 1-year
        "time_interest": time_points.tolist(),
        "cif_series": predicted_risks.tolist(),
    }
    logger.debug(f"Extracted prediction details: {prediction_result}")
    return prediction_result


def store_prediction(db: sqlite3.Connection, input: ModelInput, prediction_result: dict):
    insert_values = None  # Initialize insert_values
    try:
        cursor = db.cursor()
        insert_values = (
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
        )
        cursor.execute('''
            INSERT INTO predictions (
                age, sex_f, elective_adm, homelessness, 
                peripheral_AD, coronary_AD, stroke, CHF, hypertension, COPD, 
                CKD, malignancy, mental_illness, creatinine, Hb_A1C, albumin, 
                Hb_A1C_missing, creatinine_missing, albumin_missing, predict_at, prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', insert_values)
        db.commit()
        logger.info("Prediction inserted into database.")
    except Exception as e:
        logger.error("Error inserting into the database: %s", e)
        logger.error("Failed to insert values: %s", insert_values)
        raise HTTPException(status_code=500, detail=f"Error inserting into the database: {e}")


# FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Prediction endpoint using the loaded model in memory
@app.post("/predict")
def predict(input: ModelInput, db: sqlite3.Connection = Depends(get_db)):
    try:
        new_data = prepare_input_data(input)
        logger.info("Prepared input data for prediction.")
        predictions = make_prediction(new_data)
        logger.info("Prediction successful.")
        # Extract relevant information for the prediction result (e.g., CIF at specified time point)
        prediction_result = extract_prediction_details(predictions, input.predict_at)
        # Store prediction in the database
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


@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...), db: sqlite3.Connection = Depends(get_db)):
    """
    Accepts a CSV file containing multiple patient data entries,
    processes each row, and returns predictions.
    """
    try:
        content = file.file.read()
        # Decode bytes to string if necessary
        if isinstance(content, bytes):
            content = content.decode('utf-8')

        csv_file = StringIO(content)
        df = pd.read_csv(csv_file)
        logger.info("CSV file read successfully.")

        # Validate the DataFrame columns
        required_columns = [
            'age', 'sex_f', 'elective_adm', 'homelessness', 'peripheral_AD', 'coronary_AD', 'stroke',
            'CHF', 'hypertension', 'COPD', 'CKD', 'malignancy', 'mental_illness', 'creatinine',
            'Hb_A1C', 'albumin', 'Hb_A1C_missing', 'creatinine_missing', 'albumin_missing', 'predict_at'
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns in CSV: {missing_cols}")

        successful_predictions = []
        errors = []

        for index, row in df.iterrows():
            try:
                # Prepare input data
                input_data = row.to_dict()
                # Convert NaNs to None
                input_data = {k: (v if pd.notna(v) else None) for k, v in input_data.items()}
                # Create ModelInput instance
                input_model = ModelInput(**input_data)

                # Prepare data and make prediction
                new_data = prepare_input_data(input_model)
                predictions = make_prediction(new_data)
                prediction_result = extract_prediction_details(predictions, input_model.predict_at)

                # Store prediction in the database
                store_prediction(db, input_model, prediction_result)

                # Append the prediction result to the list
                successful_predictions.append({
                    "row_index": index,
                    "input_data": input_data,
                    "prediction": prediction_result
                })
            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
                logger.debug("Traceback: %s", traceback.format_exc())
                errors.append({
                    "row_index": index,
                    "input_data": input_data,
                    "error": str(e)
                })

        return {
            "successful_predictions": successful_predictions,
            "errors": errors
        }

    except HTTPException as e:
        raise e  # Re-raise HTTPExceptions to return appropriate responses

    except Exception as e:
        logger.error("Error processing CSV file: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error processing CSV file")

