import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import numpy as np
import pandas as pd
import logging


# Import R packages
rms = importr('rms')
risk_regression = importr('riskRegression')
prodlim = importr('prodlim')
Hmisc = importr('Hmisc')

# Load the model object from the .RData file
ro.r('load("final_FGR_clean.RData")')
# Access the model from the R global environment
model = ro.globalenv['model']

# Create new test data as a pandas DataFrame
new_data = pd.DataFrame({
    'age': [67],
    'sex_f': [1],
    'elective_adm': [1],
    'homelessness': [0],
    'peripheral_AD': [0],
    'coronary_AD': [1],
    'stroke': [0],
    'CHF': [0],
    'hypertension': [1],
    'COPD': [0],
    'CKD': [0],
    'malignancy': [0],
    'mental_illness': [0],
    'creatinine': [140.0],
    'Hb_A1C': [8.5],
    'albumin': [32.1],
    'Hb_A1C_missing': [0],
    'creatinine_missing': [0],
    'albumin_missing': [0]
})

# Convert the pandas DataFrame to an R data frame
with localconverter(ro.default_converter + pandas2ri.converter):
    new_data_r = ro.conversion.py2rpy(new_data)

# Access the model parameters
model_params = model.rx2('params')

# Retrieve knot locations
age_knots = model_params.rx2('age_knots')
creatinine_knots = model_params.rx2('creatinine_knots')
Hb_A1C_knots = model_params.rx2('hba1c_knots')
albumin_knots = model_params.rx2('albumin_knots')

# Generate spline basis functions using Hmisc.rcspline_eval
age_splines = Hmisc.rcspline_eval(new_data_r.rx2('age'), knots=age_knots, inclx=True)
creatinine_splines = Hmisc.rcspline_eval(new_data_r.rx2('creatinine'), knots=creatinine_knots, inclx=True)
Hb_A1C_splines = Hmisc.rcspline_eval(new_data_r.rx2('Hb_A1C'), knots=Hb_A1C_knots, inclx=True)
albumin_splines = Hmisc.rcspline_eval(new_data_r.rx2('albumin'), knots=albumin_knots, inclx=True)

# Convert spline results to NumPy arrays
with localconverter(ro.default_converter + pandas2ri.converter):
    age_splines_np = ro.conversion.rpy2py(age_splines)
    creatinine_splines_np = ro.conversion.rpy2py(creatinine_splines)
    Hb_A1C_splines_np = ro.conversion.rpy2py(Hb_A1C_splines)
    albumin_splines_np = ro.conversion.rpy2py(albumin_splines)

# Add non-linear components to the pandas DataFrame using NumPy indexing
new_data['age1'] = age_splines_np[:, 1]  # Second column
new_data['age2'] = age_splines_np[:, 2]  # Third column
new_data['creatinine1'] = creatinine_splines_np[:, 1]
new_data['creatinine2'] = creatinine_splines_np[:, 2]
new_data['Hb_A1C1'] = Hb_A1C_splines_np[:, 1]
new_data['Hb_A1C2'] = Hb_A1C_splines_np[:, 2]
new_data['albumin1'] = albumin_splines_np[:, 1]


# Convert the updated pandas DataFrame back to an R data frame
with localconverter(ro.default_converter + pandas2ri.converter):
    new_data_r = ro.conversion.py2rpy(new_data)


# Suppress console output during prediction
from rpy2.robjects import r, packages
import rpy2.rinterface_lib.callbacks

# Define a callback to suppress warnings/messages
def console_write_disable(x):
    pass

# Store the original console write function to restore later
original_console_write = rpy2.rinterface_lib.callbacks.consolewrite_print

# Suppress the console output
rpy2.rinterface_lib.callbacks.consolewrite_print = console_write_disable


# Predict risk at 1 year (365.25 days)
pred = ro.r('predict')(model, newdata=new_data_r, times=ro.FloatVector([365.25]))


# Extract the predicted value
pred_value = pred[0]
print(f'Predicted risk at 1 year: {pred_value}')

# Generate time points up to 5 years in steps of 5 days
time = np.arange(1, 365 * 5 + 1, 5)  # Include the last point
time_r = ro.FloatVector(time)

# Predict risks at specified time points
p = ro.r('predict')(model, newdata=new_data_r, times=time_r)
# Convert predictions to numpy array
p_np = np.array(p)
print(f'Predicted risks at 5 years: {p_np[0][0:5]}')