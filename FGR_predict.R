library(rms) 
library(riskRegression)
library(prodlim)

## load model object
load("final_FGR_clean.RData")

## create new test data
new_data <- data.frame(
  age = 67,
  sex_f = 1,
  elective_adm = 1,
  homelessness = 0,
  peripheral_AD = 0,
  coronary_AD = 1,
  stroke = 0,
  CHF = 0,
  hypertension = 1,
  COPD = 0,
  CKD = 0,
  malignancy = 0,
  mental_illness = 0,
  creatinine = 140.0,
  Hb_A1C = 8.5,
  albumin = 32.1,
  Hb_A1C_missing = 0,
  creatinine_missing = 0,
  albumin_missing = 0
)

## add splines to data
# derive splines based on knot locations
age_splines <- rcs(new_data$age, model$params$age_knots)
creatinine_splines <- rcs(new_data$creatinine, model$params$creatinine_knots)
Hb_A1C_splines <- rcs(new_data$Hb_A1C, model$params$hba1c_knots)
albumin_splines <- rcs(new_data$albumin, model$params$albumin_knots)

## add non-linear components to new_data
new_data$age1 <- age_splines[, 2]
new_data$age2 <- age_splines[, 3]
new_data$creatinine1 <- creatinine_splines[, 2]
new_data$creatinine2 <- creatinine_splines[, 3]
new_data$Hb_A1C1 <- Hb_A1C_splines[, 2]
new_data$Hb_A1C2 <- Hb_A1C_splines[, 3]
new_data$albumin1 <- albumin_splines[, 2]

# predicted risk at 1 year
pred <- predict(model, newdata = new_data, times = 365.25)[1] #0.0171758

# to plot CIF, we need to extract predcted values at multiple time points
time <- seq(1, 365 * 5, 5) # predict up to 5 years in steps of 5 days
p <- predict(model, newdata = new_data, times = time)
plot(time, p, type = "l")

