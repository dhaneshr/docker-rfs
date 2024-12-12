
options(repos = c(
  CRAN = "https://packagemanager.rstudio.com/all/latest",
  CRAN = "https://cloud.r-project.org"
))

install.packages(
  c("rms", "riskRegression", "prodlim"),  
  dependencies = TRUE, 
  lib = "/usr/local/lib/R/site-library"
)

# If the above fails, try installing specific versions of the packages 
# using remotes
if (!requireNamespace("rms", quietly = TRUE)) {
  install.packages("remotes")
  remotes::install_version("rms", version = "6.8-2", dependencies = TRUE)
}

installed.packages()  # Print list of installed packages for logging
