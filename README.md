# Gurgaon-Real-Estate-Price-Prediction-Recommender-System-
# Gurgaon Real Estate Analytics: Price Prediction & Recommender System

This repository documents the end-to-end data science capstone project. This project focuses on the challenging and dynamic real estate market of Gurgaon, India, intending to build a robust price prediction model and a property recommender system.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Workflow & Methodology](#project-workflow--methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Missing Value Imputation](#2-missing-value-imputation)
  - [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
  - [4. Outlier Treatment](#4-outlier-treatment)
  - [5. Feature Engineering](#5-feature-engineering)
  - [6. Feature Selection](#6-feature-selection)
  - [7. Model Building & Selection](#7-model-building--selection)
  - [8. Recommender System](#8-recommender-system)
- [Repository Structure](#repository-structure)
- [Tools & Libraries](#tools--libraries)
- [How to Use](#how-to-use)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

The primary goal of this project is twofold:

1.  **Predict Property Prices:** To build and evaluate a high-performing machine learning model that accurately predicts the sale price of a property based on its features (e.g., area, location, bedrooms, amenities).
2.  **Build a Recommender System:** To develop a content-based filtering system that suggests similar properties to a user based on their selected property, enhancing user experience on a potential real-estate platform.

The project follows the complete data science lifecycle, from raw data collection and cleaning to model building and application.

## Project Workflow & Methodology

The entire project is broken down into a series of logical steps, with each major step encapsulated in its own Jupyter Notebook for clarity and reproducibility.

### 1. Data Preprocessing
*(Notebooks: `data-preprocessing-flats.ipynb`, `data-preprocessing-houses.ipynb`)*

The initial raw dataset (`gurgaon_properties.csv`) was messy and combined different property types.
* **Separation:** The data was split into two more manageable dataframes: `flats.csv` and `houses.csv`.
* **Cleaning:** Each dataset underwent extensive cleaning, including handling structural errors, parsing numerical values from complex text strings (e.g., "1, 2, 3 BHK" -> 3), and standardizing units and categorical values.

### 2. Missing Value Imputation
*(Notebook: `missing-value-imputation.ipynb`)*

* Addressed missing data points using a combination of techniques, including median/mode imputation and logical inference based on other features to ensure the dataset's integrity.

### 3. Exploratory Data Analysis (EDA)
This crucial phase was divided into multiple parts to extract deep insights.
* **Pandas Profiling (`eda-pandas-profiling.ipynb`):** Generated an initial automated report to get a high-level overview of all variables, their distributions, and potential issues like high cardinality or skewness.
* **Univariate Analysis (`eda-univariate-analysis.ipynb`):** Analyzed each feature individually to understand its distribution, identify central tendencies, and spot anomalies.
* **Multivariate Analysis (`eda-multivariate-analysis.ipynb`):** Explored the relationships *between* features, focusing on correlations with the target variable (price) using heatmaps and scatter plots.

### 4. Outlier Treatment
*(Notebook: `outlier-treatment.ipynb`)*

* Identified and treated extreme outliers in key numerical columns (like `price`, `built_up_area`) using statistical methods (e.g., Interquartile Range, Z-score) and domain knowledge to prevent them from skewing the model's performance.

### 5. Feature Engineering
*(Notebook: `feature-engineering.ipynb`)*

* Created new, impactful features from the existing data to improve model accuracy. This included:
    * `price_per_sqft`: A critical metric in real estate.
    * `room_to_bathroom_ratio`: A potential indicator of luxury.
    * `property_age`: Extracted from the `availability` status.
    * Binning categorical features and encoding ordinal variables.

### 6. Feature Selection
*(Notebook: `feature-selection.ipynb`)*

* Reduced the dimensionality of the dataset to combat the "curse of dimensionality" and improve model efficiency.
* Applied techniques like correlation analysis, VIF (Variance Inflation Factor) to remove multicollinearity, and feature importance (from tree-based models) to select only the most predictive features.

### 7. Model Building & Selection
*(Notebooks: `baseline model.ipynb`, `model-selection.ipynb`)*

* **Baseline Model:** Established a simple model (e.g., Linear Regression) to set a performance benchmark.
* **Model Selection:** Systematically trained, tested, and evaluated a wide range of machine learning algorithms, including:
    * Linear Regression (with Regularisation: Ridge, Lasso)
    * Decision Trees
    * Random Forest
    * Gradient Boosting (XGBoost, LightGBM)
* **Hyperparameter Tuning:** Used `GridSearchCV` to find the optimal set of parameters for the best-performing model (XGBoost).

### 8. Recommender System
*(Notebook: `recommender-system.ipynb`)*

* Implemented a **content-based recommender system**.
* Utilized **TF-IDF Vectorization** on key property features (like `amenities`, `sector`, `property_type`) to create a feature matrix.
* Calculated **Cosine Similarity** between properties to find and recommend the most similar items based on their content.

## Repository Structure
## Tools & Libraries

* **Data Analysis & Manipulation:** `Pandas`, `NumPy`
* **Data Visualization:** `Matplotlib`, `Seaborn`, `Plotly`
* **Machine Learning:** `Scikit-learn` (for preprocessing, modeling, and evaluation)
* **Advanced Modeling:** `XGBoost`, `LightGBM`
* **NLP (Recommender):** `Scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
* **Utilities:** `Jupyter Notebook`, `Pandas-Profiling`

## How to Use

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ADsaint13/Gurgaon-Real-Estate-Price-Prediction-Recommender-System-.git](https://github.com/ADsaint13/Gurgaon-Real-Estate-Price-Prediction-Recommender-System-.git)
    cd Gurgaon-Real-Estate-Price-Prediction-Recommender-System-
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    *(It's recommended to create a `requirements.txt` file)*
    ```bash
    pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm pandas-profiling
    ```

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

5.  Open the notebooks in numerical order to follow the project workflow.

## Acknowledgements
* This project is based on the capstone for the [CampusX Data Science Mentorship Program](https://campusx.in/).
