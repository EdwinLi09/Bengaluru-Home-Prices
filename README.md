# Predicting Home Prices in Bengaluru

A machine learning project that predicts residential property prices in Bengaluru, India, using a Linear Regression model trained on real-estate listings from Kaggle. The notebook walks through the full ML workflow — from raw, messy data to a working prediction function — with a strong emphasis on practical data cleaning and outlier handling.

## Dataset

The data comes from the [Bengaluru House Price Dataset](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data?resource=download) on Kaggle. It contains ~13,000 property listings with features such as area type, location, size (BHK), total square footage, number of bathrooms, balconies, and price (in lakhs).

For modeling, only the following features are retained:

- `location` — neighborhood within Bengaluru
- `total_sqft` — total area in square feet
- `bath` — number of bathrooms
- `bhk` — engineered from `size`, represents Bedrooms / Hall / Kitchen count
- `price` — target variable, in lakhs (₹)

## Tech Stack

- **Python 3**
- **pandas**, **NumPy** — data manipulation
- **Matplotlib** — visualization
- **scikit-learn** — modeling, cross-validation, and hyperparameter search

## Project Workflow

### 1. Data Cleaning
- Dropped non-essential columns (`area_type`, `society`, `balcony`, `availability`).
- Removed rows with missing values.

### 2. Feature Engineering
- **BHK extraction:** Parsed the `size` column (e.g., `"4 Bedroom"`) into an integer `bhk` feature.
- **Square footage normalization:** Converted range-style entries like `"2100 - 2850"` into their midpoint average. Non-numeric entries (e.g., values given in square meters) were dropped to keep the pipeline simple.
- **Price per square foot:** Added a `price_per_sqft` feature, used heavily for outlier detection.

### 3. Dimensionality Reduction
The dataset contained over 1,300 unique locations — too many to one-hot encode cleanly. Any location with **10 or fewer listings** was relabeled as `"other"`, drastically reducing the number of dummy columns while preserving signal for popular neighborhoods.

### 4. Outlier Removal
Several domain-driven outlier checks were applied:

- **Square-foot-per-bedroom rule:** Removed rows where `total_sqft / bhk < 300`, since a typical bedroom needs at least 300 sqft (e.g., a 2-BHK with only 400 sqft is almost certainly a data error).
- **Price-per-sqft outliers:** For each location, removed listings whose `price_per_sqft` fell outside one standard deviation from the location's mean.
- **BHK consistency:** For each location, removed listings where a higher-BHK property was priced *below* the mean price-per-sqft of lower-BHK properties at the same location.
- **Bathroom sanity check:** Removed listings with more than `bhk + 2` bathrooms.

### 5. Encoding
- Applied **one-hot encoding** to the `location` column.
- Dropped the `"other"` dummy to avoid the dummy-variable trap.

### 6. Model Building
- Performed an 80/20 train-test split.
- Trained a **Linear Regression** model, achieving an R² of ~0.86 on the test set.
- Validated with **5-fold ShuffleSplit cross-validation**, consistently scoring above 80%.

### 7. Model Selection via GridSearchCV
Compared three algorithms with hyperparameter tuning:

| Model              | Hyperparameters Tuned                              |
| ------------------ | -------------------------------------------------- |
| Linear Regression  | `fit_intercept`, `positive` (with `StandardScaler`)|
| Lasso              | `alpha`, `selection`                               |
| Decision Tree      | `criterion`, `splitter`                            |

**Linear Regression** produced the best cross-validation score and was chosen as the final model.

## Making a Prediction

The notebook defines a `predict_price` helper that takes a location, square footage, bath count, and BHK count and returns the estimated price (in lakhs):

```python
predict_price('Indira Nagar', 1000, 3, 3)
# → estimated price in lakhs (₹)
```

Internally, it builds a feature vector of zeros, sets the numeric features, flips the appropriate location dummy to 1, and runs it through the trained model.

## Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Running the Notebook
1. Download the dataset from the Kaggle link above and place `Bengaluru_House_data.csv` in a `Datasets/` folder one level above the notebook (or update the path in the load cell).
2. Launch Jupyter:
   ```bash
   jupyter notebook Bengaluru_House_Prices_Final.ipynb
   ```
3. Run the cells in order.

## Repository Structure

```
.
├── Bengaluru_House_Prices_Final.ipynb   # Main notebook
├── Datasets/
│   └── Bengaluru_House_data.csv         # Raw dataset (download from Kaggle)
└── README.md
```

## Key Takeaways

- Domain-driven outlier removal (sqft-per-bedroom, bathroom limits, intra-location price sanity) had a much larger impact on model quality than algorithm choice.
- High-cardinality categorical features benefit enormously from sensible bucketing before one-hot encoding.
- For this dataset, a well-cleaned Linear Regression beat both Lasso and a Decision Tree under GridSearchCV.

## License

Add your preferred license here (e.g., MIT).
