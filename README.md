# House Price Prediction System

A beginner-level Flask website that uses a Random Forest regression model to estimate house prices.

## Important dataset note
`house_data.csv` contains 600 **synthetic, illustrative records** generated for this college demonstration. It is not collected from property transactions and must not be used for real buying, selling, lending, or valuation decisions. The model learns the artificial price-generation pattern in this sample.

## Requirements
- Python 3.10 or newer
- pip

## Run the project
1. Extract this ZIP file and open the project folder in VS Code.
2. Open Terminal in this folder.
3. Create a virtual environment (optional but recommended):
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install packages: `pip install -r requirements.txt`
5. Train the model: `python train_model.py`
6. Start the website: `python app.py`
7. Open `http://127.0.0.1:5000` in your browser.

The ZIP includes a pre-trained model, so you can also run `python app.py` directly after installing dependencies. Retrain it with step 5 if needed.

## Project files
- `app.py`: Flask backend and prediction route
- `train_model.py`: trains and evaluates the ML model
- `generate_dataset.py`: regenerates the illustrative dataset
- `house_data.csv`: dataset
- `house_price_model.pkl`: saved model
- `templates/index.html`: website page
- `static/style.css`: styling
- `PROJECT_REPORT.pdf`: college mini-project report

## Model
RandomForestRegressor with OneHotEncoder for location. The evaluation metrics printed by `train_model.py` are based on a held-out 20% test split of the synthetic dataset. They are educational metrics, not evidence of real-world accuracy.
