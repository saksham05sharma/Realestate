import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('Cleaned_data.csv')
filterdata = pd.read_csv('Train.csv')

# Define preprocessing steps
numeric_features = ['total_sqft', 'bath', 'bhk']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['location']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Train the model
X = data.drop('price', axis=1)  # Adjust target_column with your target variable name
y = data['price']
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', Ridge(alpha=1.0))])
model.fit(X, y)

# Define routes
@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(input_data)[0]*100000

    return str(np.round(prediction,2))

@app.route('/filter', methods=['GET', 'POST'])
def filter_data():
    if request.method == 'POST':
        target_price = float(request.form.get('target_price'))

        # Filter data based on target price
        filtered_data = filterdata[filterdata['TARGET(PRICE_IN_LACS)'] <= target_price]

        return render_template('filter.html', filtered_data=filtered_data.to_dict(orient='records'))

    return render_template('filter.html')



# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
