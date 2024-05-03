import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

users = {'Saksham06': 'Saksham@0606'}

# Global variable to track login status
logged_in = False

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
@app.route('/', methods=['GET', 'POST'])
def home():
    global logged_in
    if request.method == 'POST':
        action = request.form['action']
        if action == 'login':
            return render_template('login.html')
        elif action == 'register':
            return render_template('register.html')
    return render_template('home.html')

@app.route('/register', methods=['POST'])
def register():
    global logged_in
    # Get form data
    username = request.form['username']
    password = request.form['password']
    # Check if username already exists
    if username in users:
        return "Username already exists. Please choose a different one."
    else:
        # Add user to the database
        users[username] = password
        logged_in = True
        return render_template('first.html')

@app.route('/login', methods=['POST'])
def login():
    global logged_in
    # Get form data
    username = request.form['username']
    password = request.form['password']
    # Check if username and password match
    if username in users and users[username] == password:
        logged_in = True
        return render_template('first.html')  # Render the main carousel page after successful login
    else:
        return "Invalid username or password. Please try again."

@app.route('/first')
def first():
    # Render the main carousel page
    return render_template('first.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = float(request.form.get('total_sqft'))

        input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = model.predict(input_data)[0]*100000

        return str(np.round(prediction, 2))

    # For GET request, render index.html
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/filter', methods=['GET', 'POST'])
def filter_data():
    if request.method == 'POST':
        try:
            target_price = float(request.form.get('target_price'))

            # Filter data based on target price
            print(66, filterdata['TARGET_PRICE_IN_LACS'])
            # filtered_data = filterdata[filterdata['TARGET_PRICE_IN_LACS'] <= target_price]
            filtered_data = filterdata.query(f'TARGET_PRICE_IN_LACS <= {target_price}')
            # print(filterdata.columns)
            return render_template('filter.html', filtered_data=filtered_data.to_dict(orient='records'))
        except Exception as e:
            print('Error occured, see this: ', str(e))

    return render_template('filter.html')



# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
