# src/app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import traceback
from src.model_utils import prediction_mapping, generate_improvement_text

app = Flask(__name__)

# Load saved pipeline (scaler + model)
pipeline = joblib.load('voting_pipeline.joblib')  # ensure this file is next to project root or adjust path
scaler = pipeline['scaler']
model = pipeline['model']
print('Model loaded successfully:', type(model))


@app.route('/')
def index():
    return render_template('index.html')  # expects an index.html with form fields: ph, do, temp, nh3, no2, no3


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # parse and validate inputs
        ph = float(request.form.get('ph', '').strip())
        do = float(request.form.get('do', '').strip())
        temp = float(request.form.get('temp', '').strip())
        nh3 = float(request.form.get('nh3', '').strip())
        no2 = float(request.form.get('no2', '').strip())
        no3 = float(request.form.get('no3', '').strip())

        # build DataFrame
        input_df = pd.DataFrame(
            [[ph, do, temp, nh3, no2, no3]],
            columns=['pH', 'Dissolved Oxygen', 'Temperature', 'Ammonia', 'Nitrite', 'Nitrate']
        )

        # scale and predict
        X_scaled = scaler.transform(input_df)
        prediction_code = model.predict(X_scaled)[0]
        result = prediction_mapping.get(prediction_code, 'Unknown Result')

        # improvement suggestions
        improvement = generate_improvement_text(result)

        return render_template(
            'index.html',
            prediction_text=f'Suitable for: {result}',
            improvement_text=improvement
        )

    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return render_template(
            'index.html',
            prediction_text='Error occurred during prediction!',
            improvement_text=""
        )


if __name__ == '__main__':
    app.run(debug=True)
