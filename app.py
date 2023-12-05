from flask import Flask, render_template, request, jsonify
from ml_logic import predict_with_ml_model, train_model

app = Flask(__name__)

# Sample data
X_train = ["This was really awesome an awesome movie",
           "Great movie! I liked it a lot",
           "Happy Ending! Awesome Acting by hero",
           "loved it!",
           "Bad not up to the mark",
           "Could have been better",
           "really Disappointed by the movie"]

y_train = ["positive", "positive", "positive", "positive", "negative", "negative", "negative"]

# Train the model
model, cv = train_model(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.json['data']

    # Perform ML prediction
    prediction_result = predict_with_ml_model(model, cv, input_data)

    # Return the prediction result as JSON
    return jsonify({'result': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
