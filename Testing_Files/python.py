from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Define the paths to your models and tokenizers
models = {
    'java': {
        'model': '/home/zahoor-ai-developer/Desktop/FYP/models/best_model_distilbert_java',
        'tokenizer': '/home/zahoor-ai-developer/Desktop/FYP/tokenizers/best_model_distilbert_java_tokenizer',
        'class_names': ["summary", "Ownership", "Expand", "usage", "Pointer", "deprecation", "rational"]
    },
    'pharo': {
        'model': '/home/zahoor-ai-developer/Desktop/FYP/models/best_model_distilbert_pharo',
        'tokenizer': '/home/zahoor-ai-developer/Desktop/FYP/tokenizers/best_model_distilbert_pharo_tokenizer',
        'class_names': ["Keyimplementationpoints", "Example", "Responsibilities", "Classreferences", "Intent", "Keymessages", "Collaborators"]
    },
    'python': {
        'model': '/home/zahoor-ai-developer/Desktop/FYP/models/best_model_distilbert_python',
        'tokenizer': '/home/zahoor-ai-developer/Desktop/FYP/tokenizers/best_model_distilbert_python_tokenizer',
        'class_names': ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"]
    }
}

# Initialize CORS for local testing with React
CORS(app)

# Define the threshold for classification
threshold = 0.5

# Function to map indices to class names
def map_indices_to_classes(positive_indices, class_names):
    """
    Maps positive indices to their corresponding class names.
    
    Parameters:
    - positive_indices (list): A list of indices for positive values.
    - class_names (list): A list of class names corresponding to tensor indices.

    Returns:
    - List of class names corresponding to positive indices.
    """
    return [class_names[i] for i in positive_indices]

# Define the prediction function
def predict(text, language):
    # Load the appropriate model and tokenizer based on the language
    if language not in models:
        return {"error": "Invalid language specified."}
    
    model_info = models[language]
    model = DistilBertForSequenceClassification.from_pretrained(model_info['model'])
    tokenizer = DistilBertTokenizer.from_pretrained(model_info['tokenizer'])
    class_names = model_info['class_names']
    
    # Tokenize the input text
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    
    # Make prediction
    with torch.no_grad():
        output = model(**encoded_input)
    logits = output.logits
    probabilities = torch.sigmoid(logits)  # Apply sigmoid for multi-label classification
    predicted_indices = (probabilities > threshold).nonzero(as_tuple=True)[1]
    
    # Map the positive indices to class names
    predicted_classes = map_indices_to_classes(predicted_indices.tolist(), class_names)
    
    return predicted_classes if predicted_classes else ["No positive classes predicted."]

@app.route("/", methods=['GET'])
def home():
    return "Hello! The server is activated."

# Define the API route
@app.route('/predict', methods=['POST'])
def predict_route():
    # Get the JSON data from the POST request
    data = request.get_json()
    text = data.get('text', '')  # Assuming the text input is under 'text'
    language = data.get('language', '')  # Expect a 'language' parameter in the request (java, pharo, or python)

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if not language:
        return jsonify({'error': 'No language specified'}), 400

    # Get predictions
    predicted_classes = predict(text, language)

    # Return the prediction result as a JSON response
    return jsonify({'predicted_classes': predicted_classes})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Make sure to use port 5000 for local testing
