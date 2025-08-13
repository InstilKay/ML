import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf  # Main TF import
import streamlit as st
import requests
import os

# Then use tf.keras throughout your code
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------
# 1. Data Loading and Preprocessing
# --------------------------

def load_dailydialog_data():
    """Load and preprocess DailyDialog dataset"""
    try:
        # Load dataset from Hugging Face
        from datasets import load_dataset
        dataset = load_dataset('ConvLab/dailydialog')
        
        # Extract dialogues and emotions
        dialogues = []
        emotions = []
        for split in ['train', 'validation', 'test']:
            for item in dataset[split]:
                for i, utterance in enumerate(item['dialog']):
                    dialogues.append(utterance)
                    emotions.append(item['emotion'][i])
        
        return dialogues, emotions
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def preprocess_text(text):
    """Clean and tokenize text"""
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --------------------------
# 2. GloVe Embeddings
# --------------------------

def download_glove():
    """Download GloVe embeddings if not present"""
    #glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    glove_path = "glove.6B.100c.txt"
    
    if not os.path.exists(glove_path):
        print("Downloading GloVe embeddings...")
        try:
            import zipfile
            from io import BytesIO
            
            response = requests.get(glove_url, stream=True)
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                with z.open("glove.6B/glove.6B.100d.txt") as f:
                    with open(glove_path, 'wb') as out:
                        out.write(f.read())
            print("GloVe embeddings downloaded successfully.")
        except Exception as e:
            print(f"Error downloading GloVe: {e}")
            return None
    
    return glove_path

def load_glove(glove_path):
    """Load GloVe embeddings from file"""
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def text_to_glove(text, glove_embeddings, dim=100):
    """Convert text to average GloVe vector"""
    words = text.split()
    embedding = np.zeros(dim)
    count = 0
    for word in words:
        if word in glove_embeddings:
            embedding += glove_embeddings[word]
            count += 1
    return embedding / max(count, 1)  # Avoid division by zero

# --------------------------
# 3. Model Training
# --------------------------

def train_logistic_regression(X_train, y_train):
    """Train logistic regression model"""
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced'))
    model.fit(X_train, y_train)
    return model

def train_ann(X_train, y_train, input_dim, output_dim):
    """Train artificial neural network"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

# --------------------------
# 4. Evaluation
# --------------------------

def evaluate_model(model, X_test, y_test, model_type='lr'):
    """Evaluate model performance"""
    if model_type == 'lr':
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    else:  # ANN
        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    try:
        roc_auc = roc_auc_score(y_test, y_prob, average='macro')
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")

# --------------------------
# 5. Main Execution
# --------------------------

def main():
    # Load and preprocess data
    print("Loading data...")
    dialogues, emotions = load_dailydialog_data()
    
    if dialogues is None:
        print("Failed to load data. Exiting.")
        return
    
    # Preprocess text
    print("Preprocessing text...")
    processed_texts = [preprocess_text(text) for text in dialogues]
    
    # Convert emotions to multi-label format
    print("Preparing labels...")
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([[str(e)] for e in emotions])  # Convert to list of lists
    
    # Download and load GloVe
    print("Loading GloVe embeddings...")
    glove_path = download_glove()
    if not glove_path:
        return
    glove = load_glove(glove_path)
    
    # Convert text to GloVe vectors
    print("Creating feature vectors...")
    X = np.array([text_to_glove(text, glove) for text in processed_texts])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    print("Logistic Regression Evaluation:")
    evaluate_model(lr_model, X_test, y_test, 'lr')
    
    # Train ANN
    print("\nTraining Neural Network...")
    ann_model = train_ann(X_train, y_train, X_train.shape[1], y_train.shape[1])
    print("Neural Network Evaluation:")
    evaluate_model(ann_model, X_test, y_test, 'ann')
    
    # Save models and label binarizer
    import joblib
    joblib.dump(lr_model, 'lr_model.pkl')
    ann_model.save('ann_model.h5')
    joblib.dump(mlb, 'label_binarizer.pkl')
    np.save('glove_embeddings.npy', X[0])  # Save sample for dimensionality
    
    print("\nTraining complete. Models saved.")

# --------------------------
# 6. Streamlit Deployment
# --------------------------

def run_streamlit_app():
    """Run the Streamlit web app"""
    st.title("Multi-Label Emotion Classification in News Headlines")
    st.write("This app detects emotions in short text using GloVe embeddings and machine learning.")
    
    # Load models and resources
    try:
        lr_model = joblib.load('lr_model.pkl')
        ann_model = tf.keras.models.load_model('ann_model.h5')
        mlb = joblib.load('label_binarizer.pkl')
        glove = load_glove('glove.6B.100d.txt')
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return
    
    # User input
    user_input = st.text_area("Enter a headline or short text:", "The stock market reaches record highs today!")
    
    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter some text.")
            return
        
        # Preprocess and vectorize
        processed_text = preprocess_text(user_input)
        vector = text_to_glove(processed_text, glove)
        
        # Make predictions
        lr_pred = lr_model.predict([vector])
        lr_prob = lr_model.predict_proba([vector])
        
        ann_prob = ann_model.predict(np.array([vector]))
        ann_pred = (ann_prob > 0.5).astype(int)
        
        # Get emotion labels
        emotions = mlb.classes_
        lr_emotions = [emotions[i] for i, val in enumerate(lr_pred[0]) if val == 1]
        ann_emotions = [emotions[i] for i, val in enumerate(ann_pred[0]) if val == 1]
        
        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Logistic Regression**")
            if lr_emotions:
                st.write("Detected emotions:", ", ".join(lr_emotions))
            else:
                st.write("No strong emotions detected")
            st.write("Confidence scores:")
            for i, prob in enumerate(lr_prob[0]):
                st.write(f"{emotions[i]}: {prob[1]:.2f}")
        
        with col2:
            st.markdown("**Neural Network**")
            if ann_emotions:
                st.write("Detected emotions:", ", ".join(ann_emotions))
            else:
                st.write("No strong emotions detected")
            st.write("Confidence scores:")
            for i, prob in enumerate(ann_prob[0]):
                st.write(f"{emotions[i]}: {prob:.2f}")

if __name__ == '__main__':
    # Train models first (comment out after first run)
    # main()
    
    # Run Streamlit app
    import joblib
    run_streamlit_app()
