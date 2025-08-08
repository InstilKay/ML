import numpy as np #For numerical operations
import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # used for data visualization
import seaborn as sns # for statistical visualization
import streamlit as st #Streamlit for interactive web application
from PIL import Image # for image processing

from sklearn.preprocessing import LabelEncoder, StandardScaler ## LabelEncoder: converts categories to numbers; StandardScaler: standardizes features
from sklearn.impute import SimpleImputer # Handles missing values by imputing (e.g., with mean, median, etc.)
from sklearn.model_selection import train_test_split  # Splits dataset into training and test sets
from sklearn.linear_model import LogisticRegression  # Logistic regression model for binary classification (e.g., churn or no churn)
from sklearn.tree import DecisionTreeClassifier  # Decision tree model for classification tasks
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix, 
                           )   # Tools to evaluate classification model performance:
   # accuracy_score,                                      # Accuracy: % of total predictions that are correct
  #  precision_score,                                     # Precision: proportion of true positives among predicted positives
  #  recall_score,                                        # Recall: proportion of true positives detected among all actual positives
  #  f1_score,                                            # F1-score: harmonic mean of precision and recall
  #  classification_report,                              # Full text report with precision, recall, f1-score for each class
  #  confusion_matrix,                                   # Matrix showing actual vs predicted values (TP, FP, FN, TN)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)

# Global variables for storing data and models
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

def load_default_dataset():
    """Load the default Telco Customer Churn dataset"""
    try:
        # You can replace this path with your dataset path
        dataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        return dataset
    except FileNotFoundError:
        st.error("Default dataset not found. Please upload a dataset.")
        return None

def preprocess_data(df):
    """Comprehensive data preprocessing function"""
    df_processed = df.copy()
    
    # Handle TotalCharges column (convert to numeric)
    df_processed['TotalCharges'] = df_processed['TotalCharges'].replace(' ', np.nan)
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'])
    
    # Fill missing values in TotalCharges with median
    df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
    
    # Create binary encoding for categorical variables
    label_encoders = {}
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                          'MultipleLines', 'InternetService', 'OnlineSecurity', 
                          'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                          'StreamingTV', 'StreamingMovies', 'Contract', 
                          'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    for col in categorical_columns:   ##iteration to encoding the categorical variables to numeric values and fit it back into the dataset
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
    
    # Store label encoders in session state
    st.session_state.label_encoders = label_encoders
    
    # Standardize numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges'] #creating a dictionary for the numerical features
    scaler = StandardScaler()
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    # Store scaler in session state
    st.session_state.scaler = scaler  # Stores the scaler object in streamlits session state to reuse it across different pages or user interactions
    
    return df_processed

def get_model_features(df):
    """Get features for modeling (encoded columns + numerical)"""
    feature_columns = [col for col in df.columns if col.endswith('_encoded') and col != 'Churn_encoded']
    feature_columns.extend(['tenure', 'MonthlyCharges', 'TotalCharges'])
    return feature_columns

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate both models"""
    models = {}
    metrics = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    models['Logistic Regression'] = lr_model
    metrics['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred),
        'recall': recall_score(y_test, lr_pred),
        'f1': f1_score(y_test, lr_pred),
        'predictions': lr_pred,
        'probabilities': lr_pred_proba
    }
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10) #set split of data(random_state) max_depth=10 means the maximum number of splits (or “levels”) the tree can have from the root node to the deepest leaf node is 10.
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_pred_proba = dt_model.predict_proba(X_test)[:, 1]
    
    models['Decision Tree'] = dt_model
    metrics['Decision Tree'] = {
        'accuracy': accuracy_score(y_test, dt_pred),
        'precision': precision_score(y_test, dt_pred),
        'recall': recall_score(y_test, dt_pred),
        'f1': f1_score(y_test, dt_pred),
        'predictions': dt_pred,
        'probabilities': dt_pred_proba,
        'feature_importance': dt_model.feature_importances_
    }
    
    return models, metrics

# Page 1: Data Import and Overview
def page_data_overview():
    st.title("📊 Data Import and Overview") #streamlit title
    st.markdown("---")
    
    # Data loading section
    st.subheader("🔄 Data Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Default Dataset"):
            dataset = load_default_dataset()
            if dataset is not None:
                st.session_state.dataset = dataset
                st.success("Default dataset loaded successfully!")
    
    with col2:
        uploaded_file = st.file_uploader("Upload your own dataset", type=['csv'])
        if uploaded_file is not None:
            dataset = pd.read_csv(uploaded_file)
            st.session_state.dataset = dataset
            st.success("Custom dataset uploaded successfully!")
    
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        # Dataset overview
        st.subheader("📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(df)) #Total number of customers
        with col2:
            churn_count = df['Churn'].value_counts().get('Yes', 0)  #  How many have churned
            st.metric("Churned Customers", churn_count)
        with col3:
            churn_rate = (churn_count / len(df)) * 100  #The churn percentage
            st.metric("Churn Rate", f"{churn_rate:.1f}%")#Total number of features in your dataset
        with col4:
            st.metric("Features", len(df.columns))
        
        # Display raw data
        if st.checkbox("📋 Show Raw Dataset"):
            st.dataframe(df.head(100)) #printing first 100 rows in the dataset.
        
        # Summary statistics
        if st.checkbox("📊 Show Summary Statistics"):
            st.subheader("Numerical Features Summary")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            st.dataframe(df[numerical_cols].describe())
            
            st.subheader("Categorical Features Summary")
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                st.write(f"**{col}:**")
                st.write(df[col].value_counts())
                st.write("---")
        
        # Basic visualizations
        if st.checkbox("📈 Show Exploratory Visualizations"):
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Churn count plot
            df['Churn'].value_counts().plot(kind='bar', ax=ax[0], color=['blue', 'orange'])
            ax[0].set_title('Churn Distribution')
            ax[0].set_xlabel('Churn')
            ax[0].set_ylabel('Count')
            
            # Churn pie chart
            df['Churn'].value_counts().plot(kind='pie', ax=ax[1], autopct='%1.1f%%', colors=['blue', 'orange'])
            ax[1].set_title('Churn Percentage')
            ax[1].set_ylabel('')
            
            st.pyplot(fig)
            
            # Numerical features histograms
            st.subheader("Distribution of Numerical Features")
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            
            for col in numerical_cols:
                if col in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    df[col].hist(bins=30, ax=ax, alpha=0.7, color='teal')
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
            
            # Correlation matrix
            st.subheader("Correlation Matrix (Numerical Features)")
            # Convert TotalCharges to numeric for correlation
            df_corr = df.copy()
            df_corr['TotalCharges'] = pd.to_numeric(df_corr['TotalCharges'], errors='coerce')
            numerical_cols_clean = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            correlation_matrix = df_corr[numerical_cols_clean].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)

# Page 2: Data Preprocessing
def page_preprocessing():
    st.title("🔧 Data Preprocessing")
    st.markdown("---")
    
    if st.session_state.dataset is None:
        st.warning("Please load a dataset first from the Data Overview page.")
        return
    
    df = st.session_state.dataset
    
    st.subheader("🔍 Data Quality Check")
    
    # Missing values check
    if st.checkbox("📊 Check Missing Values"):
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.write("Missing values found:")
            st.dataframe(missing_values[missing_values > 0])
        else:
            st.success("No missing values found!")
    
    # Data types check
    if st.checkbox("🔤 Check Data Types"):
        st.dataframe(df.dtypes.to_frame('Data Type'))
    
    # Special handling for TotalCharges
    if st.checkbox("⚠️ Check TotalCharges Anomalies"):
        total_charges_issues = df[df['TotalCharges'] == ' ']
        st.write(f"Found {len(total_charges_issues)} rows with space in TotalCharges")
        if len(total_charges_issues) > 0:
            st.dataframe(total_charges_issues)
        else:
            if "balloons_shown" not in st.session_state:
                st.session_state.balloons_shown = False

            if not st.session_state.balloons_shown:
                st.balloons()
                st.session_state.balloons_shown = True

    # Preprocessing
    st.subheader("⚙️ Apply Preprocessing")

    if st.button("🚀 Process Data"):
        with st.spinner("Processing data..."):
            processed_data = preprocess_data(df)
            st.session_state.processed_data = processed_data
            st.success("Data preprocessing completed!")
    
    # Show processed data
    if st.session_state.processed_data is not None:
        processed_df = st.session_state.processed_data
        
        if st.checkbox("📋 Show Processed Data"):
            st.dataframe(processed_df.head())
        
        if st.checkbox("🔢 Show Encoded Features"):
            encoded_cols = [col for col in processed_df.columns if col.endswith('_encoded')]
            st.dataframe(processed_df[encoded_cols].head())
        
        # Preprocessing summary
        if st.checkbox("📝 Preprocessing Summary"):
            st.write("**Preprocessing Steps Applied:**")
            st.write("✅ Converted TotalCharges to numeric")
            st.write("✅ Filled missing values with median")
            st.write("✅ Applied Label Encoding to categorical variables")
            st.write("✅ Standardized numerical features")
            
            st.write(f"**Original shape:** {df.shape}")
            st.write(f"**Processed shape:** {processed_df.shape}")

# Page 3: Model Training
def page_model_training():
    st.title("🤖 Model Training")
    st.markdown("---")
    
    if st.session_state.processed_data is None:
        st.warning("Please preprocess the data first.")
        return
    
    df = st.session_state.processed_data
    
    st.subheader("🎯 Model Configuration")
    
    # Feature selection
    feature_columns = get_model_features(df)
    target_column = 'Churn_encoded'
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random State", value=40, min_value=0)
    
    # Show model parameters
    if st.checkbox("⚙️ Show Model Parameters"):
        st.write("**Logistic Regression Parameters:**")
        st.write("- Solver: liblinear")
        st.write("- Max iterations: 1000")
        st.write("- Random state:", random_state)
        
        st.write("**Decision Tree Parameters:**")
        st.write("- Max depth: 10")
        st.write("- Random state:", random_state)
        st.write("- Criterion: gini")
    
    # Train models
    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):
            # Prepare data
            X = df[feature_columns]
            y = df[target_column]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Store split data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.feature_columns = feature_columns
            
            # Train models
            models, metrics = train_models(X_train, X_test, y_train, y_test)
            
            st.session_state.models = models
            st.session_state.model_metrics = metrics
            
            st.success("Models trained successfully!")
    
    # Display training results
    if st.session_state.models:
        st.subheader("📊 Training Results")
        
        # Feature importance for Decision Tree
        if st.checkbox("🎯 Show Feature Importance (Decision Tree)"):
            dt_model = st.session_state.models['Decision Tree']
            feature_importance = pd.DataFrame({
                'feature': st.session_state.feature_columns,
                'importance': dt_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(10), y='feature', x='importance', ax=ax)
            ax.set_title('Top 10 Feature Importance (Decision Tree)')
            st.pyplot(fig)
            
            st.dataframe(feature_importance)

# Page 4: Model Evaluation
def page_model_evaluation():
    st.title("📏 Model Evaluation")
    st.markdown("---")
    
    if not st.session_state.models:
        st.warning("Please train the models first.")
        return
    
    metrics = st.session_state.model_metrics
    
    # Metrics comparison table
    st.subheader("📊 Model Performance Comparison")
    
    comparison_data = []
    for model_name, model_metrics in metrics.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{model_metrics['accuracy']:.2f}",
            'Precision': f"{model_metrics['precision']:.2f}",
            'Recall': f"{model_metrics['recall']:.2f}",
            'F1-Score': f"{model_metrics['f1']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)
    
    # Individual model metrics
    selected_model = st.selectbox("Select Model for Detailed Analysis", list(metrics.keys()))
    
    if st.checkbox("📋 Show Classification Report"):
        y_test = st.session_state.y_test
        predictions = metrics[selected_model]['predictions']
        
        st.text("Classification Report:")
        report = classification_report(y_test, predictions)
        st.text(report)
    
    # Confusion Matrix
    if st.checkbox("🔀 Show Confusion Matrix"):
        y_test = st.session_state.y_test
        predictions = metrics[selected_model]['predictions']
        
        cm = confusion_matrix(y_test, predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'], ax=ax)
        ax.set_title(f'Confusion Matrix - {selected_model}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    
    # Model comparison visualization
    if st.checkbox("📊 Show Metrics Comparison Chart"):
        metrics_data = []
        for model_name, model_metrics in metrics.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': model_metrics['accuracy'],
                'Precision': model_metrics['precision'],
                'Recall': model_metrics['recall'],
                'F1-Score': model_metrics['f1']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_melted = metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', ax=ax)
        ax.set_title('Model Performance Comparison')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Page 5: Prediction
def page_prediction():
    st.title("🔮 Customer Churn Prediction")
    st.markdown("---")
    
    if not st.session_state.models:
        st.warning("Please train the models first.")
        return
    
    st.subheader("📝 Enter Customer Information")
    
    # Model selection
    selected_model_name = st.selectbox("Select Model for Prediction", list(st.session_state.models.keys()))
    selected_model = st.session_state.models[selected_model_name]
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
        with col2:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 65.0)
            total_charges = st.slider("Total Charges", 18.0, 8500.0, 1000.0)
        
        predict_button = st.form_submit_button("🔮 Predict Churn")
    
    if predict_button:
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply same preprocessing
        label_encoders = st.session_state.label_encoders
        scaler = st.session_state.scaler
        
        # Encode categorical variables
        for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod']:
            if col in label_encoders:
                input_df[col + '_encoded'] = label_encoders[col].transform(input_df[col])
        
        # Scale numerical features
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Select features for prediction
        feature_columns = st.session_state.feature_columns
        X_input = input_df[feature_columns]
        
        # Make prediction
        prediction = selected_model.predict(X_input)[0]
        prediction_proba = selected_model.predict_proba(X_input)[0]
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("🚨 CUSTOMER WILL CHURN")
            else:
                st.snow()
                st.info("✅ CUSTOMER WILL STAY")
        
        with col2:
            churn_probability = prediction_proba[1]
            st.metric("Churn Probability", f"{churn_probability:.2%}")
        
        with col3:
            confidence = max(prediction_proba)
            st.metric("Prediction Confidence", f"{confidence:.2%}")
        
        # Probability visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        probabilities = prediction_proba
        labels = ['No Churn', 'Churn']
        colors = ['green', 'red']
        
        bars = ax.bar(labels, probabilities, color=colors, alpha=0.7)
        ax.set_ylabel('Probability')
        ax.set_title(f'Churn Prediction Probabilities ({selected_model_name})')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.2%}', ha='center', va='bottom')
        
        st.pyplot(fig)

# Page 6: Interpretation and Conclusions
def page_interpretation():
    st.title("🧠 Interpretation and Conclusions")
    st.markdown("---")
    
    if not st.session_state.models:
        st.warning("Please train the models first.")
        return
    
    st.subheader("🔍 Key Insights")
    
    # Feature importance analysis
    if st.checkbox("📊 Feature Importance Analysis"):
        if 'Decision Tree' in st.session_state.models:
            dt_model = st.session_state.models['Decision Tree']
            feature_columns = st.session_state.feature_columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': dt_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.write("**Top 10 Most Important Features for Churn Prediction:**")
            st.dataframe(importance_df.head(10))
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(data=importance_df.head(10), y='Feature', x='Importance', ax=ax)
            ax.set_title('Top 10 Feature Importances (Decision Tree)')
            st.pyplot(fig)
    
    # Model performance analysis
    if st.checkbox("⚖️ Model Performance Analysis"):
        metrics = st.session_state.model_metrics
        
        st.write("**Model Comparison Summary:**")
        
        best_accuracy = max(metrics[model]['accuracy'] for model in metrics)
        best_f1 = max(metrics[model]['f1'] for model in metrics)
        
        for model_name, model_metrics in metrics.items():
            st.write(f"**{model_name}:**")
            st.write(f"- Accuracy: {model_metrics['accuracy']:.2f}")
            st.write(f"- F1-Score: {model_metrics['f1']:.2f}")
            st.write(f"- Precision: {model_metrics['precision']:.2f}")
            st.write(f"- Recall: {model_metrics['recall']:.2f}")
            
            if model_metrics['accuracy'] == best_accuracy:
                st.write("🏆 **Best Accuracy**")
            if model_metrics['f1'] == best_f1:
                st.write("🏆 **Best F1-Score**")
            st.write("---")
    
    # Business insights
    if st.checkbox("💼 Business Insights"):
        st.write("**Key Business Insights:**")
        
        st.write("""
        **📈 Customer Retention Strategies:**
        - Focus on customers with month-to-month contracts (higher churn risk)
        - Improve technical support services (strong predictor of churn)
        - Target customers with fiber optic internet (often higher churn)
        - Monitor customers with electronic check payments
        
        **🎯 High-Risk Customer Characteristics:**
        - Short tenure customers (< 12 months)
        - High monthly charges with low total charges
        - No additional services (online security, tech support)
        - Senior citizens without family support
        
        **💡 Recommended Actions:**
        - Implement early intervention programs for new customers
        - Offer incentives for annual/two-year contracts
        - Enhance customer support quality
        - Bundle services to increase customer stickiness
        """)
    
    # Model trade-offs
    if st.checkbox("⚖️ Model Trade-offs"):
        st.write("**Model Selection Guidelines:**")
        
        st.write("""
        **Logistic Regression:**
        ✅ Pros:
        - Fast training and prediction
        - Interpretable coefficients
        - Good baseline performance
        - Less prone to overfitting
        
        ❌ Cons:
        - Assumes linear relationships
        - May miss complex patterns
        
        **Decision Tree:**
        ✅ Pros:
        - Highly interpretable rules
        - Captures non-linear relationships
        - Handles mixed data types well
        - Provides feature importance
        
        ❌ Cons:
        - Prone to overfitting
        - Can be unstable
        - May create biased trees
        """)
    
    # Recommendations
    if st.checkbox("🎯 Final Recommendations"):
        st.write("**Implementation Recommendations:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **For Model Deployment:**
            - Use ensemble of both models for better reliability
            - Regularly retrain models with new data
            - Monitor model performance over time
            - Set up automated alerts for high-risk customers
            """)
        
        with col2:
            st.write("""
            **For Business Implementation:**
            - Create customer risk scoring system
            - Develop targeted retention campaigns
            - Train customer service team on risk indicators
            - Implement proactive customer outreach program
            """)
    
    # Summary metrics dashboard
    if st.checkbox("📊 Summary Dashboard"):
        if st.session_state.dataset is not None and st.session_state.models:
            df = st.session_state.dataset
            metrics = st.session_state.model_metrics
            
            st.subheader("📈 Project Summary Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", len(df))
                
            with col2:
                churn_rate = (df['Churn'].value_counts().get('Yes', 0) / len(df)) * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            
            with col3:
                best_model = max(metrics, key=lambda x: metrics[x]['accuracy'])
                st.metric("Best Model", best_model)
            
            with col4:
                best_accuracy = max(metrics[model]['accuracy'] for model in metrics)
                st.metric("Best Accuracy", f"{best_accuracy:.1%}")
            
            # Performance comparison chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Metrics comparison
            models = list(metrics.keys())
            accuracy_scores = [metrics[model]['accuracy'] for model in models]
            f1_scores = [metrics[model]['f1'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax1.bar(x - width/2, accuracy_scores, width, label='Accuracy', alpha=0.8)
            ax1.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Score')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models)
            ax1.legend()
            ax1.set_ylim(0, 1)
            
            # Churn distribution
            churn_counts = df['Churn'].value_counts()
            ax2.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                   colors=['lightblue', 'lightcoral'])
            ax2.set_title('Overall Churn Distribution')
            
            st.pyplot(fig)

# Main navigation
def main():
    st.sidebar.title("🚀 Navigation")
    st.sidebar.markdown("---")
    
    pages = {
        "📊 Data Overview": page_data_overview,
        "🔧 Data Preprocessing": page_preprocessing,
        "🤖 Model Training": page_model_training,
        "📏 Model Evaluation": page_model_evaluation,
        "🔮 Prediction": page_prediction,
        "🧠 Interpretation": page_interpretation
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
    
    # Add some sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Project Info")
    st.sidebar.info("""
    **Customer Churn Prediction**
    
    This application uses machine learning to predict customer churn based on various customer attributes and service usage patterns.
    
    **Models Used:**
    - Logistic Regression
    - Decision Tree Classifier
    
    **Key Features:**
    - Interactive data exploration
    - Model comparison
    - Real-time predictions
    - Business insights
    """)
    
    # Dataset status
    if st.session_state.dataset is not None:
        st.sidebar.success("✅ Dataset Loaded")
    else:
        st.sidebar.warning("⏳ No Dataset Loaded")
    
    if st.session_state.processed_data is not None:
        st.sidebar.success("✅ Data Preprocessed")
    else:
        st.sidebar.warning("⏳ Data Not Preprocessed")
    
    if st.session_state.models:
        st.sidebar.success("✅ Models Trained")
    else:
        st.sidebar.warning("⏳ Models Not Trained")
    
    # Run selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()
