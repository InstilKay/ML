import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


#Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# Show image only on main page
if __name__ == "__main__":  # or use a custom flag
    image = Image.open("mlg2.jpeg")
    st.image(image, caption="Group 4", width=500)



st.write("Bernice Baadawo Abbe- 22253447")
st.write("Frederica Atsupi Nkegbe -22253148")
st.write("Instil Paakwesi Appau -22252453")
st.write("Erwin K. Opare-Essel -22254064")
st.write("Anita Dickson -22253364")


# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "Data Import and Overview",
    "Data Preprocessing", 
    "Model Training",
    "Model Evaluation",
    "Prediction Page",
    "Interpretation and Conclusions"
])

# Helper functions
def load_sample_data():
    """Load sample telco customer churn data"""
    try:
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

def preprocess_data(dataset):
    """Preprocess the data for modeling"""
    if dataset is None:
        return None
        
    dataset_processed = dataset.copy()
    
    # Handle TotalCharges column (convert to numeric)
    if 'TotalCharges' in dataset_processed.columns:
        dataset_processed['TotalCharges'] = pd.to_numeric(dataset_processed['TotalCharges'], errors='coerce')
        dataset_processed['TotalCharges'].fillna(dataset_processed['TotalCharges'].median(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                          'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    for col in categorical_columns:
        if col in dataset_processed.columns:
            dataset_processed[col] = le.fit_transform(dataset_processed[col].astype(str))
    
    return dataset_processed

# PAGE 1: Data Import and Overview
if page == "Data Import and Overview":
    st.title("Customer Churn Prediction - Data Overview")
    
    # File upload section
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], 
                                   help="Upload your customer churn data in CSV format")
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Sample data section
    if st.session_state.data is None:
        st.subheader("Try with Sample Data")
        if st.button("Load Sample Telco Dataset"):
            with st.spinner("Loading sample data..."):
                st.session_state.data = load_sample_data()
                if st.session_state.data is not None:
                    st.success("Sample data loaded successfully!")
    
    if st.session_state.data is not None:
        dataset = st.session_state.data
        
        # Basic dataset info
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(dataset))
        with col2:
            st.metric("Features", len(dataset.columns)-1)
        with col3:
            if 'Churn' in dataset.columns:
                if dataset['Churn'].dtype == 'object':
                    churned = dataset['Churn'].value_counts().get('Yes', 0)
                else:
                    churned = dataset['Churn'].sum()
                st.metric("Churned Customers", churned)
            else:
                st.warning("No 'Churn' column found")
        with col4:
            if 'Churn' in dataset.columns:
                churn_rate = (churned / len(dataset)) * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(dataset.head())
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(dataset.describe(include='all'))
        
        # Visualizations
        st.subheader("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn distribution
            if 'Churn' in dataset.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                if dataset['Churn'].dtype == 'object':
                    churn_counts = dataset['Churn'].value_counts()
                    labels = churn_counts.index
                else:
                    churn_counts = dataset['Churn'].value_counts()
                    labels = ['No' if x == 0 else 'Yes' for x in churn_counts.index]
                ax.bar(labels, churn_counts.values, color=['skyblue', 'salmon'])
                ax.set_title('Churn Distribution')
                ax.set_xlabel('Churn')
                ax.set_ylabel('Count')
                st.pyplot(fig)
        
        with col2:
            # Tenure histogram
            if 'tenure' in dataset.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(dataset['tenure'], bins=30, color='lightgreen', alpha=0.7)
                ax.set_title('Customer Tenure Distribution')
                ax.set_xlabel('Tenure (months)')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        
        # Correlation matrix for numerical features
        st.subheader("Correlation Matrix")
        numerical_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = dataset[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix of Numerical Features')
            st.pyplot(fig)

# PAGE 2: Data Preprocessing
elif page == "Data Preprocessing":
    st.title("🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload or load dataset in the 'Data Import and Overview' page.")
    else:
        dataset = st.session_state.data
        
        st.subheader("Raw Data")
        st.write(f"Shape: {dataset.shape}")
        st.dataframe(dataset.head())
        
        # Check for missing values
        st.subheader("Missing Values Analysis")
        missing_values = dataset.isnull().sum()
        if missing_values.sum() > 0:
            st.write("Missing values found:")
            st.dataframe(missing_values[missing_values > 0])
        else:
            st.success("No missing values!")
        
        # Data type information
        st.subheader("Data Types")
        st.write(dataset.dtypes)
        
        # Preprocessing button
        if st.button("Preprocess Data"):
            with st.spinner("Processing data..."):
                st.session_state.processed_data = preprocess_data(dataset)
                if st.session_state.processed_data is not None:
                    st.success("Data preprocessing completed!")
                else:
                    st.error("Preprocessing failed. Check your data.")
        
        # Show processed data
        if st.session_state.processed_data is not None:
            processed_df = st.session_state.processed_data
            
            st.subheader("Processed Data")
            st.write(f"Shape: {processed_df.shape}")
            st.dataframe(processed_df.head())
            
            st.subheader("Preprocessing Steps Applied:")
            st.write("- Converted TotalCharges to numeric (if present)")
            st.write("- Filled missing values with median")
            st.write("- Applied Label Encoding to categorical variables")
            st.write("✅ Data is ready for modeling")

# PAGE 3: Model Training
elif page == "Model Training":
    st.title("🤖 Model Training")
    
    if st.session_state.processed_data is None:
        st.warning("Please preprocess the data in the 'Data Preprocessing' page.")
    elif 'Churn' not in st.session_state.processed_data.columns:
        st.error("Processed data doesn't contain 'Churn' column. Please check your data.")
    else:
        dataset = st.session_state.processed_data
        # Prepare features and target
        X = dataset.drop(['Churn'], axis=1, errors='ignore')
        # Drop non-feature columns
        X = X.select_dtypes(include=[np.number])
        y = dataset['Churn']
        
        # Check if we have features to work with
        if X.shape[1] == 0:
            st.error("No numerical features found for modeling. Please check your data preprocessing.")
        else:
            # Train-test split
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random State", value=42, min_value=0)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            st.write(f"Training set size: {X_train.shape[0]}")
            st.write(f"Test set size: {X_test.shape[0]}")
            st.write(f"Number of features: {X_train.shape[1]}")
            
            # Model training
            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    try:
                        # Logistic Regression
                        lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
                        lr_model.fit(X_train_scaled, y_train)
                        
                        # Decision Tree
                        dt_model = DecisionTreeClassifier(random_state=random_state)
                        dt_model.fit(X_train, y_train)
                        
                        # Store models and data
                        st.session_state.models = {
                            'logistic_regression': lr_model,
                            'decision_tree': dt_model,
                            'scaler': scaler,
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_train': y_train,
                            'y_test': y_test,
                            'X_train_scaled': X_train_scaled,
                            'X_test_scaled': X_test_scaled,
                            'feature_names': X.columns.tolist()
                        }
                        
                        st.success("Models trained successfully!")
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")
            
            # Display model information
            if st.session_state.models:
                st.subheader("Trained Models")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Logistic Regression**")
                    lr_model = st.session_state.models['logistic_regression']
                    st.write(f"Intercept: {lr_model.intercept_[0]:.4f}")
                    st.write(f"Number of coefficients: {len(lr_model.coef_[0])}")
                    st.write(f"Features: {', '.join(st.session_state.models['feature_names'][:3])}...")
                
                with col2:
                    st.write("**Decision Tree Classifier**")
                    dt_model = st.session_state.models['decision_tree']
                    st.write(f"Max Depth: {dt_model.get_depth()}")
                    st.write(f"Number of leaves: {dt_model.get_n_leaves()}")
                    st.write(f"Features: {', '.join(st.session_state.models['feature_names'][:3])}...")

# PAGE 4: Model Evaluation
elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")
    
    if not st.session_state.models:
        st.warning("Please train models in the 'Model Training' page.")
    else:
        models = st.session_state.models
        
        # Make predictions
        try:
            lr_pred = models['logistic_regression'].predict(models['X_test_scaled'])
            dt_pred = models['decision_tree'].predict(models['X_test'])
            
            # Calculate metrics
            def calculate_metrics(y_true, y_pred, model_name):
                return {
                    'Model': model_name,
                    'Accuracy': accuracy_score(y_true, y_pred),
                    'Precision': precision_score(y_true, y_pred, zero_division=0),
                    'Recall': recall_score(y_true, y_pred, zero_division=0),
                    'F1-Score': f1_score(y_true, y_pred, zero_division=0)
                }
            
            lr_metrics = calculate_metrics(models['y_test'], lr_pred, 'Logistic Regression')
            dt_metrics = calculate_metrics(models['y_test'], dt_pred, 'Decision Tree')
            
            # Comparing Model Performance
            st.subheader("Model Performance Comparison")
            metrics_df = pd.DataFrame([lr_metrics, dt_metrics])
            st.dataframe(metrics_df.set_index('Model').style.format("{:.3f}"))
            
            # Visualize metrics
            fig, ax = plt.subplots(figsize=(12, 6))
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            x = np.arange(len(metrics_to_plot))
            width = 0.35
            
            lr_values = [lr_metrics[metric] for metric in metrics_to_plot]
            dt_values = [dt_metrics[metric] for metric in metrics_to_plot]
            
            ax.bar(x - width/2, lr_values, width, label='Logistic Regression', alpha=0.8)
            ax.bar(x + width/2, dt_values, width, label='Decision Tree', alpha=0.8)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_to_plot)
            ax.legend()
            ax.set_ylim(0, 1)
            
            for i, v in enumerate(lr_values):
                ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            for i, v in enumerate(dt_values):
                ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Logistic Regression**")
                lr_cm = confusion_matrix(models['y_test'], lr_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['No Churn', 'Churn'],
                            yticklabels=['No Churn', 'Churn'])
                ax.set_title('Logistic Regression Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            with col2:
                st.write("**Decision Tree**")
                dt_cm = confusion_matrix(models['y_test'], dt_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                            xticklabels=['No Churn', 'Churn'],
                            yticklabels=['No Churn', 'Churn'])
                ax.set_title('Decision Tree Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            # Store results
            st.session_state.model_results = {
                'lr_metrics': lr_metrics,
                'dt_metrics': dt_metrics,
                'lr_pred': lr_pred,
                'dt_pred': dt_pred,
                'feature_names': models['feature_names']
            }
            
        except Exception as e:
            st.error(f"Error evaluating models: {str(e)}")

# PAGE 5: Prediction Page
elif page == "Prediction Page":
    st.title("🔮 Churn Prediction")
    
    if not st.session_state.models:
        st.warning("Please train models in the 'Model Training' page.")
    else:
        models = st.session_state.models
        feature_names = models['feature_names']
        
        st.subheader("Enter Customer Data")
        
        # Create input form based on available features
        input_data = {}
        cols_per_row = 2
        features_per_col = len(feature_names) // cols_per_row + 1
        
        columns = st.columns(cols_per_row)
        
        for i, feature in enumerate(feature_names):
            with columns[i % cols_per_row]:
                if feature == 'tenure':
                    input_data[feature] = st.slider(feature, 0, 100, 12)
                elif feature == 'MonthlyCharges':
                    input_data[feature] = st.number_input(feature, 0.0, 200.0, 50.0)
                elif feature == 'TotalCharges':
                    input_data[feature] = st.number_input(feature, 0.0, 10000.0, 1000.0)
                else:
                    # For categorical features that were encoded
                    unique_values = st.session_state.processed_data[feature].unique()
                    if len(unique_values) <= 5:  # Likely categorical
                        input_data[feature] = st.selectbox(feature, sorted(unique_values))
                    else:
                        input_data[feature] = st.number_input(feature, value=0)
        
        if st.button("Predict Churn"):
            try:
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Ensure columns are in correct order
                input_df = input_df[feature_names]
                
                # Scale input for logistic regression
                input_scaled = st.session_state.models['scaler'].transform(input_df)
                
                # Make predictions
                lr_pred = st.session_state.models['logistic_regression'].predict(input_scaled)[0]
                lr_prob = st.session_state.models['logistic_regression'].predict_proba(input_scaled)[0]
                
                dt_pred = st.session_state.models['decision_tree'].predict(input_df)[0]
                dt_prob = st.session_state.models['decision_tree'].predict_proba(input_df)[0]
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Logistic Regression**")
                    if lr_pred == 1:
                        st.error("⚠️ Customer is likely to CHURN")
                        st.write(f"Probability: {lr_prob[1]:.1%}")
                    else:
                        st.success("✅ Customer will STAY")
                        st.write(f"Probability: {lr_prob[0]:.1%}")
                    st.progress(lr_prob[1])
                
                with col2:
                    st.write("**Decision Tree**")
                    if dt_pred == 1:
                        st.error("⚠️ Customer is likely to CHURN")
                        st.write(f"Probability: {dt_prob[1]:.1%}")
                    else:
                        st.success("✅ Customer will STAY")
                        st.write(f"Probability: {dt_prob[0]:.1%}")
                    st.progress(dt_prob[1])
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# PAGE 6: Interpretation and Conclusions
elif page == "Interpretation and Conclusions":
    st.title("📝 Interpretation and Conclusions")
    
    if not st.session_state.model_results:
        st.warning("Please train and evaluate models first.")
    else:
        results = st.session_state.model_results
        
        # Model performance summary
        st.subheader("Model Performance Summary")
        
        metrics_df = pd.DataFrame([results['lr_metrics'], results['dt_metrics']])
        st.dataframe(metrics_df.set_index('Model').style.format("{:.3f}"))
        
        # Feature importance (if Decision Tree was trained)
        if 'decision_tree' in st.session_state.models:
            st.subheader("Feature Importance Analysis")
            
            dt_model = st.session_state.models['decision_tree']
            feature_importance = pd.DataFrame({
                'Feature': results['feature_names'],
                'Importance': dt_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(10)
            ax.barh(range(len(top_features)), top_features['Importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Most Important Features (Decision Tree)')
            ax.invert_yaxis()
            
            for i, v in enumerate(top_features['Importance']):
                ax.text(v + 0.001, i, f'{v:.3f}', va='center')
            
            st.pyplot(fig)
            
            # Key insights
            st.subheader("Key Insights")
            
            st.write("**Most Predictive Features:**")
            for i, (_, row) in enumerate(top_features.head(5).iterrows(), 1):
                st.write(f"{i}. **{row['Feature']}** (Importance: {row['Importance']:.3f})")
        
        # Model comparison
        st.subheader("Model Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Logistic Regression**")
            st.write("- ✅ Provides probability estimates")
            st.write("- ✅ More robust to overfitting")
            st.write("- ✅ Works well with linear relationships")
            st.write(f"- Accuracy: {results['lr_metrics']['Accuracy']:.3f}")
        
        with col2:
            st.write("**Decision Tree**")
            st.write("- ✅ Easy to interpret")
            st.write("- ✅ Handles non-linear relationships")
            st.write("- ✅ Doesn't require feature scaling")
            st.write(f"- Accuracy: {results['dt_metrics']['Accuracy']:.3f}")
        
        # Recommendations
        st.subheader("Business Recommendations")
        
        st.write("Based on the analysis, consider these retention strategies:")
        st.write("1. **Target high-risk customers** identified by the models")
        st.write("2. **Improve service quality** for customers with short tenure")
        st.write("3. **Offer incentives** for long-term contracts")
        st.write("4. **Enhance digital services** to reduce churn")
        st.write("5. **Review pricing strategy** for high monthly charges")
        
        # Model selection recommendation
        better_model = "Logistic Regression" if results['lr_metrics']['F1-Score'] > results['dt_metrics']['F1-Score'] else "Decision Tree"
        st.subheader("Recommended Model")
        st.info(f"**{better_model}** is recommended based on overall performance")
        
        if better_model == "Logistic Regression":
            st.write("Logistic Regression provides more reliable probability estimates and is generally more robust for deployment.")
        else:
            st.write("Decision Tree offers better interpretability and handles complex relationships in the data.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Customer Churn Prediction App**  
Built with Streamlit
""")
