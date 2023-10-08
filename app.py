import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler,\
                                  MinMaxScaler,\
                                  OneHotEncoder,\
                                  OrdinalEncoder,\
                                  PowerTransformer,\
                                  FunctionTransformer
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.metrics import make_scorer,\
                            accuracy_score,\
                            precision_score,\
                            recall_score,\
                            f1_score

from scipy.stats import pointbiserialr

# Set a page config
st.set_page_config(page_title="Customer Churn", page_icon=":bar_chart:", layout="wide")

# Sidebar selection of section
section = st.sidebar.selectbox("Choose a section", ["EDA", "Model Prediction"])

# App header/title
st.title("Customer Churn Analysis & Prediction")

# Load data function
@st.cache_data
def load_data():
    data = pd.read_csv('data/train.csv')
    return data

data = load_data()

def apply_outlier_capping_and_tracking(df, columns, bounds):
  df_capped = df.copy()

  for column in columns:
    lower_bound, upper_bound = bounds[column]

    # Create a new binary column to track outliers
    outlier_col_name = f"{column}_is_outlier"
    df_capped[outlier_col_name] = ((df_capped[column] < lower_bound)\
                                    | (df_capped[column] > upper_bound))\
                                    .astype(int)

    # Cap outliers
    df_capped[column] = df_capped[column].clip(lower_bound, upper_bound)

  return df_capped

def add_uses_voicemail(df):
  # add 'uses_voicemail' column
  df['uses_voicemail'] = (df['number_vmail_messages'] > 0).astype(int)
  return df

def churn_label(churn_rates, high_threshold, low_threshold):
    def label(rate):
      if rate >= high_threshold:
          return 'High'
      elif rate <= low_threshold:
          return 'Low'
      else:
          return 'Medium'
    return np.vectorize(label)(churn_rates)

def engineer_features(df, churn_rates, high_threshold, low_threshold):
  # Convert state to region
  df['region'] = df['state'].apply(state_to_region)

  # Create churn bin label
  df['churn_bin'] = df['state'].map(churn_rates).\
                    apply(lambda rate: churn_label(rate,
                                                   high_threshold,
                                                   low_threshold))

  return df

def state_to_region(state):
  northeast = ['CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT']

  midwest = ['IL', 'IN', 'IA', 'KS', 'MI', 'MN',
             'MO', 'NE', 'ND', 'OH', 'SD', 'WI']

  south = ['AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA',
           'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV', 'DC']

  west = ['AK', 'AZ', 'CA', 'CO',
          'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']

  if state in northeast:
    return 'Northeast'
  elif state in midwest:
    return 'Midwest'
  elif state in south:
    return 'South'
  elif state in west:
    return 'West'
  else:
    return 'Other'

# EDA Section
if section == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Checkbox to show/hide raw data
    st.sidebar.subheader("Raw Data")
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(data.head())  # using dataframe instead of write for a cleaner look

    # Sidebar for Pairplot
    st.sidebar.subheader("Pairplot Analysis")

    # Checkbox for Pairplot
    if st.sidebar.checkbox("Show Pairplot Analysis"):

        st.subheader("Pairplot Analysis")

        num_vars = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # MultiSelect for the user to select variables for the pairplot
        selected_vars = st.sidebar.multiselect("Choose variables:", num_vars, default=num_vars[:3])

        # If the user has selected at least two variables, draw a pairplot
        if len(selected_vars) >= 2:

            # Creating a dataframe subset
            data_subset = data[selected_vars + ['churn']]

            # Plotting
            st.write(f"Pairplot for {', '.join(selected_vars)} grouped by Churn")
            fig = sns.pairplot(data_subset, hue='churn')
            st.pyplot(fig)

        else:
            st.warning("Please select at least two variables to create a pairplot.")
    
    # Churn Distribution
    st.sidebar.subheader("Numerical Visualizations")
    if st.sidebar.checkbox("Show Churn Distribution"):
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        
        # Custom palette
        custom_palette = {"yes": "red", "no": "green"}
        
        # Using Seaborn with the custom palette
        sns.countplot(x='churn', data=data, ax=ax, palette=custom_palette)
        
        ax.set_title('Churn Distribution')
        ax.set_xlabel('Churn')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    # Numerical Variable Distribution
    if st.sidebar.checkbox("Show Numerical Variable Distribution"):
        num_var = st.sidebar.selectbox("Choose a Numerical Variable", options=data.select_dtypes(include=['float64', 'int64']).columns.tolist())
        st.subheader(f"Distribution of {num_var}")
        fig, ax = plt.subplots()
        sns.histplot(data[num_var], bins=30, ax=ax)
        ax.set_title(f'Distribution of {num_var}')
        ax.set_xlabel(num_var)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # Detailed View
    if st.sidebar.checkbox("Show Detailed View for Numerical Variables"):
        selected_var = st.sidebar.selectbox("Choose a Numerical Variable for Detailed View", options=data.select_dtypes(include=['float64', 'int64']).columns.tolist())
        st.subheader(f"Detailed View of {selected_var}")
        
        # Creating 2 columns for side-by-side plots
        col1, col2 = st.columns(2)
        
        # Boxplot
        fig, ax = plt.subplots()
        # Custom palette
        custom_palette = {"yes": "red", "no": "blue"}
        
        sns.boxplot(x='churn', y=selected_var, data=data, ax=ax, palette=custom_palette)
        ax.set_title(f'Boxplot of {selected_var} vs Churn')
        col1.pyplot(fig)
        
        # Histogram
        fig, ax = plt.subplots()
        sns.histplot(x=data[selected_var], hue=data['churn'], kde=True, stat="probability", common_norm=False, ax=ax)
        ax.set_title(f'Histogram of {selected_var} vs Churn')
        col2.pyplot(fig)

    # Correlation
    st.sidebar.subheader("Correlation Analysis")
    
    if st.sidebar.checkbox("Show Correlation"):
        
        num_vars = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Convert target variable to numerical
        data['churn_numeric'] = data['churn'].apply(lambda x: 1 if x == 'yes' else 0)

        # Radio button or dropdown to choose the type of correlation view
        correlation_view = st.sidebar.radio("Choose a View:", ("Correlation Heatmap", "Point-Biserial Correlation"))

        # Correlation Heatmap View
        if correlation_view == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            numerical_data = data.select_dtypes(include=['float64', 'int64'])
            sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)

        # Point-Biserial Correlation View
        elif correlation_view == "Point-Biserial Correlation":
            st.subheader("Point-Biserial Correlation with Churn")

            correlations = []
            p_values = []

            # Calculate point-biserial correlation
            for col in num_vars:
                corr, p_value = pointbiserialr(data[col], data['churn_numeric'])
                correlations.append(corr)
                p_values.append(p_value)

            # Create a DataFrame for visualization
            corr_data = pd.DataFrame({
                'Variable': num_vars,
                'Correlation': correlations,
                'P_value': p_values
            })

            # Sort the DataFrame based on the absolute value of the correlation coefficient
            corr_data = corr_data.reindex(
                corr_data.Correlation.abs().sort_values(ascending=False).index
            )

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=corr_data, y='Variable', x='Correlation', palette='viridis', ax=ax)
            ax.set_title('Point-Biserial Correlation with Churn')
            ax.set_xlabel('Correlation Coefficient')
            ax.set_ylabel('Variable')
            st.pyplot(fig)

            # Display correlations and p-values in the app
            st.write(corr_data)

    # Categorical Variable Distribution
    st.sidebar.subheader("Categorical Visualizations")
    if st.sidebar.checkbox("Show Categorical Variable Distribution"):
        cat_var = st.sidebar.selectbox("Choose a Categorical Variable", options=data.select_dtypes(include=['object']).columns.tolist())
        st.subheader(f"Distribution of {cat_var}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=cat_var, data=data, order=data[cat_var].value_counts().index)
        ax.set_title(f'Distribution of {cat_var}')
        st.pyplot(fig)

    # Use markdown for styled text and better structuring
    st.markdown("""
    ## Detailed Analysis
    Eksplorasi Distribusi Berbagai Fitur dan Dampaknya Terhadap Churn Pelanggan.
    """)

    # Function to map states to regions
    def state_to_region(state):
        northeast = ['CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT']
        midwest = ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI']
        south = ['AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV', 'DC']
        west = ['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
        
        if state in northeast:
            return 'Northeast'
        elif state in midwest:
            return 'Midwest'
        elif state in south:
            return 'South'
        elif state in west:
            return 'West'
        else:
            return 'Other'

    # Apply the function to create a new column 'region'
    data['region'] = data['state'].apply(state_to_region)

    # Detailed View for Categorical Variables
    if st.sidebar.checkbox("Show Detailed View for Categorical Variables"):
        # Exclude 'state' and include 'region'
        cat_vars = [col for col in data.columns if data[col].dtype == 'O' and col != 'state' and col != 'churn']
        
        selected_var = st.sidebar.selectbox("Choose a Categorical Variable for Detailed View", options=cat_vars)
        
        st.subheader(f"Detailed View of {selected_var}")
        
        # Creating 2 columns for side-by-side plots
        col1, col2 = st.columns(2)
        
        # Countplot
        fig, ax = plt.subplots()
        sns.countplot(y=selected_var, data=data, ax=ax, order=data[selected_var].value_counts().index)
        ax.set_title(f'Distribution of {selected_var}')
        col1.pyplot(fig)
        
        # Churn Rate within Category (in percentage)
        fig, ax = plt.subplots()
        # Calculate the percentage
        ((data.groupby([selected_var])['churn'].value_counts(normalize=True) * 100)
        .unstack()
        .plot(kind='barh', stacked=True, ax=ax))
        ax.set_title(f'Churn Rate by {selected_var} (%)')
        ax.set_ylabel(selected_var)
        ax.set_xlabel('Proportion (%)')
        col2.pyplot(fig)
        
    # Detailed View for Categorical Variables vs. Churn Rate
    if st.sidebar.checkbox("Show Churn Rate for Categorical Variables"):
        # Exclude 'state' and include 'region'
        cat_vars = [col for col in data.columns if data[col].dtype == 'O' and col != 'state' and col != 'churn'] 
        
        selected_var = st.sidebar.selectbox("Choose a Categorical Variable to View Churn Rate", options=cat_vars)
        
        st.subheader(f"Churn Rate by {selected_var}")
        
        # Compute the churn rate
        churn_rate = (data.groupby(selected_var)['churn']
                    .apply(lambda x: (x == 'yes').mean())
                    .reset_index()
                    .sort_values(by='churn', ascending=False))
        
        # Rename the columns for clarity in visualization
        churn_rate.columns = [selected_var, 'churn_rate']
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=selected_var, y='churn_rate', data=churn_rate, palette="viridis", ax=ax)
        ax.set_title(f'Churn Rate by {selected_var}')
        ax.set_ylabel('Churn Rate')
        ax.set_xlabel(selected_var)
        
        # Adding the percentage values on top of the bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height() * 100:.2f}%", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='gray',
                        xytext=(0, 10),
                        textcoords='offset points')
        
        # Display the plot
        st.pyplot(fig)
        
    # Churn & Retention Analysis within EDA
    if st.sidebar.checkbox("Show Churn & Retention Analysis"):
        st.subheader("Churn & Retention Analysis")
        
        # Sidebar: Feature Selection for Churn & Retention Analysis
        selected_feature = st.sidebar.selectbox(
            "Choose a feature for Churn & Retention Analysis",
            ["voice_mail_plan", "international_plan"]
        )
        st.sidebar.markdown(f"### Analysis for: **{selected_feature}**")

        # Calculate churn and retention rates
        churn_rate_per_feature = data.groupby(selected_feature)['churn']\
                                    .value_counts(normalize=True).unstack()\
                                    .fillna(0) * 100

        # Rename columns for clearer representation in the table
        churn_and_retention_rates = churn_rate_per_feature[['yes', 'no']]\
            .rename(columns={'yes': 'Churn Rate (%)', 'no': 'Retention Rate (%)'})
        
        # Displaying churn rate for each category in the selected feature
        st.table(churn_and_retention_rates)

# Model Prediction Section
if section == "Model Prediction":
    st.header("Model Prediction")
    
    # Checkbox for choosing input method
    input_method = st.sidebar.radio("Choose input method", ["Upload a file", "Manual input"])
    
    # Load Models and Configurations
    model = joblib.load('models/XGBoost_best_model.joblib')
    bins_config = joblib.load('models/churn_bins_config.joblib')
    outlier_bounds = joblib.load('models/outlier_cap_bounds.joblib')
    
    # If the user chooses to upload a file
    if input_method == "Upload a file":
        uploaded_file = st.file_uploader("Choose a .csv file", type=['csv'])
        
        # Expected columns
        expected_columns = ['id', 'state', 'account_length', 'area_code', 'international_plan',
                            'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes',
                            'total_day_calls', 'total_day_charge', 'total_eve_minutes',
                            'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
                            'total_night_calls', 'total_night_charge', 'total_intl_minutes',
                            'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls']
        
        # Check if a file is uploaded
        if uploaded_file is not None:
            # Load the uploaded data
            data = pd.read_csv(uploaded_file)
            
            # Check if the uploaded data has the expected columns
            if list(data.columns) == expected_columns:
                st.success("File uploaded successfully!")
            
                # Predict button
                if st.button("Predict"):
                    try:
                        # Extract 'id' and drop it from the data
                        test_id = data['id']
                        data = data.drop('id', axis=1)
                        
                        # identify categorical variables
                        cat_vars = [
                            var for var in data.columns
                            if data[var].dtype == 'O'
                            ]

                        # identify numerical variables
                        num_vars = [
                            var for var in data.columns
                            if var not in cat_vars
                            ]

                        # Load outlier cap bounds and churn bins configuration
                        loaded_cap_bounds = joblib.load('models/outlier_cap_bounds.joblib')
                        churn_rates, high_threshold, low_threshold = joblib.load('models/churn_bins_config.joblib')

                        # Apply feature engineering and data preprocessing
                        data_capped = apply_outlier_capping_and_tracking(data, num_vars, loaded_cap_bounds)
                        data_voicemail = add_uses_voicemail(data_capped)
                        X_new = engineer_features(data_voicemail, churn_rates, high_threshold, low_threshold)

                        # Load and apply the model
                        pipeline = joblib.load('models/XGBoost_best_model.joblib')
                        predictions = pipeline.predict(X_new)

                        # Displaying the prediction results
                        st.success("Prediction complete!")
                        st.write("Predictions: ", predictions)
                        
                        results_df = pd.DataFrame({'ID': test_id, 'Predicted Churn': predictions})
                        st.dataframe(results_df)
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                    
            else:
                st.error("The uploaded file does not have the expected columns.")
        
        else:
            st.warning("Please upload a .csv file.")
        
    # If the user chooses to input data manually
    elif input_method == "Manual input":
        st.subheader("Manual Data Input")
        
        # Load training data for reference
        train_data = pd.read_csv('data/train.csv')
        
        # Extract unique values for categorical variables
        states = train_data['state'].unique().tolist()
        area_codes = train_data['area_code'].unique().tolist()
        intl_plan = train_data['international_plan'].unique().tolist()
        voice_plan = train_data['voice_mail_plan'].unique().tolist()
        
        # Manually input features
        with st.form(key='manual_input_form'):
            st.markdown("### Personal Details")
            state = st.selectbox("State", options=states, key='state')
            account_length = st.number_input("Account Length", min_value=1, max_value=250, key='account_length')
            area_code = st.selectbox("Area Code", options=area_codes, key='area_code')
            international_plan = st.selectbox("International Plan", options=intl_plan, key='intl_plan')
            voice_mail_plan = st.selectbox("Voice Mail Plan", options=voice_plan, key='voice_mail_plan')
            number_vmail_messages = st.number_input("Number of Voice Mail Messages", min_value=0, max_value=60, key='num_vmail_msg')
            
            
            st.markdown("### Usage Details")
            total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=350.0, key='total_day_minutes')
            total_day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, key='total_day_calls')
            total_day_charge = st.number_input("Total Day Charge", min_value=0.0, max_value=60.0, key='total_day_charge')
            
            st.markdown("### Evening Usage")
            total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=350.0, key='total_eve_minutes')
            total_eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=200, key='total_eve_calls')
            total_eve_charge = st.number_input("Total Evening Charge", min_value=0.0, max_value=60.0, key='total_eve_charge')
            
            st.markdown("### Night Usage")
            total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, max_value=350.0, key='total_night_minutes')
            total_night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, key='total_night_calls')
            total_night_charge = st.number_input("Total Night Charge", min_value=0.0, max_value=60.0, key='total_night_charge')
            
            st.markdown("### International Usage")
            total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=20.0, key='total_intl_minutes')
            total_intl_calls = st.number_input("Total International Calls", min_value=0, max_value=20, key='total_intl_calls')
            total_intl_charge = st.number_input("Total International Charge", min_value=0.0, max_value=5.0, key='total_intl_charge')
            
            st.markdown("### Customer Service")
            number_customer_service_calls = st.number_input("Number of Customer Service Calls", min_value=0, max_value=10, key='num_cust_serv_calls')
            
            # Submit button
            submit_button = st.form_submit_button(label='Predict')
            
            if submit_button:
                # Construct a DataFrame from the inputs
                input_data = pd.DataFrame({
                    'state': [state],
                    'account_length': [account_length],
                    'area_code': [area_code],
                    'international_plan': [international_plan],
                    'voice_mail_plan': [voice_mail_plan],
                    'number_vmail_messages': [number_vmail_messages],
                    'total_day_minutes': [total_day_minutes],
                    'total_day_calls': [total_day_calls],
                    'total_day_charge': [total_day_charge],
                    'total_eve_minutes': [total_eve_minutes],
                    'total_eve_calls': [total_eve_calls],
                    'total_eve_charge': [total_eve_charge],
                    'total_night_minutes': [total_night_minutes],
                    'total_night_calls': [total_night_calls],
                    'total_night_charge': [total_night_charge],
                    'total_intl_minutes': [total_intl_minutes],
                    'total_intl_calls': [total_intl_calls],
                    'total_intl_charge': [total_intl_charge],
                    'number_customer_service_calls': [number_customer_service_calls]
                })
                
                try:
                    cat_vars = [
                        var for var in input_data.columns
                        if input_data[var].dtype == 'O'
                    ]

                    # identify numerical variables
                    num_vars = [
                        var for var in input_data.columns
                        if var not in cat_vars
                    ]

                    # Load outlier cap bounds and churn bins configuration
                    loaded_cap_bounds = joblib.load('models/outlier_cap_bounds.joblib')
                    churn_rates, high_threshold, low_threshold = joblib.load('models/churn_bins_config.joblib')

                    # Apply feature engineering and data preprocessing
                    data_capped = apply_outlier_capping_and_tracking(input_data, num_vars, loaded_cap_bounds)
                    data_voicemail = add_uses_voicemail(data_capped)
                    X_new = engineer_features(data_voicemail, churn_rates, high_threshold, low_threshold)

                    # Load and apply the model
                    pipeline = joblib.load('models/XGBoost_best_model.joblib')
                    predictions = pipeline.predict(X_new)

                    # Displaying the prediction results
                    st.success("Prediction complete!")
                    st.write("Predictions: ", predictions)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("""
--- 
Created by Mohammad Ripan Saiful Mansur
""")
