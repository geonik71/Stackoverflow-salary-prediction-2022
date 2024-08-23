import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv(r'E:\python\Practical Machine Learning\Project_3\GitHub\df_final.csv')


countries_names = list(df['Country'].astype(str).unique())
countries_names.sort()

age_names = list(df["Age"].astype(str).unique())
age_names.sort()

RemoteWork_names = list(df['RemoteWork'].astype(str).unique())
RemoteWork_names.sort()

EdLevel_names = list(df['EdLevel'].astype(str).unique())
EdLevel_names.sort()





st.header('Stackoverflow Developer Survay Salary Prediction')
st.image('https://survey.stackoverflow.co/2022/up_/src/img/dev-survey-2022.png')



models_names=['Ridge Regression', 'Gradient Boosting Regressor']

models = {
    'Ridge Regression':joblib.load('best_model_ridge.joblib'),
    'Gradient Boosting Regressor':joblib.load('best_model_gbr.joblib')

}

age_select = st.selectbox("Choose the age range:" , age_names)
country_select = st.selectbox('Choose your country:', countries_names )
RemoteWork_select = st.radio("How do you work:", RemoteWork_names)
EdLevel_select = st.selectbox('What is your education level:', EdLevel_names)
YearsCodePro_select = st.number_input("Enter Your Years that you code professionally:", min_value = 0, max_value = 50)

selected_model_name = st.radio("Choose Your Model:", models_names)


X_new = []

age_vector = [1 if age == age_select else 0 for age in age_names]
X_new.extend(age_vector)

country_vector = [1 if con == country_select else 0 for con in countries_names]
X_new.extend(country_vector)

RemoteWork_vector = [1 if con == RemoteWork_select else 0 for con in RemoteWork_names]
X_new.extend(RemoteWork_vector)

EdLevel_vector = [1 if con == EdLevel_select else 0 for con in EdLevel_names]
X_new.extend(EdLevel_vector)

X_new.append(YearsCodePro_select)


X_array = np.array(X_new).reshape(1, -1)

selected_model = models[selected_model_name]
prediction = selected_model.predict(X_array)

if st.button('Predict'):
    rounded_prediction = round(prediction[0], -2)
    st.success(f"The estimated Salary is: {rounded_prediction:,} $")



    fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Adjusted figure size

    # Plot 1: Age Distribution
    sns.countplot(data=df, x='Age', order=df['Age'].value_counts().index, ax=axs[0, 0])
    axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45)
    axs[0, 0].set_title("Distribution of Respondents by Age Group")
    axs[0, 0].set_xlabel("Age Group")
    axs[0, 0].set_ylabel("Number of Respondents")

    # Plot 2: Remote Work Distribution
    RemoteWork_counts = df['RemoteWork'].value_counts()
    axs[0, 1].pie(RemoteWork_counts, labels=RemoteWork_counts.index, autopct='%1.1f%%', startangle=140,
                wedgeprops=dict(width=0.3, edgecolor='w'),
                textprops={'fontsize': 12})
    axs[0, 1].set_title("Distribution of Respondents by Remote Work Group")

    # Plot 3: Country Distribution
    top_countries = df['Country'].value_counts().head(10).index
    df_top_countries = df[df['Country'].isin(top_countries)]
    sns.countplot(data=df_top_countries, x='Country', order=top_countries, ax=axs[1, 0])
    axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45)
    axs[1, 0].set_title("Distribution of Respondents by Country Group")
    axs[1, 0].set_xlabel("Country Group")
    axs[1, 0].set_ylabel("Number of Respondents")

    # Plot 4: Years of Professional Coding Experience Distribution
    sns.histplot(df['YearsCodePro'], kde=True, bins=30, ax=axs[1, 1])
    axs[1, 1].set_title("Distribution of Professional Coding Experience")
    axs[1, 1].set_xlabel("Years of Professional Coding Experience")
    axs[1, 1].set_ylabel("Frequency")

    # Adjust layout
    plt.tight_layout()

    # Display the plots in Streamlit with expanded container
    with st.expander("Show Plots"):
        st.pyplot(fig)
