import streamlit as st
import pandas as pd
import sklearn
import pickle
import numpy as np

#importing the model
model=pickle.load(open('model.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))
data=pd.read_csv('data//medical-charges.csv')


nav=st.sidebar.radio('Navigation',['Home','Prediction','Contribute','Insights'])
if nav=='Home':
    st.title('Acme Insurance Inc.')
    st.subheader('Annual Health Expenditure Prediction')
    st.image('data//img.png')
    if st.checkbox('Show Data'):
        st.dataframe(data)


if nav=='Prediction':
    st.subheader('Please give the following information:')
    ###ask user for age
    Age = st.number_input('Age', min_value=0.0, max_value=100.0, step=1.0)
    ### sex
    Sex = st.radio('Sex', ['Male', 'Female'])
    ###bmi
    Bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=1.0)
    ###no of children
    Children = st.number_input('No of Children you have')
    ###smoker
    Smoker = st.radio('Smoker', ['Yes', 'No'])
    ###region
    Region = st.selectbox('Region', df['region'].unique())

    if st.button('Predict Yearly Medical Expenditure'):
        if Sex == 'Male':
            Sex = 0
        else:
            Sex = 1

        if Smoker == 'Yes':
            Smoker = 1
        else:
            Smoker = 0
        if Region == 'southeast':
            Region = 0
        elif Region == 'southwest':
            Region = 1
        elif Region == 'northwest':
            Region = 2
        else:
            Region = 3

        input = np.array([Age, Sex, Bmi, Children, Smoker, Region])
        ### converting into 1 row and 6 columns
        input = input.reshape(1, 6)

        st.title('The predicted Annual Health Expenditure is ' + str(round(int(model.predict(input)[0]))) + '$')

if nav == "Contribute":
    st.header("Contribute to our dataset by giving your data")
    Age = st.number_input('Age', min_value=0.0, max_value=100.0, step=1.0)
        ### sex
    Sex = st.radio('Sex', ['Male', 'Female'])
        ###bmi
    Bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=1.0)
        ###no of children
    Children = st.number_input('No of Children you have')
        ###smoker
    Smoker = st.radio('Smoker', ['Yes', 'No'])
        ###region
    Region = st.selectbox('Region', df['region'].unique())
        ###charges
    Charges=st.number_input('Charges:')
    if st.button("submit"):
        to_add = {"age": [Age], "sex": [Sex],'bmi':[Bmi],'children':[Children],'smoker':[Smoker],'region':[Region],'Charges':[Charges]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//expenditure.csv", mode='a', header=False, index=False)
        st.success("Submitted")

if nav=='Insights':
    st.title('Insights from the dataset')
    st.markdown('''
    ### 1.People who smoke and have BMI above 25 have high changes of suffering from different diseases and consequently have more annual medical expenditure.
    ### 2.Male tend to have slightly more annual medical expenditure in comparison to Female.
    ### 3.People who smoke and are older than 35 years of age have high changes of suffering from different diseases and consequently have more annual medical expenditure.
    ''')


