import pickle
import streamlit as st



# loading the trained model
pickle_in = open('model_pickle.pkl', 'rb')
model = pickle.load(pickle_in)


@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction(Engine_Size, Cylinders, Fuel_Consumption_City, Fuel_Consumption_Hwy, Fuel_Consumption_Comb, Fuel_Consumption_Comb_mpg):

    # Making predictions
    prediction = model.predict([[Engine_Size, Cylinders,Fuel_Consumption_City,Fuel_Consumption_Hwy,Fuel_Consumption_Comb,Fuel_Consumption_Comb_mpg]])

    return prediction


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit CO2 Emission Prediction ML App</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    Engine_Size = st.number_input("Engine_Size")
    Cylinders = st.number_input("Cylinders")
    Fuel_Consumption_City = st.number_input("Fuel_Consumption_City")
    Fuel_Consumption_Hwy = st.number_input("Fuel_Consumption_Hwy")
    Fuel_Consumption_Comb = st.number_input("Fuel_Consumption_Comb")
    Fuel_Consumption_Comb_mpg = st.number_input("Fuel_Consumption_Comb_mpg")

    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(Engine_Size, Cylinders,Fuel_Consumption_City,Fuel_Consumption_Hwy,Fuel_Consumption_Comb,Fuel_Consumption_Comb_mpg)
        st.success('Your Vec emission is:  {}'.format(result))



if __name__ == '__main__':
    main()