import streamlit as st
import pandas as pd
import pickle
import uuid

# Set page configuration with title and layout
st.set_page_config(page_title="Passenger Transportation Predictor", layout="wide")

# App title and description
st.title("Passenger Transportation Predictor")
st.write("Fill in the passenger details to see if they will be transported or not.")

# Sidebar for navigation between pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Make Prediction", "About Model"])

# Function to load the trained machine learning model
def load_model():
    try:
        with open('Spaceship Titanic RF model.pkl', 'rb') as file:
            return pickle.load(file)  # Load the trained model from a pickle file
    except:
        st.error("Model file not found. Please try again.")
        return None

# "About Model" page
if page == "About Model":
    st.header("About the Model")
    st.write("""
    This application uses a Random Forest classifier to predict whether a passenger will be transported.
    
    ### Model Features:
    - Passenger demographic information
    - Travel details
    - Cabin information
    - Spending habits
    
    ### Model Performance:
    - Accuracy: ~80-85% (approximate)
    - This model was trained on a dataset of passenger transportation records.
    """)

# "Make Prediction" page
elif page == "Make Prediction":
    st.header("Enter Passenger Details")

    # Section for passenger personal details
    with st.expander("Personal Information", expanded=True):
        age = st.number_input("Age", min_value=0, max_value=120, value=30)  # Numeric input for age
        vip = st.radio("VIP Status", ["No", "Yes"], index=0)  # Radio buttons for VIP status
        cryosleep = st.radio("CryoSleep", ["No", "Yes"], index=0)  # Radio buttons for CryoSleep status
        home_planet = st.selectbox("Home Planet", ["Earth", "Mars", "Europa"])  # Dropdown for home planet selection
        destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])  # Dropdown for travel destination
    
    # Section for cabin and spending information
    with st.expander("Cabin & Spending Information", expanded=True):
        cabin_deck = st.selectbox("Cabin Deck", ["A", "B", "C", "D", "E", "F", "G", "T"])  # Dropdown for cabin deck selection
        cabin_num = st.number_input("Cabin Number", min_value=0, max_value=2000, value=100)  # Numeric input for cabin number
        cabin_side = st.selectbox("Cabin Side", ["P", "S"])  # Dropdown for cabin side selection
        room_service = st.number_input("Room Service Expenditure", min_value=0, value=0)  # Numeric input for room service spending
        food_court = st.number_input("Food Court Expenditure", min_value=0, value=0)  # Numeric input for food court spending
        shopping_mall = st.number_input("Shopping Mall Expenditure", min_value=0, value=0)  # Numeric input for shopping mall spending
        spa = st.number_input("Spa Expenditure", min_value=0, value=0)  # Numeric input for spa spending
        vr_deck = st.number_input("VR Deck Expenditure", min_value=0, value=0)  # Numeric input for VR deck spending
    
    # Button to make the prediction
    if st.button("Predict"):
        # Generate a random unique passenger ID
        passenger_id = str(uuid.uuid4())[:8]

        # Create a dictionary to store user input data
        input_data = {
            'PassengerId': passenger_id,
            'HomePlanet': home_planet,
            'CryoSleep': cryosleep == "Yes",
            'Cabin': f"{cabin_deck}/{cabin_num}/{cabin_side}",
            'Destination': destination,
            'Age': age,
            'VIP': vip == "Yes",
            'RoomService': room_service,
            'FoodCourt': food_court,
            'ShoppingMall': shopping_mall,
            'Spa': spa,
            'VRDeck': vr_deck,
            'Name': f"Pass_{passenger_id}"  # Assigning a random name to passenger
        }

        # Convert input data to a Pandas DataFrame
        input_df = pd.DataFrame([input_data])

        # Display input data in a table
        st.write("### Input Data:")
        st.dataframe(input_df)

        # Load the trained model
        pipeline = load_model()

        print(input_df)

        # Predict probability of transportation using the model (or assign a random probability if model is not found)
        prediction = pipeline.predict_proba(input_df)[0, 1]

        # Display prediction probability meter
        st.subheader("Prediction Probability")
        probability_color = "#36b37e" if prediction >= 0.5 else "#ff4b4b"  # Green if transported, red if not
        meter_html = f"""
        <div style="width: 100%; background: #ddd; border-radius: 20px; padding: 5px;">
            <div style="width: {prediction*100}%; background: {probability_color}; height: 25px; border-radius: 15px; transition: width 0.5s;"></div>
        </div>
        <p style="text-align: center; font-size: 18px; font-weight: bold;">{prediction:.2f}</p>
        """
        st.markdown(meter_html, unsafe_allow_html=True)

        # Display final prediction result
        if prediction >= 0.5:
            st.success(f"This passenger is likely to be transported (Probability: {prediction:.2f})")
        else:
            st.error(f"This passenger is not likely to be transported (Probability: {prediction:.2f})")

# Footer
st.markdown("---")
st.markdown("Transportation Prediction App - Created with Streamlit")
