import streamlit as st , pandas as pd
import requests
import ast , io

def write_file(content, output_path):
    with open(output_path, 'w') as file:
        file.write(content)
# Load the dataset
@st.cache_data # Use @st.cache to cache the dataset fetching function
def fetch_dataset():
    response = requests.get("http://localhost:8000/dataset/")  # Use GET request since you're fetching data
    df = pd.read_csv(io.StringIO(response.text))  # Convert to DataFrame
    return df

cars_data = fetch_dataset()

# Total features
overall_features = [
    'listed_price', 'myear', 'body', 'transmission', 'fuel', 'km',
    'Engine Type', 'Length', 'Width', 'Height', 'Max Power Delivered', 'Color' ,
    'model' , 'images'
]

# List of numerical and categorical features
numerical_features = ['listed_price', 'myear', 'Length',]
categorical_features = ['body', 'transmission', 'fuel', 'Color']

# Get min and max values for numerical features
numerical_ranges = {feature: (int(cars_data[feature].min()), int(cars_data[feature].max())) for feature in numerical_features}

# Title of the app
st.title("Feature Selector")

# Initialize session state for input values
if 'inputs' not in st.session_state:
    st.session_state.inputs = {
        'listed_price': (numerical_ranges['listed_price'][0] + numerical_ranges['listed_price'][1]) // 2,
        **{feature: (numerical_ranges[feature][0] + numerical_ranges[feature][1]) // 2 for feature in numerical_features if feature not in ['listed_price', 'km']},
        **{feature: cars_data[feature].unique().tolist()[0] for feature in categorical_features}
    }

with st.form(key="feature"):
    # Inputs for listed_price and km
    inputs_features = {}
    st.header("Input values for listed price and km")
    inputs_features['listed_price'] = st.number_input(
        "Enter listed price:",
        min_value=numerical_ranges['listed_price'][0],
        max_value=numerical_ranges['listed_price'][1],
        value=st.session_state.inputs['listed_price']
    )

    # Sliders for other numerical features
    st.header("Select values for other numerical features")
    for feature in numerical_features:
        if feature not in ['listed_price', 'km']:
            min_value, max_value = numerical_ranges[feature]
            inputs_features[feature] = st.slider(
                f"Select {feature}:",
                min_value,
                max_value,
                st.session_state.inputs[feature]
            )

    # Dropdowns for categorical features
    st.header("Select values for categorical features")
    for feature in categorical_features:
        unique_values = cars_data[feature].unique().tolist()
        inputs_features[feature] = st.selectbox(
            f"Select {feature}:",
            unique_values,
            index=unique_values.index(st.session_state.inputs[feature])
        )

    submitted = st.form_submit_button("Show")

if submitted:
    st.subheader("Recommendations")
    response = requests.post("http://127.0.0.1:8000/recommendations/", json=inputs_features)
    cars_recommendation = pd.DataFrame(response.json())

    if cars_recommendation.shape[0] == 0:
        st.write("No Recommendation Available")

    for idx, row in cars_recommendation.iterrows():
        with st.expander(f"{row['model']}"):
            car_link_dict = ast.literal_eval(row['images'])
            if car_link_dict[0]['img'] != "":
                car_link = car_link_dict[0]['img']
                st.image(car_link, caption=row['model'], use_column_width=True)

            st.write(f"Car image link : {car_link}")

            for feature in overall_features:
                st.write(f"**{feature}:** {row[feature]}")