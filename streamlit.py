import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from plotly.subplots import make_subplots
import requests

# Streamlit page configuration
st.set_page_config(page_title="Cloudburst Prediction", page_icon="⛅", layout="wide")

# Custom dark theme styling
st.markdown("""
    <style>
        /* Main App Background */
        .stApp {
            background: linear-gradient(135deg, #0B0C10, #1F2833, #45A29E);
            color: #C5C6C7;
            font-family: 'Poppins', sans-serif;
        }

        /* Sidebar */
        .sidebar .sidebar-content {
            background: #222831;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }

        .sidebar h2 {
            color: #66FCF1;
            font-size: 22px;
            font-weight: 600;
            text-align: center;
            text-shadow: 0px 0px 10px rgba(102, 252, 241, 0.8);
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #45A29E, #66FCF1);
            color: black;
            font-size: 18px;
            font-weight: bold;
            border-radius: 20px;
            border: none;
            padding: 12px 24px;
            box-shadow: 0px 8px 20px rgba(102, 252, 241, 0.3);
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #66FCF1, #45A29E);
            transform: scale(1.05);
            box-shadow: 0px 12px 25px rgba(102, 252, 241, 0.5);
        }

        /* Sliders */
        .stSlider>div>div>div>input {
            background-color: #393E46;
            color: #66FCF1;
        }

        /* Input Boxes */
        .stTextInput>div>div>input {
            background-color: #393E46;
            color: #66FCF1;
            border-radius: 12px;
            padding: 10px;
            font-size: 16px;
        }

        /* Metrics Box */
        .stMetric {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            font-size: 30px;
            font-weight: 600;
            color: #66FCF1;
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.2);
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.7);
            color: #C5C6C7;
            font-size: 14px;
            border-radius: 10px;
        }

        footer p {
            margin: 0;
        }

        footer a {
            color: #66FCF1;
            text-decoration: none;
            font-weight: bold;
            text-shadow: 0px 0px 5px rgba(102, 252, 241, 0.7);
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data = pd.read_csv('DATASETMINI.csv')
    return data.dropna()

data = load_data()

features = ['MinimumTemperature', 'Rainfall', 'WindGustSpeed', 'Humidity9am', 
            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm']
target = 'CloudBurstTomorrow'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_imputed, y_train)

X_test_imputed = imputer.transform(X_test)
y_pred = clf.predict(X_test_imputed)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Fetch 5-day weather data from OpenWeatherMap API
@st.cache_data
def fetch_weather_data(city: str, api_key: str):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch weather data.")
        return None

# Streamlit app title
st.title("🌧 Cloudburst Probability Prediction App")

if 'user_input_dict' not in st.session_state:
    st.session_state.user_input_dict = {}

# Sidebar navigation
page = st.sidebar.radio("Navigate", 
                        ["Home", "Wind Rose Chart", "Model Analysis", 
                         "Feature Importance", "Distribution Plots", 
                         "Model Comparison", "Dual-Axis Chart", "5-Day Weather Data"])

# Sidebar inputs for predictions
st.sidebar.header("🔧 Adjust Parameters")
for feature in features:
    st.session_state.user_input_dict[feature] = st.sidebar.slider(
        f"{feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))

# Prediction function
def predict_cloudburst_probability(input_data):
    probability = clf.predict_proba(input_data)[:, 1]
    return probability[0]

user_input_imputed = imputer.transform(pd.DataFrame([st.session_state.user_input_dict]))
probability_result = predict_cloudburst_probability(user_input_imputed)

# Page content handling
if page == "Home":
    st.subheader("🏠 Welcome to the Prediction App")
    st.write("Get accurate predictions based on historical weather data.")

    st.subheader("🌧 Cloudburst Probability")
    st.metric(label="Predicted Probability", value=f"{probability_result:.2%}")

    st.progress(int(probability_result * 100))
elif page == "5-Day Weather Data":
    st.subheader("🌤 5-Day Weather Forecast (3-hour intervals)")

    # Input field for city
    city = st.text_input("Enter City:", "Mumbai")

    # API key for OpenWeatherMap
    api_key = "673c79d4cf4ef8aa981ebd075e6c3eb1"  # Replace with your OpenWeatherMap API key

    # Fetch weather data
    weather_data = fetch_weather_data(city, api_key)

    # Radio button for mode selection
    mode = st.radio(
        "Select Mode 🌈",
        ("Show Weather Data 📊", "Show Weather Graph 📈", "Interactive Map 🌍"),
        horizontal=True
    )

    if weather_data:
        st.write(f"**City:** {weather_data['city']['name']} 🌆, {weather_data['city']['country']} 🌍")
        forecasts = weather_data['list']

        # Create an empty list to store weather data for the next 5 days (3-hour intervals)
        weather_list = []

        # Loop through the forecasts, fetch data at 3-hour intervals for the next 5 days
        for forecast in forecasts:
            dt_txt = forecast['dt_txt']
            temp = forecast['main']['temp']
            desc = forecast['weather'][0]['description']
            rain = forecast.get('rain', {}).get('3h', 0)  # Rain volume in mm
            snow = forecast.get('snow', {}).get('3h', 0)  # Snow volume in mm
            wind_speed = forecast['wind']['speed']
            wind_deg = forecast['wind']['deg']
            gust = forecast.get('wind', {}).get('gust', 'N/A')
            pressure = forecast['main']['pressure']
            cloud_cover = forecast['clouds']['all']
            humidity = forecast['main']['humidity']  # Extract humidity

            # Append data to the list
            weather_list.append([dt_txt, temp, desc.capitalize(), rain, snow, wind_speed, gust, wind_deg, pressure, cloud_cover, humidity])

        # Create a DataFrame from the weather data
        df_weather = pd.DataFrame(weather_list, columns=[
            "Date/Time", "Temperature (°C)", "Description", "Rain (mm)", "Snow (mm)", 
            "Wind Speed (m/s)", "Wind Gust (m/s)", "Wind Direction (°)", 
            "Pressure (hPa)", "Cloud Cover (%)", "Humidity (%)"
        ])

        if mode == "Show Weather Data 📊":
            # Show weather data for 5 days (3-hour intervals)
            st.dataframe(df_weather)

        elif mode == "Show Weather Graph 📈":
    try:
        # Initialize the graph
        fig = go.Figure()

        # Plot temperature
        fig.add_trace(
            go.Scatter(
                x=df_weather["Date/Time"],
                y=df_weather["Temperature (°C)"],
                name="Temperature (°C) 🌡️",
                mode="lines+markers",
                line=dict(color="red")
            )
        )

        # Plot humidity on a secondary Y-axis
        fig.add_trace(
            go.Scatter(
                x=df_weather["Date/Time"],
                y=df_weather["Humidity (%)"],
                name="Humidity (%) 💧",
                mode="lines+markers",
                line=dict(color="green"),
                yaxis="y2"  # Humidity uses secondary Y-axis
            )
        )

        # Plot rain (separate Y-axis on the right)
        fig.add_trace(
            go.Bar(
                x=df_weather["Date/Time"],
                y=df_weather["Rain (mm)"],
                name="Rain (mm) 🌧️",
                marker=dict(color="blue"),
                yaxis="y3"
            )
        )

        # Update layout with multiple y-axes
        fig.update_layout(
            title="Weather Forecast 📅",
            xaxis=dict(title="Date & Time 🕒", tickangle=-45),  # Rotate x-axis labels
            yaxis=dict(
                title="Temperature (°C) 🌡️",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
            ),
            yaxis2=dict(
                title="Humidity (%) 💧",
                titlefont=dict(color="green"),
                tickfont=dict(color="green"),
                overlaying="y",  # Overlay the primary y-axis
                side="right"
            ),
            yaxis3=dict(
                title="Rain (mm) 🌧️",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
                overlaying="y",  # Overlay the primary y-axis
                side="right",
                showgrid=False  # Remove grid overlap
            ),
            barmode="group",  # Prevent bar stacking issue
            legend=dict(x=0, y=1.1, orientation="h"),
            template="plotly_white"
        )

        # Display the graph
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while creating the weather graph: {e}")
        st.write("Ensure your data is correctly formatted and contains valid values.")


elif page == "Wind Rose Chart":
    st.subheader("🌀 Wind Rose Chart")
    st.write("Visualizing wind speeds and directions.")
    st.write("The Wind Rose Chart displays the distribution of wind speeds and directions.🌀")
    st.write("The length of each bar represents the wind speed 🌬️, and the direction is indicated by the angle. ↗️")
    st.write("This chart helps visualize the predominant wind patterns in the dataset. 🌍")

    wind_directions = np.random.uniform(0, 360, len(data['WindGustSpeed']))
    fig_windrose = go.Figure()
    fig_windrose.add_trace(go.Barpolar(
        r=data['WindGustSpeed'],
        theta=wind_directions,
        marker_color="rgba(173, 216, 230, 0.6)",
        name='Wind Speed'
    ))

    fig_windrose.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, data['WindGustSpeed'].max()])),
        showlegend=True,
        paper_bgcolor="#1e1e2e",
        font=dict(color="white")
    )
    st.plotly_chart(fig_windrose)

elif page == "Model Analysis":
    st.subheader("📊 Model Performance")
    st.write(f"*Model Accuracy:* {accuracy:.2%}")

    st.subheader("📉 Confusion Matrix")
    # Create a figure explicitly
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='g', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)  # Pass the figure explicitly

    st.subheader("🔎 Classification Report")
    st.code(class_report, language="text")

    st.subheader("Heatmap of the Dataset:")
    st.write("The Heatmap 🔥 displays the correlation between different features in the dataset 📊.")
    st.write("It helps identify relationships 🔗 and patterns 🧩 among variables.")


    # Filter only numeric columns to avoid non-numeric values during correlation calculation
    numeric_data = data.select_dtypes(include=[np.number])

    fig_heatmap_dataset = go.Figure()
    fig_heatmap_dataset.add_trace(go.Heatmap(
        z=numeric_data.corr(),
        x=numeric_data.columns,
        y=numeric_data.columns,
        colorscale='Viridis'
    ))

    fig_heatmap_dataset.update_layout(
        title='Correlation Heatmap',
    )

    st.plotly_chart(fig_heatmap_dataset)



elif page == "Distribution Plots":
    # Distribution Plots page content
    st.subheader("Distribution Plots:")
    for feature in features:
        st.write(f"**{feature} Distribution:**")
        st.write(f"The Distribution Plot shows the distribution of {feature} in the dataset.")
        st.write("It provides insights into the spread and density of each feature.")

        fig_dist = px.histogram(data, x=feature, color=target, marginal="rug", nbins=30, histnorm='probability density')
        st.plotly_chart(fig_dist)



elif page == "Feature Importance":
    st.subheader("📊 Feature Importance")
    feature_importance = clf.feature_importances_
    fig_feature_importance = go.Figure()
    fig_feature_importance.add_trace(go.Bar(
        x=features, 
        y=feature_importance, 
        marker_color='#ffb86c'
    ))
    fig_feature_importance.update_layout(
        title_text="Feature Importance",
        paper_bgcolor="#1e1e2e",
        font=dict(color="white")
    )
    st.plotly_chart(fig_feature_importance)
    

elif page == "Dual-Axis Chart":
    # Dual-Axis Chart
    st.subheader("Dual-Axis Chart:")
    st.write("The Dual-Axis Chart shows the minimum temperature and rainfall over time.")
    st.write("It helps to visualize the relationship between temperature and rainfall trends.")

    # Check if required columns are present
    if 'MinimumTemperature' not in data.columns or 'Rainfall' not in data.columns:
        st.error("The dataset must include 'MinimumTemperature' and 'Rainfall' columns.")
    else:
        # Generate time series based on data rows
        if 'Date' in data.columns:
            # Use Date column if available
            time_series = pd.to_datetime(data['Date'])
        else:
            # Generate synthetic dates
            current_date = pd.to_datetime('today').strftime('%Y-%m-%d')
            time_series = pd.date_range(current_date, periods=len(data), freq='D')

        # Extract series
        temperature_series = data['MinimumTemperature']
        precipitation_series = data['Rainfall']

        # Create the figure
        fig_dualaxis = go.Figure()

        # Add temperature data to the primary y-axis
        fig_dualaxis.add_trace(go.Scatter(
            x=time_series,
            y=temperature_series,
            mode='lines',
            name='Minimum Temperature',
            yaxis='y1'
        ))

        # Add rainfall data to the secondary y-axis
        fig_dualaxis.add_trace(go.Scatter(
            x=time_series,
            y=precipitation_series,
            mode='lines',
            name='Rainfall',
            yaxis='y2'
        ))

        # Update layout
        fig_dualaxis.update_layout(
            xaxis=dict(title='Time'),
            yaxis=dict(
                title='Minimum Temperature (°C)',
                side='left',
                showgrid=False
            ),
            yaxis2=dict(
                title='Rainfall (mm)',
                side='right',
                overlaying='y',
                showgrid=False
            ),
            legend=dict(orientation="h"),
            paper_bgcolor="#1e1e2e",
            font=dict(color="white")
        )

        # Display the chart
        st.plotly_chart(fig_dualaxis)



elif page == "Model Comparison":
    st.subheader("⚖ Model Comparison")

    models = {
        'Random Forest': clf,
        'Logistic Regression': LogisticRegression(),
        'Support Vector Machine': SVC(),
        'Decision Tree': DecisionTreeClassifier()
    }

    comparison_results = []
    for model_name, model in models.items():
        model.fit(X_train_imputed, y_train)
        y_pred_model = model.predict(X_test_imputed)
        accuracy_model = accuracy_score(y_test, y_pred_model)
        comparison_results.append({'Model': model_name, 'Accuracy': f"{accuracy_model:.2%}"})

    st.table(pd.DataFrame(comparison_results))
