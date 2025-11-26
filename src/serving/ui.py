import streamlit as st
import requests
import pandas as pd


API_URL = "http://localhost:8000"


st.set_page_config(
    page_title="Air Quality Prediction", page_icon="🌤️", layout="wide"
)


st.title("🌤️ Air Quality Prediction System")
st.markdown("Predict air quality category based on sensor measurements")


with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value=API_URL)

    st.markdown("---")
    st.markdown("### Valid Stations")
    st.markdown(
        """
    - DKI1 (Bunderan HI)
    - DKI2 (Kelapa Gading)
    - DKI3 (Jagakarsa)
    - DKI4 (Lubang Buaya)
    - DKI5 (Kebon Jeruk) Jakarta Barat
    """
    )

    if st.button("Check API Health"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                if health.get("model_loaded"):
                    st.success("✅ API is healthy and model is loaded")
                else:
                    st.warning("⚠️ API is running but model is not loaded")
            else:
                st.error("❌ API is not responding correctly")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Cannot connect to API: {e}")


st.header("Enter Air Quality Measurements")

col1, col2 = st.columns(2)

with col1:
    stasiun = st.selectbox(
        "Station (Stasiun)",
        options=[
            "DKI1 (Bunderan HI)",
            "DKI2 (Kelapa Gading)",
            "DKI3 (Jagakarsa)",
            "DKI4 (Lubang Buaya)",
            "DKI5 (Kebon Jeruk) Jakarta Barat",
        ],
    )

    pm10 = st.number_input(
        "PM10 Concentration",
        min_value=-1.0,
        max_value=800.0,
        value=50.0,
        help="Particulate Matter 10 micrometers or less in diameter",
    )

    pm25 = st.number_input(
        "PM2.5 Concentration",
        min_value=-1.0,
        max_value=400.0,
        value=30.0,
        help="Particulate Matter 2.5 micrometers or less in diameter",
    )

    so2 = st.number_input(
        "SO2 Concentration",
        min_value=-1.0,
        max_value=500.0,
        value=15.0,
        help="Sulfur Dioxide",
    )

with col2:
    co = st.number_input(
        "CO Concentration",
        min_value=-1.0,
        max_value=100.0,
        value=5.0,
        help="Carbon Monoxide",
    )

    o3 = st.number_input(
        "O3 Concentration",
        min_value=-1.0,
        max_value=160.0,
        value=45.0,
        help="Ozone",
    )

    no2 = st.number_input(
        "NO2 Concentration",
        min_value=-1.0,
        max_value=100.0,
        value=25.0,
        help="Nitrogen Dioxide",
    )


st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])

with col_btn1:
    predict_button = st.button(
        "🔮 Predict", type="primary", use_container_width=True
    )

with col_btn2:
    clear_button = st.button("🔄 Clear", use_container_width=True)


if predict_button:
    payload = {
        "stasiun": stasiun,
        "pm10": pm10,
        "pm25": pm25,
        "so2": so2,
        "co": co,
        "o3": o3,
        "no2": no2,
    }

    with st.spinner("Making prediction..."):
        try:
            response = requests.post(
                f"{api_url}/predict", json=payload, timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction completed!")

                col_pred1, col_pred2 = st.columns(2)

                with col_pred1:
                    st.metric(
                        label="Predicted Category",
                        value=result.get("prediction", "N/A"),
                    )

                with col_pred2:
                    confidence = result.get("confidence")
                    if confidence is not None:
                        st.metric(
                            label="Confidence",
                            value=f"{confidence * 100:.2f}%",
                        )

                with st.expander("View Full Response"):
                    st.json(result)

                with st.expander("View Input Data"):
                    st.json(payload)

            elif response.status_code == 422:
                st.error("❌ Validation Error: Invalid input data")
                st.json(response.json())

            else:
                st.error(f"❌ Error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Cannot connect to API: {e}")
            st.info(f"Make sure the API is running at {api_url}")
            st.code("uvicorn src.serving.api:app --reload", language="bash")


if clear_button:
    st.rerun()


st.markdown("---")
st.markdown("### About")
st.info(
    """
This application predicts air quality categories based on pollutant concentrations.

**Features:**
- Real-time predictions via FastAPI backend
- Interactive input forms
- Confidence scores (when available)
- Health check monitoring

**Categories:**
- BAIK (Good)
- TIDAK BAIK (Not Good)
"""
)


with st.expander("📊 Sample Data"):
    sample_data = pd.DataFrame(
        {
            "Station": ["DKI1 (Bunderan HI)", "DKI2 (Kelapa Gading)"],
            "PM10": [50.0, 75.0],
            "PM2.5": [30.0, 45.0],
            "SO2": [15.0, 20.0],
            "CO": [5.0, 8.0],
            "O3": [45.0, 55.0],
            "NO2": [25.0, 35.0],
        }
    )
    st.dataframe(sample_data, use_container_width=True)
