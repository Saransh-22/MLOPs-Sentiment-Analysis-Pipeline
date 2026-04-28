
import os
import streamlit as st
import requests
import time

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="♋♓",
    layout="centered"
)

API_BASE_URL = os.getenv("SENTIMENT_API_URL", "http://localhost:8000")
API_URL = f"{API_BASE_URL}/predict"

st.title("Sentiment Analysis App")
st.markdown("---")
st.write("Enter text below and click **Predict** to analyze sentiment")

col1, col2 = st.columns([3, 1], gap="small")

with col1:
    user_input = st.text_area(
        label="Enter text to analyze",
        placeholder="Example: I love this product!",
        height=100
    )

if st.button("Predict", use_container_width=True):
    
    if not user_input.strip():
        st.error("Please enter some text to analyze")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": user_input},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction")
                    confidence = result.get("confidence", 0)
                    
                    st.markdown("---")
                    st.subheader("Result:")
                    
                    if prediction == "Positive":
                        st.success(f"Positive (Confidence: {confidence:.2%})")
                        st.balloons()
                    else:
                        st.error(f"Negative (Confidence: {confidence:.2%})")
                    
                    st.markdown("---")
                    st.write("**Analyzed Text:**")
                    st.info(user_input)
                    
                else:
                    st.error(f"API Error: Status code {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Cannot connect to API**\n\n"
                    "Make sure the FastAPI backend is running:\n"
                    "`docker run -p 8000:8000 sentiment-api`"
                )
            except requests.exceptions.Timeout:
                st.error("❌ API request timed out. Try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with st.sidebar:
    st.header("About")
    st.write(
        "This app analyzes sentiment of text using a "
        "machine learning model running on FastAPI."
    )
    
    st.subheader("API Status")
    try:
        response = requests.get(API_BASE_URL, timeout=5)
        st.success("✓ API is running")
    except:
        st.error("✗ API is not running")
    
    st.subheader("How to start API")
    st.code("docker run -p 8000:8000 sentiment-api", language="bash")
    
    st.subheader("Test Examples")
    st.markdown("""
    - "I love this product!" → Positive
    - "Terrible experience" → Negative
    - "It's okay" → Negative
    """)

st.markdown("---")
st.caption("Sentiment Analysis MLOps Project | Powered by FastAPI + Streamlit")
