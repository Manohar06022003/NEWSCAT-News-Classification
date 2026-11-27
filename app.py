
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import string

# Page configuration
st.set_page_config(
    page_title="News Classification",
    page_icon="📰",
    layout="wide"
)

# Text preprocessing (same as notebook)
def preprocess_text(text):
    data_out = pd.Series([text])
    data_out = data_out.str.replace('\n', ' ')
    punct = "\n\n" + string.punctuation
    data_out = data_out.str.translate(
        str.maketrans('', '', punct)
    )
    data_out = data_out.apply(lambda x: x.lower())
    return data_out.iloc[0]


# Load trained models
@st.cache_resource
def load_models():

    with open("models/mnb_model.pkl", "rb") as f:
        mnb, cv = pickle.load(f)

    with open("models/xgb_model.pkl", "rb") as f:
        xgb, tfidf, le = pickle.load(f)

    with open("models/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)

    cnn_model = tf.keras.models.load_model("models/best_model.keras")

    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return mnb, cv, xgb, tfidf, le, rf, cnn_model, tokenizer
# Load models
try:
    mnb, cv, xgb, tfidf, le, rf, cnn_model, tokenizer = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# Application heading
st.title("📰 News Classification System")
st.write(
    "Classify news articles into Business, Entertainment, "
    "Politics, Sport, or Tech using four Machine Learning models."
)

st.divider()

# Input section
st.subheader("Enter News Article")

text = st.text_area(
    "Paste your news article here:",
    height=200,
    placeholder="Enter news article text..."
)

if st.button("Predict News Category", type="primary"):

    if not text.strip():
        st.warning("Please enter a news article.")
    else:

        processed_text = preprocess_text(text)

        # Multinomial Naive Bayes
        mnb_pred = mnb.predict(
            cv.transform([processed_text])
        )[0]

        # XGBoost
        xgb_encoded = xgb.predict(
            tfidf.transform([processed_text])
        )[0]

        xgb_pred = le.inverse_transform(
            np.array([xgb_encoded])
        )[0]

        # Random Forest
        rf_encoded = rf.predict(
            tfidf.transform([processed_text])
        )[0]

        rf_pred = le.inverse_transform(
            np.array([rf_encoded])
        )[0]

        # CNN
        sequence = tokenizer.texts_to_sequences(
            [processed_text]
        )

        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=cnn_model.input_shape[1]
        )

        cnn_encoded = np.argmax(
            cnn_model.predict(padded, verbose=0),
            axis=1
        )[0]

        cnn_pred = le.inverse_transform(
            np.array([cnn_encoded])
        )[0]

        # Display results
        st.divider()
        st.subheader("Classification Results")

        results = {
            "Multinomial Naive Bayes": mnb_pred,
            "XGBoost": xgb_pred,
            "Random Forest": rf_pred,
            "CNN": cnn_pred
        }

        col1, col2 = st.columns(2)

        for i, (model_name, prediction) in enumerate(results.items()):
            with (col1 if i % 2 == 0 else col2):
                st.metric(model_name, str(prediction).upper())

        predictions = list(results.values())

        final_prediction = max(
            set(predictions),
            key=predictions.count
        )

        st.success(
            f"Final Majority Prediction: {final_prediction.upper()}"
        )

        st.caption(
            "The final result is based on majority voting among the four models."
        )

