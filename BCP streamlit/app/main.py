import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# ----------------------------------------------------------------------
# Data & model loading
# Cached so Streamlit doesn't re-read the CSV or unpickle the model on
# every single slider drag (Streamlit reruns the whole script on every
# widget interaction, so this matters a lot for responsiveness).
# ----------------------------------------------------------------------

@st.cache_data
def get_clean_data():
  data = pd.read_csv("data/data.csv")
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
  return data


@st.cache_resource
def load_model():
  with open("model/model.pkl", "rb") as f:
    return pickle.load(f)


# Full 30 features, in the exact column order the model was trained on.
# Needed to build a correctly-ordered input vector for prediction.
ALL_FEATURES = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"),
    ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"),
    ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"),
    ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"),
    ("Concave points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"),
    ("Fractal dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal dimension (worst)", "fractal_dimension_worst"),
]

# The 15 features that get an actual slider, listed in DESCENDING
# feature_importances_ order (from the trained model) so the sidebar
# shows the most influential measurement first.
# NOTE: if your printed feature_importances_ output ranked things
# differently, update this list to match.
TOP_15_ORDERED = [
    ("Area (worst)", "area_worst"),
    ("Concave points (worst)", "concave points_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Concave points (mean)", "concave points_mean"),
    ("Radius (worst)", "radius_worst"),
    ("Concavity (mean)", "concavity_mean"),
    ("Radius (mean)", "radius_mean"),
    ("Area (mean)", "area_mean"),
    ("Concavity (worst)", "concavity_worst"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (se)", "area_se"),
    ("Texture (worst)", "texture_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Radius (se)", "radius_se"),
    ("Compactness (mean)", "compactness_mean"),
]
TOP_15_KEYS = {key for _, key in TOP_15_ORDERED}


def add_sidebar(data: pd.DataFrame) -> dict:
  """Render the sidebar sliders and return a full 30-feature input dict,
  ordered to match the model's training columns."""
  st.sidebar.header("Cell Nuclei Measurements")
  st.sidebar.caption(
    "Adjust the 15 measurements that most influence the prediction. "
    "The remaining measurements are automatically set to typical values."
  )

  input_dict = {}

  # Sliders first, in descending importance order.
  for label, key in TOP_15_ORDERED:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )

  # Remaining low-importance features: no slider, dataset mean instead —
  # the model still receives all 30 features it was trained on.
  for _, key in ALL_FEATURES:
    if key not in TOP_15_KEYS:
      input_dict[key] = float(data[key].mean())

  # Re-order to match the model's training column order — required since
  # the input array is built from dict values by position.
  return {key: input_dict[key] for _, key in ALL_FEATURES}


def get_scaled_values(input_dict: dict, data: pd.DataFrame) -> dict:
  """Min-max scale each value to [0, 1] against the full dataset's range,
  purely for the radar chart display (not used by the model itself)."""
  X = data.drop(['diagnosis'], axis=1)
  scaled_dict = {}
  for key, value in input_dict.items():
    min_val, max_val = X[key].min(), X[key].max()
    scaled_dict[key] = (value - min_val) / (max_val - min_val)
  return scaled_dict


def get_radar_chart(input_data: dict, data: pd.DataFrame) -> go.Figure:
  scaled = get_scaled_values(input_data, data)

  categories = [label for label, _ in TOP_15_ORDERED]
  r_values = [scaled[key] for _, key in TOP_15_ORDERED]

  fig = go.Figure()
  fig.add_trace(go.Scatterpolar(
    r=r_values,
    theta=categories,
    fill='toself',
    name='Input Value'
  ))
  fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    margin=dict(l=40, r=40, t=40, b=40),
  )
  return fig


def add_predictions(input_data: dict, model) -> None:
  input_array = np.array(list(input_data.values())).reshape(1, -1)

  # No scaling needed here — RandomForest is scale-invariant.
  # Call predict_proba once and derive the class prediction from it,
  # instead of calling predict() and predict_proba() separately
  # (that was doing the same forward pass through the forest twice).
  probabilities = model.predict_proba(input_array)[0]
  benign_prob, malignant_prob = probabilities[0], probabilities[1]
  is_malignant = malignant_prob > benign_prob

  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")

  if is_malignant:
    st.markdown("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
  else:
    st.markdown("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)

  st.metric("Probability of being benign", f"{benign_prob:.1%}")
  st.metric("Probability of being malignant", f"{malignant_prob:.1%}")

  st.caption(
    "This app can assist medical professionals in making a diagnosis, "
    "but should not be used as a substitute for a professional diagnosis."
  )


def load_css(path: str) -> None:
  """Load an optional stylesheet — app still works fine without it."""
  try:
    with open(path) as f:
      st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
  except FileNotFoundError:
    pass


def main():
  st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )

  load_css("assets/style.css")

  data = get_clean_data()
  model = load_model()

  input_data = add_sidebar(data)

  with st.container():
    st.title("Breast Cancer Predictor")
    st.write(
      "Connect this app to your cytology lab to help diagnose breast cancer "
      "from a tissue sample. This app predicts, using a machine learning "
      "model, whether a breast mass is benign or malignant based on the "
      "measurements it receives. You can also update the measurements by "
      "hand using the sliders in the sidebar."
    )

  col1, col2 = st.columns([4, 1])

  with col1:
    st.plotly_chart(get_radar_chart(input_data, data), use_container_width=True)
  with col2:
    add_predictions(input_data, model)


if __name__ == '__main__':
  main()