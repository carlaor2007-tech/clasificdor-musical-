import streamlit as st
import numpy as np
import librosa
import joblib
import os
import tempfile

st.set_page_config(page_title="Clasificador de Género Musical", page_icon="🎵", layout="centered")

@st.cache_resource
def cargar_modelo():
    model        = joblib.load("modelo_sklearn.pkl")
    genre_names  = joblib.load("genre_names.pkl")
    scaler       = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, genre_names, scaler, feature_cols

try:
    model, genre_names, scaler, feature_cols = cargar_modelo()
    modelo_ok = True
except Exception as e:
    modelo_ok = False
    st.error(f"Error cargando el modelo: {e}")

def extraer_features(path):
    y, sr = librosa.load(path, mono=True)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    harm  = librosa.effects.harmonic(y)
    perc  = librosa.effects.percussive(y)
    f = {
        "chroma_stft_mean":        float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr))),
        "chroma_stft_var":         float(np.var(librosa.feature.chroma_stft(y=y, sr=sr))),
        "rms_mean":                float(np.mean(librosa.feature.rms(y=y))),
        "rms_var":                 float(np.var(librosa.feature.rms(y=y))),
        "spectral_centroid_mean":  float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "spectral_centroid_var":   float(np.var(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "spectral_bandwidth_mean": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "spectral_bandwidth_var":  float(np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "rolloff_mean":            float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "rolloff_var":             float(np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "zero_crossing_rate_mean": float(np.mean(librosa.feature.zero_crossing_rate(y))),
        "zero_crossing_rate_var":  float(np.var(librosa.feature.zero_crossing_rate(y))),
        "harmony_mean":            float(np.mean(harm)),
        "harmony_var":             float(np.var(harm)),
        "perceptr_mean":           float(np.mean(perc)),
        "perceptr_var":            float(np.var(perc)),
        "tempo":                   float(librosa.beat.tempo(y=y, sr=sr)[0]),
    }
    for i in range(1, 21):
        f[f"mfcc{i}_mean"] = float(np.mean(mfcc[i-1]))
        f[f"mfcc{i}_var"]  = float(np.var(mfcc[i-1]))
    vec = np.array([f.get(c, 0.0) for c in feature_cols], dtype=np.float32).reshape(1, -1)
    return scaler.transform(vec)

st.title("🎵 Sistema de Identificación del Género Musical")
st.markdown("""
Sube una canción en formato **WAV** y la IA predecirá su género musical.

**Géneros:** blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock
""")
st.divider()

audio_file = st.file_uploader("Sube tu canción (.wav)", type=["wav"])

if audio_file:
    st.audio(audio_file, format="audio/wav")
    if st.button("🎯 Clasificar género", type="primary", use_container_width=True):
        if not modelo_ok:
            st.error("Modelo no cargado.")
        else:
            with st.spinner("Analizando..."):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    features = extraer_features(tmp_path)
                    os.unlink(tmp_path)

                    probs  = model.predict_proba(features)[0]
                    idx    = int(np.argmax(probs))
                    genero = genre_names[idx]
                    conf   = float(probs[idx]) * 100

                    st.success(f"### Género predicho: **{genero.upper()}**")
                    st.metric("Confianza", f"{conf:.1f}%")
                    st.divider()
                    st.subheader("Probabilidades por género")
                    for i in np.argsort(probs)[::-1]:
                        g = genre_names[i]; p = float(probs[i])
                        ico = "🟢" if i == idx else "⚪"
                        st.progress(p, text=f"{ico} {g}: {p*100:.1f}%")
                except Exception as e:
                    st.error(f"Error: {e}")

st.divider()
st.caption("Dataset: GTZAN · Modelo: Random Forest · UPV 2026")
