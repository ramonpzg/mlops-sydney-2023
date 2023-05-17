
from transformers import AutoModel, AutoFeatureExtractor
from qdrant_client import QdrantClient
from pedalboard.io import AudioFile
import streamlit as st
import torch

st.title("Music Recommendation App")
st.markdown("Upload your favorite songs and get a list of recommendations from our database of music.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained('facebook/wav2vec2-base').to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
client = QdrantClient("localhost", port=6333)

music_file = st.file_uploader(label="📀 Music file 🎸",)

if music_file:
    st.audio(music_file)

    with AudioFile(music_file) as f:
        a_song = f.read(f.frames)[0]

    inputs = feature_extractor(
        a_song, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt",
        padding=True, return_attention_mask=True, max_length=16_000, truncation=True
    ).to(device)

    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    vectr = last_hidden_state.mean(dim=1).cpu().numpy()[0]

    st.markdown("## Real Recommendations")
    results = client.search(collection_name="music_recsys", query_vector=vectr, limit=4)
    col1, col2 = st.columns(2)

    with col1:
        st.header(f"Genre: {results[0].payload['genre']}")
        st.subheader(f"Artist: {results[0].payload['artist']}")
        st.audio(results[0].payload["url_song"])
        
        st.header(f"Genre: {results[1].payload['genre']}")
        st.subheader(f"Artist: {results[1].payload['artist']}")
        st.audio(results[1].payload["url_song"])

    with col2:
        st.header(f"Genre: {results[2].payload['genre']}")
        st.subheader(f"Artist: {results[2].payload['artist']}")
        st.audio(results[2].payload["url_song"])
        
        st.header(f"Genre: {results[3].payload['genre']}")
        st.subheader(f"Artist: {results[3].payload['artist']}")
        st.audio(results[3].payload["url_song"])
