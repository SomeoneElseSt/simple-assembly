import streamlit as st
import pandas as pd
import requests
import time
import io
import base64
import re

st.set_page_config(
    page_title="Simple Assembly [transcription]",
    page_icon="🎙️",
    layout="wide"
)

st.markdown("""
    <style>
    .stButton button { width: 100%; }
    .success-message { padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; color: #155724; margin: 1rem 0; }
    .info-message { padding: 1rem; border-radius: 0.5rem; background-color: #cce5ff; color: #004085; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("Configuration")
    with st.expander("API Settings", expanded=True):
        assemblyai_key = st.text_input(
            "AssemblyAI API Key",
            type="password"
        ).strip()

st.title("Simple Assembly [transcription]")
st.markdown("""
This dashboard helps you transcribe audio files from URLs in your CSV/XLSX files and download the results.
""")

st.info("Empty URL rows are skipped by default; processed row count may differ.")

uploaded_file = st.file_uploader(
    "Upload your CSV/XLSX file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        use_header = st.checkbox("Use first row as headers", value=True)

        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, header=0 if use_header else None)
        else:
            df = pd.read_csv(uploaded_file, header=0 if use_header else None)

        if not use_header:
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)

        original_dtypes = df.dtypes.to_dict()

        st.subheader("Column Configuration")

        create_new_column = st.checkbox("Create new transcription column")

        col1, col2 = st.columns(2)

        with col1:
            url_column = st.selectbox("Audio URL column", options=df.columns)

        with col2:
            if create_new_column:
                transcript_column = "transcription"
                if transcript_column not in df.columns:
                    df[transcript_column] = ""
                st.info("A new column 'transcription' will be created")
            else:
                transcript_column = st.selectbox("Transcription column", options=df.columns)

        col3, col4 = st.columns(2)

        with col3:
            use_id_matching = st.checkbox("Enable ID-based matching")

        with col4:
            if use_id_matching:
                id_column = st.selectbox("ID column", options=df.columns)

        filter_min_len = st.checkbox("Filter by minimum char length")

        if filter_min_len:
            chars_str = st.text_input("Minimum number of chars", value="200", key="chars")
            try:
                min_chars = int(chars_str)
            except ValueError:
                st.error("Please enter a whole number.")
                st.stop()

        stop_word_checkbox = st.checkbox("Exclude rows with stop-words (comma-separated)", key="stopword")
        stop_words = []
        if stop_word_checkbox:
            stop_words_str = st.text_input("Stop-words", value="", key="stopwords")
            stop_words = [w.strip().lower() for w in stop_words_str.split(",") if w.strip()]

        stop_phrase_checkbox = st.checkbox("Exclude rows with stop-phrases (one per line)", key="stophrase")
        stop_phrases = []
        if stop_phrase_checkbox:
            raw_phrases = st.text_area("Stop-phrases", height=120, key="stop_phrases")
            stop_phrases = [p.strip().lower() for p in raw_phrases.splitlines() if p.strip()]

        st.write("")

        if st.button("Start Transcription", use_container_width=True):
            if not assemblyai_key:
                st.error("Add your API key on the sidebar.")
            else:
                with st.spinner("Transcribing audio files..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    valid_df = df[df[url_column].notna() & (df[url_column].astype(str).str.strip() != "")]

                    if filter_min_len:
                        valid_df = valid_df[valid_df[transcript_column].astype(str).str.len() > min_chars]

                    patterns = []
                    if stop_words:
                        patterns.append(r'\b(' + '|'.join(map(re.escape, stop_words)) + r')\b')
                    if stop_phrases:
                        patterns.append('|'.join(map(re.escape, stop_phrases)))
                    if patterns:
                        combined_pattern = '|'.join(patterns)
                        valid_df = valid_df[~valid_df[transcript_column].astype(str).str.lower().str.contains(combined_pattern)]

                    audio_urls_to_process = []
                    processed_transcriptions = {}

                    if use_id_matching:
                        for idx, row in valid_df.iterrows():
                            audio_url = str(row[url_column]).strip()
                            if audio_url:
                                current_id = str(row[id_column])
                                if current_id not in processed_transcriptions:
                                    audio_urls_to_process.append((idx, audio_url, current_id))
                    else:
                        for idx, row in valid_df.iterrows():
                            audio_url = str(row[url_column]).strip()
                            if audio_url:
                                audio_urls_to_process.append((idx, audio_url, str(idx)))

                    if not audio_urls_to_process:
                        st.warning("No rows left after filtering.")
                        st.stop()

                    status_text.text(f"Found {len(audio_urls_to_process)} audio files to process")

                    for i, (idx, audio_url, current_id) in enumerate(audio_urls_to_process):
                        progress_bar.progress((i + 1) / len(audio_urls_to_process))
                        status_text.text(f"Processing {i + 1} of {len(audio_urls_to_process)}")

                        headers = {"authorization": assemblyai_key, "content-type": "application/json"}
                        try:
                            data = {
                                "audio_url": audio_url,
                                "language_code": "es",
                                "speaker_labels": True,
                                "speakers_expected": 2
                            }
                            response = requests.post("https://api.assemblyai.com/v2/transcript", json=data, headers=headers)
                            if response.status_code != 200:
                                raise Exception(response.text)

                            transcript_id = response.json().get("id")
                            if not transcript_id:
                                raise Exception("No transcript ID received")

                            endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
                            for _ in range(120):
                                poll_response = requests.get(endpoint, headers=headers)
                                if poll_response.status_code != 200:
                                    raise Exception(poll_response.text)
                                poll_data = poll_response.json()
                                status = poll_data.get("status")
                                if status == "completed":
                                    utterances = poll_data.get("utterances", [])
                                    if utterances:
                                        transcription = ""
                                        current_speaker = None
                                        for utterance in utterances:
                                            if not isinstance(utterance, dict) or "speaker" not in utterance or "text" not in utterance:
                                                continue
                                            if current_speaker != utterance["speaker"]:
                                                current_speaker = utterance["speaker"]
                                                speaker_label = "asistente" if current_speaker == "A" else "cliente"
                                                transcription += f"\n{speaker_label}: {utterance['text']}"
                                            else:
                                                transcription += f" {utterance['text']}"
                                        transcription = transcription.strip()
                                    else:
                                        transcription = poll_data.get("text", "No transcription available")
                                    break
                                if status == "error":
                                    raise Exception(poll_data.get("error", "Unknown error"))
                                time.sleep(3)
                            else:
                                raise Exception("Timeout during transcription")

                            processed_transcriptions[current_id] = transcription

                        except Exception as e:
                            st.error(f"Row {idx + 1}: {e}")
                            continue

                    for idx, row in df.iterrows():
                        current_id = str(row[id_column]) if use_id_matching else str(idx)
                        if current_id in processed_transcriptions:
                            df.at[idx, transcript_column] = processed_transcriptions[current_id]

                    for col, dtype in original_dtypes.items():
                        if col in df.columns:
                            df[col] = df[col].astype(dtype)

                    st.success("Transcription completed!")

                    st.subheader("Preview of Results")
                    st.dataframe(df)

                    st.subheader("Download Results")

                    csv_data = df.to_csv(index=False)
                    b64_csv = base64.b64encode(csv_data.encode()).decode()
                    st.markdown(
                        f'<a href="data:text/csv;base64,{b64_csv}" download="transcriptions.csv" class="stButton">Download CSV</a>',
                        unsafe_allow_html=True
                    )

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    excel_data = buffer.getvalue()
                    b64_excel = base64.b64encode(excel_data).decode()
                    st.markdown(
                        f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="transcriptions.xlsx" class="stButton">Download Excel</a>',
                        unsafe_allow_html=True
                    )

    except Exception as e:
        st.error(f"Error processing file: {e}")
