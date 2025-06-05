import streamlit as st
import pandas as pd
import requests
import time
import io
import base64


st.set_page_config(
    page_title="Simple Assembly [transcription]",
    page_icon="🎙️",
    layout="wide"
)
        
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        margin: 1rem 0;
    }
    .info-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #cce5ff;
        color: #004085;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("Configuration")
    with st.expander("API Settings", expanded=True):
        assemblyai_key = st.text_input(
            "AssemblyAI API Key",
            type="password",
            help="Enter your AssemblyAI API key. Get one at https://www.assemblyai.com/"
        ).strip()
        

st.title("Simple Assembly [transcription]")
st.markdown("""
This dashboard helps you transcribe audio files from URLs in your CSV/XLSX files and re-download them with better transcriptions.
Upload your file, select the relevant columns, and start transcribing!
""")

st.info("This will skip over empty rows by default. Hence, the # of processing rows may be smaller than the rows in the file")

uploaded_file = st.file_uploader(
    "Upload your CSV/XLSX file",
    type=["csv", "xlsx"],
    help="Your file should contain a column with audio URLs (e.g. S3 links)"
)

if uploaded_file is not None:
    try:
        use_header = st.checkbox(
            "Use first row as headers",
            value=True,
            help="Check this if your file has column headers in the first row"
        )

        if uploaded_file.name.endswith('.xlsx'):
            if use_header:
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, header=None)
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
        else:
            if use_header:
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, header=None)
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
        
        original_dtypes = df.dtypes.to_dict()
        
        st.subheader("Column Configuration")
        
        create_new_column = st.checkbox(
            "Create new transcription column",
            help="Check this if you want to create a new column for transcriptions"
        )

        col1, col2 = st.columns([1, 1])
        
        with col1: 
            url_column = st.selectbox(
                "Select the column containing audio URLs",
                options=df.columns,
                help="Choose the column that contains your audio file URLs"
            )
        
        with col2:
            if create_new_column:
                transcript_column = "transcription"
                st.info("A new column 'transcription' will be created")
            else:
                transcript_column = st.selectbox(
                    "Select the column for transcriptions",
                    options=df.columns,
                    help="Choose where to store the transcriptions"
                )

        st.write("")

        col3, col4 = st.columns([1, 1])
        
        with col3:
            use_id_matching = st.checkbox(
                "Enable ID-based matching",
                help="Use this if you want to match recordings using a specific ID column"
            )

        with col4:
            if use_id_matching:
                id_column = st.selectbox(
                    "Select ID column for matching",
                    options=df.columns,
                    help="Choose the column containing unique IDs"
                )

        # NEW checkbox to filter URLs shorter than 30 characters
        filter_min_len = st.checkbox(
            "Skip URLs shorter than 30 chars",
            help="Only send rows whose URL string length is > 30 characters"
        )

        st.write("")

        if st.button("Start Transcription", use_container_width=True):
            if not assemblyai_key:
                st.error("Oops. Forgot to add your API key on the sidebar?")
            else:
                with st.spinner("Transcribing audio files..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_rows = len(df)
                    processed_transcriptions = {}
                    audio_urls_to_process = []
                    ids_to_process = []

                    valid_df = df[df[url_column].notna() & (df[url_column].astype(str).str.strip() != "")]
                    
                    if filter_min_len:
                        chars_str = st.text_input(
                            "Enter the minimum number of chars a row must have to be processed.",
                            value="50",
                            key="chars",
                        )
                    
                        try:
                            min_chars = int(chars_str)
                        except ValueError:
                            st.error("Please enter a whole number.")
                            st.stop()
                    
                        valid_df = valid_df[valid_df[url_column].astype(str).str.len() > min_chars]

                    for idx, row in valid_df.iterrows():
                        audio_url = str(row[url_column]).strip()
                        if pd.isna(audio_url) or audio_url == "":
                            st.info(f"Skipping empty row {idx}")
                            continue
                    
                        if use_id_matching:
                            current_id = str(row[id_column])
                            if current_id not in processed_transcriptions:
                                audio_urls_to_process.append((idx, audio_url, current_id))
                                ids_to_process.append(current_id)
                        else:
                            audio_urls_to_process.append((idx, audio_url, str(idx)))


                    status_text.text(f"Found {len(audio_urls_to_process)} audio files to process")
                    
                    for i, (idx, audio_url, current_id) in enumerate(audio_urls_to_process):
                        progress = (i + 1) / len(audio_urls_to_process)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1} of {len(audio_urls_to_process)}")
                        
                        headers = {
                            "authorization": assemblyai_key,
                            "content-type": "application/json"
                        }
                        
                        try:
                            data = {
                                "audio_url": audio_url,
                                "language_code": "es",
                                "speaker_labels": True,
                                "speakers_expected": 2
                            }
                            
                            response = requests.post(
                                "https://api.assemblyai.com/v2/transcript",
                                json=data,
                                headers=headers
                            )
                            
                            if response.status_code != 200:
                                raise Exception(f"Error: {response.text}")
                                
                            transcript_id = response.json().get("id")
                            if not transcript_id:
                                raise Exception("No transcript ID received")
                                
                            endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
                            max_attempts = 120
                            attempt = 0
                            
                            while attempt < max_attempts:
                                attempt += 1
                                poll_response = requests.get(endpoint, headers=headers)
                                
                                if poll_response.status_code != 200:
                                    raise Exception(f"Polling failed: {poll_response.text}")
                                    
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
                                elif status == "error":
                                    raise Exception(f"Transcription error: {poll_data.get('error', 'Unknown error')}")
                                elif status in ["queued", "processing"]:
                                    time.sleep(3)
                                else:
                                    time.sleep(3)
                                    
                            if attempt >= max_attempts:
                                raise Exception(f"Timeout after {max_attempts} attempts")
                                
                            processed_transcriptions[current_id] = transcription
                            
                        except Exception as e:
                            st.error(f"Error processing row {idx + 1}: {str(e)}")
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
                        f'<a href="data:text/csv;base64,{b64_csv}" download="transcriptions.csv" '
                        'class="stButton">Download CSV</a>',
                        unsafe_allow_html=True
                    )
                    
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    excel_data = buffer.getvalue()
                    b64_excel = base64.b64encode(excel_data).decode()
                    st.markdown(
                        f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;'
                        f'base64,{b64_excel}" download="transcriptions.xlsx" class="stButton">'
                        'Download Excel</a>',
                        unsafe_allow_html=True
                    )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
