import streamlit as st
import pandas as pd
import requests
import time
import io
import base64
import re
import json
from typing import List, Dict, Optional, Tuple
import anthropic

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
        pyannote_key = st.text_input(
            "Pyannote API Key",
            type="password"
        ).strip()
        claude_key = st.text_input(
            "Claude API Key", 
            type="password"
        ).strip()

st.title("Simple Assembly [transcription]")
st.markdown("""
This dashboard helps you transcribe audio files from URLs in your CSV/XLSX files and download the results.
""")

# Pyannote functions
def start_pyannote_diarization(audio_url: str, pyannote_token: str, num_speakers: int = 2) -> str:
    """Start Pyannote diarization job and return job ID"""
    headers = {
        "Authorization": f"Bearer {pyannote_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "url": audio_url,
        "numSpeakers": num_speakers
    }

    response = requests.post(
        "https://api.pyannote.ai/v1/diarize",
        json=data,
        headers=headers
    )
    
    if response.status_code != 200:
        raise Exception(f"Pyannote diarization failed: {response.text}")
    
    job_id = response.json().get("jobId")
    if not job_id:
        raise Exception("No jobId received from Pyannote")
    
    return job_id

def poll_pyannote_job(job_id: str, pyannote_token: str, max_attempts: int = 60) -> Dict:
    """Poll Pyannote job until complete"""
    headers = {"Authorization": f"Bearer {pyannote_token}"}
    
    for i in range(max_attempts):
        time.sleep(5)
        
        response = requests.get(
            f"https://api.pyannote.ai/v1/jobs/{job_id}",
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Error getting job status: {response.text}")
        
        status_data = response.json()
        
        if status_data.get("status") == "succeeded":
            return status_data.get("output", {})
        
        if status_data.get("status") == "failed":
            raise Exception(f"Pyannote job failed: {status_data.get('error')}")
    
    raise Exception("Pyannote diarization timeout")

def get_pyannote_diarization(audio_url: str, pyannote_token: str) -> Dict:
    """Get diarization results from Pyannote"""
    try:
        job_id = start_pyannote_diarization(audio_url, pyannote_token)
        return poll_pyannote_job(job_id, pyannote_token)
    except Exception as e:
        st.warning(f"Pyannote diarization failed: {e}")
        return None

# Assembly AI functions with diarization alignment
@st.cache_data(show_spinner="Transcribing audio with AssemblyAI, this may take a while...", persist=True)
def transcribe_only_assemblyai(audio_url: str, assemblyai_key: str, language: str = "es") -> str:
    """Performs transcription using only AssemblyAI, with basic speaker labeling"""
    headers = {"authorization": assemblyai_key, "content-type": "application/json"}
    data = {
        "audio_url": audio_url,
        "language_code": language,
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
                transcription_text = ""
                current_speaker = None
                for utterance in utterances:
                    if not isinstance(utterance, dict) or "speaker" not in utterance or "text" not in utterance:
                        continue
                    if current_speaker != utterance["speaker"]:
                        current_speaker = utterance["speaker"]
                        speaker_label = "asistente" if current_speaker == "A" else "cliente"
                        transcription_text += f"\n{speaker_label}: {utterance['text']}"
                    else:
                        transcription_text += f" {utterance['text']}"
                return transcription_text.strip()
            else:
                return poll_data.get("text", "No transcription available")
        if status == "error":
            raise Exception(poll_data.get("error", "Unknown error"))
        time.sleep(3)
    raise Exception("Timeout during transcription")

def get_assemblyai_transcription(audio_url: str, assemblyai_key: str, language: str = "es") -> Dict:
    """Get transcription from AssemblyAI with word-level timestamps"""
    headers = {
        "authorization": assemblyai_key,
        "content-type": "application/json"
    }
    
    data = {
        "audio_url": audio_url,
        "language_code": language,
        "punctuate": True,
        "format_text": True,
        "speaker_labels": True,
        "speakers_expected": 2
    }
    
    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=data,
        headers=headers
    )
    
    if response.status_code != 200:
        raise Exception(f"AssemblyAI error: {response.text}")
    
    transcript_id = response.json().get("id")
    if not transcript_id:
        raise Exception("No transcript ID received")
    
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    
    for i in range(120):
        poll_response = requests.get(endpoint, headers=headers)

        if poll_response.status_code != 200:
            raise Exception(poll_response.text)
        
        poll_data = poll_response.json()
        status = poll_data.get("status")
        
        if status == "completed":
            return poll_data
        
        if status == "error":
            raise Exception(poll_data.get("error", "Unknown error"))
        
        time.sleep(3)
    
    raise Exception("Timeout during transcription")

def align_transcription_with_diarization(transcription: Dict, diarization: Dict) -> List[Dict]:
    """Align AssemblyAI transcription with Pyannote diarization segments"""
    segments = diarization.get("diarization", diarization.get("segments", []))
    words = transcription.get("words", [])
    
    if not words:
        return [{
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": ""
        } for seg in segments]
    
    aligned_segments = []
    
    for seg in segments:
        words_in_segment = []
        for word in words:
            word_start = word.get("start", 0) / 1000
            word_end = word.get("end", 0) / 1000
            
            if word_end > seg["start"] and word_start < seg["end"]:
                words_in_segment.append(word["text"])
        
        aligned_segments.append({
            "speaker": seg["speaker"],
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": " ".join(words_in_segment)
        })
    
    return aligned_segments

def merge_speaker_segments(segments: List[Dict], max_gap: float = 0.3) -> List[Dict]:
    """Merge consecutive segments from the same speaker"""
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for seg in segments[1:]:
        if seg["speaker"] == current["speaker"] and seg["start"] - current["end"] <= max_gap:
            current["end"] = seg["end"]
            current["text"] = f"{current['text']} {seg['text']}".strip()
        else:
            merged.append(current)
            current = seg.copy()
    
    merged.append(current)
    return merged

def relabel_speakers_with_claude(segments: List[Dict], claude_key: str) -> str:
    """Use Claude to relabel speakers as AI and USER"""
    if not segments:
        return "No transcription available"
    
    valid_segments = [seg for seg in segments if seg.get('text', '').strip()]
    if not valid_segments:
        return "No transcription available"
    
    client = anthropic.Anthropic(api_key=claude_key)
    
    transcript = "\n".join([f"{seg['speaker']}: {seg['text'].strip()}" for seg in valid_segments])
    
    prompt = f"""  You are a smart and precise assistant. Your task is to relabel speaker tags in transcripts of conversations between an AI assistant and a human user.

    Here is the transcript:

    {transcript}

    The transcript you receive will have inconsistent speaker labels (e.g., SPEAKER_00, SPEAKER_01). Your job is to:
    • Replace each speaker label with either:
        • AI: — for the virtual assistant
        • USER: — for the human user
    • The AI always speaks first in every conversation.
    • You will apply slight corrections related to issues of the transcription engine. Specifically: 
        • These transcriptions are from calls in Spanish. If any part of the transcript is on English, apply literal translations to Spanish that make common sense. For example, if you see 'Hello, Hello' this is wrong and should be translated to 'Hola, Hola'. This applies to just about any language other than Spanish that appears in calls.  
        • These transcriptions might come with chopped off parts. For example, parts of the transcript where the assistant is speaking, such as asking for payment promise, should generally not be cut off, because the assistant asks this continously. Wherever logical (wherever it is absolutetly obvious that the assistant or user actually did complete their sentence, but the dieratization engine cut them off), you should change the transcript to have coherent dieratization (so assigning the coherent sentence to the right speaker), without changing any of the actual content. It is more like realignment based on where it is obvious that the assistant or user actually did complete/continue their sentence at some point.
    • Output only the relabeled conversation as plain text, with no extra comments, headers, or formatting.

    Here is an example of the desired output:

    AI: Hello, how can I help you today?
    USER: I need help with my account.
    AI: Sure, I'd be happy to assist.
    USER: I'm having trouble logging in.

    When given a new transcript, return only the full conversation with corrected speaker labels. Do not explain anything—just output the result. Do not include new line markers such as "/n" or "+" on new lines, clean these.
"""
    
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    final_text = response.content[0].text
    final_text = re.sub(r'(\\n|\n|\s*\+\s*)', ' ', final_text).strip()
    return final_text

@st.cache_data(show_spinner="Transcribing audio, this may take a while...", persist=True)
def transcribe_with_diarization(audio_url: str, assemblyai_key: str, pyannote_key: str, claude_key: str, language: str = "es") -> str:
    """Full transcription pipeline with Pyannote diarization and Claude relabeling"""
    
    start_time_total = time.time()

    try:
        # Run Pyannote diarization and AssemblyAI transcription in parallel
        
        start_time_pyannote = time.time()
        diarization = get_pyannote_diarization(audio_url, pyannote_key)
        end_time_pyannote = time.time()
        pyannote_latency = end_time_pyannote - start_time_pyannote

        start_time_assemblyai = time.time()
        transcription = get_assemblyai_transcription(audio_url, assemblyai_key, language)
        end_time_assemblyai = time.time()
        assemblyai_latency = end_time_assemblyai - start_time_assemblyai
        
        final_transcription_text = "No transcription available"
        
        if diarization and diarization.get("diarization"):
            
            start_time_alignment = time.time()
            aligned_segments = align_transcription_with_diarization(transcription, diarization)
            end_time_alignment = time.time()
            alignment_latency = end_time_alignment - start_time_alignment
            
            start_time_merge = time.time()
            merged_segments = merge_speaker_segments(aligned_segments, max_gap=0.3)
            end_time_merge = time.time()
            merge_latency = end_time_merge - start_time_merge
            
            start_time_claude = time.time()
            final_transcription_text = relabel_speakers_with_claude(merged_segments, claude_key)
            end_time_claude = time.time()
            claude_latency = end_time_claude - start_time_claude

        else:
            utterances = transcription.get("utterances", [])
            if utterances:
                segments = []
                for utterance in utterances:
                    segments.append({
                        "speaker": f"Speaker_{utterance['speaker']}",
                        "start": utterance["start"] / 1000,
                        "end": utterance["end"] / 1000,
                        "text": utterance["text"]
                    })
                
                start_time_merge_fallback = time.time()
                merged_segments = merge_speaker_segments(segments)
                end_time_merge_fallback = time.time()
                merge_fallback_latency = end_time_merge_fallback - start_time_merge_fallback

                start_time_claude_fallback = time.time()
                final_transcription_text = relabel_speakers_with_claude(merged_segments, claude_key)
                end_time_claude_fallback = time.time()
                claude_fallback_latency = end_time_claude_fallback - start_time_claude_fallback

            else:
                final_transcription_text = transcription.get("text", "No transcription available")
        
        end_time_total = time.time()
        total_pipeline_latency = end_time_total - start_time_total
        
        return final_transcription_text
    
    except Exception as e:
        raise Exception(f"Transcription error: {e}")

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

        only_assembly_checkbox = st.checkbox("Only use AssemblyAI (skip Diarization/Claude)")

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
                st.error("Add your AssemblyAI API key on the sidebar.")
            elif not only_assembly_checkbox and not pyannote_key:
                st.error("Add your Pyannote API key on the sidebar, or select 'Only use AssemblyAI'.")
            elif not only_assembly_checkbox and not claude_key:
                st.error("Add your Claude API key on the sidebar, or select 'Only use AssemblyAI'.")
            else:
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

                status_text.markdown(f"**Found {len(audio_urls_to_process)} audio files to process**")

                for i, (idx, audio_url, current_id) in enumerate(audio_urls_to_process):
                    progress_bar.progress((i + 1) / len(audio_urls_to_process))
                    status_text.markdown(f"Processing {i + 1} of {len(audio_urls_to_process)}")

                    try:
                        if only_assembly_checkbox:
                            # Use only AssemblyAI
                            transcription = transcribe_only_assemblyai(
                                audio_url=audio_url,
                                assemblyai_key=assemblyai_key,
                                language="es"
                            )
                        else:
                            # Use full pipeline with Pyannote and Claude
                            transcription = transcribe_with_diarization(
                                audio_url=audio_url,
                                assemblyai_key=assemblyai_key,
                                pyannote_key=pyannote_key,
                                claude_key=claude_key,
                                language="es"
                            )
                        
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
