import pandas as pd
import requests
import time
import io
import base64
import re
import json
from typing import List, Dict, Optional, Tuple
import anthropic
import os
import sys

# ----------------------
# CONFIGURATION FUNCTION
# ----------------------
def init(
    input_file: str,
    output_format: str = "csv",  # 'csv' or 'xlsx'
    assemblyai_key: str = "",
    pyannote_key: str = "",
    claude_key: str = "",
    url_column: str = "audio_url",
    transcript_column: str = "transcription",
    create_new_column: bool = True,
    only_assembly: bool = False,
    use_id_matching: bool = False,
    id_column: str = "id",
    filter_min_len: bool = False,
    min_chars: int = 200,
    stop_words: Optional[List[str]] = None,
    stop_phrases: Optional[List[str]] = None,
    language: str = "es"
):
    """
    All configuration for the run. Call this at the top of the file with your desired arguments.
    """
    if stop_words is None:
        stop_words = []
    if stop_phrases is None:
        stop_phrases = []
    return locals()

# ----------------------
# UTILITY FUNCTIONS
# ----------------------
def start_pyannote_diarization(audio_url: str, pyannote_token: str, num_speakers: int = 2) -> str:
    headers = {
        "Authorization": f"Bearer {pyannote_token}",
        "Content-Type": "application/json"
    }
    data = {"url": audio_url, "numSpeakers": num_speakers}
    response = requests.post("https://api.pyannote.ai/v1/diarize", json=data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Pyannote diarization failed: {response.text}")
    job_id = response.json().get("jobId")
    if not job_id:
        raise Exception("No jobId received from Pyannote")
    return job_id

def poll_pyannote_job(job_id: str, pyannote_token: str, max_attempts: int = 60) -> Dict:
    headers = {"Authorization": f"Bearer {pyannote_token}"}
    for i in range(max_attempts):
        time.sleep(5)
        response = requests.get(f"https://api.pyannote.ai/v1/jobs/{job_id}", headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error getting job status: {response.text}")
        status_data = response.json()
        if status_data.get("status") == "succeeded":
            return status_data.get("output", {})
        if status_data.get("status") == "failed":
            raise Exception(f"Pyannote job failed: {status_data.get('error')}")
    raise Exception("Pyannote diarization timeout")

def get_pyannote_diarization(audio_url: str, pyannote_token: str) -> Dict:
    try:
        job_id = start_pyannote_diarization(audio_url, pyannote_token)
        return poll_pyannote_job(job_id, pyannote_token)
    except Exception as e:
        print(f"[WARN] Pyannote diarization failed: {e}")
        return None

def transcribe_only_assemblyai(audio_url: str, assemblyai_key: str, language: str = "es") -> str:
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
    headers = {"authorization": assemblyai_key, "content-type": "application/json"}
    data = {
        "audio_url": audio_url,
        "language_code": language,
        "punctuate": True,
        "format_text": True,
        "speaker_labels": True,
        "speakers_expected": 2
    }
    response = requests.post("https://api.assemblyai.com/v2/transcript", json=data, headers=headers)
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
    if not segments:
        return "No transcription available"
    valid_segments = [seg for seg in segments if seg.get('text', '').strip()]
    if not valid_segments:
        return "No transcription available"
    client = anthropic.Anthropic(api_key=claude_key)
    transcript = "\n".join([f"{seg['speaker']}: {seg['text'].strip()}" for seg in valid_segments])

    prompt = f"""

    You are a smart and precise assistant. Your task is to relabel speaker tags in transcripts of conversations between an AI assistant and a human user.

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
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )
    final_text = response.content[0].text
    return final_text

def transcribe_with_diarization(audio_url: str, assemblyai_key: str, pyannote_key: str, claude_key: str, language: str = "es") -> str:
    try:
        diarization = get_pyannote_diarization(audio_url, pyannote_key)
        transcription = get_assemblyai_transcription(audio_url, assemblyai_key, language)
        final_transcription_text = "No transcription available"
        if diarization and diarization.get("diarization"):
            aligned_segments = align_transcription_with_diarization(transcription, diarization)
            merged_segments = merge_speaker_segments(aligned_segments, max_gap=0.3)
            final_transcription_text = relabel_speakers_with_claude(merged_segments, claude_key)
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
                merged_segments = merge_speaker_segments(segments)
                final_transcription_text = relabel_speakers_with_claude(merged_segments, claude_key)
            else:
                final_transcription_text = transcription.get("text", "No transcription available")
        return final_transcription_text
    except Exception as e:
        raise Exception(f"Transcription error: {e}")

# ----------------------
# MAIN PROCESSING LOGIC
# ----------------------
def main(config):
    input_file = config["input_file"]
    output_format = config["output_format"].lower()
    assemblyai_key = config["assemblyai_key"]
    pyannote_key = config["pyannote_key"]
    claude_key = config["claude_key"]
    url_column = config["url_column"]
    transcript_column = config["transcript_column"]
    create_new_column = config["create_new_column"]
    only_assembly = config["only_assembly"]
    use_id_matching = config["use_id_matching"]
    id_column = config["id_column"]
    filter_min_len = config["filter_min_len"]
    min_chars = config["min_chars"]
    stop_words = config["stop_words"]
    stop_phrases = config["stop_phrases"]
    language = config["language"]

    # Read input file
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    original_dtypes = df.dtypes.to_dict()

    if create_new_column and transcript_column not in df.columns:
        df[transcript_column] = ""

    # Filtering
    valid_df = df[df[url_column].notna() & (df[url_column].astype(str).str.strip() != "")]
    if filter_min_len:
        valid_df = valid_df[valid_df[transcript_column].astype(str).str.len() > min_chars]
    patterns = []
    if stop_words:
        patterns.append(r'\\b(' + '|'.join(map(re.escape, stop_words)) + r')\\b')
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
        print("[WARN] No rows left after filtering.")
        return

    print(f"Found {len(audio_urls_to_process)} audio files to process.")

    for i, (idx, audio_url, current_id) in enumerate(audio_urls_to_process):
        print(f"Processing {i + 1} of {len(audio_urls_to_process)} (row {idx + 1})...")
        try:
            if only_assembly:
                transcription = transcribe_only_assemblyai(
                    audio_url=audio_url,
                    assemblyai_key=assemblyai_key,
                    language=language
                )
            else:
                transcription = transcribe_with_diarization(
                    audio_url=audio_url,
                    assemblyai_key=assemblyai_key,
                    pyannote_key=pyannote_key,
                    claude_key=claude_key,
                    language=language
                )
            processed_transcriptions[current_id] = transcription
        except Exception as e:
            print(f"[ERROR] Row {idx + 1}: {e}")
            continue

    for idx, row in df.iterrows():
        current_id = str(row[id_column]) if use_id_matching else str(idx)
        if current_id in processed_transcriptions:
            debug_trans = processed_transcriptions[current_id]
            df.at[idx, transcript_column] = debug_trans

    for col, dtype in original_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    df[transcript_column] = df[transcript_column].astype(str).str.replace(r'[\n\r\t]+', ' ', regex=True)
    base, ext = os.path.splitext(os.path.basename(input_file))
    output_file = f"{base}_transcribed.{output_format}"
    if output_format == "csv":
        df.to_csv(output_file, index=False)
        print(f"[DONE] Output written to {output_file}")
    elif output_format == "xlsx":
        df.to_excel(output_file, index=False)
        print(f"[DONE] Output written to {output_file}")
    else:
        print(f"[ERROR] Unknown output format: {output_format}")

if __name__ == "__main__":
    config = init(
        input_file="transcribe.csv",  # or "input.csv"
        output_format="csv",      # or "xlsx"
        assemblyai_key=""" ",
        pyannote_key="sk_d937f62445fb4a378cd2373502c1f4d8",
        claude_key="""",
        url_column="grabacion_url",
        transcript_column="new_transcripcion",
        create_new_column=False,
        only_assembly=False, # if true, only assemblyai will be used, so no diarization 
        use_id_matching=True,
        id_column="id",
        filter_min_len=False,
        min_chars=200,
        stop_words=[],
        stop_phrases=[],
        language="es" # language of the audio files
    )
    main(config) 