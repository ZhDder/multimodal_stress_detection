import os
import time
import shutil
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash-lite"
VIDEO_DATA_DIR = "../video/data/5s5fps_cropped"
AUDIO_DATA_DIR = "../audio/data/5s_0overlap_original"
OUT_PATH = "./results/Gemini_stress_prediction_results.csv"
FAILURE_PATH = "./results/Gemini_stress_prediction_failures.csv"

# Only these activities (spoken/audio-available subset) will be included in the run.
AUDIO_ACTIVITIES = ["Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"]

# If True, samples without matching audio files are skipped.
# If False, the model still runs on images only and audio_included=False is recorded
# (useful for a frames-only ablation on the same subset).
REQUIRE_AUDIO = True

# Free tier rate limits (~15 RPM for Free Tier Flash-Lite).
DELAY_BETWEEN_CALLS = 5.0
MAX_RETRIES = 3
TRANSIENT_API_CODES = {429, 500, 502, 503, 504}

OUTPUT_COLUMNS = [
    "sample_id",
    "subject",
    "activity",
    "window_id",
    "stress_level",
    "confidence",
    "reasoning",
    "binary_stress",
    "audio_included",
]

FAILURE_COLUMNS = [
    "timestamp",
    "sample_id",
    "subject",
    "activity",
    "window_id",
    "error_type",
    "error_message",
    "attempts",
]

system_prompt = """
You are an expert in behavioral psychology and affective computing analyzing visual and vocal signs of stress.

RULES OF EVALUATION:
1. You will be provided with a SEQUENCE OF FRAMES and, when available, an AUDIO TRACK of a subject performing a task.
2. Your goal is to assess their overall stress level based on visible and audible behavioral cues.

VISUAL CUES TO ANALYZE:
- head_movement: visible head motion, fidgeting, instability, repeated shifts
- blink_behavior: unusual blinking frequency or eye closure patterns visible in the frames
- eye_aperture: narrowed, widened, tense, or unusually open eyes
- gaze_direction: gaze aversion, downward gaze, looking away from task/camera
- gaze_stability: unstable gaze, frequent visual shifts, reduced fixation
- eyebrow_movement: brow furrowing, raised brows, eyebrow tension
- mouth_shape: tense mouth shape, flattened mouth, grimacing, tight lips
- lip_pressing: pressed lips or compressed lips
- lip_corner_tension: pulled, depressed, or tense lip corners
- jaw_mouth_tension: jaw clenching, mouth rigidity, visible lower-face tension

VOCAL CUES TO ANALYZE, IF AUDIO IS PROVIDED:
- pitch_instability: noticeable fluctuations, abnormally high pitch (F0), or voice cracking
- speech_rate: rapid, rushed speech, or unusually slow speech with unnatural pauses
- vocal_energy: sudden increases in volume, strained projection, or unusually quiet/timid responses
- voice_quality: breathiness, harshness, or noticeable vocal tremor
- disfluencies: stuttering, frequent filler words ("um", "uh"), or trailing off
- non_linguistic_sounds: heavy breathing, sighing, nervous laughter, or frequent throat clearing

SCORING:
- stress_level: An integer from 0 to 10 based on the intensity of the deviations. (0 = no stress, 10 = extreme stress)
- confidence: A float from 0.0 to 1.0 reflecting the clarity and consistency of the cues.
- reasoning: Must be under 2 short sentences explaining the explicit visual and, if available, vocal evidence.
"""

user_prompt = "Analyze the provided image sequence and, if present, the audio track. Output the JSON evaluation."


class StressPrediction(BaseModel):
    stress_level: int = Field(ge=0, le=10, description="Integer 0-10 representing the stress level.")
    confidence: float = Field(ge=0.0, le=1.0, description="Float 0.0-1.0 representing confidence in the assessment.")
    reasoning: str = Field(description="Under 2 short sentences explaining the observable evidence.")

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_sample_id(subject: str, activity: str, window_id: Any) -> str:
    """Stable unique key per window. Matches the project's subject_activity convention."""
    return f"{subject}_{activity}_{window_id}"


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_existing_results(out_path: str) -> Tuple[Set[str], bool]:
    """
    Read the existing CSV and return:
      - processed_keys: sample_id values that should be skipped on resume
      - write_headers: whether the next append should include a header row
    """
    ensure_parent_dir(out_path)

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return set(), True

    try:
        existing_df = pd.read_csv(out_path, engine="python", on_bad_lines="skip")
    except Exception as e:
        backup_path = f"{out_path}.corrupt_{timestamp_for_filename()}.bak"
        shutil.move(out_path, backup_path)
        log.warning(f"Could not read existing CSV for resuming: {e}")
        log.warning(f"Moved unreadable CSV to {backup_path}; starting a fresh results file.")
        return set(), True

    if existing_df.empty or "sample_id" not in existing_df.columns:
        return set(), True

    processed_keys = set(existing_df["sample_id"].dropna().astype(str).tolist())
    log.info(f"Found existing CSV. Resuming run... Skipping {len(processed_keys)} processed samples.")
    return processed_keys, False


def append_csv_row(path: str, row: Dict[str, Any], columns: List[str], write_header: bool) -> bool:
    """Append one row and fsync so completed rows survive crashes."""
    ensure_parent_dir(path)
    df_row = pd.DataFrame([{col: row.get(col, "") for col in columns}], columns=columns)

    with open(path, "a", encoding="utf-8", newline="") as f:
        df_row.to_csv(f, header=write_header, index=False)
        f.flush()
        os.fsync(f.fileno())

    return False  # after first successful write, future appends should not write headers


def write_failure(sample: Dict[str, Any], error_type: str, error_message: str, attempts: int) -> None:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_id": sample.get("sample_id", ""),
        "subject": sample.get("subject", ""),
        "activity": sample.get("activity", ""),
        "window_id": sample.get("window_id", ""),
        "error_type": error_type,
        "error_message": error_message,
        "attempts": attempts,
    }
    write_header = not os.path.exists(FAILURE_PATH) or os.path.getsize(FAILURE_PATH) == 0
    append_csv_row(FAILURE_PATH, row, FAILURE_COLUMNS, write_header)


def api_error_code(error: APIError) -> Optional[int]:
    code = getattr(error, "code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def load_images(image_paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images

# ---------------------------------------------------------------------------
# Dataset Discovery
# ---------------------------------------------------------------------------

def get_dataset_samples(video_dir: str, audio_dir: str) -> List[Dict[str, Any]]:
    """Scan the video directory, filter for spoken tasks, and match audio files."""
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video data directory not found: {video_dir}")
    if not os.path.isdir(audio_dir):
        log.warning(f"Audio data directory not found: {audio_dir}; continuing with image-only samples.")

    samples: List[Dict[str, Any]] = []

    for s_entry in sorted(os.scandir(video_dir), key=lambda x: x.name):
        if not s_entry.is_dir():
            continue

        subject = s_entry.name

        for a_entry in sorted(os.scandir(s_entry.path), key=lambda x: x.name):
            if not a_entry.is_dir():
                continue

            activity_folder = a_entry.name
            # Strip any prefix like "Task_Counting1" -> "Counting1".
            activity = activity_folder.split("_", 1)[1] if "_" in activity_folder else activity_folder

            if not any(supported_act in activity for supported_act in AUDIO_ACTIVITIES):
                continue

            for w_entry in sorted(os.scandir(a_entry.path), key=lambda x: x.name):
                if not w_entry.is_dir():
                    continue

                window_name = w_entry.name
                image_paths = sorted(
                    [
                        os.path.join(w_entry.path, f)
                        for f in os.listdir(w_entry.path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ],
                    key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or 0),
                )

                audio_path = None
                audio_stem = os.path.join(audio_dir, subject, activity_folder, window_name)
                for ext in (".wav", ".mp3"):
                    candidate = f"{audio_stem}{ext}"
                    if os.path.exists(candidate):
                        audio_path = candidate
                        break

                if REQUIRE_AUDIO and audio_path is None:
                    log.warning(f"Missing audio for {subject}/{activity_folder}/{window_name}. Skipping (REQUIRE_AUDIO=True).")
                    continue

                try:
                    window_id = int(window_name.split("_", 1)[1]) if "_" in window_name else int(window_name)
                except ValueError:
                    window_id = window_name

                sample_id = make_sample_id(subject, activity, window_id)

                samples.append({
                    "sample_id": sample_id,
                    "subject": subject,
                    "activity": activity,
                    "window_id": window_id,
                    # Kept only for I/O — not written to the output CSV.
                    "image_paths": image_paths,
                    "audio_path": audio_path,
                })

    return samples

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_one_sample(client: genai.Client, sample: Dict[str, Any], write_headers: bool) -> Tuple[bool, bool]:
    """
    Run inference for one sample.

    Returns:
      - success: whether a result row was written
      - write_headers: updated header flag for future result appends
    """
    if not sample["image_paths"]:
        msg = f"No images found for {sample['sample_id']}."
        log.warning(msg)
        write_failure(sample, "MissingImages", msg, attempts=0)
        return False, write_headers

    uploaded_audio = None
    attempts_used = 0

    try:
        images = load_images(sample["image_paths"])

        if sample["audio_path"]:
            uploaded_audio = client.files.upload(file=sample["audio_path"])

        contents = [user_prompt] + images
        if uploaded_audio:
            contents.append(uploaded_audio)

        for attempt in range(MAX_RETRIES):
            attempts_used = attempt + 1
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=StressPrediction,
                        temperature=0.2,
                    ),
                )

                parsed = getattr(response, "parsed", None)
                if parsed is None:
                    response_text = getattr(response, "text", "")
                    raise ValueError(f"Gemini returned no parsed JSON. Raw response starts with: {response_text[:500]}")

                result_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
                binary_stress = 1 if int(result_dict["stress_level"]) >= 5 else 0

                final_row = {
                    "sample_id": sample["sample_id"],
                    "subject": sample["subject"],
                    "activity": sample["activity"],
                    "window_id": sample["window_id"],
                    "stress_level": result_dict["stress_level"],
                    "confidence": result_dict["confidence"],
                    "reasoning": result_dict["reasoning"],
                    "binary_stress": binary_stress,
                    "audio_included": bool(sample["audio_path"]),
                }

                write_headers = append_csv_row(OUT_PATH, final_row, OUTPUT_COLUMNS, write_headers)
                return True, write_headers

            except APIError as e:
                code = api_error_code(e)
                if code in TRANSIENT_API_CODES and attempt < MAX_RETRIES - 1:
                    wait_s = DELAY_BETWEEN_CALLS * (attempt + 2)
                    log.warning(
                        f"Transient API error {code} on {sample['sample_id']}. "
                        f"Retrying in {wait_s:.1f}s..."
                    )
                    time.sleep(wait_s)
                    continue

                log.error(f"API error on {sample['sample_id']}: {e}")
                write_failure(sample, "APIError", str(e), attempts=attempts_used)
                return False, write_headers

            except Exception as e:
                log.error(f"Unexpected error on {sample['sample_id']}: {e}")
                write_failure(sample, type(e).__name__, str(e), attempts=attempts_used)
                return False, write_headers

        # Should not normally be reached.
        msg = f"Max retries exhausted for {sample['sample_id']}"
        log.error(msg)
        write_failure(sample, "MaxRetriesExceeded", msg, attempts=attempts_used)
        return False, write_headers

    finally:
        # Runs even on KeyboardInterrupt, so uploaded files are cleaned up when possible.
        if uploaded_audio is not None:
            try:
                client.files.delete(name=uploaded_audio.name)
            except Exception as e:
                log.warning(f"Could not delete uploaded audio file {uploaded_audio.name}: {e}")

# ---------------------------------------------------------------------------
# Main Inference Loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    ensure_parent_dir(OUT_PATH)
    ensure_parent_dir(FAILURE_PATH)

    client = genai.Client()
    log.info("Gemini Client initialized.")

    samples = get_dataset_samples(VIDEO_DATA_DIR, AUDIO_DATA_DIR)
    log.info(f"Dataset scanned — {len(samples)} spoken windows found in queue.")

    processed_keys, write_headers = read_existing_results(OUT_PATH)

    try:
        with tqdm(samples, desc="Inferring", unit="window", dynamic_ncols=True) as pbar:
            for sample in pbar:
                if sample["sample_id"] in processed_keys:
                    continue

                success, write_headers = run_one_sample(client, sample, write_headers)

                if success:
                    processed_keys.add(sample["sample_id"])

                # Throttle to stay within RPM limits.
                time.sleep(DELAY_BETWEEN_CALLS)

    except KeyboardInterrupt:
        log.warning("Interrupted by user. Completed rows are already saved and will be skipped on restart.")
        raise

    log.info(f"Run complete. Results saved progressively to {OUT_PATH}")
    log.info(f"Failures, if any, saved to {FAILURE_PATH}")
    log.info(f"Total runtime: {time.time() - t_start:.1f}s")