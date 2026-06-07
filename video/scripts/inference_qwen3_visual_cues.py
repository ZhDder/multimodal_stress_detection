from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm
import os
import time
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import json
import re
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DATA_DIR = "/data/5s5fps_20overlap_cropped"
OUT_PATH = "/workspace/Qwen3_VL_4B_Instruct_visual_cues_5s5fps_20overlap_cropped_results.csv"

BATCH_SIZE = 1
NUM_WORKERS = 0
MAX_NEW_TOKENS = 256

CUE_NAMES = [
    "head_movement",
    "blink_behavior",
    "eye_aperture",
    "gaze_direction",
    "gaze_stability",
    "eyebrow_movement",
    "mouth_shape",
    "lip_pressing",
    "lip_corner_tension",
    "jaw_mouth_tension",
]

system_prompt = """
You are an expert in behavioral psychology and affective computing.
You will analyze a sequence of video frames and extract ONLY observable visual cues related to stress.

IMPORTANT:
- Do NOT predict stress_level.
- Do NOT output binary_stress.
- Do NOT infer internal emotion beyond visible evidence.
- Rate only what can be visually observed in the frames.

CUE SCALE:
Use an integer intensity scale for each cue:
0 = absent / not visible
1 = mild
2 = moderate
3 = strong

CUES TO RATE:
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

OUTPUT FORMAT:
Return ONLY a valid JSON object with exactly these fields:
{
  "head_movement": <integer 0-3>,
  "blink_behavior": <integer 0-3>,
  "eye_aperture": <integer 0-3>,
  "gaze_direction": <integer 0-3>,
  "gaze_stability": <integer 0-3>,
  "eyebrow_movement": <integer 0-3>,
  "mouth_shape": <integer 0-3>,
  "lip_pressing": <integer 0-3>,
  "lip_corner_tension": <integer 0-3>,
  "jaw_mouth_tension": <integer 0-3>,
  "visual_confidence": <float 0.0-1.0>,
  "visual_summary": "<one short sentence>"
}
"""

user_prompt = """
Analyze the provided frame sequence and output the JSON visual-cue assessment.
"""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StressWindowDataset(Dataset):
    def __init__(self, data_dir: str):
        self.samples = []
        for s_entry in sorted(os.scandir(data_dir), key=lambda x: x.name):
            if not s_entry.is_dir():
                continue
            for a_entry in sorted(os.scandir(s_entry.path), key=lambda x: x.name):
                if not a_entry.is_dir():
                    continue
                for w_entry in sorted(os.scandir(a_entry.path), key=lambda x: x.name):
                    if not w_entry.is_dir():
                        continue
                    self.samples.append((
                        s_entry.name,
                        a_entry.name,
                        w_entry.name,
                        w_entry.path,
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject, act, window, window_path = self.samples[idx]

        def frame_sort_key(path):
            name = os.path.basename(path)
            match = re.search(r"(\d+)", name)
            return int(match.group(1)) if match else name

        image_paths = sorted(
            [os.path.join(window_path, f) for f in os.listdir(window_path) if f.endswith(".jpg")],
            key=frame_sort_key,
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]

        activity = act.split("_", 1)[1] if "_" in act else act
        window_id = int(window.split("_", 1)[1]) if "_" in window else int(window)

        return {
            "subject": subject,
            "activity": activity,
            "window_id": window_id,
            "images": images,
        }


def collate_fn(batch):
    return {
        "subjects": [item["subject"] for item in batch],
        "activities": [item["activity"] for item in batch],
        "window_ids": [item["window_id"] for item in batch],
        "images": [item["images"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_messages(images: list) -> list:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": user_prompt}],
        },
    ]


def clamp_int(value, min_value=0, max_value=3):
    try:
        value = int(round(float(value)))
        return max(min_value, min(max_value, value))
    except Exception:
        return -1


def clamp_float(value, min_value=0.0, max_value=1.0):
    try:
        value = float(value)
        return max(min_value, min(max_value, value))
    except Exception:
        return 0.0


def extract_json_object(raw_output: str) -> str:
    text = raw_output.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return text


def parse_output(raw_output: str):
    try:
        clean_text = extract_json_object(raw_output)
        parsed = json.loads(clean_text)

        cue_values = {
            cue: clamp_int(parsed.get(cue, -1), 0, 3)
            for cue in CUE_NAMES
        }
        visual_confidence = clamp_float(parsed.get("visual_confidence", 0.0), 0.0, 1.0)
        visual_summary = str(parsed.get("visual_summary", "")).replace("\n", " ").strip()

        if not visual_summary:
            visual_summary = "No visual summary provided."

        parse_ok = True

    except Exception:
        cue_values = {cue: -1 for cue in CUE_NAMES}
        visual_confidence = 0.0
        visual_summary = f"PARSE_ERROR: {raw_output}".replace("\n", " ").strip()
        parse_ok = False
        print("FAILED TO PARSE:", raw_output)

    return cue_values, visual_confidence, visual_summary, parse_ok


def apply_chat_template_safe(processor, batch_messages):
    try:
        return processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
    except TypeError:
        return processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "return_tensors": "pt",
                "padding": True,
            },
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)
        log.info("Logged in to Hugging Face")
    else:
        log.warning("HF_TOKEN environment variable not set. Continuing without explicit login.")

    log.info("Loading model and processor...")
    t_model = time.time()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"

    log.info(f"Model ready ({time.time() - t_model:.1f}s)")

    dataset = StressWindowDataset(DATA_DIR)
    log.info(f"Dataset scanned — {len(dataset)} windows found")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    all_predictions = []
    parse_errors = 0
    n_batches = len(loader)
    t_infer = time.time()

    log.info(f"Running inference in {n_batches} batches of {BATCH_SIZE} windows...")

    with torch.inference_mode():
        with tqdm(loader, desc="Inferring", unit="window", total=len(dataset), dynamic_ncols=True) as pbar:
            for batch in pbar:
                batch_messages = [build_messages(images) for images in batch["images"]]

                inputs = apply_chat_template_safe(processor, batch_messages).to(model.device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                )

                input_len = inputs.input_ids.shape[1]
                trimmed = generated_ids[:, input_len:]
                output_texts = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for subject, activity, window_id, raw_output in zip(
                    batch["subjects"],
                    batch["activities"],
                    batch["window_ids"],
                    output_texts,
                ):
                    cue_values, visual_confidence, visual_summary, parse_ok = parse_output(raw_output)

                    if not parse_ok:
                        parse_errors += 1

                    all_predictions.append({
                        "subject": subject,
                        "activity": activity,
                        "window_id": window_id,
                        **cue_values,
                        "visual_confidence": visual_confidence,
                        "visual_summary": visual_summary,
                    })

                pbar.update(len(batch["subjects"]) - 1)
                pbar.set_postfix({"parse_errors": parse_errors})

                del inputs, generated_ids, trimmed

    elapsed_infer = time.time() - t_infer
    log.info(
        f"Inference complete — {elapsed_infer:.1f}s "
        f"({len(all_predictions) / elapsed_infer:.1f} windows/s)"
    )

    if parse_errors:
        log.warning(f"{parse_errors}/{len(all_predictions)} outputs failed JSON parsing")

    df_predictions = pd.DataFrame(all_predictions)
    df_predictions.to_csv(OUT_PATH, index=False)

    log.info(f"Results saved to {OUT_PATH}")
    log.info(f"Total runtime: {time.time() - t_start:.1f}s")
