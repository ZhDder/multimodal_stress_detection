from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor
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

system_prompt = """
You are an expert in behavioral psychology and affective computing.
Your final answer must be ONLY a valid JSON object. 
Keep the reasoning field strictly under 2 short and precise sentences.
"""

user_prompt = """
You are analyzing visual signs of stress in a subject. You are provided with:
1. A single BASELINE IMAGE showing the subject's normal, relaxed state.
2. A VIDEO SEQUENCE of 5 frames of the subject performing a task.

Your goal is to determine if the subject is exhibiting visual signs of stress in the video.
Do NOT evaluate absolute expressions. Evaluate the DEVIATION from the baseline image. 
Focus on:
- Facial Tension: Newly furrowed brows, tightening of lips, or clenched jaw absent in the baseline.
- Hand-to-Face Contact: New self-soothing behaviors (touching face/hair).
- Posture: Increased rigidity or leaning compared to the baseline.

Respond ONLY with a valid JSON object, and nothing else:
{"stress_level": <float 0.0-1.0>, "confidence": <float 0.0-1.0>, "reasoning": "<max 2 sentences explaining the deviation from baseline>"}
"""

BATCH_SIZE  = 2  #adjust this depending on VRAM
NUM_WORKERS = 0   
JSON_RE     = re.compile(r"```json\n|\n```|```")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StressWindowDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.samples = []
        
        # 1. First, build a map of subject -> baseline image path
        self.baseline_map = self._build_baseline_mapping()

        # 2. Then, build the standard samples list (ignoring baseline activities for evaluation)
        for s_entry in os.scandir(data_dir):
            if not s_entry.is_dir():
                continue
            for a_entry in os.scandir(s_entry.path):
                if not a_entry.is_dir():
                    continue
                
                # Skip baseline activities for the actual evaluation windows
                if 'Baseline' in a_entry.name or 'Relax' in a_entry.name:
                    continue
                    
                for w_entry in os.scandir(a_entry.path):
                    if not w_entry.is_dir():
                        continue
                    self.samples.append((
                        s_entry.name,
                        a_entry.name,
                        w_entry.name,
                        w_entry.path,
                    ))

    def _build_baseline_mapping(self):
        """Scans the directory to find the first frame of the first baseline window per subject."""
        b_map = {}
        for s_entry in os.scandir(self.data_dir):
            if not s_entry.is_dir():
                continue
            
            subject = s_entry.name
            b_map[subject] = None  # Default to None if not found
            
            for a_entry in os.scandir(s_entry.path):
                if not a_entry.is_dir():
                    continue
                
                if 'Baseline' in a_entry.name or 'Relax' in a_entry.name:
                    # Get all windows for this baseline activity, sorted alphabetically/numerically
                    windows = sorted([w for w in os.scandir(a_entry.path) if w.is_dir()], key=lambda x: x.name)
                    if windows:
                        first_window = windows[0]
                        # Get all frames in this first window
                        frames = sorted([f for f in os.listdir(first_window.path) if f.endswith('.jpg')])
                        if frames:
                            # Save the path to the very first frame
                            b_map[subject] = os.path.join(first_window.path, frames[0])
                            break # Found the baseline for this subject, stop looking
        return b_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject, act, window, window_path = self.samples[idx]
        
        image_paths = sorted(
            os.path.join(window_path, f)
            for f in os.listdir(window_path) if f.endswith('.jpg')
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        # Load the baseline image if it exists, otherwise return None
        baseline_path = self.baseline_map.get(subject)
        baseline_img = Image.open(baseline_path).convert("RGB") if baseline_path else None
	
        if images:
            target_size = images[0].size  # (width, height)
            images = [
                img.resize(target_size, Image.Resampling.LANCZOS) if img.size != target_size else img 
                for img in images
            ]
            baseline_img = baseline_img.resize(target_size, Image.Resampling.LANCZOS) if baseline_img.size != target_size else baseline_img

        return {
            "subject":   subject,
            "activity":  act.split('_')[1] if '_' in act else act,
            "window_id": int(window.split('_')[1]) if '_' in window else window,
            "baseline":  baseline_img, 
            "frames":    images, 
        }

def collate_fn(batch):
    return {
        "subjects":   [item["subject"]   for item in batch],
        "activities": [item["activity"]  for item in batch],
        "window_ids": [item["window_id"] for item in batch],
        "baselines": [item["baseline"] for item in batch],
        "frames":     [item["frames"]    for item in batch],
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_messages(baseline, frames: list) -> list:

    # We have a baseline, use the comparative multimodal payload
    content = [
        {"type": "text", "text": "BASELINE IMAGE (Relaxed state):"},
        {"type": "image", "image": baseline},
        {"type": "text", "text": "\nTASK VIDEO SEQUENCE:"},
        {"type": "video", "video": frames},
        {"type": "text",  "text": user_prompt}
    ]

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content}
    ]

def parse_output(raw_output: str):
    # This regex will ignore the <|think|>...<|/think|> blocks
    match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
    if match:
        clean_text = match.group(0)
    else:
        clean_text = raw_output.strip()
        if clean_text.startswith('{') and not clean_text.endswith('}'):
            if clean_text.endswith('"'):
                clean_text += "}"
            else:
                clean_text += '"}'
    try:
        parsed       = json.loads(clean_text)
        stress_level = float(parsed.get("stress_level", 0))
        confidence   = float(parsed.get("confidence", 0.0))
        reasoning    = parsed.get("reasoning", "")
        binary_pred  = 1 if stress_level >= 5.0 else 0
    except json.JSONDecodeError:
        stress_level, binary_pred, confidence = -1, -1, 0.0
        reasoning = f"Parse Error. Raw text: {raw_output}"
        print('FAILED TO PARSE: ', raw_output)
    return stress_level, binary_pred, confidence, reasoning

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t_start = time.time()

    # Gemma 4 is fully open Apache 2.0
    login('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')
    log.info('Logged in')

    log.info('Loading model and processor...')
    t_model = time.time()
    
    # Load Gemma 4 E4B IT
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B-it",
        torch_dtype=torch.bfloat16,   
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
    processor.tokenizer.padding_side = "left"
    log.info(f'Model ready  ({time.time() - t_model:.1f}s)')

    dataset = StressWindowDataset('/data/processed_data_cropped')
    log.info(f'Dataset scanned — {len(dataset)} windows found')

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    all_predictions = []
    parse_errors    = 0
    n_batches       = len(loader)
    t_infer         = time.time()

    log.info(f'Running inference in {n_batches} batches of {BATCH_SIZE} windows...')

    with torch.inference_mode():
        with tqdm(loader, desc="Inferring", unit="window",
                  total=len(dataset), dynamic_ncols=True) as pbar:
            for batch in pbar:
                batch_messages = [
                    build_messages(baseline, frames) 
                    for baseline, frames in zip(batch["baselines"], batch["frames"])
                ]

                inputs = processor.apply_chat_template(
                    batch_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)

                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=256, # Increased slightly to accommodate the thinking tokens
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
                )

                input_len    = inputs.input_ids.shape[1]
                trimmed      = generated_ids[:, input_len:]
                output_texts = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for subject, activity, window_id, raw_output in zip(
                    batch["subjects"], batch["activities"],
                    batch["window_ids"], output_texts
                ):
                    stress_level, binary_pred, confidence, reasoning = parse_output(raw_output)
                    if stress_level == -1:
                        parse_errors += 1
                    all_predictions.append({
                        'subject':       subject,
                        'activity':      activity,
                        'window_id':     window_id,
                        'stress_level':  stress_level,
                        'binary_stress': binary_pred,
                        'confidence':    confidence,
                        'reasoning':     reasoning,
                    })

                pbar.update(len(batch["subjects"])) 
                pbar.set_postfix({"parse_errors": parse_errors})

                del inputs, generated_ids, trimmed

    elapsed_infer = time.time() - t_infer
    log.info(
        f'Inference complete — {elapsed_infer:.1f}s  '
        f'({len(all_predictions) / elapsed_infer:.1f} windows/s)'
    )

    if parse_errors:
        log.warning(f'{parse_errors}/{len(all_predictions)} outputs failed JSON parsing')

    out_path = "/workspace/Gemma4_E4B_IT_baselineRef_results.csv"
    df_predictions = pd.DataFrame(all_predictions)
    df_predictions.to_csv(out_path, index=False)

    log.info(f'Results saved to {out_path}')
    log.info(f'Total runtime: {time.time() - t_start:.1f}s')
