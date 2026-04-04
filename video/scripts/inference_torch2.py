from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm
import os
import gc
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


prompt = """

You are an expert in behavioral psychology and affective computing.
You are analyzing a 5-second video sequence (sampled at 1 fps) of a person performing a task.
Your goal is to assess their stress level based on visible behavioral cues.

Focus on the following indicators:
- Facial Expressions: Furrowed brows, pressed lips, clenched jaw, flared nostrils
- Eye Movement: Rapid blinking, gaze aversion, widened eyes
- Body Language: Face/hair touching (self-soothing), rigid posture, fidgeting

Use this scale for stress_level:
- 0-2: Visibly relaxed, neutral expression, natural movements
- 3-4: Mild tension, slight indicators present
- 5-6: Moderate stress, multiple indicators present
- 7-8: High stress, clear and consistent indicators
- 9-10: Extreme stress, overwhelming indicators

Respond ONLY with a valid JSON object, and nothing else:
{"stress_level": <integer 0-10>, "confidence": <float 0.0-1.0>, "reasoning": "<max 2 sentences>"}

"""

# One llm.generate() call per batch — tune based on your VRAM.
# At 5 frames/window, each window uses ~1000-1500 vision tokens.
# 8 windows = ~10000 tokens peak in VRAM during a forward pass.
BATCH_SIZE  = 8
NUM_WORKERS = 0       # WSL: keep at 0 to avoid spawn OOM
JSON_RE     = re.compile(r"```json\n|\n```|```")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StressWindowDataset(Dataset):
    def __init__(self, data_dir: str):
        self.samples = []
        for s_entry in os.scandir(data_dir):
            if not s_entry.is_dir():
                continue
            for a_entry in os.scandir(s_entry.path):
                if not a_entry.is_dir():
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject, act, window, window_path = self.samples[idx]
        image_paths = sorted(
            os.path.join(window_path, f)
            for f in os.listdir(window_path) if f.endswith('.jpg')
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return {
            "subject":   subject,
            "activity":  act.split('_')[1],
            "window_id": int(window.split('_')[1]),
            "images":    images,
        }


def collate_fn(batch):
    return {
        "subjects":   [item["subject"]   for item in batch],
        "activities": [item["activity"]  for item in batch],
        "window_ids": [item["window_id"] for item in batch],
        "images":     [item["images"]    for item in batch],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_messages(images: list) -> list:
    return [{"role": "user", "content":
        [{"type": "image", "image": img} for img in images] +
        [{"type": "text",  "text":  prompt}]
    }]


def parse_output(raw_output: str):
    clean_text = JSON_RE.sub("", raw_output).strip()
    try:
        parsed       = json.loads(clean_text)
        stress_level = float(parsed.get("stress_level", 0))
        confidence   = float(parsed.get("confidence", 0.0))
        reasoning    = parsed.get("reasoning", "")
        binary_pred  = 1 if stress_level >= 5.0 else 0
    except json.JSONDecodeError:
        stress_level, binary_pred, confidence = -1, -1, 0.0
        reasoning = f"Parse Error. Raw text: {raw_output}"
    return stress_level, binary_pred, confidence, reasoning


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t_start = time.time()

    login('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')
    log.info('Logged in to HuggingFace Hub')

    log.info('Loading model and processor...')
    t_model = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,   # explicit half-precision
        device_map="cuda:0",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    log.info(f'Model ready  ({time.time() - t_model:.1f}s)')

    dataset = StressWindowDataset('./processed_data')
    log.info(f'Dataset scanned — {len(dataset)} windows found')

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False,
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
                    build_messages(images) for images in batch["images"]
                ]

                # Tokenise with left-padding so sequences align on the right
                inputs = processor.apply_chat_template(
                    batch_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to("cuda:0")

                generated_ids = model.generate(**inputs, max_new_tokens=256)

                # Trim input tokens, decode only generated portion
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

                pbar.update(len(batch["subjects"]) - 1)  # tqdm counts 1 already
                pbar.set_postfix({"parse_errors": parse_errors})

                # Explicitly free GPU tensors before next batch
                del inputs, generated_ids, trimmed
                torch.cuda.empty_cache()
                gc.collect()

    elapsed_infer = time.time() - t_infer
    log.info(
        f'Inference complete — {elapsed_infer:.1f}s  '
        f'({len(all_predictions) / elapsed_infer:.1f} windows/s)'
    )

    if parse_errors:
        log.warning(f'{parse_errors}/{len(all_predictions)} outputs failed JSON parsing')

    out_path = "./Qwen3_VL_2B_Instruct_results.csv"
    df_predictions = pd.DataFrame(all_predictions)
    df_predictions.to_csv(out_path, index=False)

    log.info(f'Results saved to {out_path}')
    log.info(f'Total runtime: {time.time() - t_start:.1f}s')