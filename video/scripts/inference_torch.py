from huggingface_hub import login
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm
import os
import time
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import json
import gc
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

# Max images per window (5-second clip at 1fps = 5 frames)
MAX_IMAGES_PER_PROMPT = 5
# Both constants are intentionally the same: the DataLoader batch IS the
# inference chunk — one llm.generate() call per DataLoader batch means
# only that batch's decoded images are in RAM and VRAM at any time.
# Kept small because vLLM's encoder cache accumulates vision tokens in VRAM
# across a generate() call. At 5 frames/window, 8 windows = 40 frames max
# in the encoder cache simultaneously.
# Tune up cautiously if GPU utilisation looks low.
DATALOADER_BATCH_SIZE = 8
INFERENCE_CHUNK_SIZE  = 8
JSON_RE = re.compile(r"```json\n|\n```|```")


# ---------------------------------------------------------------------------
# Dataset — images are decoded in __getitem__
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


# ---------------------------------------------------------------------------
# Collate — PIL Images can't be stacked as tensors, so keep as nested lists
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return {
        "subjects":   [item["subject"]   for item in batch],
        "activities": [item["activity"]  for item in batch],
        "window_ids": [item["window_id"] for item in batch],
        "images":     [item["images"]    for item in batch],
    }


# ---------------------------------------------------------------------------
# Build a single vLLM input dict for one window
# ---------------------------------------------------------------------------

def build_vllm_input(images: list, processor) -> dict:
    """
    vLLM expects the text prompt (with image-placeholder tokens baked in)
    plus the raw images passed separately via multi_modal_data.
    """
    messages = [{"role": "user", "content":
        [{"type": "image"} for _ in images] +
        [{"type": "text", "text": prompt}]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt":           text,
        "multi_modal_data": {"image": images},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    log.info('Loading processor...')
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    log.info('Initialising vLLM engine...')
    t_model = time.time()
    llm = LLM(
        model="Qwen/Qwen3-VL-2B-Instruct",
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.80,  # lower ceiling leaves headroom for encoder cache
        max_model_len=8192,          # cap seq len — default 262144 needs 28 GiB KV cache
        limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT},
        mm_processor_kwargs = {"disable_mm_processor_cache":True}
    )
    log.info(f'vLLM engine ready  ({time.time() - t_model:.1f}s)')

    sampling_params = SamplingParams(max_tokens=256)

    dataset = StressWindowDataset('./processed_data')
    log.info(f'Dataset scanned — {len(dataset)} windows found')

    loader = DataLoader(
        dataset,
        batch_size=DATALOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=0,             # WSL: avoid OOM from spawning extra processes
        collate_fn=collate_fn,
        pin_memory=False,          # PIL Images aren't pinnable
        persistent_workers=False,  # must be False when num_workers=0
    )

    # Process the dataset in chunks of INFERENCE_CHUNK_SIZE windows.
    # Each iteration: load chunk from disk → infer → parse → discard.
    # At no point is more than one chunk of decoded images in RAM,
    # and vLLM only renders one chunk of image tokens into VRAM at a time.
    n_windows    = len(dataset)
    n_chunks     = (n_windows + INFERENCE_CHUNK_SIZE - 1) // INFERENCE_CHUNK_SIZE
    all_predictions = []
    parse_errors    = 0
    t_infer = time.time()

    log.info(f'Running inference in {n_chunks} chunks of {INFERENCE_CHUNK_SIZE} windows...')

    with tqdm(total=n_windows, desc="Inferring", unit="window",
              dynamic_ncols=True) as pbar:
        for chunk_idx, batch in enumerate(loader):
            # Build vLLM inputs for this chunk only
            chunk_inputs   = []
            chunk_metadata = []
            for subject, activity, window_id, images in zip(
                batch["subjects"], batch["activities"],
                batch["window_ids"], batch["images"]
            ):
                chunk_inputs.append(build_vllm_input(images, processor))
                chunk_metadata.append({
                    "subject":   subject,
                    "activity":  activity,
                    "window_id": window_id,
                })

            # Infer on this chunk — PIL images in chunk_inputs are the only
            # ones in RAM; previous chunks have already been GC'd
            chunk_outputs = llm.generate(chunk_inputs, sampling_params)

            # Parse and collect results, then let chunk data go out of scope
            for meta, output in zip(chunk_metadata, chunk_outputs):
                raw_output = output.outputs[0].text
                stress_level, binary_pred, confidence, reasoning = parse_output(raw_output)
                if stress_level == -1:
                    parse_errors += 1
                all_predictions.append({
                    'subject':       meta["subject"],
                    'activity':      meta["activity"],
                    'window_id':     meta["window_id"],
                    'stress_level':  stress_level,
                    'binary_stress': binary_pred,
                    'confidence':    confidence,
                    'reasoning':     reasoning,
                })
            pbar.update(len(chunk_inputs))
            pbar.set_postfix({
                "chunk":        f"{chunk_idx + 1}/{n_chunks}",
                "parse_errors": parse_errors,
            })

            # Explicitly free chunk data before next batch is loaded.
            # Without this, Python may not GC the PIL images promptly,
            # causing two chunks to occupy RAM simultaneously.
            del chunk_inputs, chunk_outputs, chunk_metadata
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