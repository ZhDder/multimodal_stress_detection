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
import warnings
from transformers import logging as hf_logging

#warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Gemma 4 now supports system prompts natively.
# Adding <|think|> enables its built-in reasoning mode for better analysis.
system_prompt = """
You are an expert in behavioral psychology and affective computing.
Your final answer must be ONLY a valid JSON object. 
Keep the reasoning field strictly under 2 short and precise sentences.
"""

user_prompt = """
You are an expert in behavioral psychology and affective computing.
You are analyzing a 5-second video sequence (sampled at 1 fps) of a person performing a task.
Your goal is to assess their stress level based on visible behavioral cues.

Focus on the following indicators:
- Facial Expressions: Furrowed brows, pressed lips, clenched jaw, flared nostrils
- Eye Movement: Rapid blinking, gaze aversion, widened eyes
- Body Language: Face/hair touching (self-soothing), rigid posture, fidgeting

Respond ONLY with a valid JSON object, and nothing else:
{"stress_level": <integer 0-10>, "confidence": <float 0.0-1.0>, "reasoning": "<max 2 sentences, short and precise>"}
"""

BATCH_SIZE  = 2  # You might need to adjust this depending on your VRAM
NUM_WORKERS = 0   
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
        
        # Load the frames
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        # FIX: Ensure all frames in the sequence have identical dimensions for np.stack()
        # We'll use the dimensions of the first frame as the target size.
        if images:
            target_size = images[0].size  # (width, height)
            images = [
                img.resize(target_size, Image.Resampling.LANCZOS) if img.size != target_size else img 
                for img in images
            ]

        return {
            "subject":   subject,
            "activity":  act.split('_')[1],
            "window_id": int(window.split('_')[1]),
            "frames":    images, 
        }

def collate_fn(batch):
    return {
        "subjects":   [item["subject"]   for item in batch],
        "activities": [item["activity"]  for item in batch],
        "window_ids": [item["window_id"] for item in batch],
        "frames":     [item["frames"]    for item in batch],
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_messages(frames: list) -> list:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            # Gemma 4 processes video natively; we pass the frames as a single video sequence
            {"type": "video", "video": frames}, 
            {"type": "text",  "text":  user_prompt}
        ]}
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
       #attn_implementation="flash_attention_2"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
    processor.tokenizer.padding_side = "left"
    processor.video_processor.num_frames = 5

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
                    build_messages(frames) for frames in batch["frames"]
                ]

                inputs = processor.apply_chat_template(
                    batch_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
	 	    processor_kwargs={
                        "num_frames": 5, 
                        "fps": 1.0  
                    }
                ).to(model.device)

                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=128,
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

    out_path = "/workspace/Gemma4_E4B_IT_cropped_results.csv"
    df_predictions = pd.DataFrame(all_predictions)
    df_predictions.to_csv(out_path, index=False)

    log.info(f'Results saved to {out_path}')
    log.info(f'Total runtime: {time.time() - t_start:.1f}s')
