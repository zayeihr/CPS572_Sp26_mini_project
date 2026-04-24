"""
Multi-task SFT training on GSM8K (math), tulu-3 (instruction following),
and python_code_instructions (code).

Usage:
    # Quick test on 3B model:
    python evaluation/train_and_publish.py --num_steps 200 --checkpoint_name sft_3b_v1

    # Final submission on 8B model:
    python evaluation/train_and_publish.py --model meta-llama/Llama-3.1-8B --num_steps 500 --checkpoint_name sft_8b_final

    # Skip publishing (dry run):
    python evaluation/train_and_publish.py --no_publish
"""

import argparse
import json
import os
import random

import numpy as np
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

# ── Model ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B"
# DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"  # uncomment for final submission

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data budget per task ───────────────────────────────────────────────────────
N_MATH = 5000   # was 3000
N_IF   = 2000   # was 3000  
N_CODE = 6000   # was 3000 — boost code to fix HumanEval


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_gsm8k(n: int) -> list:
    """Load GSM8K math examples formatted as chain-of-thought conversations."""
    print(f"  Loading GSM8K ({n} examples)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    conversations = []
    for ex in ds:
        conversations.append([
            {"role": "user",      "content": ex["question"].strip()},
            {"role": "assistant", "content": ex["answer"].strip()},
        ])
    print(f"    {len(conversations)} math examples loaded.")
    return conversations


def load_tulu(n: int) -> list:
    """Load tulu-3 SFT mixture for instruction following (streaming)."""
    print(f"  Loading tulu-3 ({n} examples)...")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    conversations = []
    for ex in ds:
        if len(conversations) >= n:
            break
        msgs = ex.get("messages", [])
        convo = [{"role": m["role"], "content": m["content"]}
                 for m in msgs if m["role"] in ("user", "assistant")]
        if len(convo) >= 2:
            conversations.append(convo)
    print(f"    {len(conversations)} instruction-following examples loaded.")
    return conversations


def load_code(n: int) -> list:
    """Load python code instructions — free, fast, no auth needed."""
    print(f"  Loading python code instructions ({n} examples)...")
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    conversations = []
    for ex in ds:
        instruction = (ex.get("instruction") or ex.get("prompt") or "").strip()
        response    = (ex.get("output") or "").strip()
        if instruction and response:
            conversations.append([
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": response},
            ])
    print(f"    {len(conversations)} code examples loaded.")
    return conversations


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-task SFT training")
    parser.add_argument("--model",           type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--num_steps",       type=int,   default=200)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--rank",            type=int,   default=32)
    parser.add_argument("--max_length",      type=int,   default=1024)
    parser.add_argument("--checkpoint_name", type=str,   default="sft_multitask_v1")
    parser.add_argument("--no_publish",      action="store_true")
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Model: {args.model}")
    tokenizer     = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer      = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # ── Load & mix data ────────────────────────────────────────────────────────
    print("\nLoading training data...")
    math_convos = load_gsm8k(N_MATH)
    if_convos   = load_tulu(N_IF)
    code_convos = load_code(N_CODE)

    all_convos = math_convos + if_convos + code_convos
    random.shuffle(all_convos)
    print(f"\nTotal conversations: {len(all_convos)} "
          f"(math={len(math_convos)}, IF={len(if_convos)}, code={len(code_convos)})")

    # ── Tokenize ───────────────────────────────────────────────────────────────
    print("\nTokenizing...")
    all_data = []
    skipped  = 0
    for convo in all_convos:
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=args.max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            all_data.append(datum)
        except Exception:
            skipped += 1

    print(f"  {len(all_data)} examples ready ({skipped} skipped due to length/errors)")

    if len(all_data) == 0:
        raise RuntimeError("No training data! Check dataset loading above.")

    # ── Create training client ─────────────────────────────────────────────────
    print(f"\nCreating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.model, rank=args.rank)
    print("  Training client ready")

    # ── Train ──────────────────────────────────────────────────────────────────
    adam_params = types.AdamParams(
        learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8
    )
    print(f"\nTraining for {args.num_steps} steps "
          f"(batch_size={args.batch_size}, lr={args.lr}, rank={args.rank})...")

    losses = []
    for step in range(args.num_steps):
        start = (step * args.batch_size) % len(all_data)
        batch = [all_data[i % len(all_data)] for i in range(start, start + args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future   = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights  = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss     = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        losses.append(loss)

        if (step + 1) % 10 == 0 or step == 0:
            avg = np.mean(losses[-10:])
            print(f"  Step {step+1:>4}/{args.num_steps} | Loss: {loss:.4f} | Avg(last 10): {avg:.4f}")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt            = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    # ── Publish ────────────────────────────────────────────────────────────────
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # ── Save info ──────────────────────────────────────────────────────────────
    info = {
        "checkpoint_path": checkpoint_path,
        "base_model":      args.model,
        "renderer_name":   renderer_name,
        "training": {
            "num_steps":     args.num_steps,
            "batch_size":    args.batch_size,
            "learning_rate": args.lr,
            "lora_rank":     args.rank,
            "max_length":    args.max_length,
            "data_mix":      {"math": N_MATH, "instruction_following": N_IF, "code": N_CODE},
            "final_loss":    float(losses[-1]) if losses else None,
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with:")
    print(f'  python evaluation/eval_all.py --checkpoint_path "{checkpoint_path}" --base_model {args.model}')


if __name__ == "__main__":
    main()