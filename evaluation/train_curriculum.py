"""
Curriculum Learning SFT: trains on all 3 tasks equally in Phase 1,
then shifts focus to code-heavy data in Phase 2.

Motivation: v1 (equal mix) was strong on math/IF but weak on code.
v2 (code-heavy) fixed code but hurt math/IF. Curriculum learning
aims to get the best of both by learning general skills first,
then specializing on the hardest task (HumanEval/code).

Usage:
    python evaluation/train_curriculum.py --checkpoint_name sft_8b_curriculum
    python evaluation/train_curriculum.py --no_publish  # dry run
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
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Curriculum config ──────────────────────────────────────────────────────────
# Phase 1: equal mix — learn all tasks together
PHASE1_STEPS = 500
PHASE1_MATH  = 5000
PHASE1_IF    = 2000
PHASE1_CODE  = 3000

# Phase 2: code-heavy — specialize on the hardest task
PHASE2_STEPS = 500
PHASE2_MATH  = 1000
PHASE2_IF    = 1000
PHASE2_CODE  = 6000


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_gsm8k(n: int, seed: int = 42) -> list:
    print(f"  Loading GSM8K ({n} examples)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    conversations = []
    for ex in ds:
        conversations.append([
            {"role": "user",      "content": ex["question"].strip()},
            {"role": "assistant", "content": ex["answer"].strip()},
        ])
    print(f"    {len(conversations)} math examples loaded.")
    return conversations


def load_tulu(n: int) -> list:
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


def load_code(n: int, seed: int = 42) -> list:
    print(f"  Loading python code instructions ({n} examples)...")
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
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


def tokenize(convos: list, renderer, max_length: int) -> list:
    data = []
    skipped = 0
    for convo in convos:
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            data.append(datum)
        except Exception:
            skipped += 1
    return data, skipped


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Curriculum learning SFT")
    parser.add_argument("--model",           type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--rank",            type=int,   default=32)
    parser.add_argument("--max_length",      type=int,   default=1024)
    parser.add_argument("--checkpoint_name", type=str,   default="sft_8b_curriculum")
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

    # ── Load all data upfront ──────────────────────────────────────────────────
    print("\n=== Loading Phase 1 data (equal mix) ===")
    math_p1 = load_gsm8k(PHASE1_MATH, seed=42)
    if_p1   = load_tulu(PHASE1_IF)
    code_p1 = load_code(PHASE1_CODE, seed=42)
    phase1_convos = math_p1 + if_p1 + code_p1
    random.shuffle(phase1_convos)
    print(f"Phase 1 total: {len(phase1_convos)} conversations")

    print("\n=== Loading Phase 2 data (code-heavy) ===")
    math_p2 = load_gsm8k(PHASE2_MATH, seed=99)   # different seed = different examples
    if_p2   = load_tulu(PHASE2_IF)
    code_p2 = load_code(PHASE2_CODE, seed=99)
    phase2_convos = math_p2 + if_p2 + code_p2
    random.shuffle(phase2_convos)
    print(f"Phase 2 total: {len(phase2_convos)} conversations")

    # ── Tokenize ───────────────────────────────────────────────────────────────
    print("\nTokenizing Phase 1...")
    phase1_data, skip1 = tokenize(phase1_convos, renderer, args.max_length)
    print(f"  {len(phase1_data)} ready ({skip1} skipped)")

    print("Tokenizing Phase 2...")
    phase2_data, skip2 = tokenize(phase2_convos, renderer, args.max_length)
    print(f"  {len(phase2_data)} ready ({skip2} skipped)")

    if len(phase1_data) == 0 or len(phase2_data) == 0:
        raise RuntimeError("No training data! Check dataset loading.")

    # ── Create training client ─────────────────────────────────────────────────
    print(f"\nCreating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.model, rank=args.rank)
    print("  Training client ready")

    adam_params = types.AdamParams(
        learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8
    )

    total_steps = PHASE1_STEPS + PHASE2_STEPS
    losses = []

    # ── Phase 1: equal mix ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: Equal mix ({PHASE1_STEPS} steps)")
    print(f"  Math={PHASE1_MATH}, IF={PHASE1_IF}, Code={PHASE1_CODE}")
    print(f"{'='*60}")

    for step in range(PHASE1_STEPS):
        start = (step * args.batch_size) % len(phase1_data)
        batch = [phase1_data[i % len(phase1_data)] for i in range(start, start + args.batch_size)]

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
            print(f"  [P1] Step {step+1:>4}/{PHASE1_STEPS} | Loss: {loss:.4f} | Avg: {avg:.4f}")

    print(f"\nPhase 1 complete. Avg loss: {np.mean(losses):.4f}")

    # ── Phase 2: code-heavy ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: Code-heavy ({PHASE2_STEPS} steps)")
    print(f"  Math={PHASE2_MATH}, IF={PHASE2_IF}, Code={PHASE2_CODE}")
    print(f"{'='*60}")

    phase2_losses = []
    for step in range(PHASE2_STEPS):
        start = (step * args.batch_size) % len(phase2_data)
        batch = [phase2_data[i % len(phase2_data)] for i in range(start, start + args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future   = tc.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights  = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss     = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        phase2_losses.append(loss)
        losses.append(loss)

        if (step + 1) % 10 == 0 or step == 0:
            avg = np.mean(phase2_losses[-10:])
            print(f"  [P2] Step {step+1:>4}/{PHASE2_STEPS} | Loss: {loss:.4f} | Avg: {avg:.4f}")

    print(f"\nPhase 2 complete. Avg loss: {np.mean(phase2_losses):.4f}")

    # ── Save & publish ─────────────────────────────────────────────────────────
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt            = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    info = {
        "checkpoint_path": checkpoint_path,
        "base_model":      args.model,
        "renderer_name":   renderer_name,
        "training": {
            "method":        "curriculum_learning",
            "phase1": {"steps": PHASE1_STEPS, "math": PHASE1_MATH, "IF": PHASE1_IF, "code": PHASE1_CODE},
            "phase2": {"steps": PHASE2_STEPS, "math": PHASE2_MATH, "IF": PHASE2_IF, "code": PHASE2_CODE},
            "batch_size":    args.batch_size,
            "learning_rate": args.lr,
            "lora_rank":     args.rank,
            "final_loss":    float(losses[-1]),
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
