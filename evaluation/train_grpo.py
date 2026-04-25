"""
Run GRPO-style RL with Tinker Cookbook on math environments.

This wrapper exists because the cookbook math RL CLI does not expose every
`tinker_cookbook.rl.train.Config` field we need for these experiments, notably
`remove_constant_reward_groups` and KL reference configuration.
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.math_rl import math_env
from tinker_cookbook.rl.train import Config, KLReferenceConfig, main as rl_main


EVAL_DIR = Path(__file__).resolve().parent


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def run(args: argparse.Namespace) -> None:
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=args.model,
        explicit_renderer_name=args.renderer_name,
        load_checkpoint_path=args.load_checkpoint_path,
        base_url=args.base_url,
    )

    log_path = Path(args.log_path)
    cli_utils.check_log_dir(
        str(log_path), behavior_if_exists=args.behavior_if_log_dir_exists
    )

    dataset_builder = math_env.get_math_dataset_builder(
        dataset_name=args.env,
        batch_size=args.groups_per_batch,
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        group_size=args.group_size,
        seed=args.seed,
    )

    kl_reference_config = None
    if args.kl_penalty_coef > 0:
        kl_reference_config = KLReferenceConfig(
            base_model=args.kl_reference_base_model or args.model,
            load_checkpoint_path=args.kl_reference_path or args.load_checkpoint_path,
        )

    metadata = {
        "method": "grpo",
        "env": args.env,
        "model": args.model,
        "renderer_name": renderer_name,
        "load_checkpoint_path": args.load_checkpoint_path,
        "log_path": str(log_path),
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
        "learning_rate": args.lr,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "kl_penalty_coef": args.kl_penalty_coef,
        "kl_reference_base_model": args.kl_reference_base_model or args.model,
        "kl_reference_path": args.kl_reference_path or args.load_checkpoint_path,
        "remove_constant_reward_groups": args.remove_constant_reward_groups,
        "loss_fn": args.loss_fn,
        "max_steps": args.max_steps,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    info_path = EVAL_DIR / "grpo_run_info.json"
    write_json(info_path, {**metadata, "status": "running"})

    config = Config(
        learning_rate=args.lr,
        dataset_builder=dataset_builder,
        model_name=args.model,
        renderer_name=renderer_name,
        lora_rank=args.rank,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_path=str(log_path),
        load_checkpoint_path=args.load_checkpoint_path,
        kl_penalty_coef=args.kl_penalty_coef,
        kl_reference_config=kl_reference_config,
        remove_constant_reward_groups=args.remove_constant_reward_groups,
        loss_fn=args.loss_fn,
        loss_fn_config=None,
        num_substeps=args.num_substeps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_steps=args.max_steps,
        base_url=args.base_url,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        compute_post_kl=args.compute_post_kl,
    )

    await rl_main(config)

    last_sampler = checkpoint_utils.get_last_checkpoint(
        str(log_path), required_key="sampler_path"
    )
    last_state = checkpoint_utils.get_last_checkpoint(
        str(log_path), required_key="state_path"
    )
    write_json(
        info_path,
        {
            **metadata,
            "status": "completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "last_sampler_path": last_sampler.sampler_path if last_sampler else None,
            "last_state_path": last_state.state_path if last_state else None,
        },
    )
    print(f"GRPO run metadata saved to {info_path}")
    if last_sampler and last_sampler.sampler_path:
        print(f"Latest sampler checkpoint: {last_sampler.sampler_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tinker GRPO/RL training.")
    parser.add_argument("--env", type=str, default="gsm8k")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--renderer_name", type=str, default=None)
    parser.add_argument("--load_checkpoint_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default="evaluation/logs/grpo_3b_a")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--groups_per_batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl_penalty_coef", type=float, default=0.05)
    parser.add_argument("--kl_reference_base_model", type=str, default=None)
    parser.add_argument("--kl_reference_path", type=str, default=None)
    parser.add_argument("--remove_constant_reward_groups", action="store_true")
    parser.add_argument("--loss_fn", type=str, default="importance_sampling")
    parser.add_argument("--num_substeps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--compute_post_kl", action="store_true")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--behavior_if_log_dir_exists", type=str, default="ask")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
