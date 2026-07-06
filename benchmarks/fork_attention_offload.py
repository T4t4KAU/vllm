# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer


@dataclass(frozen=True)
class RequestResult:
    latency_ms: float
    input_tokens: int
    output_tokens: int
    output_digest: str
    output_token_ids: list[int]


def fit_token_ids(tokenizer: Any, seed: str, target_tokens: int) -> list[int]:
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    repeats = (target_tokens + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repeats)[:target_tokens]


def token_digest(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


async def complete(
    client: httpx.AsyncClient,
    model: str,
    prompt: list[int],
    max_tokens: int,
) -> RequestResult:
    started = time.perf_counter()
    response = await client.post(
        "/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
    )
    response.raise_for_status()
    payload = response.json()
    usage = payload["usage"]
    choice = payload["choices"][0]
    output_text = choice["text"]
    output_token_ids = choice["token_ids"]
    return RequestResult(
        latency_ms=(time.perf_counter() - started) * 1000,
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
        output_digest=hashlib.sha256(output_text.encode()).hexdigest(),
        output_token_ids=output_token_ids,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    root = fit_token_ids(
        tokenizer,
        "Shared system policy and agent execution context. ",
        args.prefix_tokens,
    )
    private_contexts = [
        fit_token_ids(
            tokenizer,
            f" Branch {branch_id} private tool history and observations. ",
            args.private_tokens,
        )
        for branch_id in range(args.branches)
    ]
    branch_prompts = [root + private_context for private_context in private_contexts]
    revisit_prompts = [
        prompt
        + tokenizer.encode(
            f"\nShort follow-up query for branch {branch_id}.",
            add_special_tokens=False,
        )
        for branch_id, prompt in enumerate(branch_prompts)
    ]

    limits = httpx.Limits(
        max_connections=args.branches,
        max_keepalive_connections=args.branches,
    )
    timeout = httpx.Timeout(args.timeout)
    workload_started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=timeout,
        limits=limits,
        headers={"Authorization": "Bearer vllm-local"},
    ) as client:
        root_warm = await complete(client, args.model, root, 1)

        warm_started = time.perf_counter()
        branch_warm = await asyncio.gather(
            *(complete(client, args.model, prompt, 1) for prompt in branch_prompts)
        )
        warm_wall_ms = (time.perf_counter() - warm_started) * 1000

        churn_results: list[RequestResult] = []
        churn_started = time.perf_counter()
        for churn_id in range(args.churn_requests):
            churn = fit_token_ids(
                tokenizer,
                f"Unrelated cache pressure sequence {churn_id}. ",
                args.churn_tokens,
            )
            churn_results.append(await complete(client, args.model, churn, 1))
        churn_wall_ms = (time.perf_counter() - churn_started) * 1000

        revisit_started = time.perf_counter()
        revisits = await asyncio.gather(
            *(
                complete(
                    client,
                    args.model,
                    prompt,
                    args.output_tokens,
                )
                for prompt in revisit_prompts
            )
        )
        revisit_wall_ms = (time.perf_counter() - revisit_started) * 1000
    total_wall_ms = (time.perf_counter() - workload_started) * 1000

    if root_warm.input_tokens != len(root):
        raise RuntimeError("Root prompt token count changed during API processing")
    for result, prompt in zip(branch_warm, branch_prompts):
        if result.input_tokens != len(prompt):
            raise RuntimeError(
                "Branch prompt token count changed during API processing"
            )
    for result, prompt in zip(revisits, revisit_prompts):
        if result.input_tokens != len(prompt):
            raise RuntimeError(
                "Revisit prompt token count changed during API processing"
            )
        if result.output_tokens != args.output_tokens:
            raise RuntimeError("Revisit did not generate the requested token count")
        if len(result.output_token_ids) != result.output_tokens:
            raise RuntimeError("API output token IDs do not match usage")

    latencies = sorted(result.latency_ms for result in revisits)
    output_tokens = sum(result.output_tokens for result in revisits)
    p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95))
    config = vars(args).copy()
    config["output"] = str(args.output) if args.output is not None else None
    return {
        "config": config,
        "total": {
            "wall_ms": total_wall_ms,
        },
        "workload": {
            "root_tokens": len(root),
            "private_tokens": [len(tokens) for tokens in private_contexts],
            "revisit_tokens": [len(tokens) for tokens in revisit_prompts],
            "root_digest": token_digest(root),
            "revisit_digests": [token_digest(tokens) for tokens in revisit_prompts],
        },
        "root_warm": asdict(root_warm),
        "branch_warm": {
            "wall_ms": warm_wall_ms,
            "mean_latency_ms": statistics.mean(
                result.latency_ms for result in branch_warm
            ),
            "input_tokens": [result.input_tokens for result in branch_warm],
            "output_digests": [result.output_digest for result in branch_warm],
            "output_token_ids": [result.output_token_ids for result in branch_warm],
        },
        "churn": {
            "wall_ms": churn_wall_ms,
            "requests": len(churn_results),
            "mean_latency_ms": (
                statistics.mean(result.latency_ms for result in churn_results)
                if churn_results
                else 0
            ),
        },
        "revisit": {
            "wall_ms": revisit_wall_ms,
            "mean_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": latencies[p95_index],
            "output_tokens": output_tokens,
            "output_tokens_per_second": output_tokens / (revisit_wall_ms / 1000),
            "output_digests": [result.output_digest for result in revisits],
            "output_token_ids": [result.output_token_ids for result in revisits],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--model", default="qwen3-0.6b-local")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--private-tokens", type=int, default=1024)
    parser.add_argument("--branches", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--churn-requests", type=int, default=4)
    parser.add_argument("--churn-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
