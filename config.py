# -*- coding: utf-8 -*-
"""Shared argument parser for all pipeline entry points."""

import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments shared across inference_refining, stepuq, and analyze_result."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="maximum number of new tokens to generate per call"
    )
    parser.add_argument(
        "--try_times", type=int, default=20,
        help="retry limit per question for a valid model response"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--dataset", default="gsm8k",
        choices=["ASDiv", "2WikimhQA", "gsm8k", "hotpotQA", "svamp"]
    )
    parser.add_argument("--datapath", default=None, type=str, help="override default dataset path")
    parser.add_argument(
        "--model_id", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model identifier passed to the API"
    )
    parser.add_argument(
        "--provider", default="featherless", choices=["featherless", "openai"],
        help="API provider"
    )
    parser.add_argument(
        "--uq_engine", default="self-probing-baseline",
        help="used by analyze_result.py to locate confidences/output_v1_<uq_engine>.json"
    )
    parser.add_argument(
        "--output_path", default="output/",
        help="directory for all output files"
    )
    parser.add_argument("--test_start", default="0", help="start index for dataset slice")
    parser.add_argument("--test_end", default="full", help="end index or 'full'")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip already-processed questions instead of starting fresh"
    )

    return parser.parse_args()


args = parse_arguments()
