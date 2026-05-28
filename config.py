import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_length_cot", type=int, default=256,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--try_times", type=int, default=5,
        help="try times for meaningful reasoning process"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help=""
    )
    parser.add_argument(
        '--dataset', default='NLI',
        help="dataset",
        choices=["ASDiv", "2WikimhQA", "gsm8k", "hotpotQA", "logiQA", "AddSub", "SingleEq", "CommonsenseQA", "StrategyQA", "CAD", "TriviaQA", "Math", "NLI", "svamp"]
    )
    parser.add_argument(
        "--datapath", default=None, type=str, help='file path'
    )
    parser.add_argument(
        "--api_key", default="", type = str, help='gpt api_key'
    )
    parser.add_argument(
        "--model_engine", default='llama2-7b', help="model engine",
        choices=["llama3-1_8B", "llama2-13b"]
    )
    parser.add_argument(
        "--uq_engine", default='self-probing-keystep', help="uncertainty quantification engine",
        choices=[
            "self-probing-baseline", "self-probing-keyword",
            "self-probing-allkeyword", "self-probing-keystep", "self-probing-allstep",
        ]
    )
    parser.add_argument(
        "--keyword_threshold", type=float, default=0.5,
        help="minimum contribution score fraction (0-1) to include a keyword in self-probing-keyword"
    )
    parser.add_argument(
        "--test_start", default='0', help='string, number'
    )
    parser.add_argument(
        "--test_end", default='1000', help='string, number'
    )
    parsed_args = parser.parse_args()
    parsed_args.output_path = f"output/{parsed_args.model_engine}/{parsed_args.dataset}"
    return parsed_args


args = parse_arguments()
