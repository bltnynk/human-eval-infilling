import sys

import fire
import json

from human_eval_infilling.evaluation import evaluate_functional_correctness

def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_copies: int = 1,
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"

    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout)
    results_file = sample_file[:sample_file.find("output")] + "eval.jsonl"
    with open(results_file, "w") as f:
        f.write(json.dumps(results, indent=2))
    print(f"Results saved to {results_file}")
    print(results)


def main():
    fire.Fire(entry_point)


sys.exit(main())
