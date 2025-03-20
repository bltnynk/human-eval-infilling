import itertools
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Union

import numpy as np
import tqdm

from human_eval_infilling.data import read_problems, stream_jsonl, write_jsonl
from human_eval_infilling.execution import check_correctness
STOP_WORDS = ["<|endoftext|>", "<|filename|>", "<file_sep>"]
FIM_MIDDLE = '<fim_middle>'
BENCHMARK_NAME = ["single-line", "multi-line", "random-span", "random-span-light"]
# BENCHMARK_NAME = ["single-line"]

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def group_samples_by_benchmark(sample_file):
    samples = dict()
    for sample in tqdm.tqdm(stream_jsonl(sample_file)):
        benchmark = sample["task_name"][len("HumanEval_"):]
        if benchmark not in samples:
            samples[benchmark] = []
        samples[benchmark].append(sample)
    return samples

def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    pass_at_ks = {}
    samples = group_samples_by_benchmark(sample_file)
    for benchmark_name in BENCHMARK_NAME:
        benchmark_samples = samples[benchmark_name]
        problems = read_problems(benchmark_name)
        # Check the generated samples against test suites.
        with ThreadPoolExecutor(max_workers=n_workers) as executor:

            futures = []
            completion_id = Counter()
            n_samples = 0
            results = defaultdict(list)

            print("Reading samples...")
            for sample in tqdm.tqdm(benchmark_samples):
                task_id = sample["task_id"]
                completion = sample["output"]
                completion_start = completion.find(FIM_MIDDLE) + len(FIM_MIDDLE)
                completion = completion[completion_start:]
                completion_end = len(completion)
                for stop_word in STOP_WORDS:
                    if stop_word in completion:
                        completion_end = min(completion_end, completion.find(stop_word))
                completion = completion[:completion_end]
                args = (problems[task_id], completion, timeout, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

            # assert len(completion_id) == len(problems), "Some problems are not attempted."

            print("Running test suites...")
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

        # Calculate pass@k.
        total, correct = [], []
        for result in results.values():
            result.sort()
            passed = [r[1]["passed"] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        ks = k
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()
        }

        # Finally, save the results in one file:
        def combine_results():
            for sample in benchmark_samples:
                task_id = sample["task_id"]
                result = results[task_id].pop(0)
                sample["result"] = result[1]["result"]
                sample["passed"] = result[1]["passed"]
                yield sample

        out_file = sample_file[:sample_file.find('.jsonl')] + f"_{benchmark_name}" + "_results.jsonl"
        print(f"Writing results to {out_file}...")
        write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))
        pass_at_ks[benchmark_name] = pass_at_k

    return pass_at_ks
