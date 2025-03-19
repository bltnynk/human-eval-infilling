import argparse
from human_eval_infilling.data import read_problems, stream_jsonl, write_jsonl
import copy

BENCHMARK_NAME = ["single-line", "multi-line", "random-span", "random-span-light"]
STOP_WORDS = ["<|endoftext|>", "<|filename|>", "<file_sep>"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HumanEval Task Inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--n_copies",
        type=int,
        default=1,
        help="Number of samples to solve and evaluate from the benchmark",
    )

    args = parser.parse_args()
    
    return args

def generate_input_text(sample):
    return '<fim_prefix>' + sample['prompt'] + '<fim_suffix>' + sample['suffix'] + '<fim_middle>'

if __name__ == "__main__":
    args = parse_args()
    inputs = []
    for benchmark in BENCHMARK_NAME:
        problems = read_problems(benchmark)
        for i, (id, sample) in enumerate(problems.items()):
            if args.limit and i >= args.limit:
                break
            for copy_ind in range(args.n_copies):
                input = copy.deepcopy(sample)
                input['task_name'] = "HumanEval_" + benchmark
                input['id'] = id + f"_copy{copy_ind}"
                input['copy'] = copy_ind
                input['text'] = generate_input_text(sample)
                input['stop_words'] = STOP_WORDS
                inputs.append(input)
    write_jsonl("../data/HumanEval-Inputs.json", inputs)
    print('Wrote tasks to data/HumanEval-Inputs.json')
