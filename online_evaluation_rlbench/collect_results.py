import argparse
import json
import os


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    parser.add_argument('--folder', type=str)

    return parser.parse_args()


args = parse_arguments()
FOLDER = args.folder

sum_ = 0
tasks = sorted(os.listdir(FOLDER))
results = []
for folder in tasks:
    eval_file = f'{FOLDER}/{folder}/eval.json'
    if not os.path.exists(eval_file):
        print(f"Warning: {folder} - eval.json not found, skipping")
        continue
    try:
        with open(eval_file) as fid:
            res = 100 * json.load(fid)[folder]["mean"]
        results.append(res)
        print(folder, res)
        sum_ += res
    except Exception as e:
        print(f"Warning: {folder} - failed to load results: {e}")

if len(results) > 0:
    print(f'Mean on {len(results)} tasks: {sum_ / len(results):.2f}')
else:
    print(f'No valid results found in {len(tasks)} tasks')
