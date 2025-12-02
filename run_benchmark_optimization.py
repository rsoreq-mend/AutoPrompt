"""
Benchmark Optimization Script

Iteratively optimizes a prompt using a fixed annotated dataset.
The dataset should contain 'text' and 'annotation' columns.

Usage:
    python run_benchmark_optimization.py \
        --dataset my_data.csv \
        --prompt "Your initial prompt" \
        --task_description "Task description" \
        --labels Yes No \
        --num_steps 10 \
        --output results.json
"""

import argparse
import json
import logging

from benchmark_optimizer import BenchmarkOptimizer
from utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser(
        description='Optimize a prompt using a fixed benchmark dataset'
    )

    parser.add_argument(
        '--config',
        default='config/config_benchmark.yml',
        type=str,
        help='Configuration file path'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        help='Path to CSV with text and annotation columns'
    )
    parser.add_argument(
        '--prompt',
        default='',
        required=False,
        type=str,
        help='Initial prompt to optimize'
    )
    parser.add_argument(
        '--task_description',
        default='',
        required=False,
        type=str,
        help='Description of the classification task'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        required=False,
        help='Label schema (e.g., Yes No)'
    )
    parser.add_argument(
        '--num_steps',
        default=10,
        type=int,
        help='Number of optimization iterations'
    )
    parser.add_argument(
        '--output',
        default='benchmark_results.json',
        type=str,
        help='Output JSON file path'
    )

    opt = parser.parse_args()

    # Load configuration
    config = load_yaml(opt.config)

    # Override label schema if provided
    if opt.labels:
        config.dataset.label_schema = opt.labels

    # Get task description interactively if not provided
    if opt.task_description == '':
        task_description = input("Describe the task: ")
    else:
        task_description = opt.task_description

    # Get initial prompt interactively if not provided
    if opt.prompt == '':
        initial_prompt = input("Initial prompt: ")
    else:
        initial_prompt = opt.prompt

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print('\n' + '='*60)
    print('BENCHMARK OPTIMIZATION')
    print('='*60)
    print(f'Dataset: {opt.dataset}')
    print(f'Labels: {config.dataset.label_schema}')
    print(f'Max iterations: {opt.num_steps}')
    print('='*60 + '\n')

    # Initialize and run optimizer
    optimizer = BenchmarkOptimizer(config, task_description, initial_prompt)
    result = optimizer.run(opt.dataset, opt.num_steps)

    # Print results
    print('\n' + '='*60)
    print('OPTIMIZATION COMPLETE')
    print('='*60)
    print(f'\033[92mInitial Score: {result["initial_score"]:.2%}\033[0m')
    print(f'\033[92mBest Score: {result["best_score"]:.2%}\033[0m')
    print(f'Iterations: {result["num_iterations"]}')
    print(f'Total Usage: ${result["total_usage"]:.4f}')
    print('\n\033[92mBest Prompt:\033[0m')
    print(result['best_prompt'])
    print('='*60)

    # Save results to JSON
    with open(opt.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nResults saved to: {opt.output}')


if __name__ == '__main__':
    main()
