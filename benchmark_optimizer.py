import pandas as pd
import json

from eval.evaluator import Eval
from dataset.base_dataset import DatasetBase
from utils.llm_chain import MetaChain
from estimator import give_estimator


class BenchmarkOptimizer:
    """
    Iterative prompt optimizer using a fixed benchmark dataset.

    This class implements a simplified optimization loop:
    1. Load existing annotated dataset (text + annotation columns)
    2. For each iteration:
       - PREDICT: Run current prompt against all samples
       - EVALUATE: Compare predictions vs ground truth annotations
       - REFINE: Generate improved prompt based on error analysis
    3. Track and return the best performing prompt
    """

    def __init__(self, config, task_description: str, initial_prompt: str):
        """
        Initialize the BenchmarkOptimizer.

        :param config: Configuration EasyDict
        :param task_description: Description of the classification task
        :param initial_prompt: Initial prompt to optimize
        """
        self.config = config
        self.task_description = task_description
        self.cur_prompt = initial_prompt

        # Initialize components
        self.meta_chain = MetaChain(config)
        self.predictor = give_estimator(config.predictor)
        self.dataset: DatasetBase | None = None
        self.eval: Eval | None = None

        # Tracking
        self.history: list[dict] = []
        self.best_prompt = initial_prompt
        self.best_score = 0.0
        self.patient = 0

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load a CSV dataset with required columns: text, annotation

        :param dataset_path: Path to the CSV file
        :return: Loaded DataFrame
        """
        df = pd.read_csv(dataset_path, dtype={'annotation': str})

        # Validate required columns
        required_cols = ['text', 'annotation']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}. "
                           f"Found columns: {list(df.columns)}")

        # Add required columns for compatibility
        if 'id' not in df.columns:
            df['id'] = range(len(df))
        if 'batch_id' not in df.columns:
            df['batch_id'] = 0
        if 'prediction' not in df.columns:
            df['prediction'] = ''
        if 'metadata' not in df.columns:
            df['metadata'] = ''
        if 'score' not in df.columns:
            df['score'] = 0.0

        return df

    def run_step_prompt(self) -> str:
        """
        Generate a new prompt suggestion based on error analysis.

        :return: The new suggested prompt
        """
        if self.eval is None:
            raise RuntimeError("Evaluator not initialized. Call run() first.")

        step_num = len(self.eval.history)

        # Get history for the meta-prompt
        if step_num < self.config.meta_prompts.warmup or (step_num % 3) > 0:
            last_history = self.eval.history[-self.config.meta_prompts.history_length:]
        else:
            sorted_history = sorted(
                self.eval.history[max(0, self.config.meta_prompts.warmup - 1):],
                key=lambda x: x['score'],
                reverse=False
            )
            last_history = sorted_history[-self.config.meta_prompts.history_length:]

        # Build history prompt
        history_prompt = '\n'.join([
            self.eval.sample_to_text(
                sample,
                num_errors_per_label=self.config.meta_prompts.num_err_prompt,
                is_score=True
            ) for sample in last_history
        ])

        # Prepare input for step_prompt chain
        prompt_input = {
            "history": history_prompt,
            "task_description": self.task_description,
            "error_analysis": last_history[-1]['analysis']
        }

        if 'label_schema' in self.config.dataset.keys():
            prompt_input["labels"] = json.dumps(self.config.dataset.label_schema)

        # Get new prompt suggestion
        prompt_suggestion = self.meta_chain.step_prompt_chain.invoke(prompt_input)

        # Handle Google LLM response format
        if self.meta_chain.step_prompt_chain.llm_config.type == 'google':
            if isinstance(prompt_suggestion, list) and len(prompt_suggestion) == 1:
                prompt_suggestion = prompt_suggestion[0]['args']

        return prompt_suggestion['prompt']

    def stop_criteria(self) -> bool:
        """
        Check if stopping criteria is met.

        :return: True if should stop, False otherwise
        """
        if self.eval is None:
            return False

        # Check usage limit
        if 0 < self.config.stop_criteria.max_usage < self.calc_usage():
            print('Stop: Max usage reached')
            return True

        # Check patience (no improvement for N steps)
        if len(self.eval.history) <= self.config.meta_prompts.warmup:
            self.patient = 0
            return False

        current_score = self.eval.history[-1]['score']
        if current_score > self.best_score + self.config.stop_criteria.min_delta:
            self.patient = 0
        else:
            self.patient += 1

        if self.patient > self.config.stop_criteria.patience:
            print(f'Stop: No improvement for {self.patient} steps')
            return True

        return False

    def calc_usage(self) -> float:
        """Calculate total usage cost."""
        total_usage = self.meta_chain.calc_usage()
        total_usage += self.predictor.calc_usage()
        return total_usage

    def run(self, dataset_path: str, num_steps: int = 10) -> dict:
        """
        Run the optimization loop.

        :param dataset_path: Path to CSV with text + annotation columns
        :param num_steps: Maximum number of optimization iterations
        :return: Dict with best_prompt, best_score, history, etc.
        """
        # Load dataset
        df = self.load_dataset(dataset_path)
        print(f'Loaded {len(df)} samples')

        # Initialize dataset wrapper
        self.dataset = DatasetBase(self.config.dataset)
        self.dataset.records = df
        self.dataset.label_schema = self.config.dataset.label_schema

        # Initialize evaluator
        self.eval = Eval(
            self.config.eval,
            self.meta_chain.error_analysis,
            self.config.dataset.label_schema
        )

        # Track initial state
        self.best_prompt = self.cur_prompt
        self.best_score = 0.0
        initial_score = None

        for step in range(num_steps):
            print(f'\n=== Step {step + 1}/{num_steps} ===')

            # PREDICT: Run current prompt against all samples
            if hasattr(self.predictor, 'cur_instruct'):
                setattr(self.predictor, 'cur_instruct', self.cur_prompt)
            records = self.predictor.apply(self.dataset, 0)
            if isinstance(records, pd.DataFrame) and len(records) > 0:
                self.dataset.update(records)

            # EVALUATE: Compare predictions vs annotations
            self.eval.dataset = self.dataset.records.copy()
            score = self.eval.eval_score()
            errors = self.eval.extract_errors()
            self.eval.add_history(self.cur_prompt, self.task_description)

            if initial_score is None:
                initial_score = score

            # Track history
            self.history.append({
                'iteration': step + 1,
                'prompt': self.cur_prompt,
                'score': score,
                'num_errors': len(errors)
            })

            print(f'Score: {score:.2%} | Errors: {len(errors)}/{len(self.dataset.records)}')

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_prompt = self.cur_prompt
                print(f'New best: {score:.2%}')

            # Check stopping criteria
            if self.stop_criteria():
                print('Stopping criteria reached.')
                break

            # REFINE: Generate improved prompt (except on last iteration)
            if step < num_steps - 1:
                self.cur_prompt = self.run_step_prompt()

        # Return results
        return {
            'best_prompt': self.best_prompt,
            'best_score': self.best_score,
            'initial_score': initial_score,
            'num_iterations': len(self.history),
            'history': self.history,
            'initial_prompt': self.history[0]['prompt'] if self.history else self.cur_prompt,
            'task_description': self.task_description,
            'total_samples': len(self.dataset.records),
            'total_usage': self.calc_usage()
        }
