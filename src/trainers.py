import time
import math
import torch
import collections
from torch.utils.data import Dataset
from transformers.trainer import Trainer
from transformers.utils import logging
from transformers.trainer_utils import speed_metrics, PredictionOutput

from metrics import MetricsForGlobalPtr

from typing import Optional, List, Dict


logger = logging.get_logger(__name__)


class GlobalPtrTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(
            self, test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test") -> PredictionOutput:
        self._memory_tracker.start()
        dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()
        model = self._wrap_model(self.model, training=False)
        batch_size = dataloader.batch_size

        prediction_loss_only = self.args.prediction_loss_only

        tot_predictions = []
        model.eval()
        with torch.no_grad():
            num_samples = 0
            for step, inputs in enumerate(dataloader):
                num_samples += len(inputs['input_ids'])
                loss, logits, labels = self.prediction_step(
                    model, inputs, prediction_loss_only,
                    ignore_keys=ignore_keys)

                tot_predictions.extend(logits)

        metrics = {}

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size)))
        return PredictionOutput(
            predictions=tot_predictions, metrics=metrics, label_ids=None)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval") -> Dict[str, float]:
        self._memory_tracker.start()
        dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        model = self._wrap_model(self.model, training=False)
        batch_size = dataloader.batch_size

        prediction_loss_only = self.args.prediction_loss_only

        logger.info("***** Running Evaluation *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        with torch.no_grad():
            num_samples = 0
            tot_loss = 0.
            f1_metric = MetricsForGlobalPtr()
            for step, inputs in enumerate(dataloader):
                batch_size = len(inputs['input_ids'])
                num_samples += batch_size
                loss, logits, labels = self.prediction_step(
                    model, inputs, prediction_loss_only,
                    ignore_keys=ignore_keys)
                tot_loss += loss.item() * batch_size
                f1_metric.accumulate(logits, labels)
        avg_loss = tot_loss / num_samples
        metrics = f1_metric.summary()

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )
        metrics['eval_loss'] = avg_loss

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
