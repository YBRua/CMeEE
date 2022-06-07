from typing import List

from transformers import set_seed, BertTokenizer, Trainer, HfArgumentParser, TrainingArguments

from args import ModelConstructArgs, CBLUEDataArgs
from logger import get_logger

from ee_data import (
    GlobalPtrDataset,
    SeqTagDataset,
    CollateFnForNER,
    CollateFnForGlobalPtr,
    EE_NUM_LABELS1,
    EE_NUM_LABELS2,
    EE_NUM_LABELS,
)
from model import (
    BertForCRFHeadNER,
    BertForGlobalPointer,
    BertForLinearHeadNER,
    BertForLinearHeadNestedNER,
    BertForCRFHeadNestedNER
)
from metrics import (
    MetricsForBIOTagging,
    MetricsForNestedBIOTagging,
    MetricsForGlobalPtr,
)
from result_gen import (
    gen_result_bio_tagging,
    gen_result_global_ptr
)

MODEL_CLASS = {
    'linear': BertForLinearHeadNER, 
    'linear_nested': BertForLinearHeadNestedNER,
    'crf': BertForCRFHeadNER,
    'crf_nested': BertForCRFHeadNestedNER,
    'global_ptr': BertForGlobalPointer
}

def get_logger_and_args(logger_name: str, _args: List[str] = None):
    parser = HfArgumentParser([TrainingArguments, ModelConstructArgs, CBLUEDataArgs])
    train_args, model_args, data_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(logger_name, exp_dir=train_args.logging_dir, rank=train_args.local_rank)
    for _log_name, _logger in logger.manager.loggerDict.items():
        # 在4.6.0版本的transformers中无效
        if _log_name.startswith("transformers.trainer"):
            # Redirect other loggers' output
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")

    return logger, train_args, model_args, data_args


def get_model_with_tokenizer(model_args):
    model_class = MODEL_CLASS[model_args.head_type]

    if 'nested' not in model_args.head_type:
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS)
    else:
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1, num_labels2=EE_NUM_LABELS2)
    
    tokenizer = BertTokenizer.from_pretrained(model_args.model_path)
    return model, tokenizer





def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, train_args, model_args, data_args = get_logger_and_args(__name__, _args)

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args)
    for_nested_ner = 'nested' in model_args.head_type

    # ===== Get datasets =====
    if train_args.do_train:
        if model_args.head_type == 'global_ptr':
            train_dataset = GlobalPtrDataset(
                data_args.cblue_root,
                "train", data_args.max_length,
                tokenizer, for_nested_ner=for_nested_ner)
            dev_dataset = GlobalPtrDataset(
                data_args.cblue_root,
                "dev", data_args.max_length,
                tokenizer, for_nested_ner=for_nested_ner)
        else:
            train_dataset = SeqTagDataset(
                data_args.cblue_root,
                "train", data_args.max_length,
                tokenizer, for_nested_ner=for_nested_ner)
            dev_dataset = SeqTagDataset(
                data_args.cblue_root,
                "dev", data_args.max_length,
                tokenizer, for_nested_ner=for_nested_ner)
        logger.info(f"Trainset: {len(train_dataset)} samples")
        logger.info(f"Devset: {len(dev_dataset)} samples")
    else:
        train_dataset = dev_dataset = None

    # ===== Trainer =====
    if model_args.head_type == 'global_ptr':
        compute_metrics = MetricsForGlobalPtr()
        collate_fn = CollateFnForGlobalPtr(tokenizer.pad_token_id)
    else:
        compute_metrics = MetricsForNestedBIOTagging() if for_nested_ner else MetricsForBIOTagging()
        collate_fn = CollateFnForNER(tokenizer.pad_token_id, for_nested_ner=for_nested_ner)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    if train_args.do_train:
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    if train_args.do_predict:
        if model_args.head_type == 'global_ptr':
            test_dataset = GlobalPtrDataset(
                data_args.cblue_root,
                "test", data_args.max_length,
                tokenizer, for_nested_ner=for_nested_ner)
        else:
            test_dataset = SeqTagDataset(
                data_args.cblue_root,
                "test", data_args.max_length,
                tokenizer, for_nested_ner=for_nested_ner)
        logger.info(f"Testset: {len(test_dataset)} samples")

        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")

        if model_args.head_type == 'global_ptr':
            gen_result_global_ptr(
                train_args, logger,
                predictions, test_dataset,
                for_nested_ner=for_nested_ner)
        else:
            gen_result_bio_tagging(
                train_args, logger,
                predictions, test_dataset,
                for_nested_ner=for_nested_ner)


if __name__ == '__main__':
    main()
