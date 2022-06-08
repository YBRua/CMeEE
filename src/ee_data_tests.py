import torch
from transformers import BertTokenizer
from ee_data import (
    SeqTagDataset, CollateFnForSeqTag,
    GlobalPtrDataset, CollateFnForGlobalPtr,
    W2NERDataset, CollateFnForW2NER
)

from metrics import decode_w2matrix


if __name__ == '__main__':
    # Unit tests for dataloaders and collate functions
    MODEL_NAME = "../bert-base-chinese"
    CBLUE_ROOT = "../data/CBLUEDatasets"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    testees = [
        # (SeqTagDataset, CollateFnForSeqTag),
        # (GlobalPtrDataset, CollateFnForGlobalPtr),
        (W2NERDataset, CollateFnForW2NER)]
    modes = ['dev', 'test']
    MAX_LEN = 20
    BATCH_SIZE = 8

    for dataset, collator in testees:
        for mode in modes:
            print(f'Testing {dataset.__name__}. Mode: {mode}')
            dset = dataset(CBLUE_ROOT, mode=mode, max_length=MAX_LEN, tokenizer=tokenizer, for_nested_ner=False)
            batch = [dset[i] for i in range(BATCH_SIZE)]
            inputs = collator(pad_token_id=tokenizer.pad_token_id, for_nested_ner=False)(batch)
            for key, value in inputs.items():
                if value is None:
                    print(f'  {key} is None')
                elif isinstance(value, torch.Tensor):
                    print(f'  {key} (Tensor): {value.shape}')
                else:
                    print(f'  {key} ({str(type(value))}): {value}')
            if mode == 'dev':
                print('rel pos')
                print(inputs['rel_pos'][2])
                print('grid mask')
                print(inputs['grid_mask'][2])
                print('wordpair label')
                print(inputs['labels'][2])
                print('decoded')
                print(decode_w2matrix(inputs['labels'], inputs['text_len']))