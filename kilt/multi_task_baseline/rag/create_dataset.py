import sys
import os
os.environ['HF_HOME']='/hdd/'
from datasets import load_dataset

import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import torch
from datasets import Features, Sequence, Value, load_dataset

import faiss
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)


logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

CONTEXT_ENCODER_PATH = 'facebook/dpr-ctx_encoder-multiset-base'
WIKI_PATH = '/hdd/kilt_wiki'
Path(WIKI_PATH).mkdir(parents=True, exist_ok=True)

KILT_TASKS_DATA = "/hdd/kilt_tasks2"
Path(KILT_TASKS_DATA).mkdir(parents=True, exist_ok=True)


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    wikipedia_ids, titles, texts = [], [], []
    for title, text, _id in zip(documents["wikipedia_title"], documents['text'], documents['wikipedia_id']):
        if text is not None:
            # convert from list to str
            paragraphs= text['paragraph']
            text = " ".join(paragraphs)
            for passage in split_text(text):
                wikipedia_ids.append(_id)
                texts.append(passage)
                titles.append(title)
    return {"title": titles, "text": texts, 'wikipedia_id': wikipedia_ids}


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}

def compute_passage_embeddings():
    # compute the embeddings
    ctx_encoder = DPRContextEncoder.from_pretrained(CONTEXT_ENCODER_PATH).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(CONTEXT_ENCODER_PATH)
    new_features = Features(
        {"text": Value("string"), "title": Value("string"), "wikipedia_id": Value("string"), "embeddings": Sequence(Value("float32"))}
    )  # optional, save as float32 instead of float64 to save space

    kilt_wiki = load_dataset("kilt_wikipedia", data_dir=WIKI_PATH, split='full')

    # process the dataset
    # split into passages
    dataset = kilt_wiki
    dataset = dataset.map(split_documents, batched=True, remove_columns=kilt_wiki.column_names, num_proc=8)
    # # embed the docs
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=100,
        features=new_features,
    )
    return dataset

def create_index(dataset):
    D = 768

    # compression ratio is
    # (D x 32) / (M x nbits)

    # IVF is the filtering step. nprobe specified at test time.

    # Param of PQ
    M = 128  # The number of sub-vector.
    nbits = 8 # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte

    # Param of IVF
    nlist = 2048  # The number of cells (space partition), according to the quantizer. Typical value is sqrt(N)
    # Param of HNSW
    hnsw_m = 64  # The number of neighbors for HNSW. This is typically 32

    # Setup
    quantizer = faiss.IndexHNSWFlat(D, hnsw_m)
    index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT)
    print(f"default value for nprobe is : {index.nprobe}")
    index.nprobe = 8

    dataset.add_faiss_index(column='embeddings', custom_index=index, train_size=200_000)

    _index_path = os.path.join(WIKI_PATH, 'index.faiss')
    dataset.save_faiss_index('embeddings', file=_index_path)
    dataset.save_to_disk(WIKI_PATH)

def get_kilt_dataset():
    data = dict()
    # WOW
    
    data['wow'] = load_dataset("kilt_tasks", name="wow", data_dir='/hdd/kilt/')
    data['wow'].save_to_disk('/hdd/kilt_tasks2/wow')
    # eli5
    data['eli5'] = load_dataset("kilt_tasks", name="eli5", data_dir='/hdd/kilt/')
    data['eli5'].save_to_disk('/hdd/kilt_tasks2/eli5')
    # fever
    data['fever'] = load_dataset("kilt_tasks", name="fever", data_dir='/hdd/kilt/')
    data['fever'].save_to_disk('/hdd/kilt_tasks2/fever')
    # aidayago2
    data['aidayago2'] = load_dataset("kilt_tasks", name="aidayago2", data_dir='/hdd/kilt/')
    data['aidayago2'].save_to_disk('/hdd/kilt_tasks2/aidayago2')
    # wned
    data['wned'] = load_dataset("kilt_tasks", name="wned", data_dir='/hdd/kilt/')
    data['wned'].save_to_disk('/hdd/kilt_tasks2/wned')
    # cweb
    data['cweb'] = load_dataset("kilt_tasks", name="cweb", data_dir='/hdd/kilt/')
    data['cweb'].save_to_disk('/hdd/kilt_tasks2/cweb')
    # trex
    data['trex'] = load_dataset("kilt_tasks", name="trex", data_dir='/hdd/kilt/')
    data['trex'].save_to_disk('/hdd/kilt_tasks2/trex')
    # structured_zeroshot
    data['structured_zeroshot'] = load_dataset("kilt_tasks", name="structured_zeroshot", data_dir='/hdd/kilt/')
    data['structured_zeroshot'].save_to_disk('/hdd/kilt_tasks2/structured_zeroshot')
    # nq
    data['nq'] = load_dataset("kilt_tasks", name="nq", data_dir='/hdd/kilt/')
    data['nq'].save_to_disk('/hdd/kilt_tasks2/nq')
    # hotpotqa
    data['hotpotqa'] = load_dataset("kilt_tasks", name="hotpotqa", data_dir='/hdd/kilt/')
    data['hotpotqa'].save_to_disk('/hdd/kilt_tasks2/hotpotqa')
    # triviaqa_support_only
    triviaqa_support_only = load_dataset("kilt_tasks2", name="triviaqa_support_only", data_dir='/hdd/kilt/')
    trivia_qa = load_dataset('trivia_qa', 'unfiltered.nocontext')
    # The KILT IDs can then be mapped to the TriviaQA questions with:
    triviaqa_map = {}

    for k in ['train', 'validation', 'test']:
        triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])
        triviaqa_support_only[k] = triviaqa_support_only[k].filter(lambda x: x['id'] in triviaqa_map)
        triviaqa_support_only[k] = triviaqa_support_only[k].map(lambda x: {'input': trivia_qa[k][triviaqa_map[x['id']]]['question']})
    triviaqa_support_only.save_to_disk("/hdd/kilt_tasks2/triviaqa_support_only")
    data['triviaqa_support_only'] = triviaqa_support_only
    print(data)


    # TRAIN DATASET
    from collections import defaultdict
    max_number_samples = 100_000

    assert len(data.keys()) == 11


    from datasets import Features, Sequence, Value, load_dataset
    from functools import partial

    new_features = Features(
        {"input_": Value("string"), "output_": Value("string"), "id": Value("string"), 'task_name': Value("string")}
    )


    def extract_text(documents: dict, task_name: str) -> dict:
        x = [task_name + ' ' +  i for i in documents['input']]
        task_names = [task_name] * len(x)
        try:
            labels =  [o[0]['answer'] if 'answer' in o[0] else '' for o in documents['output'] ]
        except:
            labels = [''] * len(x)
        return {'output_': labels, 'input_': x, "id": documents['id'], "task_name": task_names}


    # process the dataset
    from datasets import Dataset

    processed_data = dict()
    for task_name, _dataset_dict in data.items():
        # cap the number of examples
        if 'train' in _dataset_dict:
            _dataset_dict['train'] = Dataset.from_dict(_dataset_dict['train'][:max_number_samples])
        _dataset_dict = _dataset_dict.map(
            partial(extract_text, task_name=task_name),
            batched=True,
            batch_size=100,
            features=new_features,
            remove_columns=['id', 'input', 'meta', 'output']
        )
        processed_data[task_name] = _dataset_dict

    # concat dataset
    from datasets import concatenate_datasets
    from datasets import DatasetDict

    _dict = dict()
    for data_ref in ['train','validation','test']:
        dts = [d[data_ref] for d in processed_data.values() if data_ref in d]
        _dict[data_ref] = concatenate_datasets(dts)

    kilt_test_dataset = DatasetDict(_dict)
    kilt_test_dataset.save_to_disk(KILT_TASKS_DATA)



if __name__ == "__main__":
    # index
    dataset = compute_passage_embeddings()
    create_index(dataset)
    # dataset
    get_kilt_dataset()
