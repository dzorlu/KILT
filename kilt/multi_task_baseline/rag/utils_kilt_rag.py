import itertools
import json
import linecache
import os
import pickle
import re
import socket
import string
from rouge import Rouge
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import git
import torch
from torch.utils.data import Dataset
from itertools import chain

from datasets import load_from_disk
import numpy as np

from transformers import BartTokenizer, RagTokenizer, T5Tokenizer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def encode_line(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class KILTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        n_obs=None,
        type_path: str="train",
        nb_max_answers: int=5,
        nb_max_wiki_per_answer: int=5,
        prefix=""):
        
        self.task_map = {'wow': 0,
             'eli5': 1,
             'fever': 2,
             'aidayago2': 3,
             'wned': 4,
             'cweb': 5,
             'trex': 6,
             'structured_zeroshot': 7,
             'nq': 8,
             'hotpotqa': 9,
             'triviaqa_support_only': 10
        }
        if type_path == 'val': type_path = 'validation'
        self.type_path = type_path

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # for eval and test.
        self.nb_max_answers = nb_max_answers
        self.nb_max_wiki_per_answer = nb_max_wiki_per_answer
        self.type_path = type_path
        
        self.dataset = load_from_disk(data_dir).get(type_path)

        self.source_tokenizer = (
            tokenizer.question_encoder if isinstance(tokenizer, RagTokenizer)  else tokenizer
        )
        self.target_tokenizer = tokenizer.generator if isinstance(tokenizer, RagTokenizer) else tokenizer
        
        dataset = self.dataset.map(lambda e: self._map(e), batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_masks', 'decoder_input_ids', 'wiki_ids', 'task_names'])
        self.dataset = dataset
        self.src_lens = len(dataset)
        if n_obs is not None:
            self.src_lens = n_obs

    def __len__(self):
        return self.src_lens

    def _map(self, documents: dict, truncation=True, return_tensors='np', padding='max_length') -> dict:
        """Tokenize inputs and labels"""

        input_ids = self.source_tokenizer(
            documents["input_"], 
            truncation=truncation, 
            padding=padding, 
            max_length=self.max_source_length,
            return_tensors=return_tensors
        )
        
        if self.type_path == 'train':
            # single answer
            batch_answers_flat = [o[0]['answer'] for o in documents['output_']]
            batch_wiki_ids = [-1] * len(batch_answers_flat)

            decoder_input_ids = self.target_tokenizer(
                batch_answers_flat,
                truncation=truncation, 
                padding=padding, 
                max_length=self.max_target_length,
                return_tensors=return_tensors
            )['input_ids']
        else:
            # validation and test
            # multiple answers per each output
            batch_answers = list()
            batch_wiki_ids = [list()]
            
            for o in documents['output_']:
                _answers = list()
                _wiki_ids = list()
                for answer in o[:self.nb_max_answers]:
                    _answers.append(answer['answer'])
                    # an answer might have multiple provanences
                    _wiki_ids = list()
                    for wid in answer['wid']:
                        if wid['id'] is not None:
                            _wiki_ids.append(int(wid['id']))
                    if len(_wiki_ids) < self.nb_max_wiki_per_answer:
                        nb_slots = self.nb_max_wiki_per_answer - len(_wiki_ids)
                        [_wiki_ids.append(-1) for _ in range(nb_slots)]
                    batch_wiki_ids.append(_wiki_ids)
                if len(_answers) < self.nb_max_answers:
                    nb_slots = self.nb_max_answers - len(_answers)
                    [_answers.append('') for _ in range(nb_slots)]
                    [batch_wiki_ids.append([-1] * self.nb_max_wiki_per_answer) for _ in range(nb_slots)]
                batch_answers.append(_answers)
                
            batch_answers_flat = list(chain.from_iterable(batch_answers))
            batch_wiki_ids = list(chain.from_iterable(batch_wiki_ids))
            
            batch_wiki_ids = np.array(batch_wiki_ids)

            outputs = self.target_tokenizer(
                batch_answers_flat,
                truncation=truncation, 
                padding=padding, 
                max_length=self.max_target_length,
                return_tensors=return_tensors
            )
            # [B, nb_max_answers, max_length_target]
            decoder_input_ids = outputs['input_ids'].reshape(-1, self.nb_max_answers, self.max_target_length)
            #print(decoder_input_ids.shape)
            # [B, nb_max_answers, nb_max_wiki_per_answer]
            batch_wiki_ids = batch_wiki_ids.reshape(-1, self.nb_max_answers, self.nb_max_wiki_per_answer)
            
        #print(documents['task_name'])
        task_names = np.array([self.task_map[n] for n in documents['task_name']]).squeeze()
        #print(task_names)
        return {
            "input_ids": input_ids['input_ids'],
            "attention_masks": input_ids['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'wiki_ids': batch_wiki_ids,
            'task_names': task_names,
            }

    def __getitem__(self, index):
        _dt = self.dataset[index]
        if not isinstance(_dt['decoder_input_ids'], torch.Tensor):
            decoder_input_ids = torch.stack(_dt['decoder_input_ids'])
            wiki_ids = torch.stack(_dt['wiki_ids'])
        else:
            decoder_input_ids = _dt['decoder_input_ids']
            wiki_ids = _dt['wiki_ids']   
        return {
            "input_ids": _dt['input_ids'],
            'decoder_input_ids':  decoder_input_ids,
            "attention_masks":_dt['attention_masks'],
            'wiki_ids': wiki_ids,
            'task_names': _dt['task_names'],                                     
        }


logger = getLogger(__name__)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, cls=NumpyEncoder)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
        "hostname": str(socket.gethostname()),
    }
    return repo_infos


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def lmap_inv(x: Iterable, y: Iterable) -> List[List]:
    """turn y into x-like"""
    x_like = list()
    _size = len(y) // len(x)
    while y:
        _temp = list()
        for _ in range(_size):
            _temp.append(y.pop(0))
        x_like.append(_temp)
    return x_like

def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def accuracy_score(prediction, ground_truth):
    return prediction == ground_truth
    
def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:
        return 0.0
    return scores["rouge-l"]["f"]


# def calculate_exact_match(output_lns: List[str], reference_lns: List[str]) -> Dict:
#     em = 0
#     for hypo, pred in zip(output_lns, reference_lns):
#         em += exact_match_score(hypo, pred)
#     if len(output_lns) > 0:
#         em /= len(output_lns)
#     return em

def _computeRprec(guess_ids, gold_ids):
    gold_ids = set([g for g in gold_ids if g > 0]) #remove padding
    R = len(gold_ids)
    num = 0

    for prediction in guess_ids[:R]:
        if prediction in gold_ids:
            num += 1

    Rprec = num / R if R > 0 else 0
    return Rprec


# R-precision https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_486
def calculate_rprecision(guess_items: np.ndarray, gold_items: np.ndarray) -> List:
    Rprec_vector_batch = []
    for prs, gts in zip(guess_items, gold_items):
        Rprec_vector = []
        for gt in gts:
            Rprec = _computeRprec(prs, gt)
            Rprec_vector.append(Rprec)
        Rprec_vector_batch.append(max(Rprec_vector))
    return Rprec_vector_batch


def is_rag_model(model_prefix):
    return model_prefix.startswith("rag")



def set_extra_model_params(extra_params, hparams, config):
    equivalent_param = {p: p for p in extra_params}
    # T5 models don't have `dropout` param, they have `dropout_rate` instead
    equivalent_param["dropout"] = "dropout_rate"
    for p in extra_params:
        if getattr(hparams, p, None):
            if not hasattr(config, p) and not hasattr(config, equivalent_param[p]):
                logger.info("config doesn't have a `{}` attribute".format(p))
                delattr(hparams, p)
                continue
            set_p = p if hasattr(config, p) else equivalent_param[p]
            setattr(config, set_p, getattr(hparams, p))
            delattr(hparams, p)
    return hparams, config
