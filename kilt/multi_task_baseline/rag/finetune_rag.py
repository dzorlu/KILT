"""Finetuning script for RAG models. Adapted from examples.seq2seq.finetune.py"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from itertools import chain

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.accelerators.ddp_accelerator import DDPAccelerator
from pytorch_lightning.cluster_environments import TorchElasticEnvironment
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    BatchEncoding,
    RagConfig,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    T5ForConditionalGeneration,
)
from transformers import logging as transformers_logging
from transformers.integrations import is_ray_available


if is_ray_available():
    import ray
    from distributed_ray_retriever import RagRayDistributedRetriever, RayRetriever


from callbacks_rag import (  # noqa: E402 # isort:skipq
    get_checkpoint_callback,
    get_early_stopping_callback,
    Seq2SeqLoggingCallback,
)

from distributed_pytorch_retriever import RagPyTorchDistributedRetriever  # noqa: E402 # isort:skip
from utils_kilt_rag import (  # noqa: E402 # isort:skip
    calculate_rprecision,
    exact_match_score,
    metric_max_over_ground_truths,
    f1_score,
    rougel_score,
    lmap_inv,
    accuracy_score,
    flatten_list,
    get_git_info,
    is_rag_model,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    set_extra_model_params,
    KILTDataset,
    NumpyEncoder,
)

from torch.utils.data import Dataset

# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa

torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_info()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# In PTL >v1.0, `init_ddp_connection` method in the `LightningModule`
# is no longer used, and is moved into DDPAccelerator instead.
# We override DDPAccelerator to add our custom logic for initializing the
# retriever.
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/backends/test_accelerator_connector.py


class CustomAccel(DDPAccelerator):
    def __init__(self, trainer=None, **kwargs):
        # Trainer is set later.
        super().__init__(trainer, **kwargs)

    def init_ddp_connection(self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True):
        logger.info("Custom init_ddp_connection.")
        module = self.trainer.model
        if self.cluster_environment is None:
            self.cluster_environment = TorchElasticEnvironment()
        self.distributed_port = module.hparams.distributed_port
        os.environ["MASTER_PORT"] = str(self.distributed_port)
        super().init_ddp_connection(global_rank, world_size, is_slurm_managing_tasks)
        if module.is_rag_model:
            if module.distributed_retriever == "pytorch":
                module.model.rag.retriever.init_retrieval(self.distributed_port)
            elif module.distributed_retriever == "ray" and global_rank == 0:
                # For the Ray retriever, only initialize it once when global
                # rank is 0.
                module.model.rag.retriever.init_retrieval()


class KILTModule(BaseTransformer):
    mode = "kilt"
    loss_names = ["loss"]
    val_metric = "em"

    def __init__(self, hparams, **kwargs):
        # when loading from a pytorch lightning checkpoint, hparams are passed as dict
        if isinstance(hparams, dict):
            hparams = AttrDict(hparams)
        if hparams.model_type == "rag_sequence":
            self.model_class = RagSequenceForGeneration
        elif hparams.model_type == "rag_token":
            self.model_class = RagTokenForGeneration
        elif hparams.model_type == "bart":
            self.model_class = BartForConditionalGeneration
        else:
            self.model_class = T5ForConditionalGeneration
        self.is_rag_model = is_rag_model(hparams.model_type)

        config_class = RagConfig if self.is_rag_model else AutoConfig
        config = config_class.from_pretrained(hparams.model_name_or_path)

        # set retriever parameters
        config.index_name = hparams.index_name or config.index_name
        config.passages_path = hparams.passages_path or config.passages_path
        config.index_path = hparams.index_path or config.index_path
        #config.use_dummy_dataset = True

        # set extra_model_params for generator configs and load_model
        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "attention_dropout", "dropout")
        if self.is_rag_model:
            if hparams.prefix is not None:
                config.generator.prefix = hparams.prefix
            config.label_smoothing = hparams.label_smoothing
            hparams, config.generator = set_extra_model_params(extra_model_params, hparams, config.generator)
            #initialize custom retriever
            retriever = RagPyTorchDistributedRetriever.from_pretrained(
                hparams.model_name_or_path, 
                config=config,
            )
        
            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config, retriever=retriever)
            
            prefix = config.question_encoder.prefix
        else:
            if hparams.prefix is not None:
                config.prefix = hparams.prefix
            hparams, config = set_extra_model_params(extra_model_params, hparams, config)
            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config)
            prefix = config.prefix

        tokenizer = (
            RagTokenizer.from_pretrained(hparams.model_name_or_path)
            if self.is_rag_model
            else AutoTokenizer.from_pretrained(hparams.model_name_or_path)
        )

        super().__init__(hparams, config=config, tokenizer=tokenizer, model=model)

        save_git_info(self.hparams.output_dir)
        self.output_dir = Path(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.distributed_port = self.hparams.distributed_port

        # For single GPU training, init_ddp_connection is not called.
        # So we need to initialize the retrievers here.
        if hparams.gpus <= 1:
            if hparams.distributed_retriever == "ray":
                self.model.retriever.init_retrieval()
            elif hparams.distributed_retriever == "pytorch":
                self.model.retriever.init_retrieval(self.distributed_port)

        self.distributed_retriever = hparams.distributed_retriever

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs) 
    
    def maybe_reshape(self, generated_ids):
        if len(generated_ids.shape) == 3:
            generated_ids = generated_ids.reshape(-1, generated_ids.shape[-1])
        return generated_ids
            
    def ids_to_clean_text(self, generated_ids: List[int]):
        generated_ids = self.maybe_reshape(generated_ids)
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict, prefix: str) -> float:
        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_masks"], batch["decoder_input_ids"]

        rag_kwargs = {}
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(target_ids)
            lm_labels = target_ids
        elif isinstance(self.model, BartForConditionalGeneration):
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
        else:
            assert self.is_rag_model
            generator = self.model.rag.generator
            if isinstance(generator, T5ForConditionalGeneration):
                decoder_start_token_id = generator.config.decoder_start_token_id
                decoder_input_ids = (
                    torch.cat(
                        [torch.Tensor([[decoder_start_token_id]] * target_ids.shape[0]).to(target_ids), target_ids],
                        dim=1,
                    )
                    if target_ids.shape[0] < self.target_lens["train"]
                    else generator._shift_right(target_ids)
                )
            elif isinstance(generator, BartForConditionalGeneration):
                decoder_input_ids = target_ids
            lm_labels = decoder_input_ids
            rag_kwargs["reduce_loss"] = True

        assert decoder_input_ids is not None
        outputs = self(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            labels=lm_labels,
            **rag_kwargs,
        )
        loss = outputs["loss"]
        return loss

    @property
    def pad(self) -> int:
        raise NotImplementedError("pad not implemented")
    
    ##################
    # step functions #
    ##################

    def training_step(self, batch, batch_idx) -> Dict:
        #self.log('train_bla', 1)
        loss = self._step(batch, 'train')
        self.log('train_loss', loss)
        return loss


    def test_step(self, batch, batch_idx):
        return self._generative_step(batch, 'test')

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch, 'val')

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, 'test')


    def validation_epoch_end(self, outputs, prefix='val') -> Dict:
        self.step_count += 1
        metrics = [list(b.keys()) for b in outputs]
        metrics = set(chain.from_iterable(metrics))
        epoch_metrics = dict()
        for m in metrics:
            magg = list()
            for o in outputs:
                # a step might not have all metric combinations.
                if m in o:
                    magg.extend(o[m])
            epoch_metrics[m] = np.array(magg).mean()
        epoch_metrics['step_count'] = self.step_count
        # writes epoch metrics to self.metrics_save_path
        #self.save_metrics(epoch_metrics, prefix)
        #logger.info(f"epoch_metrics:{epoch_metrics}")
        self.log_dict(epoch_metrics)

    # def save_metrics(self, latest_metrics, type_path) -> None:
    #     self.metrics[type_path].append(latest_metrics)
    #     save_json(self.metrics, self.metrics _save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        if len(preds) != len(target):
            target = lmap_inv(preds, target)
        batch_em, batch_f1, batch_acc, batch_rougel = [], [], [], []
        for pr, gt in zip(preds, target): #List[str], List[List[str]]
            #total += 1
            # exact match
            batch_em.append(
                metric_max_over_ground_truths(exact_match_score, pr, gt)
            )
            # f1 metric
            batch_f1.append(
                metric_max_over_ground_truths(f1_score, pr, gt)         
            )
            # accuracy (strict exact match)
            batch_acc.append(
                metric_max_over_ground_truths(accuracy_score, pr, gt)
            )
            # rougel_score
            batch_rougel.append(
                metric_max_over_ground_truths(rougel_score, pr, gt)
            )

        return {
            f'em': batch_em, 
            f'f1':batch_f1,
            f'acc': batch_acc,
            f'rougel': batch_rougel,
            #f'total': total
        }

    def calculate_downstream_metrics(self, batch: dict) -> Dict:
        # downstream
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_masks"],
            do_deduplication=False,  # rag specific parameter
            use_cache=True,
            min_length=1,
            max_length=self.target_lens["val"],
        )
        preds: List[str] = self.ids_to_clean_text(generated_ids) #[B]

        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"]) #[BxA]
        gen_metrics: Dict = self.calc_generative_metrics(preds, target)
        return gen_metrics


    def calculate_retrieval_metrics(self, batch: dict, n_docs: int=5) -> Dict:
        # retrieval
        question_enc_outputs = self.model.rag.question_encoder(
            batch["input_ids"], 
            attention_mask=batch["attention_masks"]
        )
        # faiss requires np.ndarray.
        question_enc_pool_output = question_enc_outputs[0].detach().cpu().numpy()
        #question_enc_pool_output = question_enc_outputs[0]

        doc_ids = self.model.rag.retriever(
            batch["input_ids"],
            question_enc_pool_output,
            prefix=self.model.rag.generator.config.prefix,
            n_docs=n_docs, #TODO: INHERIT
            return_tensors="pt",
        )['doc_ids'] #[B, n_docs]
        gt = batch['wiki_ids'].detach().cpu().numpy() # [B, A, P]
        batch_rprecision = calculate_rprecision(doc_ids, gt)
        return {'rprecision': batch_rprecision}


    def aggregate_metrics_by_task(self, task_names: np.ndarray, metrics: dict) -> Dict:
        """
        returns {'prefix_metric_task_id': [m1, m2, ...]}
        """
        prefix = 'val'
        metrics_agg = defaultdict(list)
        for m_name, vals in metrics.items():  
            for val, tid in zip(vals, task_names):
                _name = '_'.join([prefix, m_name, str(tid)])
                metrics_agg[_name].append(val)
        return metrics_agg

    def _generative_step(self, batch: dict, prefix: str, n_docs: int=5) -> Dict:
        """
            returns: {agg_metric1: [], agg_metric2: [], .. }
        """
        start_time = time.time()
        # DO I NEED THIS?
        #batch = BatchEncoding(batch)
        batch_rprecision = self.calculate_retrieval_metrics(batch)
        batch_metrics = self.calculate_downstream_metrics(batch)
        gen_time = (time.time() - start_time) / batch["input_ids"].shape[0]
        assert len(list(batch_rprecision.values())[0]) == len(list(batch_metrics.values())[0])
        batch_metrics.update(batch_rprecision)

        task_names = batch['task_names'].detach().cpu().numpy()
        batch_metrics_by_task = self.aggregate_metrics_by_task(task_names, batch_metrics)

        # validation / test step data returns [B, A, D] for eval purposes
        # but for loss calculation take the first one. 
        # [B, A, D] -> [B, D]
        batch['decoder_input_ids'] = batch['decoder_input_ids'] \
            .narrow(dim=1,start=0,length=1) \
            .squeeze(dim=1)
        loss_dict = {f'{prefix}_loss': [self._step(batch, prefix).detach().cpu().numpy()]}
        # base_metrics = {name: loss for name, loss in zip(self.loss_names, [loss_tensors])}
        # base_metrics.update({'gen_time': gen_time})
        
        # reduce for each step w/o task aggregation
        #batch_metrics = {f"{prefix}_{k}": np.array(v).mean() for k,v in batch_metrics.items()}
        batch_metrics = {f"{prefix}_{k}": v for k,v in batch_metrics.items()}
        #logger.info(batch_metrics)
        batch_metrics.update(batch_metrics_by_task)
        batch_metrics.update(loss_dict)
        #self.log_dict(batch_metrics, prog_bar=False)
        
        return batch_metrics # for epoch-end consumption {k: [v],..}


    def get_dataset(self, type_path) -> Dataset:
        max_target_length = self.target_lens[type_path]
        dataset = KILTDataset(
            tokenizer=self.tokenizer,
            type_path=type_path,
            max_target_length=max_target_length,
            n_obs=self.n_obs[type_path],
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        logger.info('train')
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        logger.info('val')
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        logger.info('test')
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("checkpoint{}".format(self.step_count))
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="wandb")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument(
            "--prefix",
            type=str,
            default=None,
            help="Prefix added at the beginning of each text, typically used with T5-based models.",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument(
            "--distributed-port", type=int, default=-1, required=False, help="Port number for distributed training."
        )
        parser.add_argument(
            "--model_type",
            choices=["rag_sequence", "rag_token", "bart", "t5"],
            type=str,
            help="RAG model type: sequence or token, if none specified, the type is inferred from the model_name_or_path",
        )

        return parser

    @staticmethod
    def add_retriever_specific_args(parser):
        parser.add_argument(
            "--index_name",
            type=str,
            default=None,
            help="Name of the index to use: 'hf' for a canonical dataset from the datasets library (default), 'custom' for a local index, or 'legacy' for the orignal one)",
        )
        parser.add_argument(
            "--passages_path",
            type=str,
            default=None,
            help="Path to the dataset of passages for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            "--index_path",
            type=str,
            default=None,
            help="Path to the faiss index for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            "--distributed_retriever",
            choices=["ray", "pytorch"],
            type=str,
            default="pytorch",
            help="What implementation to use for distributed retriever? If "
            "pytorch is selected, the index is loaded on training "
            "worker 0, and torch.distributed is used to handle "
            "communication between training worker 0, and the other "
            "training workers. If ray is selected, the Ray library is "
            "used to create load the index on separate processes, "
            "and Ray handles the communication between the training "
            "workers and the retrieval actors.",
        )
        parser.add_argument(
            "--use_dummy_dataset",
            type=bool,
            default=False,
            help="Whether to use the dummy version of the dataset index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )

        parser.add_argument(
            "--num_retrieval_workers",
            type=int,
            default=1,
            help="The number of retrieval actors to use when Ray is selected"
            "for the distributed retriever. Has no effect when "
            "distributed_retriever is set to pytorch.",
        )
        return parser

    @staticmethod
    def add_ray_specific_args(parser):
        parser.add_argument(
            "--num_retrieval_workers",
            type=int,
            default=1,
            help="The number of retrieval actors to use when Ray is selected"
            "for the distributed retriever. Has no effect when "
            "distributed_retriever is set to pytorch.",
        )

        # Ray cluster address.
        parser.add_argument(
            "--ray-address",
            default="auto",
            type=str,
            help="The address of the Ray cluster to connect to. If not "
            "specified, Ray will attempt to automatically detect the "
            "cluster. Has no effect if pytorch is used as the distributed "
            "retriever.",
        )

        return parser


def main(args=None, model=None) -> KILTModule:

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = KILTModule.add_model_specific_args(parser, os.getcwd())
    parser = KILTModule.add_retriever_specific_args(parser)

    args = args or parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    named_actors = []
    if args.distributed_retriever == "ray" and args.gpus > 1:
        if not is_ray_available():
            raise RuntimeError("Please install Ray to use the Ray " "distributed retriever.")
        # Connect to an existing Ray cluster.
        try:
            ray.init(address=args.ray_address)
        except (ConnectionError, ValueError):
            logger.warning(
                "Connection to Ray cluster failed. Make sure a Ray"
                "cluster is running by either using Ray's cluster "
                "launcher (`ray up`) or by manually starting Ray on "
                "each node via `ray start --head` for the head node "
                "and `ray start --address='<ip address>:6379'` for "
                "additional nodes. See "
                "https://docs.ray.io/en/master/cluster/index.html "
                "for more info."
            )
            raise

        # Create Ray actors only for rank 0.
        if ("LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == 0) and (
            "NODE_RANK" not in os.environ or os.environ["NODE_RANK"] == 0
        ):
            remote_cls = ray.remote(RayRetriever)
            named_actors = [
                remote_cls.options(name="retrieval_worker_{}".format(i)).remote()
                for i in range(args.num_retrieval_workers)
            ]
        else:
            logger.info(
                "Getting named actors for NODE_RANK {}, LOCAL_RANK {}".format(
                    os.environ["NODE_RANK"], os.environ["LOCAL_RANK"]
                )
            )
            named_actors = [ray.get_actor("retrieval_worker_{}".format(i)) for i in range(args.num_retrieval_workers)]
    args.actor_handles = named_actors
    assert args.actor_handles == named_actors

    if model is None:
        model: KILTModule = KILTModule(args)

    dataset = Path(args.data_dir).name

    from pytorch_lightning.loggers import WandbLogger
    logger.info(f"wandb logger saving under {args.output_dir}..")

    #project = os.environ.get("WANDB_PROJECT", dataset)
    training_logger = WandbLogger(
        name=model.output_dir.name, 
        project="kilt",
        save_dir=args.output_dir
    )

    # es_callback = (
    #     get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    #     if args.early_stopping_patience >= 0
    #     else False
    # )

    trainer: pl.Trainer = generic_train(
        model,
        args,
        #logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        #early_stopping_callback=es_callback,
        logger=training_logger,
        accelerator=CustomAccel() if args.gpus > 1 else None,
        profiler=pl.profiler.AdvancedProfiler() if args.profile else None,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    if not args.do_predict:
        return model

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = KILTModule.add_model_specific_args(parser, os.getcwd())
    parser = KILTModule.add_retriever_specific_args(parser)
    #parser = KILTModule.add_ray_specific_args(parser)

    # Pytorch Lightning Profiler
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If True, use pytorch_lightning.profiler.AdvancedProfiler to profile the Trainer.",
    )

    args = parser.parse_args()
    print(args)
    main(args)
