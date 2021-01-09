import abc
import numexpr
import numpy as np

from typing import Union, Optional, Dict


################################################################################

class BaseMultiTaskSampler(metaclass=abc.ABCMeta):
    def __init__(self, task_dict: dict, rng: Union[int, np.random.RandomState, None]):
        self.task_dict = task_dict
        if isinstance(rng, int) or rng is None:
            rng = np.random.RandomState(rng)
        self.rng = rng

    def pop(self):
        raise NotImplementedError()

    def iter(self):
        yield self.pop()


class UniformMultiTaskSampler(BaseMultiTaskSampler):
    def pop(self):
        task_name = self.rng.choice(list(self.task_dict))
        return task_name, self.task_dict[task_name]


class ProportionalMultiTaskSampler(BaseMultiTaskSampler):
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_num_examples_dict: dict,
        gradient_accumulation_steps: int = 1
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_num_examples_dict.keys()
        self.task_to_examples_dict = task_to_num_examples_dict
        self.task_names = list(task_to_num_examples_dict.keys())
        self.task_num_examples = np.array([task_to_num_examples_dict[k] for k in self.task_names])
        self.task_p = self.task_num_examples / self.task_num_examples.sum()
        # gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.i = 0
        self.task_name = None

    def pop(self):
        # sample after accumulation steps.
        # TODO: move this to base.
        if  self.i % self.gradient_accumulation_steps == 0 and  self.i > 0:
            self.task_name = self.rng.choice(self.task_names, p=self.task_p)
        return self.task_name, self.task_dict[self.task_name]

################################################################################

class TaskConfiguration(object):
    def __init__(self, 
                train_task_name_list: List[str],
                val_task_name_list: List[str],
                test_task_name_list: List[str],
                max_steps = int,
                do_train: bool= True,
                do_val: bool= False,
                do_test: bool= False,
                max_epochs: int=10,
                train_examples_cap: int=16,
                gradient_accumulation_steps: int=1,
                train_batch_size: int = 64,
                num_gpus: int = 1,
            ):

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_examples_cap = train_examples_cap # K
        self.train_batch_size = train_batch_size
        self.num_gpus = num_gpus


        def create_config(self):
            # === Gather task names === #
            # Get the full list of tasks across all phases
            task_name_list_dict = {
                "train": self.train_task_name_list,
                "val": self.val_task_name_list,
                "test": self.test_task_name_list,
            }

            # === Compute training steps === #
            if not self.do_train:
                assert self.max_epochs is None
            elif self.epochs is not None:
                if self.num_gpus:
                    # We multiply by num_gpus because 1 step is done across (potentially) multiple GPUs
                    effective_batch_size = (
                        self.train_batch_size * self.gradient_accumulation_steps * self.num_gpus
                    )
                else:
                    effective_batch_size = self.train_batch_size * self.gradient_accumulation_steps
                num_examples = get_num_examples_from_cache(
                    cache_path=os.path.expandvars(task_cache_config["train"]),
                )
                max_steps = self.epochs * math.ceil(train_examples_cap / effective_batch_size)
            else:
                raise RuntimeError("Require either `epochs` or `max_steps`")

            num_examples = train_batch_size * gradient_accumulation_steps * 

            # === Build configuration === #
            # Finally, we build our big config dictionary. Congrats!
            config_dict = {
                "task_config_path_dict": task_config_path_dict,
                "task_cache_config_dict": task_cache_config_dict,
                "sampler_config": sampler_config,
                "global_train_config": {
                    "max_steps": int(max_steps),
                    "warmup_steps": int(max_steps * self.warmup_steps_proportion),
                },
                "task_specific_configs_dict": {
                    task_name: {
                        "train_batch_size": self.train_batch_size,
                        "eval_batch_size": eval_batch_size,
                        "gradient_accumulation_steps": self.gradient_accumulation_steps,
                        "eval_subset_num": self.eval_subset_num,
                    }
                    for task_name in full_task_name_list
                },
                "taskmodels_config": {
                    "task_to_taskmodel_map": {
                        task_name: task_name for task_name in full_task_name_list
                    },
                    "taskmodel_config_map": {task_name: None for task_name in full_task_name_list},
                },
                "task_run_config": {
                    "train_task_list": task_name_list_dict["train"],
                    "train_val_task_list": task_name_list_dict["train_val"],
                    "val_task_list": task_name_list_dict["val"],
                    "test_task_list": task_name_list_dict["test"],
                },
                "metric_aggregator_config": {"metric_aggregator_type": "EqualMetricAggregator"},
            }
            return config_dict

