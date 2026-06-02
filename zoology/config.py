import argparse
from datetime import datetime
from functools import partial
from typing import List, Optional, Tuple, Union, Literal

from pydantic import BaseModel


from zoology.utils import import_from_str


class BaseConfig(BaseModel):
    @classmethod
    def from_cli(cls):
        import yaml
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--config', type=str, default=None, help='Path to the config file')
        parser.add_argument('--run_id', type=str, default=None, help='Run ID for the training')
        args, extra_args = parser.parse_known_args()


        if args.config is not None:
            with open(args.config) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            config = {}
        
        # Override with any extra arguments from the command line
        def _nested_update(config, args):
            for key, value in args.items():
                keys = key.split(".")
                for key in keys[:-1]:
                    config = config.setdefault(key, {})
                config[keys[-1]] = value

        extra_args = dict([arg.lstrip("-").split("=") for arg in extra_args])
        extra_args = {k.replace("-", "_"): v for k, v in extra_args.items()}
        _nested_update(config, extra_args)
        config = cls.parse_obj(config)

        if config.run_id is None:
            config.run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return config

    def print(self):
        try:
            import rich
            rich.print(self)
        except ImportError:
            print(self)


class FunctionConfig(BaseConfig):
    name: str
    kwargs: dict = {}

    def instantiate(self):
        return partial(import_from_str(self.name), **self.kwargs)

class ModuleConfig(BaseConfig):
    name: str
    kwargs: dict = {}

    def instantiate(self, **kwargs):
        return import_from_str(self.name)(**kwargs, **self.kwargs)

class DataSegmentConfig(BaseConfig):
    """
    This class should be subclassed to define per task. For example, MQARConfig
    """
    vocab_size: int = 8_192
    num_examples: int = 1_000
    input_seq_len: int = 64

    def build(self, **kwargs):
        raise NotImplementedError()
    
class DataConfig(BaseConfig):
    train_configs: List[DataSegmentConfig]
    test_configs: List[DataSegmentConfig]

    # can pass a tuple if you want a different batch size for train and test
    batch_size: Union[int, Tuple[int, int]] = 32
    seed: int = 123
    cache_dir: Optional[str] = None
    force_cache: bool = False 

    # JRT style sequences (https://arxiv.org/abs/2407.05483)
    num_passes: int = 1


class ContinualDataConfig(BaseConfig):
    train_stage_configs: List[DataSegmentConfig]
    test_stage_configs: List[DataSegmentConfig]

    # can pass a tuple if you want a different batch size for train and test
    batch_size: Union[int, Tuple[int, int]] = 32
    seed: int = 123
    cache_dir: Optional[str] = None
    force_cache: bool = False


class ModelConfig(BaseConfig):
    sequence_mixer: ModuleConfig = None
    state_mixer: ModuleConfig = ModuleConfig(
        name="zoology.mixers.mlp.MLP", 
        kwargs={"hidden_mult": 4}
    )

    d_model: int = 128
    n_layers: int = 2
    multiplier: int = 1
    max_position_embeddings: int = 64
    learnable_word_embeddings: bool = True
    embedding_init_type: Literal["default", "spherical", "normal"] = "default"
    vocab_size: int = 8_192

    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    drop_path: float = 0.0
    layer_norm_epsilon: float = 1e-5
    pad_vocab_size_multiple: int = 1

    block_type: Literal["TransformerBlock", "MambaBlock", "Mamba2Block"] = "TransformerBlock"
    name: str = "default"

class LoggerConfig(BaseConfig):

    backend: Optional[Literal["wandb", "comet", "none"]] = None
    project_name: Optional[str] = None
    entity: Optional[str] = None

    # Comet uses "workspace" where WandB uses "entity".
    workspace: Optional[str] = None
    api_key: Optional[str] = None
    experiment_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = []
    

class TrainConfig(BaseConfig):
    data: DataConfig
    model: ModelConfig
    logger: LoggerConfig = LoggerConfig()

    max_epochs: int = 100

    # Per-batch training metric logging interval. Set to 0 to disable batch logs.
    # Epoch and stage summary metrics are still logged.
    train_log_interval: int = 1

    loss_type: Literal["ce", "mse", "ce_embed"] = "ce"
    input_type: Literal["discrete", "continuous"] = "discrete"

    # stop training once this metric reaches the threshold
    # set metric to None to disable early stopping
    early_stopping_metric: Optional[str] = "valid/accuracy"
    early_stopping_threshold: float = 0.99
    slice_keys: List[str] = []

    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    slow_update_mode: Literal["skip", "accumulate"] = "skip"
    seed: int = 123

    launch_id: Optional[str] = None
    sweep_id: Optional[str] = None
    run_id: str = "default"


class ContinualTrainConfig(TrainConfig):
    data: ContinualDataConfig
    training_mode: Literal["continual"] = "continual"
    evaluate_future_stages: bool = False
    lr_scheduler_mode: Literal[
        "global_cosine",
        "stage_cosine",
        "stage_onecycle",
        "constant",
    ] = "global_cosine"

    # Log current-stage validation curves during continual training. Stage-end
    # evaluation still computes all seen-stage CL metrics. Set <= 0 to disable.
    continual_epoch_eval_interval: int = 1
