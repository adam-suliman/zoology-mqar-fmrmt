from zoology.model import LanguageModel
from zoology.config import TrainConfig


def _env_flag(name: str) -> bool:
    import os

    value = os.getenv(name)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _flatten_dict(data: dict, prefix: str = ""):
    import json

    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value, default=str)
        else:
            flat[name] = value
    return flat


class ExperimentLogger:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.run = None
        self.backend = self._resolve_backend(config)
        self.step = 0

        if self.backend == "none":
            print("No logger specified, skipping...")
            self.no_logger = True
            return

        self.no_logger = False
        if self.backend == "wandb":
            self._init_wandb(config)
        elif self.backend == "comet":
            self._init_comet(config)
        else:
            raise ValueError(f"Unknown logger backend: {self.backend}")

    def _resolve_backend(self, config: TrainConfig):
        import os

        if config.logger.backend is not None:
            return config.logger.backend

        if os.getenv("COMET_PROJECT_NAME") is not None or os.getenv("COMET_API_KEY") is not None:
            return "comet"

        if config.logger.project_name is not None and config.logger.entity is not None:
            return "wandb"

        return "none"

    def _init_wandb(self, config: TrainConfig):
        import wandb

        if config.logger.project_name is None or config.logger.entity is None:
            print("WandB logger selected but project_name/entity missing, skipping...")
            self.no_logger = True
            self.backend = "none"
            return

        self.run = wandb.init(
            name=config.run_id,
            entity=config.logger.entity,
            project=config.logger.project_name,
        )

    def _init_comet(self, config: TrainConfig):
        import os
        try:
            import comet_ml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Comet logging requested but comet_ml is not installed. "
                "Run `pip install comet_ml` or `pip install -r requirements.txt`."
            ) from exc

        api_key = config.logger.api_key or os.getenv("COMET_API_KEY")
        project_name = config.logger.project_name or os.getenv("COMET_PROJECT_NAME")
        workspace = config.logger.workspace or os.getenv("COMET_WORKSPACE")
        experiment_key = config.logger.experiment_key or os.getenv("COMET_EXPERIMENT_KEY")
        offline = config.logger.offline or _env_flag("COMET_OFFLINE")

        self.run = comet_ml.start(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            experiment_key=experiment_key,
            online=not offline,
            mode="create" if experiment_key is None else "get_or_create",
        )
        self.run.set_name(config.run_id)

        tags = list(config.logger.tags)
        env_tags = os.getenv("COMET_TAGS")
        if env_tags:
            tags.extend([tag.strip() for tag in env_tags.split(",") if tag.strip()])
        if tags:
            if hasattr(self.run, "add_tags"):
                self.run.add_tags(tags)
            elif hasattr(self.run, "add_tag"):
                for tag in tags:
                    self.run.add_tag(tag)

        self.run.log_others({
            "run_id": config.run_id,
            "launch_id": config.launch_id,
            "sweep_id": config.sweep_id,
        })
        if getattr(self.run, "url", None):
            print(f"Comet experiment: {self.run.url}")

    def log_config(self, config: TrainConfig):
        if self.no_logger:
            return
        if self.backend == "wandb":
            self.run.config.update(config.model_dump(), allow_val_change=True)
        elif self.backend == "comet":
            self.run.log_parameters(_flatten_dict(config.model_dump()))

    def log_model(
        self, 
        model: LanguageModel,
        config: TrainConfig
    ):
        if self.no_logger:
            return
        
        test_configs = getattr(config.data, "test_configs", None)
        if test_configs is None:
            test_configs = getattr(config.data, "test_stage_configs", [])
        max_seq_len = max([c.input_seq_len for c in test_configs])
        model_info = {
            "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "state_size": model.state_size(sequence_length=max_seq_len),
        }
        if self.backend == "wandb":
            import wandb

            wandb.log(model_info)
            wandb.watch(model)
        elif self.backend == "comet":
            self.run.log_parameters(model_info, prefix="model")

    def log(self, metrics: dict):
        if self.no_logger:
            return
        if self.backend == "wandb":
            import wandb

            wandb.log(metrics)
        elif self.backend == "comet":
            epoch = metrics.get("epoch")
            self.run.log_metrics(metrics, step=self.step, epoch=epoch)
            self.step += 1
    
    def finish(self):
        if self.no_logger:
            return
        if hasattr(self.run, "finish"):
            self.run.finish()
        elif hasattr(self.run, "end"):
            self.run.end()


WandbLogger = ExperimentLogger
