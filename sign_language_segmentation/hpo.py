"""optuna hyperparameter optimization for the segmentation model.

The search space is defined in a YAML file with two sections:
  architecture: model structure params (skipped when fine-tuning)
  training: optimization and regularization params (always searched)

See optuna.yaml for the default search space.
"""
from pathlib import Path

import optuna
import yaml


def load_search_space(path: str | Path, skip_architecture: bool = False) -> tuple[dict, str]:
    """load search space from YAML and flatten into a single param dict.

    When skip_architecture is True (fine-tuning), only the 'training'
    section is returned.

    Returns (params, monitor_metric).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Optuna search space file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    monitor_metric = raw.get("monitor_metric", "value")

    params: dict = {}
    if not skip_architecture:
        params.update(raw.get("architecture", {}))
    params.update(raw.get("training", {}))
    return params, monitor_metric


def sample_hyperparams(trial: optuna.Trial, search_space: dict) -> dict:
    """sample hyperparameters from an Optuna trial using a search space dict."""
    overrides: dict = {}
    for name, spec in search_space.items():
        param_type = spec["type"]
        if param_type == "float":
            overrides[name] = trial.suggest_float(
                name, low=spec["low"], high=spec["high"], log=spec.get("log", False),
            )
        elif param_type == "int":
            overrides[name] = trial.suggest_int(
                name, low=spec["low"], high=spec["high"], step=spec.get("step", 1),
            )
        elif param_type == "categorical":
            overrides[name] = trial.suggest_categorical(name, choices=spec["choices"])
        else:
            raise ValueError(f"Unknown param type '{param_type}' for '{name}'")
    return overrides


def run_study(
    train_fn,
    search_space: dict,
    n_trials: int,
    metric_name: str = "value",
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    no_wandb: bool = False,
) -> optuna.Study:
    """create and run an Optuna study."""
    wandb_callback = None
    if not no_wandb:
        from optuna_integration import WeightsAndBiasesCallback
        wandb_callback = WeightsAndBiasesCallback(
            metric_name=metric_name,
            wandb_kwargs={"entity": wandb_entity, "project": wandb_project},
            as_multirun=True,
        )

    def objective(trial: optuna.Trial) -> float:
        overrides = sample_hyperparams(trial=trial, search_space=search_space)
        overrides["_trial"] = trial
        return train_fn(overrides=overrides)

    if wandb_callback:
        objective = wandb_callback.track_in_wandb()(objective)

    study = optuna.create_study(direction="maximize", study_name="segmentation-hpo")
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[wandb_callback] if wandb_callback else [],
    )
    return study
