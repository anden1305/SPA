from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml
from src.config.config import GlobalConfig

def _write_temp_yaml(cfg: GlobalConfig) -> Path:
    """Write a temporary YAML for a single sweep trial.

    Location is fixed under results/training/sweeps_temp to keep sweep artifacts
    centralized and independent of cfg.results_dir.
    """
    tmp_dir = Path("results/training/sweeps_temp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "sweep_run.yaml"
    with tmp_file.open("w") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)
    return tmp_file

def _build_sweep_config(base_cfg: GlobalConfig) -> Dict[str, Any]:
    # Minimal, focused bayesian sweep.
    metric = {"name": "val/nmi", "goal": "maximize"} if getattr(base_cfg.validator, "nmi", False) else {"name": "train/total_loss", "goal": "minimize"}

    parameters: Dict[str, Any] = {
        # Core levers that typically matter most
        "trainer.learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "trainer.optimizer": {"values": ["adam", "adamw"]},
        # Keep scheduler extremely simple: on/off and decay factor
        "trainer.scheduler.enabled": {"values": [False, True]},
        "trainer.scheduler.gamma": {"distribution": "uniform", "min": 0.8, "max": 0.999},
    }

    return {
        "method": "bayes",
        "metric": metric,
        "parameters": parameters,
        "early_terminate": {"type": "hyperband", "min_iter": 3},
    }

def _apply_wandb_config(base_cfg: GlobalConfig, wb_cfg: Dict[str, Any]) -> GlobalConfig:
    """Deep-merge W&B sweep parameters into the base config without replacing entire submodels.

    We start from the full dict of base_cfg and directly set parsed leaf values along the dotted paths.
    Then we re-validate into a GlobalConfig to ensure required fields remain intact.
    """
    parsers: Dict[str, Any] = {
        "trainer.learning_rate": float,
        "trainer.optimizer": str,
        "trainer.validate_per_epoch": int,
        "trainer.scheduler.enabled": bool,
        "trainer.scheduler.gamma": float,
        "trainer.scheduler.type": str,
        "trainer.scheduler.step_size": (lambda x: None if x is None else int(x)),
        "model.init_strategy": str,
        "model.init_noisy": bool,
        "model.covariance_type": str,
    }

    # Work on a full dict snapshot to preserve all required fields
    merged: Dict[str, Any] = base_cfg.model_dump()

    for key, value in wb_cfg.items():
        parser = parsers.get(key)
        if parser is None:
            continue
        try:
            parsed_value = parser(value)
        except Exception:
            # If parsing fails, skip this key rather than breaking the sweep
            continue
        # Descend into nested dicts according to dotted path
        path = key.split(".")
        node: Dict[str, Any] = merged
        for p in path[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt  # type: ignore[assignment]
        node[path[-1]] = parsed_value

    # Reconstruct a validated config
    return GlobalConfig.model_validate(merged)


def run_sweep(sweep_yaml_path: str) -> None:
    import wandb  # type: ignore

    # Read sweep YAML which may contain: method/metric/parameters/early_terminate, plus custom: base_config, count
    sweep_yaml = yaml.safe_load(Path(sweep_yaml_path).read_text())
    base_config_path = sweep_yaml.get("base_config")
    count = sweep_yaml.get("count")

    if not base_config_path:
        raise ValueError("Sweep YAML must include a 'base_config' path to the experiment config.")

    base_cfg = GlobalConfig.from_yaml(base_config_path)
    wb_settings = base_cfg.wandb
    project = "SPA"
    entity = "dtu_projects"
    group = getattr(wb_settings, "group", None) if wb_settings is not None else None
    tags = getattr(wb_settings, "tags", None) if wb_settings is not None else None

    # Use sweep spec from YAML if provided; else build a default minimal sweep
    sweep_spec = {k: v for k, v in sweep_yaml.items() if k in {"method", "metric", "parameters", "early_terminate"}}
    if not sweep_spec:
        sweep_spec = _build_sweep_config(base_cfg)

    # Determine metric to optimize for summary aggregation
    metric_spec = sweep_spec.get("metric") or {}
    metric_name: str = metric_spec.get("name") or ("val/nmi" if getattr(base_cfg.validator, "nmi", False) else "train/total_loss")
    metric_goal: str = metric_spec.get("goal", "maximize")

    # Create the sweep on the W&B server
    sweep_id = wandb.sweep(sweep=sweep_spec, project=project, entity=entity)

    def _train() -> None:
        # Only for sweeps: force headless Matplotlib before any pyplot import
        import matplotlib
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        # Import after backend is set so Visualizer uses Agg
        from src.orchestrator.orchestrator import Orchestrator
        wandb.init(project=project, entity=entity, group=group, tags=tags)
        # Apply sweep overrides and run orchestrator ONCE; it will handle cfg.runs repeats internally
        cfg = _apply_wandb_config(base_cfg, dict(wandb.config))
        tmp_yaml = _write_temp_yaml(cfg)
        orch = Orchestrator(str(tmp_yaml))
        orch.run()

        # Aggregate best-of across orchestrator runs
        best_val: float | None = None
        best_run_number: int | None = None
        for td in getattr(orch, 'train_details', []) or []:
            current_val: float | None = None
            if metric_name.startswith("val/"):
                key = metric_name.split("/", 1)[1]
                v = td.get_trained_validations().get(key)
                current_val = float(v) if isinstance(v, (int, float)) else None
            elif metric_name == "train/total_loss":
                if td.losses:
                    last_epoch = max(td.losses.keys())
                    current_val = float(td.losses[last_epoch])

            if current_val is None:
                continue
            if best_val is None:
                best_val, best_run_number = current_val, td.run_number
            else:
                if (metric_goal == "maximize" and current_val > best_val) or (metric_goal == "minimize" and current_val < best_val):
                    best_val, best_run_number = current_val, td.run_number

        # Log the aggregated result as the sweep metric
        if best_val is not None:
            wandb.summary[metric_name] = float(best_val)
            wandb.summary["aggregate/best_run_number"] = best_run_number
            wandb.summary["aggregate/runs"] = int(getattr(cfg, "runs", 1))
        wandb.finish()

    # If environment variable WAND_B_AGENT_ONLY is set, act as an agent for this sweep
    import os
    if os.getenv("WAND_B_AGENT_ONLY", "") == "1":
        # run a single agent (count should be 1 for array-job parallelism)
        wandb.agent(sweep_id, function=_train, count=1)
        return

    # Default behaviour: create sweep and run agents inline for the requested count
    wandb.agent(sweep_id, function=_train, count=count)


def create_sweep_only(sweep_yaml_path: str) -> str:
    """Create the sweep and return sweep_id without running agents.

    This is useful to call once and then launch multiple agent jobs (array) that each
    call the agent-only mode to run a single trial.
    """
    import wandb  # type: ignore
    sweep_yaml = yaml.safe_load(Path(sweep_yaml_path).read_text())
    base_config_path = sweep_yaml.get("base_config")
    if not base_config_path:
        raise ValueError("Sweep YAML must include a 'base_config' path to the experiment config.")
    base_cfg = GlobalConfig.from_yaml(base_config_path)
    sweep_spec = {k: v for k, v in sweep_yaml.items() if k in {"method", "metric", "parameters", "early_terminate"}}
    if not sweep_spec:
        sweep_spec = _build_sweep_config(base_cfg)
    project = "SPA"
    entity = "dtu_projects"
    sweep_id = wandb.sweep(sweep=sweep_spec, project=project, entity=entity)
    print(sweep_id)
    return sweep_id


def run_agent_only(sweep_yaml_path: str, sweep_id: str | None = None, trials_per_agent: int = 1) -> None:
    """Run a single agent for the given sweep. If sweep_id is None, create sweep first.

    This function is intended for array-job style parallelism: create sweep once, then
    submit multiple agent jobs that each call run_agent_only with the same sweep_id.
    """
    import wandb  # type: ignore
    sweep_yaml = yaml.safe_load(Path(sweep_yaml_path).read_text())
    base_config_path = sweep_yaml.get("base_config")
    if not base_config_path:
        raise ValueError("Sweep YAML must include a 'base_config' path to the experiment config.")
    base_cfg = GlobalConfig.from_yaml(base_config_path)
    sweep_spec = {k: v for k, v in sweep_yaml.items() if k in {"method", "metric", "parameters", "early_terminate"}}
    if not sweep_spec:
        sweep_spec = _build_sweep_config(base_cfg)
    project = "SPA"
    entity = "dtu_projects"
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_spec, project=project, entity=entity)

    def _train() -> None:
        import matplotlib
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        from src.orchestrator.orchestrator import Orchestrator
        wandb.init(project=project, entity=entity)
        cfg = _apply_wandb_config(base_cfg, dict(wandb.config))
        tmp_yaml = _write_temp_yaml(cfg)
        orch = Orchestrator(str(tmp_yaml))
        orch.run()
        # Aggregate best-of and write summary as in run_sweep
        best_val: float | None = None
        best_run_number: int | None = None
        metric_spec = sweep_spec.get("metric") or {}
        metric_name: str = metric_spec.get("name") or ("val/nmi" if getattr(base_cfg.validator, "nmi", False) else "train/total_loss")
        metric_goal: str = metric_spec.get("goal", "maximize")
        for td in getattr(orch, 'train_details', []) or []:
            current_val: float | None = None
            if metric_name.startswith("val/"):
                key = metric_name.split("/", 1)[1]
                v = td.get_trained_validations().get(key)
                current_val = float(v) if isinstance(v, (int, float)) else None
            elif metric_name == "train/total_loss":
                if td.losses:
                    last_epoch = max(td.losses.keys())
                    current_val = float(td.losses[last_epoch])
            if current_val is None:
                continue
            if best_val is None:
                best_val, best_run_number = current_val, td.run_number
            else:
                if (metric_goal == "maximize" and current_val > best_val) or (metric_goal == "minimize" and current_val < best_val):
                    best_val, best_run_number = current_val, td.run_number
        if best_val is not None:
            wandb.summary[metric_name] = float(best_val)
            wandb.summary["aggregate/best_run_number"] = best_run_number
            wandb.summary["aggregate/runs"] = int(getattr(cfg, "runs", 1))
        wandb.finish()

    wandb.agent(sweep_id, function=_train, count=int(trials_per_agent))


if __name__ == '__main__':
    # Provide a tiny CLI for convenience when invoked directly on HPC nodes.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_yaml')
    parser.add_argument('--create-only', action='store_true', help='Only create the sweep and print the sweep id')
    parser.add_argument('--agent-only', action='store_true', help='Run agent(s) for the specified sweep id (use --sweep-id)')
    parser.add_argument('--sweep-id', type=str, default=None, help='Existing sweep id to run agent for')
    parser.add_argument('--trials-per-agent', type=int, default=1, help='Number of trials each agent should run')
    args = parser.parse_args()
    if args.create_only:
        create_sweep_only(args.sweep_yaml)
    elif args.agent_only:
        run_agent_only(args.sweep_yaml, sweep_id=args.sweep_id, trials_per_agent=args.trials_per_agent)
    else:
        run_sweep(args.sweep_yaml)
