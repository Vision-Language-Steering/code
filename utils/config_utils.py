"""
Configuration utilities for VLS.

Provides helpers for Hydra configuration, including dynamic task loading
based on environment selection.
"""

import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


def get_config_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent.parent / "configs"


def load_task_config(env_backend: str, task_name: str) -> Optional[DictConfig]:
    """
    Load task-specific config based on environment.
    
    Args:
        env_backend: Environment backend (calvin, libero, realworld)
        task_name: Task name (drawer_open, goal, etc.)
    
    Returns:
        Task configuration as DictConfig, or None if not found
    """
    config_dir = get_config_dir()
    task_dir = config_dir / "task" / env_backend
    
    # If no task folder for this env, return None (e.g., libero uses suite_name directly)
    if not task_dir.exists():
        return None
    
    task_path = task_dir / f"{task_name}.yaml"
    
    if not task_path.exists():
        available = [p.stem for p in task_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Task '{task_name}' not found for env '{env_backend}'. "
            f"Available tasks: {available}"
        )
    
    return OmegaConf.load(task_path)


def merge_task_config(cfg: DictConfig, task_name: str) -> DictConfig:
    """
    Merge task-specific config into the main config.
    
    For environments without task configs (e.g., libero), returns cfg unchanged.
    
    Args:
        cfg: Main configuration
        task_name: Task name to load
    
    Returns:
        Merged configuration (or original if no task config exists)
    """
    env_backend = cfg.backend.backend
    task_cfg = load_task_config(env_backend, task_name)
    
    if task_cfg is None:
        # No task config for this env, just return original
        return cfg
    
    # Merge task config into main config
    return OmegaConf.merge(cfg, task_cfg)


def list_available_tasks(env_backend: str) -> list[str]:
    """
    List available tasks for an environment.
    
    Returns empty list if env doesn't use task configs.
    """
    config_dir = get_config_dir()
    task_dir = config_dir / "task" / env_backend
    
    if not task_dir.exists():
        return []
    
    return [p.stem for p in task_dir.glob("*.yaml")]


def validate_task_env_match(cfg: DictConfig) -> None:
    """
    Validate that the task matches the selected environment.
    
    Raises ValueError if there's a mismatch.
    """
    if "task" not in cfg or not isinstance(cfg.task, DictConfig):
        return  # task is just a string (like for libero), skip validation
    
    if "backend" not in cfg.task:
        return  # No task.backend specified, skip validation
    
    task_env = cfg.task.backend
    selected_backend = cfg.backend.backend
    
    if task_env != selected_backend:
        raise ValueError(
            f"Task '{cfg.task.name}' is designed for '{task_env}' environment, "
            f"but '{selected_env}' is selected. "
            f"Either change env={task_env} or choose a different task."
        )
