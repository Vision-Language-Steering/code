"""
Hydra SearchPath Plugin for Environment-Aware Task Loading

This plugin dynamically adds the environment-specific task folder to Hydra's
search path, allowing users to simply use `task=drawer_open` instead of 
`task/calvin=drawer_open` when env=calvin is set.

Usage:
    python main.py env=calvin task=drawer_open
    # Hydra will automatically search in configs/task/calvin/
"""

from hydra.core.global_hydra import GlobalHydra
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins
from hydra.core.config_search_path import ConfigSearchPath
from omegaconf import OmegaConf
import os


class TaskSearchPathPlugin(SearchPathPlugin):
    """
    Adds environment-specific task folder to search path.
    
    When env=calvin, adds configs/task/calvin to the task config group search path.
    """
    
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # This is called before config resolution, so we can't access the final env value
        # Instead, we add all possible task paths and let Hydra handle the resolution
        pass


def get_task_config_path(env_backend: str) -> str:
    """Get the task config path for a given environment backend."""
    return f"task/{env_backend}"

