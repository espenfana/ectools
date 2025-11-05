"""
Enhanced configuration system for ectools

Provides a more robust configuration management system with validation,
default values, and extensibility.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import logging
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Constants
LOCAL_TZ = ZoneInfo('Europe/Oslo')

# Check for optional dependencies
try:
    import bokeh
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False


class PlotBackend(Enum):
    """Available plotting backends."""
    MATPLOTLIB = 'matplotlib'
    BOKEH = 'bokeh'


class CycleConvention(Enum):
    """Cycle numbering conventions."""
    V2 = 'v2'  # Second vertex convention
    INIT = 'init'  # Initial value convention


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


class EcToolsConfig:
    """Enhanced configuration management for ectools."""
    
    _defaults = {
        'plotter': PlotBackend.MATPLOTLIB,
        'cycle_convention': CycleConvention.V2,
        'log_level': LogLevel.WARNING,
        'timezone': LOCAL_TZ,
        'cache_parsed_files': True,
        'parallel_processing': False,
        'max_workers': 4,
        'chunk_size': 1000,
        'memory_limit_mb': 1024,
        'auto_cleanup': True,
        'plot_theme': 'default',
        'figure_size': (10, 6),
        'dpi': 100
    }
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self._config = self._defaults.copy()
        self._config_file = Path(config_file) if config_file else None
        
        if self._config_file and self._config_file.exists():
            self.load_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value with validation."""
        if key == 'plotter':
            if isinstance(value, str):
                value = PlotBackend(value)
            elif not isinstance(value, PlotBackend):
                raise ValueError(f"Invalid plot backend: {value}")
            
            if value == PlotBackend.BOKEH and not BOKEH_AVAILABLE:
                raise RuntimeError("Bokeh is not available. Install Bokeh to use this feature.")
        
        elif key == 'cycle_convention':
            if isinstance(value, str):
                value = CycleConvention(value)
            elif not isinstance(value, CycleConvention):
                raise ValueError(f"Invalid cycle convention: {value}")
        
        elif key == 'log_level':
            if isinstance(value, str):
                value = LogLevel(value)
            elif not isinstance(value, LogLevel):
                raise ValueError(f"Invalid log level: {value}")
        
        elif key == 'max_workers':
            if not isinstance(value, int) or value < 1:
                raise ValueError("max_workers must be a positive integer")
        
        elif key == 'memory_limit_mb':
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("memory_limit_mb must be a positive number")
        
        self._config[key] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        for key, value in updates.items():
            self.set(key, value)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = self._defaults.copy()
        logger.info("Configuration reset to defaults")
    
    def save_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        save_path = Path(file_path) if file_path else self._config_file
        
        if not save_path:
            raise ValueError("No config file path specified")
        
        # Convert enums to strings for JSON serialization
        serializable_config = {}
        for key, value in self._config.items():
            if isinstance(value, Enum):
                serializable_config[key] = value.value
            elif hasattr(value, '__str__'):  # Handle timezone and other objects
                serializable_config[key] = str(value)
            else:
                serializable_config[key] = value
        
        try:
            with open(save_path, 'w') as f:
                json.dump(serializable_config, f, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def load_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from file."""
        load_path = Path(file_path) if file_path else self._config_file
        
        if not load_path or not load_path.exists():
            logger.warning(f"Config file not found: {load_path}")
            return
        
        try:
            with open(load_path, 'r') as f:
                file_config = json.load(f)
            
            # Validate and set each configuration value
            for key, value in file_config.items():
                if key in self._defaults:
                    try:
                        self.set(key, value)
                    except (ValueError, RuntimeError) as e:
                        logger.warning(f"Invalid config value for {key}: {value} ({e})")
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            logger.info(f"Configuration loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_summary(self) -> str:
        """Get a formatted summary of current configuration."""
        lines = ["EcTools Configuration:"]
        lines.append("-" * 30)
        
        for key, value in self._config.items():
            if isinstance(value, Enum):
                display_value = value.value
            else:
                display_value = str(value)
            lines.append(f"{key:20}: {display_value}")
        
        return "\n".join(lines)


# Global configuration instance
_global_config = EcToolsConfig()

# Convenience functions for backward compatibility
def set_config(key: str, value: Any) -> None:
    """Set a global configuration value."""
    _global_config.set(key, value)

def get_config(key: str, default: Any = None) -> Any:
    """Get a global configuration value."""
    return _global_config.get(key, default)

def get_global_config() -> EcToolsConfig:
    """Get the global configuration instance."""
    return _global_config
