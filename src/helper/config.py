import yaml
from pathlib import Path
from typing import Any, Dict


# Define a constructor for pathlib.PosixPath that handles sequences
def posixpath_constructor(loader, node):
    # The node for !!python/object/apply:pathlib.PosixPath has a sequence as its value
    # We need to construct the sequence and join the elements to form the path string
    path_components = loader.construct_sequence(node)
    # Join the path components, ensuring the root '/' is handled correctly
    if path_components and path_components[0] == "/":
        # If the first component is '/', join from the second element onwards and prepend '/'
        path_str = "/" + "/".join(str(p) for p in path_components[1:])
    else:
        # Otherwise, just join all components
        path_str = "/".join(str(p) for p in path_components)
    return Path(path_str)


# Add the constructor to the SafeLoader
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    posixpath_constructor,
)


def load_config(path: Path) -> Dict[str, Any]:
    """
    Loads a configuration file from the given path.

    Args:
        path (Path): The path to the configuration file.

    Returns:
        Dict[str, Any]: The configuration dictionary with Path objects resolved.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        IOError: If there is an error reading the configuration file.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {path}")
    except IOError as e:
        raise IOError(f"Error reading configuration file at {path}: {e}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file at {path}: {e}")

    project_root = Path(__file__).parent.parent.parent

    if "base_dir" in config and isinstance(config["base_dir"], str):
        config["base_dir"] = (project_root / config["base_dir"]).resolve()
    if "dataset_dir" in config and isinstance(config["dataset_dir"], str):
        config["dataset_dir"] = (
            project_root / config["dataset_dir"]
        ).resolve()
    if "dataset_processed_dir" in config and isinstance(
        config["dataset_processed_dir"], str
    ):
        config["dataset_processed_dir"] = (
            project_root / config["dataset_processed_dir"]
        ).resolve()

    return config


def config_updater(
    key: str, value: Any, config: Dict[str, Any], config_path: Path
) -> Dict[str, Any]:
    """
    Updates a specific key-value pair in a configuration dictionary and saves the changes to a YAML file.

    Args:
        key (str): The key to update in the configuration dictionary.
        value (Any): The new value for the specified key.
        config (Dict[str, Any]): The current configuration dictionary.
        config_path (Path): The path to the configuration file.

    Returns:
        Dict[str, Any]: The updated configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        IOError: If there is an error reading or writing the configuration file.

    Example:
        >>> from pathlib import Path
        >>> import yaml
        >>> config_path = Path("config/config.yaml")
        >>> config_path.parent.mkdir(parents=True, exist_ok=True)
        >>> with open(config_path, "w") as f:
        ...     yaml.dump({"learning_rate": 0.001, "epochs": 10}, f)
        >>> current_config = {"learning_rate": 0.001, "epochs": 10}
        >>> updated_config = config_updater("epochs", 20, current_config, config_path)
        >>> print(f"Updated config: {updated_config}")
        Updated config: {'learning_rate': 0.001, 'epochs': 20}
    """
    # Create a copy to avoid modifying the original dictionary directly
    updated_config = config.copy()
    updated_config[key] = value

    # Ensure the directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the updated config back to the YAML file
    try:
        with open(config_path, "w") as f:
            yaml.dump(updated_config, f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}"
        )
    except IOError as e:
        raise IOError(
            f"Error writing to configuration file at {config_path}: {e}"
        )

    return updated_config
