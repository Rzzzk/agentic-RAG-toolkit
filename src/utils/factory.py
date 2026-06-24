import importlib
from typing import Any, Dict

class ComponentFactory:
    """
    Dynamically loads and instantiates toolkit components based on dictionary configs.
    This entirely removes hardcoded dependencies from your main execution flows.
    """
    
    @staticmethod
    def create_instance(config: Dict[str, Any]) -> Any:
        """
        Expects a config dictionary with the following structure:
        {
            "class_path": "src.components.llms.openai_wrapper.OpenAILLM",
            "init_args": {
                "model_name": "gpt-4o",
                "temperature": 0.2
            }
        }
        """
        if "class_path" not in config:
            raise ValueError("Configuration missing required 'class_path' key.")
            
        class_path = config["class_path"]
        init_args = config.get("init_args", {})
        
        try:
            # Split "src.components...OpenAILLM" into module and class name
            module_path, class_name = class_path.rsplit(".", 1)
            
            # Dynamically load the module
            module = importlib.import_module(module_path)
            
            # Grab the class from the module
            target_class = getattr(module, class_name)
            
            # Instantiate and return "boom, done"
            return target_class(**init_args)
            
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to dynamically load {class_path}: {e}")