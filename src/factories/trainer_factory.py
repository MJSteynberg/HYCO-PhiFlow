"""Factory for creating trainers."""

from typing import Dict, Any
from src.training.abstract_trainer import AbstractTrainer
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer


class TrainerFactory:
    """Factory for creating trainer instances."""
    
    _trainers = {
        'synthetic': SyntheticTrainer,
        'physical': PhysicalTrainer,
    }
    
    @staticmethod
    def create_trainer(config: Dict[str, Any]) -> AbstractTrainer:
        """
        Create trainer from config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Trainer instance (AbstractTrainer subclass)
            
        Raises:
            ValueError: If model_type is unknown
        """
        model_type = config['run_params']['model_type']
        
        if model_type not in TrainerFactory._trainers:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Available: {list(TrainerFactory._trainers.keys())}"
            )
        
        TrainerClass = TrainerFactory._trainers[model_type]
        return TrainerClass(config)
    
    @staticmethod
    def register_trainer(name: str, trainer_class: type):
        """
        Register a new trainer type.
        
        Args:
            name: Name to register the trainer under
            trainer_class: Trainer class to register
        """
        TrainerFactory._trainers[name] = trainer_class
    
    @staticmethod
    def list_available_trainers():
        """
        List all available trainer types.
        
        Returns:
            List of registered trainer names
        """
        return list(TrainerFactory._trainers.keys())
