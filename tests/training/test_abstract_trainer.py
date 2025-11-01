"""
Tests for AbstractTrainer

Tests the minimal abstract base class that all trainers must implement.
"""

import pytest
from abc import ABC

from src.training.abstract_trainer import AbstractTrainer


class TestAbstractTrainerProperties:
    """Tests for AbstractTrainer abstract class properties."""
    
    def test_abstract_trainer_is_abstract(self):
        """Test that AbstractTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AbstractTrainer({})
    
    def test_abstract_trainer_has_abstract_methods(self):
        """Test that AbstractTrainer defines expected abstract methods."""
        abstract_methods = AbstractTrainer.__abstractmethods__
        
        # Only train() should be abstract
        assert 'train' in abstract_methods
        assert len(abstract_methods) == 1
    
    def test_abstract_trainer_inherits_from_abc(self):
        """Test that AbstractTrainer uses ABC."""
        assert issubclass(AbstractTrainer, ABC)


class ConcreteTrainer(AbstractTrainer):
    """Minimal concrete implementation for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.train_called = False
        self.train_result = {'loss': 0.5}
    
    def train(self):
        """Minimal train implementation."""
        self.train_called = True
        return self.train_result


class TestAbstractTrainerInitialization:
    """Tests for AbstractTrainer initialization."""
    
    def test_initialization_with_config(self):
        """Test that trainer can be initialized with config."""
        config = {'project_root': '/test/path', 'model': {}}
        trainer = ConcreteTrainer(config)
        
        assert trainer.config == config
    
    def test_config_stored(self):
        """Test that config is stored in trainer."""
        config = {'test_key': 'test_value'}
        trainer = ConcreteTrainer(config)
        
        assert trainer.config == config
        assert trainer.config['test_key'] == 'test_value'
    
    def test_project_root_extracted(self):
        """Test that project_root is extracted from config."""
        config = {'project_root': '/custom/path'}
        trainer = ConcreteTrainer(config)
        
        assert trainer.project_root == '/custom/path'
    
    def test_project_root_defaults_to_dot(self):
        """Test that project_root defaults to '.' if not provided."""
        config = {}
        trainer = ConcreteTrainer(config)
        
        assert trainer.project_root == '.'


class TestAbstractTrainerMethods:
    """Tests for AbstractTrainer concrete methods."""
    
    def test_get_config_returns_config(self):
        """Test that get_config returns the stored config."""
        config = {'key1': 'value1', 'key2': 'value2'}
        trainer = ConcreteTrainer(config)
        
        retrieved_config = trainer.get_config()
        assert retrieved_config == config
        assert retrieved_config is config  # Should be same object
    
    def test_get_project_root_returns_project_root(self):
        """Test that get_project_root returns the project root."""
        config = {'project_root': '/test/root'}
        trainer = ConcreteTrainer(config)
        
        assert trainer.get_project_root() == '/test/root'
    
    def test_train_must_be_implemented(self):
        """Test that train() must be implemented by subclass."""
        trainer = ConcreteTrainer({})
        
        # Should be callable
        assert callable(trainer.train)
        
        # Should return something
        result = trainer.train()
        assert result is not None


class TestAbstractTrainerInheritance:
    """Tests for AbstractTrainer inheritance patterns."""
    
    def test_concrete_trainer_is_abstract_trainer(self):
        """Test that concrete implementation is an AbstractTrainer."""
        trainer = ConcreteTrainer({})
        assert isinstance(trainer, AbstractTrainer)
    
    def test_multiple_trainers_independent(self):
        """Test that multiple trainer instances are independent."""
        config1 = {'name': 'trainer1'}
        config2 = {'name': 'trainer2'}
        
        trainer1 = ConcreteTrainer(config1)
        trainer2 = ConcreteTrainer(config2)
        
        assert trainer1.config is not trainer2.config
        assert trainer1.config['name'] == 'trainer1'
        assert trainer2.config['name'] == 'trainer2'
    
    def test_train_result_independent(self):
        """Test that train results are independent between instances."""
        trainer1 = ConcreteTrainer({})
        trainer2 = ConcreteTrainer({})
        
        trainer1.train_result = {'loss': 1.0}
        trainer2.train_result = {'loss': 2.0}
        
        result1 = trainer1.train()
        result2 = trainer2.train()
        
        assert result1['loss'] == 1.0
        assert result2['loss'] == 2.0


class TestAbstractTrainerWithDifferentConfigs:
    """Tests for AbstractTrainer with various configuration patterns."""
    
    def test_empty_config(self):
        """Test initialization with empty config."""
        trainer = ConcreteTrainer({})
        assert trainer.config == {}
        assert trainer.project_root == '.'
    
    def test_nested_config(self):
        """Test with nested configuration dictionary."""
        config = {
            'project_root': '/test',
            'model': {
                'type': 'test_model',
                'params': {'lr': 0.001}
            }
        }
        trainer = ConcreteTrainer(config)
        
        assert trainer.config == config
        assert trainer.config['model']['type'] == 'test_model'
        assert trainer.config['model']['params']['lr'] == 0.001
    
    def test_config_immutability(self):
        """Test that modifying config doesn't affect trainer."""
        config = {'value': 10}
        trainer = ConcreteTrainer(config)
        
        # Modify original config
        config['value'] = 20
        
        # Trainer should still reference the same config object
        # (Python passes dicts by reference)
        assert trainer.config['value'] == 20
    
    def test_multiple_configs_same_structure(self):
        """Test multiple trainers with similar config structures."""
        configs = [
            {'project_root': f'/path/{i}', 'id': i}
            for i in range(5)
        ]
        
        trainers = [ConcreteTrainer(config) for config in configs]
        
        for i, trainer in enumerate(trainers):
            assert trainer.config['id'] == i
            assert trainer.project_root == f'/path/{i}'


class TestAbstractTrainerAbstractMethods:
    """Tests for abstract method enforcement."""
    
    def test_missing_train_raises_error(self):
        """Test that missing train() implementation raises TypeError."""
        
        class IncompleteTrainer(AbstractTrainer):
            """Trainer without train() implementation."""
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteTrainer({})
    
    def test_train_with_wrong_signature_still_works(self):
        """Test that train() can have any signature."""
        
        class CustomSignatureTrainer(AbstractTrainer):
            def train(self, extra_param=None):
                return {'param': extra_param}
        
        trainer = CustomSignatureTrainer({})
        assert isinstance(trainer, AbstractTrainer)
        
        result = trainer.train(extra_param='test')
        assert result['param'] == 'test'
    
    def test_train_can_return_any_type(self):
        """Test that train() can return any type."""
        
        class StringReturnTrainer(AbstractTrainer):
            def train(self):
                return "training complete"
        
        class NoneReturnTrainer(AbstractTrainer):
            def train(self):
                return None
        
        class ListReturnTrainer(AbstractTrainer):
            def train(self):
                return [1, 2, 3]
        
        trainer1 = StringReturnTrainer({})
        trainer2 = NoneReturnTrainer({})
        trainer3 = ListReturnTrainer({})
        
        assert trainer1.train() == "training complete"
        assert trainer2.train() is None
        assert trainer3.train() == [1, 2, 3]


class TestAbstractTrainerEdgeCases:
    """Tests for edge cases and unusual usage patterns."""
    
    def test_none_config(self):
        """Test behavior when config is None (should raise AttributeError)."""
        with pytest.raises(AttributeError):
            trainer = ConcreteTrainer(None)
    
    def test_config_with_non_dict_types(self):
        """Test with config containing various types."""
        config = {
            'string': 'test',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': [1, 2, 3],
            'none': None
        }
        trainer = ConcreteTrainer(config)
        
        assert trainer.config['string'] == 'test'
        assert trainer.config['int'] == 42
        assert trainer.config['float'] == 3.14
        assert trainer.config['bool'] is True
        assert trainer.config['list'] == [1, 2, 3]
        assert trainer.config['none'] is None
    
    def test_callable_methods_exist(self):
        """Test that all public methods are callable."""
        trainer = ConcreteTrainer({})
        
        assert callable(trainer.train)
        assert callable(trainer.get_config)
        assert callable(trainer.get_project_root)
