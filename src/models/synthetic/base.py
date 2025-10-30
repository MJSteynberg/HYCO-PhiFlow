# In src/models/synthetic/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch.nn as nn
from phi.field import Field, StaggeredGrid, CenteredGrid, stack, native_call
from phi.math import math, channel
from phi import field as phi_field
from phi.field import native_call
class SyntheticModel(nn.Module, ABC):
    """
    Abstract base class for all synthetic models (neural networks).

    This class handles the boilerplate logic for:
    1.  Pre-processing: Converting a state dict of Phiflow Fields (including
        StaggeredGrids) into a single, multi-channel CenteredGrid tensor.
    2.  Post-processing: Converting the network's output CenteredGrid back
        into a state dict of individual Fields, restoring original
        StaggeredGrid types where appropriate.

    Subclasses are only required to implement the `_predict` method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the synthetic model.

        Args:
            config: A dictionary containing model-specific configurations.
                    Expected to contain 'input_specs' and 'output_specs'
                    dictionaries, e.g., {'density': 1, 'velocity': 2}.
        """
        super().__init__()
        self.config = config

        # Get specs from config, default to empty dict if not provided
        self.INPUT_SPECS: Dict[str, int] = config.get('input_specs', {})
        self.OUTPUT_SPECS: Dict[str, int] = config.get('output_specs', {})

        # Derive the field lists directly from the specs
        self.INPUT_FIELDS: List[str] = list(self.INPUT_SPECS.keys())
        self.OUTPUT_FIELDS: List[str] = list(self.OUTPUT_SPECS.keys())

    def _fields_to_grid(self, state: Dict[str, Field]) -> CenteredGrid:
        """
        Pre-processes the input state dict into a single CenteredGrid for the NN.
        
        Converts StaggeredGrids to CenteredGrids and stacks all fields.
        """
        
        # 1. Get list of scalar CenteredGrids (your code for this is correct)
        inputs_centered = [
            val
            for name in self.INPUT_FIELDS
            for val in (state[name].at_centers().vector if state[name].is_staggered else [state[name]])
        ]
        
        # 2. Get the raw value tensors from each field
        input_tensors = [f.values for f in inputs_centered]
        
        # 3. Stack the TENSORS, not the Fields. This avoids the metadata stack.
        stacked_tensor = math.stack(input_tensors, channel('channels'))
        
        # 4. Create a NEW CenteredGrid container for the stacked tensor.
        #    We borrow the metadata (bounds, extrap) from the first field.
        template = inputs_centered[0]
        stacked_input = CenteredGrid(
            stacked_tensor, 
            bounds=template.bounds, 
            extrapolation=template.extrapolation
        )
        
        # 5. Rename 'channels' to 'vector' for native_call
        nn_input_grid = math.rename_dims(stacked_input, 'channels', 'vector')
        return nn_input_grid

    def _grid_to_fields(self, 
                        pred_field_centered: CenteredGrid, 
                        original_state: Dict[str, Field]) -> Dict[str, Field]:
        """
        Post-processes the NN's output grid back into a state dict of Fields.
        
        Resamples CenteredGrid outputs back to StaggeredGrids if the
        original state's field was staggered.
        """
        output_dict = {}
        channel_offset = 0

        for field_name in self.OUTPUT_FIELDS:
            # Use the original field from the state as a template
            template = original_state[field_name]
            
            # Get channel count from the template
            num_channels = template.shape.get_size('vector') if 'vector' in template.shape else 1
            
            # Slice the *Field object* directly to preserve metadata (bounds, etc.)
            nn_output_centered = pred_field_centered.vector[channel_offset : channel_offset + num_channels]
            
            # Restore vector dimension names (e.g., 'x', 'y') or squeeze for scalars
            if num_channels == 1:
                nn_output_centered = math.squeeze(nn_output_centered, 'vector')
            elif 'vector' in template.shape:
                vector_names = template.shape.get_item_names('vector')
                if vector_names:
                    # Re-apply item names like ('x', 'y')
                    nn_output_centered = math.rename_dims(nn_output_centered, 'vector', channel(vector=list(vector_names)))

            # Resample to staggered if the template was staggered
            if template.is_staggered:
                # The '@' operator resamples the CenteredGrid 'nn_output_centered'
                # onto the StaggeredGrid 'template' locations.
                output_dict[field_name] = nn_output_centered @ template
            else:
                # It's already a CenteredGrid, just assign it.
                output_dict[field_name] = nn_output_centered
                
            channel_offset += num_channels
            
        return output_dict

    @abstractmethod
    def _predict(self, nn_input_grid: CenteredGrid, dt: float) -> CenteredGrid:
        """
        Performs the core prediction on the processed, multi-channel grid.

        This is the only method subclasses need to implement.
        
        Args:
            nn_input_grid: A single CenteredGrid containing all input channels
                           stacked along the 'vector' dimension.
            dt: The time step duration (float).

        Returns:
            A single CenteredGrid containing all output channels stacked
            along the 'vector' dimension.
        """
        pass

    def forward(self, state: Dict[str, Field], dt: float = 0.0) -> Dict[str, Field]:
        """
        Performs one full prediction step, including pre- and post-processing.

        This method is now concrete and should not be overridden by subclasses.
        Subclasses must implement `_predict` instead.
        
        Args:
            state: A dictionary of Fields (CenteredGrid, StaggeredGrid)
                   representing the current state.
            dt: The time step duration (float).

        Returns:
            A dictionary of Fields representing the predicted next state.
        """
        # Step 1: Convert input state dict to a single, processed grid
        nn_input_grid = self._fields_to_grid(state)
        
        # Step 2: Run the model's core prediction logic
        nn_output_grid = self._predict(nn_input_grid, dt)
        
        # Step 3: Convert the output grid back to a state dict of fields
        output_dict = self._grid_to_fields(nn_output_grid, state)
        
        return output_dict