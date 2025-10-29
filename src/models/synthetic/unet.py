# In src/models/synthetic/unet.py

from typing import Dict, Any, List
from phiml.math import Tensor, math
from phiml.nn import u_net
# We only need 'Field' as a type hint now
from phi.field import CenteredGrid, Field 

from .base import SyntheticModel

class UNet(SyntheticModel):
    """
    A generic U-Net model that is agnostic to the input/output fields.
    
    It reads the specific fields and their channel counts from the
    configuration dictionary.
    
    It uses the 'staggered_input_fields' config key to determine
    which fields need to be converted from staggered to centered.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the U-Net model based on the provided config.

        Args:
            config: A dictionary containing model configurations.
                    Must include:
                    - 'in_channels': (int) Total input channels.
                    - 'out_channels': (int) Total output channels.
                    - 'levels': (int) Number of U-Net levels.
                    - 'filters': (int or list) Number of filters.
                    - 'input_fields': (list[str]) Names of input fields.
                    - 'input_channels': (list[int]) Channels for each input field.
                    - 'output_fields': (list[str]) Names of output fields.
                    - 'output_channels': (list[int]) Channels for each output field.
                    Optional:
                    - 'staggered_input_fields': (list[str]) List of field names
                                                  that are StaggeredGrids and
                                                  need conversion to centered.
        """
        super().__init__(config)

        # Build the U-Net architecture
        self.network = u_net(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            levels=config['levels'],
            filters=config['filters']
        )
        
        # Store field mapping info
        self.input_fields_list: List[str] = config['input_fields']
        self.input_channels_list: List[int] = config['input_channels']
        self.output_fields_list: List[str] = config['output_fields']
        self.output_channels_list: List[int] = config['output_channels']
        
        # --- NEW ---
        # Get list of fields that are expected to be StaggeredGrids
        # This avoids using isinstance() which can conflict with JIT
        self.staggered_input_fields: List[str] = config.get(
            'staggered_input_fields', []
        )
        
        # Sanity checks
        assert sum(self.input_channels_list) == config['in_channels'], \
            "Sum of 'input_channels' must equal 'in_channels'"
        assert sum(self.output_channels_list) == config['out_channels'], \
            "Sum of 'output_channels' must equal 'out_channels'"
        assert len(self.input_fields_list) == len(self.input_channels_list), \
            "'input_fields' and 'input_channels' must have the same length"
        assert len(self.output_fields_list) == len(self.output_channels_list), \
            "'output_fields' and 'output_channels' must have the same length"


    def forward(self, state: Dict[str, Field], dt: float) -> Dict[str, Field]:
        """
        Performs one prediction step using the U-Net.

        Args:
            state: Dictionary of input fields.
            dt: Time step (unused by this model, but required by API).

        Returns:
            Dictionary of predicted output fields (all as CenteredGrid).
        """
        # --- 1. Prepare Input Tensor ---
        
        # Get geometry from the first input field
        any_input_field = state[self.input_fields_list[0]]
        domain = any_input_field.domain
        extrapolation = any_input_field.extrapolation

        input_tensors = []
        for field_name in self.input_fields_list:
            field = state[field_name]
            
            # --- MODIFIED LOGIC ---
            # Check field_name against config list instead of using isinstance
            if field_name in self.staggered_input_fields:
                # This call to .at_centers() will be traced by the JIT
                values = field.at_centers().values
            else:
                # Assumed to be a CenteredGrid or have compatible .values
                values = field.values
            input_tensors.append(values)
            
        # Concatenate along the channel dimension
        input_tensor = math.concat(input_tensors, 'channel')

        # --- 2. Run Network ---
        predicted_tensor = self.network(input_tensor)
        
        # --- 3. Unpack Output Tensors ---
        output_state = {}
        current_channel_idx = 0
        
        for i, field_name in enumerate(self.output_fields_list):
            num_channels = self.output_channels_list[i]
            
            # Slice the correct channels from the output tensor
            field_values = predicted_tensor[
                ..., current_channel_idx : current_channel_idx + num_channels
            ]
            
            # Wrap values back into a new CenteredGrid
            output_state[field_name] = CenteredGrid(
                values=field_values,
                domain=domain,
                extrapolation=extrapolation
            )
            
            current_channel_idx += num_channels

        return output_state