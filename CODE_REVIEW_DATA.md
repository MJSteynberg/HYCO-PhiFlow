
# DataGenerator:
```
def _create_field_names(self, velocity_tensor) -> str:
        """
        Create field names based on velocity tensor dimensions.

        For 1D: 'velocity'
        For 2D: 'vel_x,vel_y'
        For 3D: 'vel_x,vel_y,vel_z'
        """
        if 'vector' not in velocity_tensor.shape:
            # Scalar field (1D with no vector dim)
            return 'velocity'

        vector_shape = velocity_tensor.shape['vector']
        if vector_shape.item_names[0] is not None:
            # Has named vector components like 'x', 'y', 'z'
            return ','.join([f'vel_{name}' for name in vector_shape.item_names])
        else:
            # Unnamed vector components
            num_components = vector_shape.size
            if num_components == 1:
                return 'velocity'
            else:
                return ','.join([f'vel_{i}' for i in range(num_components)])
```
For 1d I want it named vel_x, to keep it uniform across dimensions.

```
def _convert_to_unified_tensor(self, data) -> Tensor:
        """
        Convert data to unified tensor format if needed.

        If data is already a unified tensor (has 'field' dimension), return as-is.
        If data is a dictionary, convert from old format.

        Output: Tensor(time, x, y?, field='vel_x,vel_y')
        """
        # Check if already unified tensor format
        if isinstance(data, Tensor) and 'field' in data.shape:
            return data

        # Old dictionary format: {'velocity': Tensor(time, x, y?, vector='x,y')}
        if isinstance(data, dict):
            velocity = data['velocity']

            # Extract tensor values if it's a Field object
            if isinstance(velocity, Field):
                velocity = velocity.values

            # Create field names based on tensor structure
            field_names = self._create_field_names(velocity)

            # Rename 'vector' dimension to 'field' with proper names
            if 'vector' in velocity.shape:
                return math.rename_dims(velocity, 'vector', channel(field=field_names))
            else:
                return math.expand(velocity, channel(field=field_names))

        # Fallback: assume it's a tensor with vector dimension
        if isinstance(data, Tensor):
            field_names = self._create_field_names(data)
            if 'vector' in data.shape:
                return math.rename_dims(data, 'vector', channel(field=field_names))
            else:
                return math.expand(data, channel(field=field_names))

        raise ValueError(f"Unexpected data type: {type(data)}")
```
We can assume data is always in unified tensor format from generation to prediction to training. This should make this function removable I think?


In general, this class can be written with less comments. It is rather self explanitory so please just minimalise everything here.

# Dataset

```
def _load_simulation(self, sim_idx: int) -> Tensor:
        """
        Load a single simulation from disk.

        Handles both old dict format and new unified tensor format.
        Returns unified tensor: Tensor(time, x, y?, field)
        """
        sim_path = os.path.join(self.data_dir, f"sim_{sim_idx:04d}.npz")
        sim_data = math.load(sim_path)

        # Check if already unified tensor (new format) or dict (old format)
        if isinstance(sim_data, dict):
            # Old format: Dict[str, Tensor] with 'velocity' key
            # Convert to unified tensor with field dimension
            sim_data = self._convert_old_format(sim_data)

        return sim_data
```
We can assume it is always the unified format.

```
def _convert_old_format(self, data: dict) -> Tensor:
        """
        Convert old dict format to unified tensor format.

        Old format: {'velocity': Tensor(time, x, y?, vector)}
        New format: Tensor(time, x, y?, field='vel_x,vel_y,...')
        """
        from phi.field import Field

        # Get velocity tensor (primary field in old format)
        velocity = data.get('velocity')
        if velocity is None:
            raise ValueError("Old format data missing 'velocity' key")

        # Handle Field vs Tensor - use .values property properly for Fields
        if isinstance(velocity, Field):
            velocity = velocity.values

        # Create field names based on vector dimension
        if 'vector' in velocity.shape:
            vector_shape = velocity.shape['vector']
            # Get the actual item names (may be nested tuples)
            raw_names = vector_shape.item_names
            # Flatten if nested: (('x',),) -> ('x',) or ('x', 'y') -> ('x', 'y')
            if raw_names and isinstance(raw_names[0], tuple):
                item_names = raw_names[0]  # Unwrap nested tuple
            else:
                item_names = raw_names

            if item_names and item_names[0] is not None:
                field_names = ','.join([f'vel_{name}' for name in item_names])
            else:
                # Fallback: use indices
                n_components = vector_shape.size
                if n_components == 1:
                    field_names = 'velocity'
                else:
                    field_names = ','.join([f'vel_{i}' for i in range(n_components)])

            # Rename vector -> field
            unified = math.rename_dims(velocity, 'vector', channel(field=field_names))
        else:
            # Scalar field (no vector dimension)
            unified = math.expand(velocity, channel(field='velocity'))

        return unified
```
Again, we dont assume backward compatability so i think this can be removed.

```
def _load_all_simulations(self) -> List[Tensor]:
        """Load all training simulations into memory."""
        simulations = []
        for sim_idx in self.train_sim:
            sim_data = self._load_simulation(sim_idx)
            simulations.append(sim_data)
        return simulations
```
Perhaps simulations should be a dataclass that can be cached. Then, when data is required we just slice simulations correctly without storing the already-sliced tensors. As we do in _get_sample, but we get it from a Simulation. Then, augmented trajectories are also simulations. Perhaps we can have a flag to distinguish between real simulations and augmented simulations.

```
def set_augmented_predictions(self, predictions: List[Tensor]):
        """
        Add augmented predictions to dataset.
        Alias for set_augmented_trajectories.
        """
        self.set_augmented_trajectories(predictions)
```
This aliasing just overcomplicates things.

Final comment. I think we should make access policy more clear, perhaps using enums? 
Also, let us bring back "alpha" functionality. That is, when we do iterate batches, we choose a certain proportion of real data given by sample, i.e. we only return 20% of the real data (randomly chosen indices) when alpha is 0.2.

Again, we can clean the comments structure a bit.



