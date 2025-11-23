# Synthetic
```
def __call__(self, state: Tensor) -> Tensor:
        """
        Forward pass predicting next state.

        Args:
            state: Tensor(batch?, x, y?, field='vel_x,vel_y,...')

        Returns:
            Tensor with same shape (predicted next state)
        """
        return math.native_call(self.network, state)
```
I would like to reintroduce static data fields which is part of the input of the network, but not the output, and then gets reattached with the network output to produce the result of __call__. I think there should be a clever way to flag fields as static or dynamic using the phi named channels. Perhaps the channel names can be channel(static='vel_x,vel_y'), channel(dynamic='density') for example. Contemplate whether this can work.
```
def get_network(self):
    """Get the underlying network (for optimization)."""
    return self.network
``` 
I would rather want us to use pythonic getters and setters of attributes.

It seems like convnet and resnet should be updated to have the same functionality as unet w.r.t. spacial dims etc.

# Physical
```
def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """
        Generate random initial velocity as tensor.

        Returns:
            Tensor(batch, x, y?, field='vel_x,...')
        """
        # Build full spatial shape
        grid_shape = spatial(**{name: self.resolution.get_size(name) for name in self.spatial_dims})

        # Create coordinate tensors (all with full spatial shape)
        coords = {}
        for i, dim in enumerate(self.spatial_dims):
            n = self.resolution.get_size(dim)
            size = float(self.domain[i].size)
            coord_1d = math.linspace(0, size, spatial(**{dim: n}))
            # Expand to full spatial shape
            coords[dim] = math.expand(coord_1d, grid_shape)

        # Create velocity components (each with full spatial shape)
        components = []
        for i, dim in enumerate(self.spatial_dims):
            wave_number = 4 * math.pi * (i + 1) / float(self.domain[i].size)
            comp = math.sin(wave_number * coords[dim])
            components.append(comp)

        # Stack into vector
        velocity = math.stack(components, channel(vector=','.join(self.spatial_dims)))

        # Add noise
        noise = 0.05 * math.random_normal(velocity.shape)
        velocity = velocity + noise

        # Add batch dimension
        velocity = math.expand(velocity, batch(batch=batch_size))

        # Convert to field dimension
        result = math.rename_dims(
            velocity, 'vector', channel(field=','.join(self.field_names))
        )
        return result
```
This is quite convoluted, lets go back to using Noise() and instantiate it as a centered grid and then returning the centered grid.values. I.e. something like this: 
```
def get_initial_state(self, batch_size: int = 1) -> Dict[str, Field]:
        """
        Returns an initial state of (noisy velocity).
        We use periodic boundaries as they are common for Burgers.
        Dimension-agnostic: works for 1D, 2D, and 3D.
        """
        b = batch(batch=batch_size)

        # Build kwargs dynamically from resolution Shape
        grid_kwargs = {
            name: self.resolution.get_size(name)
            for name in self.resolution.names
        }
        noise = StaggeredGrid(
            Noise(scale=10, smoothness=10),
            extrapolation.PERIODIC,  # Use periodic boundaries
            bounds=self.domain,
            **grid_kwargs
        )
        velocity_0 = CenteredGrid(noise, extrapolation.PERIODIC, bounds=self.domain, **grid_kwargs)

        velocity_0 = math.expand(velocity_0, b)
        return {"velocity": velocity_0}
```
Furthermore, let us reintroduce havinf field values for the diffusion coefficient. What we had before:
```
 def _initialize_fields(self, pde_params: Dict[str, Any]):
        """Initialize model fields from PDE parameters."""
        if self.n_spatial_dims == 1:
            def f(x):
                evaluation = eval(pde_params['value'], {'x':x, 'math': math, 'size_x': self.domain.size[0]})
                return evaluation
        elif self.n_spatial_dims == 2:
            def f(x, y):
                evaluation = eval(pde_params['value'], {'x':x, 'y':y, 'math': math, 'size_x': self.domain.size[0], 'size_y': self.domain.size[1]})
                return evaluation
        else:  # 3D
            def f(x, y, z):
                evaluation = eval(pde_params['value'], {'x':x, 'y':y, 'z':z, 'math': math, 'size_x': self.domain.size[0], 'size_y': self.domain.size[1], 'size_z': self.domain.size[2]})
                return evaluation
```
And it worked pretty well, but if we can do it with a single statement for all dims, it would be better.

```
def _create_velocity_field(self) -> CenteredGrid:
        """Create the static velocity field as a CenteredGrid."""
        domain = self.domain
        resolution = self.resolution
        spatial_dims = self.spatial_dims

        # Build grid kwargs
        grid_kwargs = {name: resolution.get_size(name) for name in spatial_dims}
        if len(spatial_dims) == 2:
            # Create a swirling/rotating velocity field
            def velocity_fn(x, y):
                center_x = float(domain[0].size) / 2
                center_y = float(domain[1].size) / 2
                dy = y - center_y
                dx = x - center_x
                r = math.sqrt(dx**2 + dy**2 + 1e-6)

                # Circular flow with some variation
                size_x = float(domain[0].size)
                size_y = float(domain[1].size)
                vx = -dy * math.exp(-(r**2) / (0.2 * size_x) ** 2) + 0.2 * math.sin(2 * math.pi * y / size_y)
                vy = dx * math.exp(-(r**2) / (0.2 * size_x) ** 2) + 0.2 * math.cos(2 * math.pi * x / size_x)

                return math.stack([vx, vy], channel("vector"))
        
        else:
            def velocity_fn(x):
                # Simple constant flow to the right in 1D
                size_x = float(domain[0].size)
                vx = 0.5 + 0.1 * math.sin(2 * math.pi * x / size_x)
                return math.stack([vx], channel("vector"))

        return CenteredGrid(velocity_fn, PERIODIC, bounds=domain, **grid_kwargs)
``` 
Let us replace this velocity with
```class AngularVelocity
(
location: phiml.math._tensors.Tensor | tuple | list | numbers.Number,
strength: phiml.math._tensors.Tensor | numbers.Number = 1.0,
falloff: Callable = None)
```
Please consult phiflow docs how to use it. 
Then, coming back to the idea of static fields. Let us reintroduce this as a static field when generating the data. 

