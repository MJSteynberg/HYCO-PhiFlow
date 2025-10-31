from phi.torch.flow import *

domain = Box(x=1.0, y=1.0)
resolution = spatial(x=128, y=128)


velocity_0 = StaggeredGrid(
            Noise(batch=1),
            extrapolation.ZERO,
            x=128,
            y=128,
            bounds=domain,
        )

print(velocity_0)