# Synthetic
This is generally well-written, but I want to add learning rate scheduler to it. Phiml has get and setters for learning rate, but not inherent compatability with torch schedulers. Thus, let us use torch schedulers, but manually update the learning rate based on them. You'll see that there is already config attributes for the type of scheduler we want to use.

# Physical

```
def _setup(self):
        """
        Create optimizer for learnable parameters.

        Returns:
            PhiFlow optimizer instance
        """
        self.optimizer = math.Solve(
            method=self.method,
            abs_tol=self.abs_tol,
            x0=self.learnable_parameters,
            max_iterations=self.max_iterations,
            suppress=(math.NotConverged,),

        )

    def _prepare_batch(self, batch) -> Tuple[Tensor, Tensor]:
        """
        Prepare batch data for model.

        Args:
            batch: Batch dataclass with:
                  - initial_state: Tensor(batch, x, y?, field)
                  - targets: Tensor(batch, time, x, y?, field)

        Returns:
            Tuple of (initial_state, targets) tensors
        """
        return batch.initial_state, batch.targets
``` 
both of these functions seem rather useless as standalone functions.
I think we should introduce seperate batch sizes for physical and synthetic trainers as their data i.e. rollout steps are different etc.

```
def save_checkpoint(self, epoch: int, loss: float):
        """
        Save model checkpoint to specified path.

        Args:
            path: File path to save the checkpoint
            epoch: Current training epoch
            loss: Loss value at the checkpoint

        """
        
        params = [param.native('x,y') if isinstance(param, Tensor) else param for param in self.learnable_parameters]
        # Convert them to native tensors for saving
        checkpoint = {
            "learnable_parameters": params,
            "epoch": epoch,
            "loss": loss,
        }
        torch.save(checkpoint, self.checkpoint_path)
```
I think we need to go to phi's native saving and loading like we did for the synthetic model.

```
def _update_params(self, learnable_tensors: Tuple[Tensor, ...]):
        """
        Update model parameters (scalars or fields) from optimizer.

        Args:
            learnable_tensors: Tuple of updated parameter values from optimizer
        """
        for param_name, param_value, param_type in zip(
            self.param_names, learnable_tensors, self.param_types
        ):
            if param_type == "field":
                # Wrap tensor in CenteredGrid
                original_field = getattr(self.model, param_name)
                updated_field = CenteredGrid(
                    param_value,
                    extrapolation=original_field.extrapolation,
                    bounds=original_field.bounds,
                )
                setattr(self.model, param_name, updated_field)
            else:
                # Scalar - set directly
                setattr(self.model, param_name, param_value)

        # Update optimizer state
        self.optimizer.x0 = list(learnable_tensors)
        self.learnable_parameters = list(learnable_tensors)
```
Even with "field" params, I want us to store them as tensors in the models and then convert them to fields in the jit step.