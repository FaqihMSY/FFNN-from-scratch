class Model:
    def __init__(self,
                 layers: list,
                 loss: LossFunction,
                 optimizer: Optimizer,
                 batch_size: int = 1):
        self.layers     = layers
        self.loss       = loss
        self.optimizer  = optimizer
        self.batch_size = batch_size

        self.training_loss   = []
        self.validation_loss = []

    def _forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _backward(self, grad: np.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _fused_loss_last_layer(self) -> bool:
        return isinstance(self.loss, (SoftmaxCCELoss, SigmoidBCELoss))

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int,
              verbose: bool = False,
              validation_X: np.ndarray = None,
              validation_y: np.ndarray = None,
              output_file=None):

        n_samples  = X.shape[0]
        batch_size = n_samples if self.batch_size is None else self.batch_size
        from tqdm import tqdm

        pbar = tqdm(range(epochs), desc="Training")
        self.training_loss   = []
        self.validation_loss = []

        for epoch in pbar:
            epoch_loss = 0
            idx = np.random.permutation(n_samples)
            X, y = X[idx], y[idx]

            for start in range(0, n_samples, batch_size):
                X_batch = X[start:start + batch_size]
                y_batch = y[start:start + batch_size]
                actual_batch = len(X_batch)

                self.optimizer.zero_grad(self.layers)

                if self._fused_loss_last_layer():
                    logits = self._forward_no_last(X_batch)
                    loss   = self.loss.forward(logits, y_batch)
                    grad   = self.loss.backward()
                    self._backward_no_last(grad)
                else:
                    out    = self._forward(X_batch)
                    loss   = self.loss.forward(y_batch, out)
                    grad   = self.loss.backward()
                    self._backward(grad)

                epoch_loss += loss
                self.optimizer.step(self.layers, actual_batch)

            self.training_loss.append(epoch_loss)
            pbar.set_postfix(loss=f"{epoch_loss:.6f}")

            if verbose and validation_X is not None:
                val_out  = self.predict(validation_X)
                val_loss = self.loss.forward(validation_y, val_out)
                self.validation_loss.append(val_loss)

            if output_file is not None:
                output_file.write(f"{epoch+1},{epoch_loss:.6f}\n")
                output_file.flush()

    def _forward_no_last(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers[:-1]:
            out = layer.forward(out)
        return out

    def _backward_no_last(self, grad: np.ndarray):
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)

    def save(self, path: str):
        arrays = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                arrays[f'w_{i}'] = layer.weights
        arrays['layer_types'] = np.array([type(l).__name__ for l in self.layers])
        np.savez(path, **arrays)

    @classmethod
    def from_file(cls, path: str, layers: list, loss: LossFunction, optimizer: Optimizer):
        data = np.load(path, allow_pickle=True)
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                layer.weights = data[f'w_{i}']
        return cls(layers, loss, optimizer)
