mono_depth: https://github.com/jspenmar/monodepth_benchmark

主要基于(torchmetrics)[https://lightning.ai/docs/torchmetrics/stable/pages/overview.html]库，用于计算各种指标，如精度、召回率、F1分数等。

TorchMetrics 是一个为 PyTorch 和 PyTorch Lightning 设计的度量 API，旨在简化度量的开发和使用。该 API 经过严格测试，覆盖了所有边界情况，并包含越来越多的常见度量实现。

该度量 API 提供了 `update()`、`compute()` 和 `reset()` 函数供用户使用。度量基类继承自 `torch.nn.Module`，这使得我们可以直接调用 `metric(...)`。基类的 `forward()` 方法既用于对输入调用 `update()`，又同时返回提供输入的度量值。

这些度量在 PyTorch 和 PyTorch Lightning 的分布式数据并行（DDP）环境中默认可以使用。当在分布式模式下调用 `.compute()` 时，每个度量的内部状态会在每个进程间同步并进行归约，以便在所有进程的状态信息上应用 `.compute()` 中的逻辑。

该度量 API 独立于 PyTorch Lightning。度量可以直接在 PyTorch 中使用，如下示例所示：

```python
from torchmetrics.classification import BinaryAccuracy

train_accuracy = BinaryAccuracy()
valid_accuracy = BinaryAccuracy()

for epoch in range(epochs):
    for x, y in train_data:
        y_hat = model(x)

        # training step accuracy
        batch_acc = train_accuracy(y_hat, y)
        print(f"Accuracy of batch{i} is {batch_acc}")

    for x, y in valid_data:
        y_hat = model(x)
        valid_accuracy.update(y_hat, y)

    # total accuracy over all training batches
    total_train_accuracy = train_accuracy.compute()

    # total accuracy over all validation batches
    total_valid_accuracy = valid_accuracy.compute()

    print(f"Training acc for epoch {epoch}: {total_train_accuracy}")
    print(f"Validation acc for epoch {epoch}: {total_valid_accuracy}")

    # Reset metric states after each epoch
    train_accuracy.reset()
    valid_accuracy.reset()
```