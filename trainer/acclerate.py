import torch
import evaluate
from accelerate import Accelerator

accelerator = Accelerator(log_with="wandb",gradient_accumulation_steps=2)
train_dataloader, eval_dataloader, model, optimizer, scheduler = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer, scheduler
)
accelerator.init_trackers()
metric = evaluate.load("accuracy")

# Training loop
model.train()
for batch in train_dataloader:
    with accelerator.accumulate(model):
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.log({"loss":loss})
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

accelerator.end_training()

# Evaluation loop  
model.eval()
for batch in eval_dataloader:
    inputs, targets = batch
    with torch.no_grad():
        outputs = model(inputs)
    predictions = outputs.argmax(dim=-1)
    predictions, references = accelerator.gather_for_metrics((predictions, targets))
    metric.add_batch(predictions=predictions, references=references)

print(metric.compute())


accelerator.save_state("checkpoint_dir")
accelerator.load_state("checkpoint_dir")