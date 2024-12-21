import torch
from torch import nn
from transformers import Trainer

class BeliefStateTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss for the BeliefStateTransformer.
        """
        inputs, labels = inputs["input_ids"], inputs["labels"]
        reverse_inputs = torch.flip(inputs, dims=[1])

        model.base_model.set_adapter("forward_encoder")
        forward_states = model.base_model(inputs).last_hidden_state

        model.base_model.set_adapter("backward_encoder")
        backward_states = torch.flip(model.base_model(reverse_inputs).last_hidden_state, dims=[1])

        loss = model.compute_loss(forward_states, backward_states, labels)
        return (loss, {"loss": loss}) if return_outputs else loss