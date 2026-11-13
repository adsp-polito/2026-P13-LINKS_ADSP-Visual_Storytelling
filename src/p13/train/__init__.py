"""Parameter-efficient adaptation (LoRA / QLoRA) of Qwen3-VL. Owner: Behrad.

The base model is never updated: vision encoder and language decoder stay frozen and only the
LoRA adapter weights are trained, saved and pushed (privately) to the Hugging Face Hub.
"""
