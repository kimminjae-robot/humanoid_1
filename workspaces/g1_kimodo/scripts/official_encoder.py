"""Pinned local construction using Kimodo's own LLM2Vec classes and call path."""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer
from kimodo.model.llm2vec.llm2vec import LLM2Vec
from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel

ROOT = Path(__file__).resolve().parents[1]


class PinnedLLM2VecEncoder(LLM2VecEncoder):
    def __init__(self):
        manifest = json.loads((ROOT / "reports/official_encoder_checkpoint.json").read_text())
        if not manifest["complete"]:
            raise RuntimeError("Official encoder download is incomplete")
        repos = manifest["repositories"]
        expected = json.loads((ROOT / "configs/official_encoder.json").read_text())
        for role, spec in expected["repositories"].items():
            if any(repos[role][key] != spec[key] for key in ["repo_id", "revision"]):
                raise ValueError("Official encoder revision mismatch")
        tokenizer = AutoTokenizer.from_pretrained(repos["mntp"]["snapshot"], local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        config = AutoConfig.from_pretrained(repos["mntp"]["snapshot"], local_files_only=True)
        # Resolve the base explicitly, so PEFT cannot fetch a floating base revision.
        model = LlamaBiModel.from_pretrained(repos["base"]["snapshot"], config=config,
                                            torch_dtype=torch.bfloat16, local_files_only=True)
        model = PeftModel.from_pretrained(model, repos["mntp"]["snapshot"], local_files_only=True)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, repos["supervised"]["snapshot"], local_files_only=True)
        # Official LLM2Vec selects the Llama chat format by this canonical name.
        model.config._name_or_path = repos["base"]["repo_id"]
        self.model = LLM2Vec(model=model, tokenizer=tokenizer).to("cpu").eval()
        self.llm_dim = 4096
        self._device = "cpu"
        for parameter in self.model.parameters():
            parameter.requires_grad = False
