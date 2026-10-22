"""Local-only single-device Qwen runtime and a CPU-only scripted test backend."""
from __future__ import annotations

import importlib
import os
import time
from pathlib import Path
from typing import Callable

from evidence_ownership.utils.config import get_paths, inference_config, model_config
from evidence_ownership.utils.hashing import sha256_file, sha256_json, sha256_text
from evidence_ownership.utils.logs import get_logger

THINK_SUFFIX = '<think>\n\n</think>\n\n'


def verify_image(path: Path, expected: str | None) -> str:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f'Image SHA256 mismatch: {path}')
    return actual


class FakeBackend:
    """No model/torch imports. Scripts may map basenames to text, exceptions or sequences.

    A callable receives (image_path, system, user). It never receives gold labels.
    """
    def __init__(self, outputs: dict | Callable | None = None, **_: object):
        self.outputs = outputs or {}
        self.calls = 0
        self.generation_config = dict(inference_config()['generation'])
        self.chat_template_sha256 = sha256_text('fake-template-v1')
        self.model_info = dict(model_id='fake', revision='fake-v1', dtype='none',
                               attn_implementation='none', **{'class': 'FakeBackend'},
                               processor_class='FakeProcessor', image_processor_class='FakeImageProcessor')
        self.metadata = dict(model=self.model_info, generation_config=self.generation_config,
                             generation_config_sha256=sha256_json(self.generation_config),
                             chat_template_sha256=self.chat_template_sha256,
                             loaded_unique_parameters=0, attention_implementation='none',
                             kernels={'fla': False, 'causal_conv1d': False}, cuda_device_name=None,
                             load_time_s=0.0, load_peak_allocated=0, load_peak_reserved=0,
                             parameter_count_matches=None)
        self.last_rendered_prompt = ''

    def generate(self, image_path: str | Path, system: str, user: str,
                 image_sha256: str | None = None) -> dict:
        checked = verify_image(Path(image_path), image_sha256)
        self.calls += 1
        if callable(self.outputs):
            response = self.outputs(Path(image_path), system, user)
        else:
            response = self.outputs.get(str(image_path), self.outputs.get(Path(image_path).name, '{"label":"UNKNOWN"}'))
            if isinstance(response, list):
                response = response.pop(0) if len(response) > 1 else response[0]
        if isinstance(response, Exception):
            raise response
        self.last_rendered_prompt = system + '\n<image>\n' + user + '\n' + THINK_SUFFIX
        tokens = list(str(response).encode('utf-8'))
        return dict(raw_text=str(response), generated_token_ids=tokens, n_input_tokens=100,
                    n_generated_tokens=len(tokens), hit_max_new_tokens=False, image_grid_thw=[1, 64, 64],
                    latency_s=0.0, cuda_max_memory_allocated=0, cuda_max_memory_reserved=0,
                    rendered_prompt_sha256=sha256_text(self.last_rendered_prompt),
                    image_sha256_checked=checked, thinking_disabled_verified=True)


class QwenBackend:
    """Loads only the configured local checkpoint, on exactly one explicit device.

    Physical GPUs are selected by CUDA_VISIBLE_DEVICES; --device addresses that visible set.
    Construction loads weights: use FakeBackend for CPU development and dry runs.
    """
    def __init__(self, device: str = 'cuda:0'):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        from transformers import GenerationConfig

        if device == 'auto' or ',' in device:
            raise ValueError('Select a single explicit device, e.g. cuda:0')
        self.device = torch.device(device)
        self.is_cuda = self.device.type == 'cuda'
        cfg, model_cfg = inference_config(), model_config()
        self.generation_config = dict(cfg['generation'])
        if self.is_cuda:
            torch.cuda.set_device(self.device)  # initialise the CUDA context before touching memory stats
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        local_dir = get_paths().model_dir
        self.processor = AutoProcessor.from_pretrained(
            str(local_dir), local_files_only=True, revision=model_cfg['processor_revision'])
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(local_dir), local_files_only=True, revision=model_cfg['revision'],
            dtype=torch.bfloat16, attn_implementation=cfg['attn_implementation'], device_map={'': device})
        self.model.eval()
        seen, count = set(), 0
        for param in self.model.parameters():
            ptr = param.data_ptr()
            key = ('ptr', ptr) if ptr else ('id', id(param))
            if key not in seen:
                seen.add(key)
                count += param.numel()
        if count != model_cfg['loaded_unique_parameters']:
            raise RuntimeError(f'Loaded {count} unique parameters; expected {model_cfg["loaded_unique_parameters"]}')
        template = self.processor.chat_template
        self.chat_template_sha256 = sha256_json(template) if not isinstance(template, str) else sha256_text(template)
        self.model_info = dict(model_id=model_cfg['model_id'], revision=model_cfg['revision'],
                               processor_revision=model_cfg['processor_revision'], dtype='bfloat16',
                               attn_implementation=self.model.config._attn_implementation,
                               **{'class': type(self.model).__name__},
                               processor_class=type(self.processor).__name__,
                               image_processor_class=type(self.processor.image_processor).__name__)
        kernels = {}
        for module in ('fla', 'causal_conv1d'):
            try:
                importlib.import_module(module)
                kernels[module] = True
            except (ImportError, OSError, RuntimeError):
                kernels[module] = False
        # Inspect the complete effective config while explicitly overriding checkpoint sampling defaults.
        effective = GenerationConfig.from_dict(self.model.generation_config.to_dict())
        effective.update(**self.generation_config)
        self.effective_generation_config = effective.to_dict()
        get_logger(__name__).info('Effective generation config: %s', self.effective_generation_config)
        self.metadata = dict(model=self.model_info, generation_config=self.generation_config,
                             effective_generation_config=self.effective_generation_config,
                             generation_config_sha256=sha256_json(self.effective_generation_config),
                             chat_template_sha256=self.chat_template_sha256,
                             loaded_unique_parameters=count, parameter_count_matches=True,
                             attention_implementation=self.model.config._attn_implementation,
                             kernels=kernels, load_time_s=time.perf_counter() - start,
                             cuda_device_name=torch.cuda.get_device_name(self.device) if self.is_cuda else None,
                             load_peak_allocated=torch.cuda.max_memory_allocated(self.device) if self.is_cuda else 0,
                             load_peak_reserved=torch.cuda.max_memory_reserved(self.device) if self.is_cuda else 0)
        self.last_rendered_prompt = ''

    def generate(self, image_path: str | Path, system: str, user: str,
                 image_sha256: str | None = None) -> dict:
        import torch

        checked = verify_image(Path(image_path), image_sha256)
        messages = [dict(role='system', content=system),
                    dict(role='user', content=[dict(type='image', image=str(Path(image_path).resolve())),
                                               dict(type='text', text=user)])]
        template_kwargs = dict(add_generation_prompt=True, enable_thinking=False)
        rendered = self.processor.apply_chat_template(messages, tokenize=False, **template_kwargs)
        if not rendered.endswith(THINK_SUFFIX):
            raise RuntimeError('Chat template did not end with the required empty thinking block')
        self.last_rendered_prompt = rendered
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                        return_tensors='pt', **template_kwargs)
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            n_input = inputs['input_ids'].shape[1]
            grid = inputs.get('image_grid_thw')
            grid = grid.detach().cpu().tolist() if grid is not None else None
            if grid is not None and len(grid) == 1:
                grid = grid[0]
            out = self.model.generate(**inputs, **self.generation_config)
            generated = out[0, n_input:]
            ids = generated.detach().cpu().tolist()
            raw = self.processor.decode(generated, skip_special_tokens=True)
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        return dict(raw_text=raw, generated_token_ids=ids, n_input_tokens=n_input,
                    n_generated_tokens=len(ids), hit_max_new_tokens=len(ids) >= self.generation_config['max_new_tokens'],
                    image_grid_thw=grid, latency_s=time.perf_counter() - start,
                    cuda_max_memory_allocated=torch.cuda.max_memory_allocated(self.device) if self.is_cuda else 0,
                    cuda_max_memory_reserved=torch.cuda.max_memory_reserved(self.device) if self.is_cuda else 0,
                    rendered_prompt_sha256=sha256_text(rendered), image_sha256_checked=checked,
                    thinking_disabled_verified=True)
