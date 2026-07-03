import importlib
import sys
import types
from types import SimpleNamespace


class _FakeProvider:
    def __init__(self):
        self.calls = []
        self.finalized = False
        self.attention_backend = None
        self.tensor_model_parallel_size = 1
        self.sequence_parallel = False
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.variable_seq_lengths = False
        self.num_layers = 8
        self.moe_layer_freq = None
        self.fp16 = False
        self.bf16 = False
        self.params_dtype = None
        self.vision_dp_when_cp = False

    def finalize(self):
        self.finalized = True

    def provide(self, pre_process=True, post_process=True, vp_stage=None):
        self.calls.append(
            {
                "pre_process": pre_process,
                "post_process": post_process,
                "vp_stage": vp_stage,
            }
        )
        return SimpleNamespace(named_modules=lambda: [])


def _install_fake_megatron(monkeypatch, provider=None):
    provider = provider or _FakeProvider()

    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")
    models = types.ModuleType("megatron.core.models")
    gpt = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    transformer = types.ModuleType("megatron.core.transformer")
    spec_utils = types.ModuleType("megatron.core.transformer.spec_utils")
    transformer_config = types.ModuleType("megatron.core.transformer.transformer_config")
    training = types.ModuleType("megatron.training")
    arguments = types.ModuleType("megatron.training.arguments")
    bridge = types.ModuleType("megatron.bridge")

    class _FakeGPTModel:
        pass

    class _FakeTransformerConfig:
        pass

    class _FakeAutoBridge:
        @classmethod
        def from_hf_pretrained(cls, *args, **kwargs):
            return cls()

        def to_megatron_provider(self, load_weights=False):
            return provider

    mpu.get_virtual_pipeline_model_parallel_world_size = lambda: 2
    mpu.get_virtual_pipeline_model_parallel_rank = lambda: 1
    mpu.get_context_parallel_world_size = lambda: 1
    mpu.get_context_parallel_rank = lambda: 0
    mpu.get_tensor_model_parallel_rank = lambda: 0
    core.mpu = mpu
    core.tensor_parallel = tensor_parallel
    gpt.GPTModel = _FakeGPTModel
    gpt_layer_specs.get_gpt_decoder_block_spec = lambda *args, **kwargs: object()
    gpt_layer_specs.get_gpt_layer_local_spec = lambda *args, **kwargs: object()
    gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec = lambda *args, **kwargs: object()
    spec_utils.import_module = lambda path: object()
    transformer_config.TransformerConfig = _FakeTransformerConfig
    arguments.core_transformer_config_from_args = lambda args: _FakeTransformerConfig()
    bridge.AutoBridge = _FakeAutoBridge

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "megatron.core.tensor_parallel": tensor_parallel,
        "megatron.core.models": models,
        "megatron.core.models.gpt": gpt,
        "megatron.core.models.gpt.gpt_layer_specs": gpt_layer_specs,
        "megatron.core.transformer": transformer,
        "megatron.core.transformer.spec_utils": spec_utils,
        "megatron.core.transformer.transformer_config": transformer_config,
        "megatron.training": training,
        "megatron.training.arguments": arguments,
        "megatron.bridge": bridge,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return provider


def _load_model_provider(monkeypatch, provider=None):
    provider = _install_fake_megatron(monkeypatch, provider=provider)
    sys.modules.pop("relax.backends.megatron.model_provider", None)
    module = importlib.import_module("relax.backends.megatron.model_provider")
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 1)
    return module, provider


def _bridge_args(**overrides):
    values = {
        "megatron_to_hf_mode": "bridge",
        "hf_checkpoint": "fake-hf",
        "attention_backend": "flash",
        "tensor_model_parallel_size": 2,
        "sequence_parallel": True,
        "pipeline_model_parallel_size": 4,
        "virtual_pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "variable_seq_lengths": True,
        "dsa_indexer_loss_coeff": None,
        "dsa_indexer_use_sparse_loss": None,
        "attention_softmax_in_fp32": True,
        "bias_dropout_fusion": True,
        "apply_rope_fusion": False,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "distribute_saved_activations": False,
        "moe_router_load_balancing_type": "none",
        "moe_router_dtype": None,
        "moe_aux_loss_coeff": None,
        "moe_token_dispatcher_type": "alltoall",
        "moe_shared_expert_overlap": False,
        "moe_enable_deepep": False,
        "moe_flex_dispatcher_backend": None,
        "use_audio_in_video": False,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "vision_dp_when_tp": False,
        "vision_dp_when_cp": False,
        "calculate_per_token_loss": False,
        "num_layers": 8,
        "moe_layer_freq": None,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "fp16": False,
        "bf16": True,
        "save": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bridge_provider_receives_virtual_pipeline_size(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)

    model_provider = module.get_model_provider_func(_bridge_args(), role="actor")
    model_provider(pre_process=True, post_process=False, vp_stage=1)

    assert provider.virtual_pipeline_model_parallel_size == 2
    assert provider.finalized
    assert provider.calls == [{"pre_process": True, "post_process": False, "vp_stage": 1}]


def test_bridge_provider_receives_vision_dp_when_cp(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)

    model_provider = module.get_model_provider_func(_bridge_args(vision_dp_when_cp=True), role="actor")
    model_provider(pre_process=True, post_process=True)

    assert provider.vision_dp_when_cp is True


def test_wrapper_derives_vp_stage_from_parallel_state(monkeypatch):
    module, _ = _load_model_provider(monkeypatch)
    calls = []

    def original_provider(pre_process=True, post_process=True, vp_stage=None):
        calls.append(
            {
                "pre_process": pre_process,
                "post_process": post_process,
                "vp_stage": vp_stage,
            }
        )
        return SimpleNamespace(named_parameters=lambda: [])

    wrapped_provider = module.wrap_model_provider_with_freeze(
        original_provider,
        SimpleNamespace(only_train_params_name_list=None, freeze_params_name_list=None),
    )
    wrapped_provider(pre_process=True, post_process=False)

    assert calls == [{"pre_process": True, "post_process": False, "vp_stage": 1}]


def test_wrapper_passes_vp_stage_through_bridge_provider(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)

    bridge_provider = module.get_model_provider_func(_bridge_args(), role="actor")
    wrapped_provider = module.wrap_model_provider_with_freeze(
        bridge_provider,
        SimpleNamespace(only_train_params_name_list=None, freeze_params_name_list=None),
    )
    wrapped_provider(pre_process=True, post_process=False)

    assert provider.calls == [{"pre_process": True, "post_process": False, "vp_stage": 1}]
