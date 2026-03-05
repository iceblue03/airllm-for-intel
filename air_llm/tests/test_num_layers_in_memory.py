import unittest
import warnings
import sys
import types
from unittest.mock import MagicMock


def _setup_mock_modules():
    """Set up mock modules for all heavy dependencies."""
    mock_modules = {}

    # Create a comprehensive mock torch module that supports attribute access
    mock_torch = types.ModuleType('torch')
    mock_torch.__path__ = ['/fake/torch']
    mock_torch.float16 = 'float16'
    mock_torch.device = MagicMock(return_value='cpu')
    mock_torch.inference_mode = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(return_value=None),
        __exit__=MagicMock(return_value=False),
    ))
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.is_available = MagicMock(return_value=False)
    mock_torch.cuda.Stream = MagicMock
    mock_torch.Tensor = MagicMock
    mock_torch.LongTensor = MagicMock
    mock_torch.FloatTensor = MagicMock
    mock_torch.long = 'long'
    mock_modules['torch'] = mock_torch

    # torch sub-modules
    for sub in ['torch.nn', 'torch.nn.functional', 'torch.cuda', 'torch.utils',
                'torch.utils.data', 'torch.distributed', 'torch.autograd']:
        mock_modules[sub] = MagicMock()

    # Create a real class for GenerationMixin so it can be used as a base class
    class MockGenerationMixin:
        pass

    mock_transformers = MagicMock()
    mock_transformers.GenerationMixin = MockGenerationMixin
    mock_transformers.GenerationConfig = MagicMock
    mock_transformers.AutoConfig = MagicMock()
    mock_transformers.AutoConfig.from_pretrained = MagicMock(return_value=MagicMock(quantization_config=None))
    mock_transformers.AutoTokenizer = MagicMock()
    mock_transformers.AutoModelForCausalLM = MagicMock()
    mock_transformers.AutoModel = MagicMock()
    mock_transformers.LlamaForCausalLM = MagicMock()
    mock_modules['transformers'] = mock_transformers

    # Mock all other dependencies
    for mod_name in [
        'tqdm', 'safetensors', 'safetensors.torch',
        'transformers.modeling_outputs',
        'transformers.quantizers', 'transformers.cache_utils',
        'accelerate', 'accelerate.utils', 'accelerate.utils.modeling',
        'optimum', 'optimum.bettertransformer',
        'bitsandbytes',
        'huggingface_hub',
    ]:
        mock_modules[mod_name] = MagicMock()

    return mock_modules


def _import_base_class(mock_modules):
    """Import AirLLMBaseModel with mocked dependencies."""
    saved = {}
    for name, mod in mock_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        import importlib

        # First, handle airllm package/submodules
        airllm_pkg = types.ModuleType('airllm')
        airllm_pkg.__path__ = ['/home/runner/work/airllm-for-intel/airllm-for-intel/air_llm/airllm']
        airllm_pkg.__package__ = 'airllm'
        sys.modules['airllm'] = airllm_pkg

        # Mock airllm.profiler
        profiler_mod = types.ModuleType('airllm.profiler')
        profiler_mod.LayeredProfiler = MagicMock
        sys.modules['airllm.profiler'] = profiler_mod
        airllm_pkg.profiler = profiler_mod

        # Mock airllm.utils
        utils_mod = types.ModuleType('airllm.utils')
        utils_mod.clean_memory = MagicMock()
        utils_mod.load_layer = MagicMock()
        utils_mod.find_or_create_local_splitted_path = MagicMock(return_value=('/fake', '/fake'))
        sys.modules['airllm.utils'] = utils_mod
        airllm_pkg.utils = utils_mod

        memory_utils_mod = types.ModuleType('airllm.memory_utils')
        memory_utils_mod.suggest_num_layers = MagicMock(return_value=3)
        memory_utils_mod.confirm_num_layers = MagicMock(return_value=3)
        memory_utils_mod.get_available_memory_gb = MagicMock(return_value=8.0)
        memory_utils_mod.get_avg_layer_size_gb = MagicMock(return_value=1.5)
        memory_utils_mod.calculate_min_required_memory_gb = MagicMock(return_value=2.0)
        memory_utils_mod.check_memory_and_confirm = MagicMock(return_value=True)
        sys.modules['airllm.memory_utils'] = memory_utils_mod
        airllm_pkg.memory_utils = memory_utils_mod

        spec = importlib.util.spec_from_file_location(
            "airllm.airllm_base",
            "/home/runner/work/airllm-for-intel/airllm-for-intel/air_llm/airllm/airllm_base.py",
            submodule_search_locations=[]
        )
        module = importlib.util.module_from_spec(spec)
        module.__package__ = 'airllm'
        sys.modules['airllm.airllm_base'] = module
        spec.loader.exec_module(module)
        return module.AirLLMBaseModel
    finally:
        for name in list(sys.modules.keys()):
            if name.startswith('airllm'):
                sys.modules.pop(name, None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


class TestNumLayersInMemoryValidation(unittest.TestCase):
    """Test parameter validation for num_layers_in_memory."""

    @classmethod
    def setUpClass(cls):
        cls.mock_modules = _setup_mock_modules()
        cls.AirLLMBaseModel = _import_base_class(cls.mock_modules)

    def _create_instance(self, num_layers_in_memory=1, compression=None):
        """Create an AirLLMBaseModel with heavily mocked internals."""
        cls = self.AirLLMBaseModel

        orig_init_model = cls.init_model
        orig_set_layer_names = cls.set_layer_names_dict
        orig_get_gen_config = cls.get_generation_config
        orig_get_tokenizer = cls.get_tokenizer

        try:
            if 'airllm' not in sys.modules:
                airllm_pkg = types.ModuleType('airllm')
                airllm_pkg.__path__ = ['/home/runner/work/airllm-for-intel/airllm-for-intel/air_llm/airllm']
                sys.modules['airllm'] = airllm_pkg
            if 'airllm.memory_utils' not in sys.modules:
                memory_utils_mod = types.ModuleType('airllm.memory_utils')
                memory_utils_mod.suggest_num_layers = MagicMock(return_value=3)
                memory_utils_mod.confirm_num_layers = MagicMock(return_value=3)
                memory_utils_mod.get_available_memory_gb = MagicMock(return_value=8.0)
                memory_utils_mod.get_avg_layer_size_gb = MagicMock(return_value=1.5)
                memory_utils_mod.calculate_min_required_memory_gb = MagicMock(return_value=2.0)
                memory_utils_mod.check_memory_and_confirm = MagicMock(return_value=True)
                sys.modules['airllm.memory_utils'] = memory_utils_mod

            def mock_init_model(self_):
                # Simulate model with nested structure for layer counting
                mock_layer = MagicMock()
                mock_layers_container = MagicMock()
                mock_layers_container.__len__ = MagicMock(return_value=2)
                mock_layers_container.__iter__ = MagicMock(return_value=iter([mock_layer, mock_layer]))
                mock_layers_container.__getitem__ = MagicMock(return_value=mock_layer)

                mock_model_inner = MagicMock()
                mock_model_inner.layers = mock_layers_container

                self_.model = MagicMock()
                self_.model.model = mock_model_inner
                self_.model.named_buffers = MagicMock(return_value=[])

            cls.init_model = mock_init_model
            cls.set_layer_names_dict = lambda self: setattr(self, 'layer_names_dict', {
                'embed': 'model.embed_tokens',
                'layer_prefix': 'model.layers',
                'norm': 'model.norm',
                'lm_head': 'lm_head',
            })
            cls.get_generation_config = lambda self: MagicMock()
            cls.get_tokenizer = lambda self, hf_token=None: MagicMock()

            instance = cls(
                '/fake/model',
                device='cpu',
                compression=compression,
                prefetching=False,
                num_layers_in_memory=num_layers_in_memory,
            )
            return instance
        finally:
            cls.init_model = orig_init_model
            cls.set_layer_names_dict = orig_set_layer_names
            cls.get_generation_config = orig_get_gen_config
            cls.get_tokenizer = orig_get_tokenizer

    def test_default_num_layers_in_memory_is_1(self):
        instance = self._create_instance(num_layers_in_memory=1)
        self.assertEqual(instance.num_layers_in_memory, 1)

    def test_num_layers_in_memory_greater_than_1(self):
        instance = self._create_instance(num_layers_in_memory=4)
        self.assertEqual(instance.num_layers_in_memory, 4)

    def test_num_layers_in_memory_zero_raises(self):
        with self.assertRaises(ValueError):
            self._create_instance(num_layers_in_memory=0)

    def test_num_layers_in_memory_negative_raises(self):
        with self.assertRaises(ValueError):
            self._create_instance(num_layers_in_memory=-1)

    def test_num_layers_in_memory_with_compression_warns_and_forces_1(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            instance = self._create_instance(num_layers_in_memory=4, compression='4bit')
            self.assertEqual(instance.num_layers_in_memory, 1)
            self.assertTrue(any("num_layers_in_memory" in str(warning.message) for warning in w))

    def test_num_layers_in_memory_auto_sets_suggested_value(self):
        instance = self._create_instance(num_layers_in_memory='auto')
        self.assertEqual(instance.num_layers_in_memory, 3)

    def test_num_layers_in_memory_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            self._create_instance(num_layers_in_memory='invalid')

    def test_num_layers_in_memory_1_with_compression_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            instance = self._create_instance(num_layers_in_memory=1, compression='4bit')
            self.assertEqual(instance.num_layers_in_memory, 1)
            self.assertFalse(any("num_layers_in_memory" in str(warning.message) for warning in w))


if __name__ == '__main__':
    unittest.main()
