import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, '../airllm')

if torch is not None:
    from airllm import compress_layer_state_dict, uncompress_layer_state_dict

@unittest.skipIf(torch is None, "torch is not installed")
class TestCompression(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_should_compress_uncompress(self):
        if torch.cuda.is_available():
            test_device = "cuda"
        elif hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
            test_device = "xpu"
        else:
            test_device = "cpu"

        a0 = torch.normal(0, 1, (32, 128), dtype=torch.float16).to(test_device)
        a1 = torch.normal(0, 1, (32, 128), dtype=torch.float16).to(test_device)

        a_state_dict = {'a0':a0, 'a1':a1}

        loss_fn = torch.nn.MSELoss()

        compressions = [None]
        if torch.cuda.is_available():
            compressions.extend(['4bit', '8bit'])

        for iloop in range(10):
            for compression in compressions:
                b = compress_layer_state_dict(a_state_dict, compression)

                if iloop < 2:
                    print(f"for compression {compression}, compressed to: { {k:v.shape for k,v in b.items()} }")

                aa = uncompress_layer_state_dict(b, device="cuda" if torch.cuda.is_available() else test_device)

                for k in aa.keys():

                    if compression is None:
                        self.assertTrue(torch.equal(aa[k], a_state_dict[k]))
                    else:
                        RMSE_loss = torch.sqrt(loss_fn(aa[k], a_state_dict[k])).detach().cpu().item()
                        print(f"compression {compression} loss: {RMSE_loss}")
                        self.assertLess(RMSE_loss, 0.1)
