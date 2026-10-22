import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.always_on_runtime import load_environment


class RuntimeNetworkConfigurationTest(unittest.TestCase):
    def test_explicit_proxy_overrides_inherited_lowercase_and_preserves_local_bypass(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"https_proxy": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1", "all_proxy": "socks5://127.0.0.1:1"}):
            config = Path(temp)/"runtime.env"
            config.write_text("HTTPS_PROXY=http://proxy.example:8080\nALL_PROXY=''\nSIMULATION_HTTP_HOST=172.18.0.1\n")
            values = load_environment(config)
            self.assertEqual(values["https_proxy"], "http://proxy.example:8080")
            self.assertEqual(values["HTTPS_PROXY"], values["https_proxy"])
            self.assertIn("172.18.0.1", values["no_proxy"])
            self.assertIn("127.0.0.1", values["NO_PROXY"])
            self.assertEqual(values["all_proxy"], "")


if __name__ == "__main__":
    unittest.main()
