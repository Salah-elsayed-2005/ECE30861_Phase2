import importlib
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestCLIApp(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        logging.shutdown()

    def tearDown(self) -> None:
        logging.shutdown()

    def _reload_modules(self) -> None:
        import src.logging_utils as logging_utils
        import src.CLIApp as cli_app
        importlib.reload(logging_utils)
        importlib.reload(cli_app)
        self.cli_app = cli_app

    def test_validate_env_passes_with_valid_settings(self) -> None:
        log_path = Path(self.tmpdir.name) / "app.log"
        env = {
            "GITHUB_TOKEN": "ghp_valid_token",
            "LOG_FILE": str(log_path),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self._reload_modules()
            self.cli_app._validate_env_or_exit()

        self.assertTrue(log_path.exists())

    def test_validate_env_fails_with_bad_token(self) -> None:
        log_path = Path(self.tmpdir.name) / "app.log"
        env = {
            "GITHUB_TOKEN": "bad",
            "LOG_FILE": str(log_path),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self._reload_modules()
            with self.assertRaises(SystemExit):
                self.cli_app._validate_env_or_exit()

    def test_validate_env_missing_token_exits(self) -> None:
        log_path = Path(self.tmpdir.name) / "app.log"
        env = {
            "LOG_FILE": str(log_path),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self._reload_modules()
            with self.assertRaises(SystemExit):
                self.cli_app._validate_env_or_exit()

    def test_validate_env_invalid_log_extension(self) -> None:
        env = {
            "GITHUB_TOKEN": "ghp_valid_token",
            "LOG_FILE": "output.txt",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self._reload_modules()
            with self.assertRaises(SystemExit):
                self.cli_app._validate_env_or_exit()

    def test_validate_env_unwritable_log_path(self) -> None:
        log_path = Path(self.tmpdir.name) / "app.log"
        env = {
            "GITHUB_TOKEN": "ghp_valid_token",
            "LOG_FILE": str(log_path),
        }
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("pathlib.Path.mkdir", side_effect=OSError("no perm")):
            self._reload_modules()
            with self.assertRaises(SystemExit):
                self.cli_app._validate_env_or_exit()


if __name__ == "__main__":
    unittest.main()
