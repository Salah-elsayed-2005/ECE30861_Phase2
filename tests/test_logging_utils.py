import importlib
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestLoggingUtils(unittest.TestCase):
    def setUp(self) -> None:
        logging.shutdown()

    def test_logger_disabled_when_level_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "app.log"

            with patch.dict(
                os.environ,
                {"LOG_LEVEL": "invalid", "LOG_FILE": str(log_file)},
                clear=True,
            ):
                import src.logging_utils as logging_utils
                importlib.reload(logging_utils)

                logger = logging_utils.get_logger()

            self.assertTrue(logger.disabled)
            self.assertTrue(log_file.exists())

    def test_logger_writes_when_level_is_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "app.log"

            with patch.dict(
                os.environ,
                {"LOG_LEVEL": "2", "LOG_FILE": str(log_file)},
                clear=True,
            ):
                import src.logging_utils as logging_utils
                importlib.reload(logging_utils)

                logger = logging_utils.get_logger("test")
                logger.debug("hello world")

            logging.shutdown()
            contents = log_file.read_text(encoding="utf-8")
            self.assertIn("hello world", contents)

    def test_logger_disabled_when_level_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "app.log"

            with patch.dict(
                os.environ,
                {"LOG_LEVEL": "0", "LOG_FILE": str(log_file)},
                clear=True,
            ):
                import src.logging_utils as logging_utils
                importlib.reload(logging_utils)

                logger = logging_utils.get_logger()

            logging.shutdown()
            self.assertTrue(logger.disabled)
            self.assertTrue(log_file.exists())
            self.assertEqual(log_file.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
