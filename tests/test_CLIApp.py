import importlib
import logging
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
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
        import src.CLIApp as cli_app
        import src.logging_utils as logging_utils
        importlib.reload(logging_utils)
        importlib.reload(cli_app)
        self.cli_app = cli_app

    def _exec_cli_main(self) -> None:
        cli_path = Path(__file__).resolve().parents[1] / "src" / "CLIApp.py"
        code = cli_path.read_text()
        exec(compile(code, str(cli_path), "exec"), {"__name__": "__main__"})

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

    def test_cli_main_processes_groups_and_prints_results(self) -> None:
        log_path = Path(self.tmpdir.name) / "cli.log"
        env = {
            "GITHUB_TOKEN": "ghp_valid_token",
            "LOG_FILE": str(log_path),
        }
        groups = [{"model_url": "https://example.com/model"}]

        with mock.patch.dict(os.environ, env, clear=True):
            with ExitStack() as stack:
                mock_parser_cls = stack.enter_context(
                    mock.patch("src.Parser.Parser")
                )
                mock_dispatcher_cls = stack.enter_context(
                    mock.patch("src.Dispatcher.Dispatcher")
                )
                mock_print_results = stack.enter_context(
                    mock.patch("src.Display.print_results")
                )

                metric_paths = [
                    ("license", "src.Metrics.LicenseMetric"),
                    ("size", "src.Metrics.SizeMetric"),
                    ("ramp", "src.Metrics.RampUpTime"),
                    ("availability", "src.Metrics.AvailabilityMetric"),
                    ("dataset", "src.Metrics.DatasetQuality"),
                    ("code", "src.Metrics.CodeQuality"),
                    ("performance", "src.Metrics.PerformanceClaimsMetric"),
                    ("bus", "src.Metrics.BusFactorMetric"),
                ]
                metric_mocks = {
                    label: stack.enter_context(mock.patch(path))
                    for label, path in metric_paths
                }

                stack.enter_context(
                    mock.patch.object(
                        sys,
                        "argv",
                        ["cli_app.py", "input.json"],
                    )
                )

                mock_parser = mock_parser_cls.return_value
                mock_parser.getGroups.return_value = groups

                mock_dispatcher = mock_dispatcher_cls.return_value
                mock_dispatcher.dispatch.return_value = ["result"]

                for label, metric_cls in metric_mocks.items():
                    metric_cls.return_value = f"metric_{label}"

                self._exec_cli_main()

                mock_parser_cls.assert_called_once_with("input.json")
                mock_parser.getGroups.assert_called_once_with()

                mock_dispatcher_cls.assert_called_once()
                dispatched_metrics = mock_dispatcher_cls.call_args.args[0]
                self.assertEqual(len(dispatched_metrics), 8)
                mock_dispatcher.dispatch.assert_called_once_with(groups[0])
                mock_print_results.assert_called_once_with(
                    groups[0],
                    ["result"],
                )

    def test_cli_main_exits_when_missing_input_argument(self) -> None:
        log_path = Path(self.tmpdir.name) / "cli.log"
        env = {
            "GITHUB_TOKEN": "ghp_valid_token",
            "LOG_FILE": str(log_path),
        }

        with mock.patch.dict(os.environ, env, clear=True):
            with ExitStack() as stack:
                mock_parser_cls = stack.enter_context(
                    mock.patch("src.Parser.Parser")
                )
                mock_dispatcher_cls = stack.enter_context(
                    mock.patch("src.Dispatcher.Dispatcher")
                )
                stack.enter_context(mock.patch("src.Display.print_results"))

                for path in [
                    "src.Metrics.LicenseMetric",
                    "src.Metrics.SizeMetric",
                    "src.Metrics.RampUpTime",
                    "src.Metrics.AvailabilityMetric",
                    "src.Metrics.DatasetQuality",
                    "src.Metrics.CodeQuality",
                    "src.Metrics.PerformanceClaimsMetric",
                    "src.Metrics.BusFactorMetric",
                ]:
                    stack.enter_context(mock.patch(path))

                stack.enter_context(
                    mock.patch.object(sys, "argv", ["cli_app.py"])
                )

                with self.assertRaises(SystemExit):
                    self._exec_cli_main()

            mock_parser_cls.assert_not_called()
            mock_dispatcher_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
