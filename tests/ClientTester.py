import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from src.Client import Client, GitClient, HFClient, PurdueClient

'''
CLIENT TESTS
'''


class TestRateLimiting(unittest.TestCase):
    """
    Test that the rate limiting feature of our Client abstract
    class works properly by setting up a dummy class to derive from it
    """
    def setUp(self):
        # Create our dummy class
        class DummyClient(Client):
            n_reqs = 3

            def __init__(self):
                pass

            def can_send(self):
                return DummyClient.n_reqs > 0

            def _send(self):
                DummyClient.n_reqs -= 1
                return
        self.client = DummyClient()

    def test_rate_limiting(self):
        self.client.request()  # This should be fine (n_reqs: 3 -> 2)
        self.client.request()  # This should be fine (n_reqs: 2 -> 1)
        self.client.request()  # This should be fine (n_reqs: 1 -> 0)
        with self.assertRaises(Exception, msg="Limit should be hit"):
            self.client.request()  # This should error (n_reqs: 1 -> 0)


'''
GROK TESTS
'''


class TestPurdueClientRateLimit(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure clean slate for the shared history
        PurdueClient.request_history.clear()

    def tearDown(self) -> None:
        PurdueClient.request_history.clear()

    def test_global_rate_limit_window(self) -> None:
        client = PurdueClient(max_requests=2, token="t", window_seconds=10.0)

        with patch("time.monotonic", side_effect=[100.0, 100.0, 100.0, 111.0]):
            # First two allowed
            self.assertTrue(client.can_send())
            self.assertTrue(client.can_send())
            # Third within the same timestamp denied
            self.assertFalse(client.can_send())
            # Advance beyond window; old timestamps expire
            self.assertTrue(client.can_send())


class TestPurdueClientSendAndLLM(unittest.TestCase):
    def setUp(self) -> None:
        PurdueClient.request_history.clear()

    def tearDown(self) -> None:
        PurdueClient.request_history.clear()

    @patch.object(PurdueClient, "can_send", return_value=True)
    @patch("requests.request")
    def test__send_json_success(self,
                                mock_req: MagicMock,
                                _mock_can: MagicMock) -> None:
        client = PurdueClient(max_requests=3, token="abc")

        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {"hello": "world"}
        mock_req.return_value = resp

        out = client._send("GET", "/status")
        self.assertEqual(out, {"hello": "world"})

    @patch.object(PurdueClient, "can_send", return_value=True)
    @patch("requests.request")
    def test__send_text_fallback(self,
                                 mock_req: MagicMock,
                                 _mock_can: MagicMock) -> None:
        client = PurdueClient(max_requests=3, token="abc")

        resp = MagicMock()
        resp.ok = True
        resp.json.side_effect = ValueError("not json")
        resp.text = "plain text"
        mock_req.return_value = resp

        out = client._send("GET", "/status")
        self.assertEqual(out, "plain text")

    @patch.object(PurdueClient, "can_send", return_value=True)
    @patch("requests.request")
    def test__send_error_status(self,
                                mock_req: MagicMock,
                                _mock_can: MagicMock) -> None:
        client = PurdueClient(max_requests=3, token="abc")

        resp = MagicMock()
        resp.ok = False
        resp.status_code = 400
        resp.text = "Bad Request"
        mock_req.return_value = resp

        with self.assertRaises(RuntimeError) as ctx:
            client._send("POST", "/do")
        self.assertIn("400", str(ctx.exception))

    @patch.object(PurdueClient, "can_send", return_value=True)
    @patch("requests.request")
    def test_llm_uses_message_and_parses_text(self,
                                              mock_req: MagicMock,
                                              _mock_can: MagicMock) -> None:
        client = PurdueClient(max_requests=3, token="abc")

        completion = {
            "choices": [
                {"message": {"content": "Hello back"}}
            ]
        }
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = completion
        mock_req.return_value = resp

        reply = client.llm("hey there")
        self.assertEqual(reply, "Hello back")

        # Verify the payload included our message
        args, kwargs = mock_req.call_args
        payload = kwargs.get("json")
        self.assertIsNotNone(payload)
        self.assertEqual(payload["messages"][0]["content"], "hey there")


'''
HF TESTS
'''


class TestHFClientRateLimit(unittest.TestCase):
    def setUp(self) -> None:
        HFClient.request_history.clear()

    def tearDown(self) -> None:
        HFClient.request_history.clear()

    def test_global_rate_limit_window(self) -> None:
        client = HFClient(max_requests=2, token="t", window_seconds=10.0)

        with patch("time.monotonic", side_effect=[100.0, 100.0, 100.0, 111.0]):
            self.assertTrue(client.can_send())
            self.assertTrue(client.can_send())
            self.assertFalse(client.can_send())
            self.assertTrue(client.can_send())


class TestHFClientSend(unittest.TestCase):
    def setUp(self) -> None:
        HFClient.request_history.clear()

    def tearDown(self) -> None:
        HFClient.request_history.clear()

    @patch.object(HFClient, "can_send", return_value=True)
    @patch("requests.request")
    def test__send_json_success(self,
                                mock_req: MagicMock,
                                _mock_can: MagicMock) -> None:
        client = HFClient(max_requests=3,
                          token="hf_abc",
                          base_url="https://example.test")

        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {"ok": True}
        mock_req.return_value = resp

        out = client._send("GET", "/api/models/bert-base-cased")
        self.assertEqual(out, {"ok": True})

        # Verify URL and Authorization header
        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs["url"],
                         "https://example.test/api/models/bert-base-cased")
        self.assertIn("Authorization", kwargs["headers"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer hf_abc")

    @patch.object(HFClient, "can_send", return_value=True)
    @patch("requests.request")
    def test__send_text_fallback(self,
                                 mock_req: MagicMock,
                                 _mock_can: MagicMock) -> None:
        client = HFClient(max_requests=3, token="hf_abc")

        resp = MagicMock()
        resp.ok = True
        resp.json.side_effect = ValueError("not json")
        resp.text = "plain text"
        mock_req.return_value = resp

        out = client._send("GET", "/api/spaces")
        self.assertEqual(out, "plain text")

    @patch.object(HFClient, "can_send", return_value=True)
    @patch("requests.request")
    def test__send_error_status(self,
                                mock_req: MagicMock,
                                _mock_can: MagicMock) -> None:
        client = HFClient(max_requests=3, token="hf_abc")

        resp = MagicMock()
        resp.ok = False
        resp.status_code = 403
        resp.text = "Forbidden"
        mock_req.return_value = resp

        with self.assertRaises(RuntimeError) as ctx:
            client._send("GET", "/api/spaces")
        self.assertIn("403", str(ctx.exception))

    @patch.object(HFClient, "can_send", return_value=True)
    @patch("requests.request", side_effect=requests.RequestException("boom"))
    def test__send_request_exception(self,
                                     _mock_req: MagicMock,
                                     _mock_can: MagicMock) -> None:
        client = HFClient(max_requests=3, token="hf_abc")
        with self.assertRaises(RuntimeError) as ctx:
            client._send("GET", "/api/spaces")
        self.assertIn("HF API request failed", str(ctx.exception))


'''
GIT TESTS
'''


class TestGitClientRateLimit(unittest.TestCase):
    def setUp(self) -> None:
        GitClient.request_history.clear()

    def tearDown(self) -> None:
        GitClient.request_history.clear()

    def test_global_rate_limit_window(self) -> None:
        client = GitClient(max_requests=2, repo_path=".", window_seconds=10.0)

        with patch("time.monotonic", side_effect=[100.0, 100.0, 100.0, 111.0]):
            self.assertTrue(client.can_send())
            self.assertTrue(client.can_send())
            self.assertFalse(client.can_send())
            self.assertTrue(client.can_send())


class TestGitClientSend(unittest.TestCase):
    def setUp(self) -> None:
        GitClient.request_history.clear()

    def tearDown(self) -> None:
        GitClient.request_history.clear()

    @patch.object(GitClient, "can_send", return_value=True)
    @patch("subprocess.run")
    def test__send_text_success(self,
                                mock_run: MagicMock,
                                _mock_can: MagicMock) -> None:
        client = GitClient(max_requests=3, repo_path="/tmp/repo")

        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "file1.txt\ndir/file2.py\n"
        proc.stderr = ""
        mock_run.return_value = proc

        out = client._send("ls-files")
        self.assertEqual(out, "file1.txt\ndir/file2.py")

        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], ["git", "ls-files"])  # command
        self.assertEqual(kwargs.get("cwd"), "/tmp/repo")

    @patch.object(GitClient, "can_send", return_value=True)
    @patch("subprocess.run")
    def test__send_error_status(self,
                                mock_run: MagicMock,
                                _mock_can: MagicMock) -> None:
        client = GitClient(max_requests=3, repo_path="/not/a/repo")

        proc = MagicMock()
        proc.returncode = 128
        proc.stdout = ""
        proc.stderr = "fatal: not a git repository"
        mock_run.return_value = proc

        with self.assertRaises(RuntimeError) as ctx:
            client._send("status", "--porcelain")
        self.assertIn("not a git repository", str(ctx.exception))

    @patch.object(GitClient, "can_send", return_value=True)
    @patch("subprocess.run")
    def test_list_files_parses_output(self,
                                      mock_run: MagicMock,
                                      _mock_can: MagicMock) -> None:
        client = GitClient(max_requests=3, repo_path="/tmp/repo")

        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "a.txt\nb/c.md\n"
        proc.stderr = ""
        mock_run.return_value = proc

        files = client.list_files()
        self.assertEqual(files, ["a.txt", "b/c.md"])

        args, _kwargs = mock_run.call_args
        self.assertEqual(args[0], ["git", "ls-files"])  # used ls-files


'''
ENV VAR TOKEN TESTS
'''


class TestPurdueClientEnvToken(unittest.TestCase):
    @patch.object(PurdueClient, "can_send", return_value=True)
    @patch("requests.request")
    def test_env_token_used_in_headers(self,
                                       mock_req: MagicMock,
                                       _mock_can: MagicMock) -> None:
        with patch.dict(os.environ, {"GEN_AI_STUDIO_API_KEY": "env_groq"},
                        clear=False):
            client = PurdueClient(max_requests=3)

        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {"ok": True}
        mock_req.return_value = resp

        _ = client._send("GET", "/status")

        _, kwargs = mock_req.call_args
        self.assertIn("Authorization", kwargs["headers"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer env_groq")

    def test_missing_env_raises_value_error(self) -> None:
        with patch.dict(os.environ, {"GEN_AI_STUDIO_API_KEY": ""},
                        clear=False):
            with self.assertRaises(ValueError):
                _ = PurdueClient(max_requests=1)


class TestHFClientEnvToken(unittest.TestCase):
    @patch.object(HFClient, "can_send", return_value=True)
    @patch("requests.request")
    def test_env_token_used_in_headers(self,
                                       mock_req: MagicMock,
                                       _mock_can: MagicMock) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "env_hf"}, clear=False):
            client = HFClient(max_requests=3)

        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {"ok": True}
        mock_req.return_value = resp

        _ = client._send("GET", "/api/spaces")

        _, kwargs = mock_req.call_args
        self.assertIn("Authorization", kwargs["headers"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer env_hf")

    def test_missing_env_raises_value_error(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=False):
            with self.assertRaises(ValueError):
                _ = HFClient(max_requests=1)


if __name__ == '__main__':
    unittest.main()
