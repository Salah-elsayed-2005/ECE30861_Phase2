import unittest
from src.Client import Client

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
        self.client.request() # This should be fine (DummyClient.n_reqs: 3 -> 2)
        self.client.request() # This should be fine (DummyClient.n_reqs: 2 -> 1)
        self.client.request() # This should be fine (DummyClient.n_reqs: 1 -> 0)
        with self.assertRaises(Exception, msg="Upon fourth request error should be raised"):
            self.client.request() # This should be fine (DummyClient.n_reqs: 1 -> 0)
            
if __name__ == '__main__':
    unittest.main()