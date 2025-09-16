# src/CLIApp.py
# THIS CODE WILL HANDLE THE HIGH LEVEL LOGIC OF THE APP
import sys

from .Client import HFClient
from .Metrics import RampUpTime
from .Parser import Parser

if __name__ == "__main__":
    parse = Parser(sys.argv[1])
    print(f"Parser groups: {parse.getGroups()}")
    client = HFClient(max_requests=3)
    metric = RampUpTime(client)
    result = metric.compute(parse.getGroups())
