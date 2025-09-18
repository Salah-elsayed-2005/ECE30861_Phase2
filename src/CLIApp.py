# src/CLIApp.py
# THIS CODE WILL HANDLE THE HIGH LEVEL LOGIC OF THE APP
import sys

from src.Parser import Parser
from src.Metrics import RampUpTime

if __name__ == "__main__":
    parse = Parser(sys.argv[1])
    print(f"Parser groups: {parse.getGroups()}")
    print(f"Model URL:  {parse.getGroups()["model_url"]}")
    ramp = RampUpTime()
    print(ramp.compute(parse.getGroups()))