# src/CLIApp.py
# THIS CODE WILL HANDLE THE HIGH LEVEL LOGIC OF THE APP
import sys

from src.Dispatcher import Dispatcher
from src.Metrics import LicenseMetric, RampUpTime, SizeMetric
from src.Parser import Parser

if __name__ == "__main__":
    parse = Parser(sys.argv[1])
    urls_dict = parse.getGroups()

    dispatcher = Dispatcher([LicenseMetric(),
                             SizeMetric(),
                             RampUpTime()])
    results = dispatcher.dispatch(parse.getGroups())
    print(results)
