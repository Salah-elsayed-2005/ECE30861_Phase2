# src/CLIApp.py
# THIS CODE WILL HANDLE THE HIGH LEVEL LOGIC OF THE APP
import sys

from src.Dispatcher import Dispatcher
from src.Metrics import (AvailabilityMetric, CodeQuality, DatasetQuality,
                         LicenseMetric, RampUpTime, SizeMetric)
from src.Parser import Parser

if __name__ == "__main__":
    parse = Parser(sys.argv[1])
    urls_dict = parse.getGroups()

    dispatcher = Dispatcher([LicenseMetric(),
                             SizeMetric(),
                             RampUpTime(),
                             AvailabilityMetric(),
                             DatasetQuality(),
                             CodeQuality()])
    results = dispatcher.dispatch(parse.getGroups())
    print(results)
    print("\n\n")
    for res in results:
        print(f"{res.metric}: {res.value}")
