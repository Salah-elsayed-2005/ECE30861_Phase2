# src/CLIApp.py
# THIS CODE WILL HANDLE THE HIGH LEVEL LOGIC OF THE APP
import sys

from src.Dispatcher import Dispatcher
from src.Display import print_results
from src.Metrics import (AvailabilityMetric, CodeQuality, DatasetQuality,
                         LicenseMetric, RampUpTime, SizeMetric)
from src.Parser import Parser

if __name__ == "__main__":
    parse = Parser(sys.argv[1])
    url_groups = parse.getGroups()

    dispatcher = Dispatcher([LicenseMetric(),
                             SizeMetric(),
                             RampUpTime(),
                             AvailabilityMetric(),
                             DatasetQuality(),
                             CodeQuality()])
    for group in url_groups:
        results = dispatcher.dispatch(group)
        print_results(group, results)
