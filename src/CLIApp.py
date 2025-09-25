# src/CLIApp.py
# THIS CODE WILL HANDLE THE HIGH LEVEL LOGIC OF THE APP
import sys

from src.Dispatcher import Dispatcher
from src.Display import print_results
from src.logging_utils import get_logger
from src.Metrics import (AvailabilityMetric, BusFactorMetric, CodeQuality,
                         DatasetQuality, LicenseMetric,
                         PerformanceClaimsMetric, RampUpTime, SizeMetric)
from src.Parser import Parser

logger = get_logger(__name__)

if __name__ == "__main__":
    input_path = sys.argv[1]
    logger.info("Starting CLI processing for %s", input_path)
    parse = Parser(input_path)
    url_groups = parse.getGroups()

    dispatcher = Dispatcher([LicenseMetric(),
                             SizeMetric(),
                             RampUpTime(),
                             AvailabilityMetric(),
                             DatasetQuality(),
                             CodeQuality(),
                             PerformanceClaimsMetric(),
                             BusFactorMetric()])
    for group in url_groups:
        logger.debug("Dispatching metrics for group %s", group)
        results = dispatcher.dispatch(group)
        logger.debug("Metrics complete for group %s", group)
        print_results(group, results)
    logger.info("Finished processing %d group(s)", len(url_groups))
