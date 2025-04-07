"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import importlib

from utils.logger import FedLogger


def load_aggregator(id, aggregator):
    logger = FedLogger(id=id, loggername="AGGREGATION_LOADER")

    module_name = f"server.aggregation.aggregator_{aggregator}"
    try:
        module = importlib.import_module(module_name)
        logger.info(
            "fedserver.aggregation.module",
            f"Aggregation module name:,{module_name}",
        )
        return module
    except ImportError:
        logger.error(
            "fedserver.aggregation.invalid.module",
            f"Could not import the module ,{module_name}",
        )
