"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import importlib

from utils.logger import FedLogger


def load_optimizer(id, optimizer_function, custom=False):
    logger = FedLogger(id=id, loggername="OPTIMIZER_LOADER")

    if custom:
        module_name = f"server.optimizer.optimizer_{optimizer_function}"
    else:
        module_name = optimizer_function

    print(optimizer_function, module_name)
    try:
        module = importlib.import_module(module_name)
        logger.info(
            "fedserver.optimizer.module", f"Optimizer module name:,{module_name}"
        )
        return module
    except ImportError:
        logger.error(
            "fedserver.optimizer.invalid.module",
            f"Could not import the module ,{module_name}",
        )
        return None
