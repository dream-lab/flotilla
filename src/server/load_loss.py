"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import importlib

from utils.logger import FedLogger


def load_loss(id, loss_function, custom):
    logger = FedLogger(id=id, loggername="LOSS_FUNC_LOADER")

    if custom:
        module_name = f"server.loss.loss_function_{loss_function}"
    else:
        module_name = loss_function
    print(loss_function, module_name)
    try:
        module = importlib.import_module(module_name)
        logger.info(
            "fedserver.loss_function.module",
            f"Loss function module name:,{module_name}",
        )
        return module
    except ImportError:
        logger.error(
            "fedserver.loss_function.invalid.module",
            f"Could not import the module ,{module_name}",
        )
        return None
