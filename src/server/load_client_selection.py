"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import importlib

from utils.logger import FedLogger


def load_client_selection(id, client_selection):
    logger = FedLogger(id=id, loggername="CLIENT_SELECTION_LOADER")

    client_selection_stratergy = client_selection
    module_name = (
        f"server.clientselection.client_selection_{client_selection_stratergy}"
    )
    try:
        module = importlib.import_module(module_name)
        logger.info(
            "fedserver.client_selection.module",
            f"Client selection module name:,{module_name}",
        )
        return module
    except ImportError:
        logger.error(
            "fedserver.client_selection.invalid.module",
            f"Could not import the module ,{module_name}",
        )
