import pkg_resources
import matplotlib
matplotlib.use('Agg')

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'

import json
import logging
import logging.config

try:
    with open("logging.json", 'rt') as f:
        logger_config = json.load(f)
    logging.config.dictConfig(logger_config)
except AssertionError as e:
    print(e)
except FileNotFoundError as e:
    print(e)

logger = logging.getLogger(__package__)

from . import utils