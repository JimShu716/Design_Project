import os
import logging

ROOT_PATH=os.path.join('/usr/local/extstore01/pengnanf/', 'VisualSearch')

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
# logger.setLevel(logging.INFO)

