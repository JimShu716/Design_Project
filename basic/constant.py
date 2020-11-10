import os
import logging
sun_path = '/usr/local/extstore01/gengyi/'
zhou_path = '/usr/local/extstore01/zhouhan/'
ROOT_PATH=os.path.join(sun_path, 'VisualSearch')

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
# logger.setLevel(logging.INFO)

