import logging
import logging.config
import warnings

from src import utils
from src import data_loader, model_trainer

logging.config.dictConfig(utils.load_config_file("logging_config.yaml"))
warnings.filterwarnings("ignore")

if __name__=="__main__" :
    config = utils.load_config_file()
    logger = logging.getLogger()
    seed = config['data']['seed']

    for task in ["test"]: # task name
        # create dataset
        data_loader.main_data_loader(config = config, task_name = task, logger = logger, seed = seed)

        # train the models
        model_trainer.train_evaluate_save(config, task_name = task, logger = logger)
