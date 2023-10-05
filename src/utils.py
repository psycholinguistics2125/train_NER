import yaml
import torch


def load_config_file(file_path="config.yaml"):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]


def compute_gpu_free_memory() -> int:
    """compute the present free memory

    Returns:
        int: _description_
    """
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    return t - (r + a)


"""
def get_gpu_info(logger=False):
    
    print(f"You are using tf version : {tf.version.VERSION}")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus :
        if logger :
            logger.info("GPU is available")
        else :
            print("GPU is available")
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                if logger :
                    logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
                else :
                    print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            if logger :
                logger.error(f"{e}")
            else :
                print(e)
    else :
        if logger :
            logger.info("GPU is NOT available")
        else :
            print("GPU is NOT available")

"""
