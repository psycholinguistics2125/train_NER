import os
import pandas as pd

import logging
import logging.config


from src import utils
from src import data_utils


def prepare_dataset(
    annotation: pd.DataFrame, labels: list, seq_len=200, overlap=20
) -> pd.DataFrame:
    """Cut the labels based on the "fin" (end) labels

    Args:
        annotation (pd.DataFrame): _description_
        labels (list): _description_
        seq_len (int, optional): _description_. Defaults to 200.
        overlap (int, optional): _description_. Defaults to 20.

    Returns:
        pd.DataFrame: _description_
    """
    annotation["cutted_text"] = annotation.apply(
        lambda x: data_utils.cut_text(x), axis=1
    )
    data = data_utils.split_all_annotation(annotation, seq_len=seq_len, overlap=overlap)
    data = data[data.labels.apply(len) > 1].reset_index()
    data["cutted_label"] = data["labels"].apply(
        lambda x: data_utils.select_labels(x, labels)
    )

    return data


def create_dataset(
    data: pd.DataFrame,
    validation_data=pd.DataFrame(),
    logger=logging.getLogger(),
    seed=42,
) -> list:
    """
    Return train test dataset or train, test and validation if validation is True

    Args:
        data (pd.DataFrame): _description_
        validation (bool, optional): _description_. Defaults to False.

    Returns:
        list: _description_
    """
    # randomly sample the data
    data = data.sample(frac=1, random_state=seed)

    if len(validation_data) < 1:
        split_test = int(len(data) * 0.65)
        split_validation = int(len(data) * 0.85)
        train = data[:split_test].reset_index()
        logger.info(f"Training data : {len(train)}")
        test = data[split_test:split_validation].reset_index()
        logger.info(f"Testing data : {len(test)}")
        val = data[split_validation:].reset_index()
        logger.info(f"validation data : {len(val)}")
        return train, test, val

    else:
        split = int(len(data) * 0.8)
        train = data[:split].reset_index()
        logger.info(f"Training data : {len(train)}")
        test = data[split:].reset_index()
        logger.info(f"Testing data : {len(test)}")
        logger.info(f"validation data : {len(validation_data)}")
        return train, test, validation_data


def main_data_loader(config: dict, task_name="on", logger=logging.getLogger(), seed=42):
    logger.info(f"Process started for task: {task_name}")
    logger.info("Data loading started...")
    # load path based on the config file
    source_folder = config["data"]["source_folder"]
    logger.info(f"The annotation file are from {source_folder}")

    # prepare the saving folder
    saving_folder = config["data"][f"data_{task_name}"]["dataset_folder"]
    if not os.path.exists(os.path.join(saving_folder)):
        os.mkdir(os.path.join(saving_folder))
    logger.info(f"The prepare dataset (.spacy) will be saved in {saving_folder}")

    filename = config["data"][f"data_{task_name}"]["origin_file"]
    validation_filename = config["data"][f"data_{task_name}"]["validation_file"]
    labels = config["data"][f"data_{task_name}"]["labels"]
    seq_len = config["data"]["seq_len"]

    logger.info(f"Labels : {labels}")

    # deal with absence or presence of a validation dataset
    if len(validation_filename) > 0:
        validation_data_path = os.path.join(source_folder, validation_filename)
        validation_annotation = pd.read_json(validation_data_path, lines=True)
        validation_annotation = data_utils.clean_annotation(validation_annotation)
        validation_source = prepare_dataset(
            validation_annotation, [x.upper() for x in labels], seq_len=seq_len
        )
        logger.info(f"A validation dataset was found in {validation_data_path}")
    else:
        validation_source = pd.DataFrame()
        logger.info(f"No validation dataset was found , taking 10% of all the dataset")

    # load training and testing data :
    data_path = os.path.join(source_folder, filename)
    logger.info(f"Loading data from {data_path}")
    annotation = pd.read_json(data_path, lines=True)
    annotation = data_utils.clean_annotation(annotation)
    data = prepare_dataset(annotation, labels, seq_len=seq_len)
    train, test, val = create_dataset(
        data, validation_data=validation_source, logger=logger, seed=seed
    )
    for name, elt in {"train": train, "test": test, "validation": val}.items():
        df_count = data_utils.count_labels(elt, labels_list=labels)
        df_count.to_csv(os.path.join(saving_folder, f"count_{name}.csv"))
    logger.info("The source data have been loaded ! ")

    logger.info("Converting the raw data into Spacy doc..")
    training_data = data_utils.build_training_data(train, labels)
    testing_data = data_utils.build_training_data(test, labels)
    validation_data = data_utils.build_training_data(val, labels)

    logger.info(f"Saving the Spacy doc using DocBin() into {saving_folder}.")
    data_utils.save_data(
        training_data, os.path.join(saving_folder, "training_data.spacy")
    )
    data_utils.save_data(
        testing_data, os.path.join(saving_folder, "testing_data.spacy")
    )
    data_utils.save_data(
        validation_data, os.path.join(saving_folder, "validation_data.spacy")
    )

    logger.info("All done! ")
