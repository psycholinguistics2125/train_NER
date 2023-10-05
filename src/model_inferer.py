"""
description: functions that helps making predictions
"""
import os
import torch
import spacy
import pandas as pd
import logging
from collections import Counter

from thinc.api import set_gpu_allocator, require_gpu

from src.utils import compute_gpu_free_memory


def make_one_prediction(text: str, model: spacy.Language, init_dic: dict) -> list:
    """
    Predict the entities in the text, using the model

    Args:
        text (str): _description_
        model (spacy.Language): _description_

    Returns:
        list: _description_
    """
    doc = model(text)
    ent_list = []
    for ent in doc.ents:
        ent_list.append((ent.text, ent.start, ent.end, ent.label_))

    # strore result in dict
    df_ent = pd.DataFrame(ent_list, columns=["text", "start", "end", "label"])
    result = {**init_dic, **df_ent["label"].value_counts()}

    # empty memory of gpu
    doc._.trf_data = None

    torch.cuda.empty_cache()

    return df_ent, result


def make_corpus_prediction(
    data: pd.DataFrame, config: dict, task_name="on", logger=logging.getLogger()
):
    """make prediction for a corpus for on task and save it into the folder on config

    Args:
        data (pd.DataFrame): _description_
        config (dict): _description_
        task_name (str, optional): _description_. Defaults to "on".
        logger (_type_, optional): _description_. Defaults to logging.getLogger().
    """

    logger.info("Seting up GPU..")

    set_gpu_allocator("pytorch")
    require_gpu()
    init_dic = {elt: 0 for elt in config["data"][f"data_{task_name}"]["labels"]}

    predictions = pd.DataFrame(columns=list(init_dic.keys()))

    model_path = os.path.join(
        config["training"][f"model_{task_name}"]["model_folder"], "model-best"
    )
    logger.info(f"Loading_model from {model_path}")
    saving_folder = os.path.join(
        config["inference"]["saving_path"], f"inference_{task_name}"
    )
    if not os.path.exists(config["inference"]["saving_path"]):
        os.mkdir(config["inference"]["saving_path"])
        logger.info(f" {config['inference']['saving_path']} was created ! ")
    if not os.path.exists(saving_folder):
        os.mkdir(saving_folder)
        logger.info(f" {saving_folder} was created ! ")

    logger.info(f"Results will be saved in {saving_folder}")

    model = spacy.load(model_path)
    logger.info(f"Model loaded ! ")

    logger.info(f"Beginning the inference of {(len(data))} documents ..")

    for i in range(len(data)):
        line = data.iloc[i]
        code = line["code"]

        file_name = f"{task_name}_{code}.csv"
        file_path = os.path.join(saving_folder, file_name)
        logger.info(f"Doing inference for document : {code}")
        logger.info(f"Result saved in : {file_path}")

        if file_name in os.listdir(saving_folder):
            logger.info(f"{code} is already done ! ")
            df_ents = pd.read_csv(file_path)
            predictions.loc[i] = pd.Series(
                {**init_dic, **df_ents["label"].value_counts()}
            )
        else:
            text = line["text"]
            try:
                df_ents, ents = make_one_prediction(text, model, init_dic=init_dic)
                df_ents.to_csv(file_path)
                logger.info(f"Result saved in : {file_path}")
                predictions.loc[i] = pd.Series(ents)
            except Exception as e:
                logger.error(f"{code} NOT done because of {e}, continuing...")
                predictions.loc[i] = pd.Series(init_dic)
                continue

        logger.info(f"Free memory on GPU is: {compute_gpu_free_memory()}")

    result = data[["code", "critereA"]].merge(
        predictions, left_index=True, right_index=True
    )
    result_path = os.path.join(saving_folder, "inference_table.csv")
    result.to_csv(result_path)
    logger.info(f"All results saved in {result_path} !")
    return result


def check_entities_count(
    config: dict, task_name: str, logger=logging.getLogger()
) -> list:
    """
    check the data consistency in inference
    Args:
        config (dict): _description_
        task_name (str): _description_
        logger (_type_, optional): _description_. Defaults to logging.getLogger().

    Returns:
        list: _description_
    """
    folder = os.path.join(config["inference"]["saving_path"], f"inference_{task_name}")
    all = pd.read_csv(os.path.join(folder, "inference_table.csv"))
    labels_list = config["data"][f"data_{task_name}"]["labels"]
    init_dic = {elt: 0 for elt in labels_list}
    ERROR = []
    for code in all["code"].tolist():
        try:
            path_example = os.path.join(folder, f"/{task_name}_{code}.csv")
            df_example = pd.read_csv(path_example)
            count_dict = {**init_dic, **df_example["label"].value_counts()}
            for label in labels_list:
                try:
                    df_value = all[all["code"] == code][label].values[0]
                    assert df_value == count_dict[label]
                except Exception as e:
                    logger(f" Assertion Fail because of {e}, in code: {code}")
                    logger(f"{label} in independent file : {count_dict[label]}")
                    logger(f"{label} in dataframe : {df_value}")
                    ERROR.append(code)
        except Exception as e:
            logger.info(f"Cannot read {code} file bevcause of {e}")

    if len(ERROR) == 0:
        logger.info("All test passed successfully !")
    else:
        logger.info(f"There are some mistakes, need to check {str(ERROR)}!")

    return ERROR
