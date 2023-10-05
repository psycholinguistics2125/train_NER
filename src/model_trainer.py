import logging
import os
import pandas as pd


from spacy.cli.train import train
from spacy.cli.evaluate import evaluate


def train_evaluate_save(config: dict, task_name="on", logger=logging.getLogger()):
    logger.info(f" Starting training for task : {task_name}")
    data_folder = config["data"][f"data_{task_name}"]["dataset_folder"]
    logger.info(f"Data will be loaded from: {data_folder}")

    training_data_path = os.path.join(data_folder, "training_data.spacy")
    testing_data_path = os.path.join(data_folder, "testing_data.spacy")
    validation_data_path = os.path.join(data_folder, "validation_data.spacy")

    saving_folder = config["training"][f"model_{task_name}"]["model_folder"]
    logger.info(f"Model will be saved in {saving_folder}")
    if not os.path.exists(os.path.join(saving_folder)):
        os.mkdir(os.path.join(saving_folder))
        logger.info(f"The saving folder was created")

    training_config_path = os.path.join(
        config["training"]["config_folder"],
        config["training"][f"model_{task_name}"]["config_name"],
    )
    logger.info(f"The training config use is : {training_config_path}")

    gpu = config["training"][f"model_{task_name}"]["gpu"]
    test_performances_filepath = os.path.join(
        saving_folder,
        f"test_{config['training'][f'model_{task_name}']['performance_filename']}",
    )
    validation_performances_filepath = os.path.join(
        saving_folder,
        f"validation_{config['training'][f'model_{task_name}']['performance_filename']}",
    )

    logger.info("Training the model..")
    # Train the model using the configuration file
    train(
        output_path=saving_folder,
        overrides={"paths.train": training_data_path},
        config_path=training_config_path,
        use_gpu=gpu,
    )

    logger.info("Evaluate the model on test dataset")
    # Evaluate the model on the testing data
    evaluate(
        model=os.path.join(saving_folder, "model-best"),
        data_path=testing_data_path,
        use_gpu=gpu,
        output=test_performances_filepath,
    )

    logger.info("Evaluate the model on training dataset")
    # Evaluate the model on the validation data
    evaluate(
        model=os.path.join(saving_folder, "model-best"),
        data_path=validation_data_path,
        use_gpu=gpu,
        output=validation_performances_filepath,
    )

    logger.info(f"The test performances are saved in {test_performances_filepath}")
    test_perf = pd.read_json(test_performances_filepath)["ents_per_type"].values
    logger.info(f"PERFORMANCES ON TEST: \n {test_perf}")

    logger.info(f"The val performances are saved in {validation_performances_filepath}")
    val_perf = pd.read_json(validation_performances_filepath)["ents_per_type"].values
    logger.info(f"PERFORMANCES ON VAL: \n {val_perf}")

    logger.info("Done ! ")
