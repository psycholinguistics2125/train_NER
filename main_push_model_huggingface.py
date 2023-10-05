import spacy
import os
from spacy.cli import package as spacy_package
from pathlib import Path

from spacy_huggingface_hub import push



def from_task_to_paths(task_name = "test", model_name = "test_model_name"):
   

    # Replace '/content/ner_model' with the path to your custom model
    model_path = os.path.join(f"models/model_{task_name}/model-best/")

    # Replace '/content/output' with the directory where you want to save the wheel package
    output_path = Path("./spacy_wheel")

    input_dir = Path(model_path)

    return input_dir, output_path, model_name


if __name__ == "__main__":
    # Save the model as a wheel package
    # nlp = spacy.load(model_path)
    
    todo = {
        "test": "test_model_name",

    }
    for task, model_name in todo.items() :
        input_dir, output_path, model_name = from_task_to_paths(task_name = task,model_name = model_name)
        wheel_path =  f"spacy_wheel/fr_{model_name}-0.0.1/dist/fr_{model_name}-0.0.1-py3-none-any.whl"
        print(wheel_path)
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        spacy_package(input_dir, output_path, force=True, create_wheel=True, name=model_name, version="0.0.1")

        result = push(wheel_path)

        print(f"The models is now upload to {result['url']}")
