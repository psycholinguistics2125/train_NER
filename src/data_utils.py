import pandas as pd

import re
from tqdm import tqdm

import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

from src.utils import flatten
from collections import Counter


def cut_text(line: pd.Series, fin_labels=["fin", "FIN", "Fin", "findecodage"]) -> str:
    """Cut the text after the first "fin" (end) labels is found

    Args:
        line (pd.Series): _description_
        fin_labels (list): _description_

    Returns:
        str: _description_
    """
    labels = line["label"]
    text = line["text"]
    i_cut = 0
    for label in labels:
        if label[-1] in fin_labels:
            i_cut = label[1]
            break
    return text[:i_cut]


def clean_annotation(annotation: pd.DataFrame) -> pd.DataFrame:
    """Remove all double spacing and update the annotation start and end

    Args:
        annotation (pd.DataFrame): data entry

    Returns:
        pd.DataFrame: data with two more columns (clean_text, clean_label)
    """

    clean_annotation = []
    for i in range(len(annotation)):
        ex_label = annotation["label"].iloc[i]
        ex_text = annotation["text"].iloc[i]
        clean_ex_label = []
        for ann in ex_label:
            s = ann[0]
            e = ann[1]
            label = ann[2]
            diff_e = len(ex_text[:e]) - len(
                ex_text[:e].replace("  ", " ")
            )  # count how many caraters was deleted
            diff_s = len(ex_text[:s]) - len(ex_text[:s].replace("  ", " "))
            clean_ex_label.append([s - diff_s, e - diff_e, label])
        clean_annotation.append(clean_ex_label)
    annotation["label"] = clean_annotation
    annotation["text"] = annotation["text"].apply(lambda x: x.replace("  ", " "))

    return annotation


def count_labels(annotations: pd.DataFrame, labels_list: list) -> pd.DataFrame:
    """Count the label and store result in Data Frame

    Args:
        annotations (pd.DataFrame): _description_
        labels_list (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    print(annotations["cutted_label"])
    labels = flatten(
        annotations["cutted_label"].apply(lambda x: [elt[-1] for elt in x]).tolist()
    )
    df_result = pd.DataFrame(dict(Counter(labels)), index=["count"])
    print(labels)
    print(df_result)
    print(labels_list)

    return df_result[labels_list]


def select_labels(x: list, new_labels: list) -> list:
    """slect only the labels present in new_labels list

    Args:
        x (list): _description_
        new_labels (list): _description_

    Returns:
        list: _description_
    """
    new_label = []
    for elt in x:
        if elt[-1] in new_labels:
            new_label.append(elt)
    return new_label


def get_indices_list(text: str, seq_len: int, overlap: int) -> list:
    """create a list with all the cutting indices based on the seq_len and the overlap

    Args:
        text (str): _description_
        seq_len (int): _description_
        overlap (int): _description_

    Returns:
        list: _description_
    """
    k = int(len(text) / seq_len)
    r = len(text) % seq_len
    incices_list = []
    for i in range(k):
        s = i * seq_len
        if s > overlap:
            s = s - overlap
        e = (i + 1) * seq_len
        incices_list.append((s, e))
    incices_list.append((e, r))
    return incices_list


def split_text(text: str, indices_list: list) -> list:
    """Split a text based on a list of indices

    Args:
        text (str): _description_
        indices_list (list): _description_

    Returns:
        list: _description_
    """
    text_list = []
    for elt in indices_list:
        text_list.append(text[elt[0] : elt[1] + 1])
    return text_list


def split_labels(labels: list, indices_list: list) -> list:
    """Split a labels based on a list of indices

    Args:
        labels (list): _description_
        indices_list (list): _description_

    Returns:
        list: _description_
    """
    label_list = []
    for elt in indices_list:
        new_label = []
        start = elt[0]
        end = elt[1]
        for label in labels:
            if label[0] >= start and label[1] <= end:
                new_start = label[0] - start
                new_end = label[1] - start
                new_label.append((new_start, new_end, label[-1]))
        label_list.append(new_label)
    return label_list


def split_one_annotation(text: str, labels: list, seq_len=1000, overlap=50):
    """For one annotation, split text and label using seq_len and over_lap as parameter

    Args:
        text (str): _description_
        labels (list): _description_
        seq_len (int, optional): _description_. Defaults to 1000.
        over_lap (int, optional): _description_. Defaults to 50.

    Returns:
        _type_: _description_
    """
    indices_list = get_indices_list(text, seq_len, overlap)
    text_list = split_text(text, indices_list)
    label_liste = split_labels(labels, indices_list)

    return text_list, label_liste


def split_all_annotation(
    annotation: pd.DataFrame, seq_len=1000, overlap=50
) -> pd.DataFrame:
    """split our text,annotation dataset into document of lenght = seq_len

    Args:
        annotation (pd.DataFrame): _description_
        seq_len (int, optional): _description_. Defaults to 1000.
        overlap (int, optional): _description_. Defaults to 50.

    Returns:
        pd.DataFrame: _description_
    """
    df_result = pd.DataFrame()
    TEXT = []
    LABEL = []
    for i in range(len(annotation)):
        text = annotation.loc[i]["text"]
        labels = annotation.loc[i]["label"]
        text_list, label_list = split_one_annotation(
            text, labels, seq_len=seq_len, overlap=overlap
        )
        TEXT = TEXT + text_list
        LABEL = LABEL + label_list

    df_result["text"] = TEXT
    df_result["labels"] = LABEL

    return df_result


def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r"\s")

    cleaned_data = []
    for elt in data:
        text = elt["text"]
        entities = elt["entities"]
        valid_entities = []
        for start, end, label in entities:
            try:
                valid_start = start
                valid_end = end
                while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]
                ):
                    valid_start += 1
                while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
                    valid_end -= 1
                valid_entities.append([valid_start, valid_end, label])

            except Exception as e:
                # print(e)
                continue

        cleaned_data.append({"text": text, "entities": valid_entities})

    return cleaned_data


def build_training_data(data: pd.DataFrame, label_liste: list) -> pd.DataFrame:
    """Apply all the cleaning and formating operation on the raw dataset

    Args:
        data (pd.DataFrame): _description_
        label_liste (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    training_data = {
        "classes": [elt.upper().strip() for elt in label_liste],
        "annotations": [],
    }
    for i in range(len(data)):
        example = data.loc[i]
        temp_dict = {}
        temp_dict["text"] = example["text"]
        temp_dict["entities"] = []
        for an in example["cutted_label"]:
            start = an[0]
            end = an[1]
            label = an[-1].upper().strip()
            temp_dict["entities"].append((start, end, label))
        training_data["annotations"].append(temp_dict)

    clean_training_data = {
        "classes": [elt.upper().strip() for elt in label_liste],
        "annotations": trim_entity_spans(training_data["annotations"]),
    }

    return clean_training_data


def save_data(training_data: pd.DataFrame, file_path: str):
    """Convert data to spacy doc and save data in spacy format

    Args:
        training_data (pd.DataFrame): _description_
        file_path (str): _description_
    """
    nlp = spacy.blank("fr")  # load a new spacy model
    doc_bin = DocBin()

    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner")
        ner = nlp.get_pipe("ner")
    for ent in training_data["classes"]:
        ner.add_label(ent)

    for training_example in tqdm(training_data["annotations"]):
        text = training_example["text"]
        labels = training_example["entities"]
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in labels:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents
        doc_bin.add(doc)
    doc_bin.to_disk(file_path)  # save the docbin object

    print(f"Data saved  in {file_path}")


def clean_label_dys(label_list):
    new = []
    for elt in label_list:
        if elt[-1] == "Auto-correction":
            elt[-1] = "AUTO_CORRECTION"
        elif elt[-1] == "Pause démarcative":
            elt[-1] = "PAUSE_DEMARCATIVE"
        elif elt[-1] == "Pause d'hésitation":
            elt[-1] = "PAUSE_HESITATION"
        elif elt[-1] == "Répétition":
            elt[-1] = "REPETITION"
        elif elt[-1] == "Inachèvement":
            elt[-1] = "INACHEVEMENT"
        elif elt[-1] == "Interruption":
            elt[-1] = "INTERRUPTION"
        elif elt[-1] == "Fin":
            elt[-1] = "FIN"
        elif elt[-1] == "Amorce":
            elt[-1] = "AMORCE"

        elt[-1] = elt[-1].upper()
        new.append(elt)
    return new
