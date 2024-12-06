import os
import codecs as cs
import orjson  # loading faster than json
import json
import pandas as pd

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion


def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motion_loader,
        text_to_sent_emb,
        text_to_token_emb,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
        dname: str = None,
        remove_annots_with_name: str = None,
        categorical_labels = None,
    ):
        if tiny:
            split = split + "_tiny"

        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)

        print(f"Loading {split} dataset with {len(self.keyids)} keys")

        self.text_to_sent_emb = text_to_sent_emb
        self.text_to_token_emb = text_to_token_emb
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        # remove annotations, for example, from specific datasets like KIT (unless I'm removing kit from kit which makes no sense)
        if remove_annots_with_name is not None and '/'+remove_annots_with_name.lower() not in path:
            self.annotations = {
                keyid: val
                for keyid, val in self.annotations.items()
                if remove_annots_with_name not in val["path"]
            }

        self.is_training = split == "train"
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

        if categorical_labels is not None:
            self.labels = self.load_categorical_labels(categorical_labels)
        else:
            self.labels = None

    def load_categorical_labels(self, csv_path):
        gt_lab_df = pd.read_csv(csv_path)
        gt_lab_df = gt_lab_df.drop(columns=["descriptions"])

        # convert labels into indexes of the categories
        categories = pd.concat([gt_lab_df['primary_label'], gt_lab_df['secondary_label'], gt_lab_df['top_level_label']])
        categories = categories.dropna()
        categories = categories.unique()
        categories = np.concatenate([np.array(["none"]), categories])
        gt_lab_df = gt_lab_df.fillna("none")
        gt_lab_df['motion_id'] = gt_lab_df['motion_file'].apply(lambda x: x.split('_')[0])

        # make a copy and mirror left to right and viceversa
        def mirror_fn(x):
            if 'left' in x.lower():
                return x.replace('left', 'right').replace('Left', 'Right')
            if 'right' in x.lower():
                return x.replace('right', 'left').replace('Right', 'Left')
            if 'counterclockwise' in x.lower():
                return x.replace('counterclockwise', 'clockwise').replace('counterClockwise', 'clockwise').replace('CounterClockwise', 'Clockwise')
            if 'clockwise' in x.lower() and 'counter' not in x.lower():
                return x.replace('clockwise', 'counterclockwise').replace('clockwise', 'counterClockwise').replace('Clockwise', 'CounterClockwise')
            return x

        gt_lab_df_inv = gt_lab_df.copy()
        gt_lab_df_inv['motion_id'] = gt_lab_df_inv['motion_id'].apply(lambda x: 'M'+x)
        gt_lab_df_inv['primary_label'] = gt_lab_df_inv['primary_label'].apply(mirror_fn)
        gt_lab_df_inv['secondary_label'] = gt_lab_df_inv['secondary_label'].apply(mirror_fn)
        gt_lab_df_inv['top_level_label'] = gt_lab_df_inv['top_level_label'].apply(mirror_fn)

        # merge the two
        gt_lab_df = pd.concat([gt_lab_df, gt_lab_df_inv])

        # codes = {i: k for i, k in enumerate(categories)}
        gt_lab_df['primary_label_idx'] = pd.Categorical(gt_lab_df['primary_label'], categories=categories).codes  # gt_lab_df['label1'].map(codes) # 
        gt_lab_df['secondary_label_idx'] = pd.Categorical(gt_lab_df['secondary_label'], categories=categories).codes  # gt_lab_df['label2'].map(codes) 
        gt_lab_df['top_level_label_idx'] = pd.Categorical(gt_lab_df['top_level_label'], categories=categories).codes 
        
        gt_lab_df = gt_lab_df.set_index('motion_id')
        gt_lab_df = gt_lab_df.to_dict(orient='index')

        return gt_lab_df

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][index]

        text = annotation["text"]
        text_x_dict = self.text_to_token_emb(text)
        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
        )
        sent_emb = self.text_to_sent_emb(text)
        if not isinstance(text_x_dict, str):    # disable = True
            text_x_dict.update({"text": text})

        if self.labels is not None:
            labels = self.labels[keyid]
        else:
            labels = {}

        output = {
            "motion_x_dict": motion_x_dict,
            "text_x_dict": text_x_dict,
            "text": text,   # TODO: this is already in text_x_dict, is it necessary here?
            "keyid": keyid,
            "sent_emb": sent_emb,
            "motion_labels": labels
        }
        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
