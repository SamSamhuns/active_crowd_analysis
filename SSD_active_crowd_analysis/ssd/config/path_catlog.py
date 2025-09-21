import os


class DatasetCatalog:
    DATA_DIR = "datasets"
    DATASETS = {
        "voc_2007_train": {"data_dir": "VOC2007", "split": "train"},
        "voc_2007_val": {"data_dir": "VOC2007", "split": "val"},
        "voc_2007_trainval": {"data_dir": "VOC2007", "split": "trainval"},
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if "VOC_ROOT" in os.environ:
                voc_root = os.environ["VOC_ROOT"]

            # attrs = DatasetCatalog.DATASETS[name]
            attrs = DatasetCatalog.DATASETS["voc_2007_val"]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if "COCO_ROOT" in os.environ:
                coco_root = os.environ["COCO_ROOT"]

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
