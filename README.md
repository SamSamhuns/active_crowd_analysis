# Active Crowd Analysis: Pandemic Risk Mitigation for the Blind or Visually Impaired

## Setup

```shell script
$ conda env create -n active_crowd -f=requirements/mobilnet_ssd.yml
$ conda activate active_crowd
```

## Object Detection and Localization

### Dataset Download

### Generating Metrics

## Distance Regression

### Generating Metrics

### Dataset Download

## Multiple Object Tracking Evaluation

### Dataset Download

### Generating Metrics

Image source path `datasets/MOT16/train/MOT16-02/img1` must be available. 
The predicted `MOT16-02.txt` is saved in `py-motmetrics/motmetrics/data/MOT16/predicted/` by default.

```shell script
$ cd py-motmetrics
$ python -m motmetrics.apps.eval_motchallenge motmetrics/data/MOT16/gt motmetrics/data/MOT16/predicted/
```

## Acknowledgements

-   py-motmetrics
-   circlenet fred
-   SSD
-   pytorch SSD



