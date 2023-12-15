**PVDN Urban: Provident Vehicle Detection at Night in Urban Scenarios** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is used in the automotive industry. 

The dataset consists of 14688 images with 33545 labeled objects belonging to 1 single class (*reflection*).

Images in the PVDN Urban dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 7059 (48% of the total) unlabeled images (i.e. without annotations). There are 3 splits in the dataset: *train* (11054 images), *val* (2148 images), and *test* (1486 images). Alternatively, the dataset could be split into 2 tags: ***contains annotations*** (7756 images) and ***oncoming vehicle visible*** (4024 images). The dataset was released in 2023.

<img src="https://github.com/dataset-ninja/pvdn-urban/raw/main/visualizations/poster.png">
