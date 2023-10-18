This repository contains code and models.

#### Data Preprocessing

You can run the [`GPT_generate.py`](https://github.com/shyorangeshy/Code/blob/main/GPT_generate.py) to generate the train set and test set for CKS. The prompt is show in [`Prompt.PDF`](https://github.com/shyorangeshy/Code/tree/master)

#### Models

We release our proposed models, including the CKS and Event Detect system. The CKS is used to select the best commonsense knowledge for each word. The EDBase is utilized to identify the trigger word.


## Training

* run [`scripts/run.sh`](https://github.com/shyorangeshy/Code/blob/main/selector/scripts/run.sh) to train the ***CKS***.

* run [`scripts/run.sh`](https://github.com/shyorangeshy/Code/blob/main/EDBase/scripts/run.sh) to train on the ***Event Detection*** task.





