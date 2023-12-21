# Image-captioing
image captioning for AI class project 
---
인공지능 심화 프로그래밍에서 기말과제로 수행한 프로젝트입니다.

0. 환경 설정 (데이터 셋 설정)
    1. image
        1. Train – http://images.cocodataset.org/zips/train2014.zip 을 “data/coco/imgs/train2014”에 저장한다
        2. Validation – http://images.cocodataset.org/zips/val2014.zip 을 “data/coco/imgs/val2014”에 저장한다.
    2. annotation
        1. https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip 을 다운 및 압축해제 후 “dataset_coco.json”을 “dataset.json”으로 ”data/coco/”에 저장한다.
    3. 데이터 셋 preprocessing
        1. Generate_json_data.py 을 수행한다.

1. train
    1. train.py를 수행한다.
        1. 처음 3번은 learning rate를 5e-5로 설정 
        2. 다음 1번은 learning rate를 1e-5로 설정  

2. evaluation
    1. eval.py를 수행한다.