from transformers import AdamW

optim_dict = {
    "AdamW": AdamW
}

esci2label = {
    'exact': 0,
    'substitute': 1,
    'irrelevant': 2,
    'complement': 3
}

label2esci = {
    0: 'exact',
    1: 'substitute',
    2: 'irrelevant',
    3: 'complement'
}

task2datapath = {
    'task1': "/ssd/kddcup2022/data/task_1_query-product_ranking/processed/public",
    # 'task2': "/home/ecs-user/kddcup2022/data/task2/raw/public",
    # 'task2': "/ssd/kddcup2022/data/data_v3.1",
    'task2': "/home/ecs-user/data/task2.1",
    'task3': "/ssd/kddcup2022/data/task_3_product_substitute_identification/processed/public"
}

savepath = "./saved"
confpath = "./config"
cachepath = "./cache"
