# PerAct has a slightly different API because we load pcd directly, not depth
from .peract import PeractTrainTester
from .rlbench import RLBenchTrainTester


def fetch_train_tester(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return RLBenchTrainTester
    if 'peract' in dataset_name:
        return PeractTrainTester
    if 'rlbench' in dataset_name:
        return RLBenchTrainTester
    return None
