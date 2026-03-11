from .rlbench import RLBenchDepth2Cloud


def fetch_depth2cloud(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return RLBenchDepth2Cloud((256, 256))
    if 'rlbench' in dataset_name:
        return RLBenchDepth2Cloud((256, 256))
    return None
