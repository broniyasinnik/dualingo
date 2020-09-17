import os
import json
import random
import utils
from collections import Counter
# from polyglot.text import Text
from torch.utils.data import Dataset


class Target:
    def __init__(self, target: str, prob: float):
        self.target = target
        self.prob = prob

    # @property
    # def pos_tags(self):
    #     text = Text(self.target, hint_language_code=self.lang)
    #     pos_tags = " ".join([pos for word, pos in text.pos_tags])
    #     return pos_tags

    def pos_tags(self, target_pos_dict):
        return self.target_pos_dict[self.target]

class Clusters:
    def __init__(self, clusters_json, top_k=200):
        clusters = self._read_clusters_json(clusters_json)
        self.clusters = [key for key, count in Counter(clusters).most_common(top_k)]

    def _read_clusters_json(self, clusters_json):
        with open(clusters_json) as f:
            clusters = json.load(f)
        return clusters

    def get_target_cluster(self, target: Target):
        if target.pos_tags in self.clusters:
            index = self.clusters.index(target.pos_tags)
            return f'<cls{index}>'
        else:
            return f'<cls0>'


class DoulingoDataEntry:
    def __init__(self, prompt, sent, targets):
        self.prompt = prompt
        self.sent = sent
        self.targets = targets

    def targets_dict(self):
        d = {trg.target: trg.prob for trg in self.targets}
        return d

class DoulingoDataset(Dataset):
    def __init__(self, params, split='train'):
        super().__init__()
        self.path = os.path.join(params.data_dir, params.dict[f'{split}_data'])
        self.data_entries = self._read_data_from_file()
        self.items = [(entry.sent, trg.target) for entry in self.data_entries for trg in entry.targets]
        # self.pos_json_path = os.path.join(params.data_dir, 'train.pos_tags.json')
        # with open(self.pos_json_path, encoding='utf-8') as js:
        #     self.pos_json = json.load(js)
        # self.clusters = Clusters(os.path.join(params.model_dir, 'clusters.json'),
        #                          top_k=params.num_clusters)

    def get_prompt_sent_trg(self):
        data = [(entry.prompt, entry.sent, entry.targets_dict()) for entry in self.data_entries]
        return data

    # def _find_params_in_dir(self):
    #     json_files = list(pathlib.Path(self.data_dir).glob('*.json'))
    #     params = Params(json_files)
    #     return params

    def _read_data_from_file(self):
        data = []
        with open(self.path, encoding='utf-8') as f:
            targets = []
            for line in f:
                new_line = line.strip()
                if new_line.startswith('prompt'):
                    prompt_hash, sentence = new_line.split('|')
                elif new_line:
                    trg_prob_split = new_line.split('|')
                    # the target format is `trg | prob`
                    if len(trg_prob_split) == 2:
                        trg, prob = trg_prob_split
                        target = Target(trg, float(prob))
                        # target.cluster = self.clusters.get_target_cluster(target)
                        targets.append(target)
                    # the target format is just `trg`
                    else:
                        target = Target(trg_prob_split[0], prob=1)
                        # target.cluster = self.clusters.get_target_cluster(target)
                        targets.append(target)
                else:
                    entry = DoulingoDataEntry(prompt_hash, sentence, targets)
                    data.append(entry)
                    targets = []
        return data

    def __len__(self):
        total_len = sum([len(entry.targets) for entry in self.data_entries])
        return total_len

    def samples_weights(self, scale_factor: int = 1):
        weights = [scale_factor * target.prob for entry in self.data_entries for target in entry.targets]
        return weights

    def __getitem__(self, idx):
        return self.items[idx]

class SentenceCodesDataset(DoulingoDataset):
    def __init__(self, params: utils.DataParams, num_codes: int,  split: str):
        super(SentenceCodesDataset, self).__init__(params, split)
        self.num_codes = num_codes

    def generate_random_codes(self):
        random_codes = [f'<cls_{random.randint(0, self.num_codes)}>' for i in range(len(self))]
        self.items = [(sent, random_codes[i]+trg) for i, (sent, trg) in enumerate(self.items)]

    def generate_codes_from_file(self, path: str):
        pass
