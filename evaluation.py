"""Evaluates the model"""

import os
import utils
import argparse
import torch
import yaml
import sacrebleu
from model.data_loader import DoulingoDataset
from model.net import Net
from staple_2020_scorer import score
from tqdm import tqdm
from typing import Dict, List
from itertools import zip_longest, combinations

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir',
                    help="Directory containing the model")
parser.add_argument('--checkpoint',
                    help="Name of checkpoint to load")
parser.add_argument('--results_dir',
                    help="Directory to store the evaluation results")


def evaluate_model(model, val_data, report_dir=None):
    # set model to evaluation mode
    model.eval()
    pred = dict()
    gold = dict()
    report = Report()
    with tqdm(val_data.get_prompt_sent_trg(), total=len(val_data.data_entries)) as t:
        for prompt, sent, trgs_dict in t:
            samples = model.sample(sent)
            trgs = [trg for trg in trgs_dict]
            prob = 1
            sample_dic = {sample.lower(): prob for sample in samples}
            pred[prompt] = sample_dic
            gold[prompt] = trgs_dict


            # Filling report dictionary
            report.add_entry_to_report(prompt, sent, trgs, samples)

    metrics = Metrics(gold, pred).metrics()
    # write_evaluation_report(data_trg, model_output, report_dir)
    if report_dir is not None:
        assert os.path.exists(report_dir)
        report.save_report(report_dir)

    return metrics


class Metrics:
    def __init__(self, gold: Dict[str, Dict[str, float]], pred: Dict[str, Dict[str, float]]):
        assert (gold.keys() == pred.keys())
        self.gold = gold
        self.pred = pred

    def ds_score(self):
        ds_avg = 0
        for prompt in self.pred:
            y = self.pred[prompt].keys()
            ds = 0
            for y1, y2 in combinations(y, 2):
                ds += 1-sacrebleu.sentence_bleu(y1, [y2]).score/100
            ds = ds/(len(y)*(len(y)-1))
            ds_avg += ds
        return ds_avg/len(self.pred)

    def bleu_score(self):
        hypotheses = []
        for prompt in self.pred:
            d = self.pred[prompt]
            pred_lst = sorted(d.items(), key=lambda item: item[1], reverse=True)
            hypotheses.append(pred_lst[0][0])
        references = []
        for prompt in self.gold:
            d = self.gold[prompt]
            ref_lst = sorted(d.items(), key=lambda item: item[1], reverse=True)
            ref_lst = [item[0] for item in ref_lst]
            references.append(ref_lst)
        references_t = [list(ref_t) for ref_t in zip_longest(*references, fillvalue='')]


        bleu = sacrebleu.corpus_bleu(hypotheses, references_t)
        return bleu.score

    def metrics(self):
        base_metrics = score(self.gold, self.pred)
        # bleu = self.bleu_score()
        # base_metrics["BLEU"] = bleu
        return base_metrics


class Report:
    def __init__(self):
        self.prompts: List[str] = []
        self.sentence_dict: Dict[str, str] = dict()
        self.targets_dict: Dict[str, List[str]] = dict()
        self.predictions_dict: Dict[str, List[str]] = dict()

    def add_entry_to_report(self, prompt: str, sent: str,
                            targets: List[str], predictions: List[str]):
        assert prompt not in self.prompts
        self.prompts.append(prompt)
        self.sentence_dict[prompt] = sent
        self.targets_dict[prompt] = targets
        self.predictions_dict[prompt] = predictions

    def get_gold(self):
        gold = dict.fromkeys(self.prompts)
        for prompt in self.prompts:
            gold[prompt] = dict()
            targets = self.targets_dict[prompt]
            for trg_and_score in targets:
                trg, _, scr = trg_and_score.partition('|')
                if scr:
                    gold[prompt][trg] = float(scr)
                else:
                    gold[prompt][trg] = 1
        return gold

    def get_pred(self):
        pred = dict.fromkeys(self.prompts)
        for prompt in self.prompts:
            pred[prompt] = dict()
            targets = self.predictions_dict[prompt]
            for trg_and_score in targets:
                trg, _, scr = trg_and_score.partition('|')
                if scr:
                    pred[prompt][trg] = float(scr)
                else:
                    pred[prompt][trg] = 1
        return pred

    def save_report(self, report_dir: str):
        path = os.path.join(report_dir, 'report.yaml')
        with open(path, 'w', encoding='utf-8') as f:
            yaml_dict = {prompt: {'sentence': self.sentence_dict[prompt],
                                  'targets': self.targets_dict[prompt],
                                  'predictions': self.predictions_dict[prompt]}
                         for prompt in self.prompts}
            yaml.dump(yaml_dict, f, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, report_path: str):
        self = cls()
        with open(report_path) as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
            self.prompts = list(d.keys())
            self.sentence_dict = {prompt: d.get(prompt)['sentence'] for prompt in d}
            self.targets_dict = {prompt: d.get(prompt)['targets'] for prompt in d}
            self.predictions_dict = {prompt: d.get(prompt)['predictions'] for prompt in d}

        return self


if __name__ == '__main__':
    args = parser.parse_args()
    # Loading the evaluation dataset
    print("Loading dataset")
    data_params_json_path = os.path.join(args.data_dir, 'params.json')
    data_params = utils.DataParams.from_json(data_params_json_path)
    val_dataset = DoulingoDataset(data_params, split='val')
    # Loading the model
    print("Loading model...")
    checkpoint = os.path.join(args.model_dir, f"runs/{args.checkpoint}.pth.tar")
    config = utils.Params(cuda=torch.cuda.is_available(), src='en', trg='hu')
    model = Net(config)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    print("Finished Loading")
    # Evaluation ...
    print("Starting Evaluation..")
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    metrics = evaluate_model(model.cuda(), val_dataset, args.results_dir)
    result_json = os.path.join(args.results_dir, 'metrics.json')
    utils.save_dict_to_json(metrics, result_json)
