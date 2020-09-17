import pytest
from evaluation import Report, Metrics


@pytest.fixture
def report():
    prompt1 = 'prompt_bde43302e530441c0afdea532cbbd419'
    sent1 = 'i think about her night'
    targets1 = ['éjjel nappal rá gondolok.|0.8',
                'rá gondolok éjjel nappal.|0.25',
                'rá gondolok éjjel és nappal.|0.05']
    predictions1 = ['bla bla bla|1', 'all i want|1']

    prompt2 = 'prompt_1b578a1e8fca427f4912b95acc612a1d'
    sent2 = 'he cooks the food.'
    target2 = ['megfőzi az ételt.|0.9',
               'főzi az ételt.|0.09',
               'ő főzi az ételt.|0.01']

    prediction2 = ['foze ala ala|1',
                   'dracula adfa|1']

    my_report = Report()
    my_report.add_entry_to_report(prompt1, sent1, targets1, predictions1)
    my_report.add_entry_to_report(prompt2, sent2, target2, prediction2)
    return my_report

@pytest.fixture
def report1():
    report_path = './Reports/report.yaml'
    report = Report.from_yaml(report_path)
    return report


def test_evaluate_model(report1):
    metrics = Metrics(report1.get_gold(), report1.get_pred())
    print(metrics.bleu_score())
    print(metrics.ds_score())
    # return metrics.bleu_score()



def test_save_report(report):
    report.save_report('./Reports')


def test_from_yaml():
    report = Report.from_yaml('./Reports/report.yaml')
