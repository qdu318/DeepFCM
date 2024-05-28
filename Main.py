import torch
import logging
import json
from Train import Train
from Evalution import Evalution
import os
import numpy as np

from Model.AGRU import DeepPTP
# from Model.base_lines.ARNN_notbaseline import DeepPTP
# from Model.base_lines.ARNN import DeepPTP
# from Model.base_lines.GRU import DeepPTP
# from Model.base_lines.LSTM import DeepPTP
# from Model.base_lines.ALSTM import DeepPTP
# from Model.base_lines.Encoder_Decoder import DeepPTP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(1111)
import warnings
warnings.filterwarnings('ignore')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__ == '__main__':

    epochs = 1000
    batchsize = 16
    in_channels_S = 16
    out_channels_S = 32
    kernel_size_S = 3
    num_inputs_T = 32
    num_channels_T = [64] * 5
    num_outputs_T = 4
    lr = 1e-3
    max_accuracy, max_precise, max_recall, max_f1_score = 0., 0., 0., 0.

    if os.path.exists('./logs/Example.log'):
        os.remove('./logs/Example.log')
    logging.basicConfig(filename='./logs/Example.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

    model = DeepPTP(
        in_channels_S=in_channels_S,
        out_channels_S=out_channels_S,
        kernel_size_S=kernel_size_S,
        num_inputs_T=num_inputs_T,
        num_channels_T=num_channels_T,
        num_outputs_T=num_outputs_T
    )
    print('number of parameters: ' + str(sum(param.numel() for param in model.parameters())))
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(epochs):
        model = Train(model=model, epoch=epoch, batchsize=batchsize, lr=lr)
        max_accuracy, max_precise, max_recall, max_f1_score, FTPRLIST= \
            Evalution(
                model=model,
                batchsize=batchsize,
                max_accuracy=max_accuracy,
                max_precise=max_precise,
                max_recall=max_recall,
                max_f1_score=max_f1_score,
                log=log
            )
        if epoch % 2 == 0:
            with open('./ROC_curve/Example.json', 'a+', encoding='utf-8') as f:
                str_data = json.dumps(FTPRLIST, cls=NpEncoder)
                f.write(str_data + "\n")
            print('---------- OK WITH ROC CURVE PRINTING PARTLY ----------')
        torch.cuda.empty_cache()
    print('End')