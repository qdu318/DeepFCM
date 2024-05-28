import torch
import json
from DataLoad import get_loader
import utils
import torch.nn.functional as F

config = json.load(open('./config.json', 'r'))

def Evalution(model, batchsize, max_accuracy, max_precise, max_recall, max_f1_score, log):
    model.eval()
    eval_loss = 0
    accuracy = 0
    confusion_matrix = torch.zeros(4, 4)

    if torch.cuda.is_available():
        confusion_matrix.cuda()
    with torch.no_grad():
        for file in config['eval_set']:
            dataset = get_loader(file, batchsize)
            for idx, parameters in enumerate(dataset):
                parameters = utils.to_var(parameters)
                out = model(parameters)
                loss = F.nll_loss(out, parameters['Class'], size_average=False).item()
                eval_loss += loss
                pred = out.data.max(1, keepdim=True)[1]
                pred = torch.squeeze(pred)
                for i in range(len(pred)):
                    confusion_matrix[parameters['Class'][i]][pred[i]] += 1

                accuracy += pred.eq(parameters['Class'].data.view_as(pred)).cpu().sum()

            precise, recall, f1_score = utils.CalConfusionMatrix(confusion_matrix)

            accuracy_value = accuracy.item() / len(dataset.dataset)
            if accuracy_value > max_accuracy:
                max_accuracy = accuracy_value
                print('---------------------------------------------MAX------------------------------------------------')
            if precise > max_precise:
                max_precise = precise
            if recall > max_recall:
                max_recall = recall
            if f1_score > max_f1_score:
                max_f1_score = f1_score
            eval_loss /= len(dataset.dataset)
            print(' Evalution: Average loss: {:.4f}'.format(eval_loss))
            print(
                'Current Evalution: Accuracy: {}/{} ({:.4f}), Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    accuracy, len(dataset.dataset), accuracy_value, precise, recall, f1_score))
            print(
                'Max Evalution: Accuracy: {:.4f}, Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    max_accuracy, max_precise, max_recall, max_f1_score))

            log.info('Accuracy:{:.4f}, Precise:{:.4f}, Recall:{:.4f}, F1 Score:{:.4f}'.format(accuracy_value, precise, recall, f1_score))
        # confusion_matrix and json for ROC_CURVE
        print(confusion_matrix)

        FTPR_list = []
        '''FPR = utils.FPR(confusion_matrix[1][0],confusion_matrix[1][1]).numpy()
        TPR = utils.TPR(confusion_matrix[0][0],confusion_matrix[0][1]).numpy()'''
        ''' 3 classes :
        FPR = utils.FPR(confusion_matrix[1][0]+confusion_matrix[2][0],confusion_matrix[1][1]+confusion_matrix[2][2]+confusion_matrix[1][2]+confusion_matrix[2][1]).numpy()
        TPR = utils.TPR(confusion_matrix[0][0],confusion_matrix[0][1]+confusion_matrix[0][2]).numpy()
        '''
        # FPR = utils.FPR(confusion_matrix[1][0] + confusion_matrix[2][0],confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[1][2] + confusion_matrix[2][1]).numpy()
        # TPR = utils.TPR(confusion_matrix[0][0], confusion_matrix[0][1] + confusion_matrix[0][2]).numpy()
        ''' 4 classes :
        FPR = utils.FPR(confusion_matrix[1][0]+confusion_matrix[2][0]+confusion_matrix[3][0],confusion_matrix[1][1]+confusion_matrix[2][2]+confusion_matrix[3][3]
                        +confusion_matrix[1][2]+confusion_matrix[1][3]+confusion_matrix[2][1]+confusion_matrix[2][3]+confusion_matrix[3][1]+confusion_matrix[3][2]).numpy()
        TPR = utils.TPR(confusion_matrix[0][0],confusion_matrix[0][1]+confusion_matrix[0][2]+confusion_matrix[0][3]).numpy()
        '''
        FPR = utils.FPR(confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[3][0],
                        confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3]
                        + confusion_matrix[1][2] + confusion_matrix[1][3] + confusion_matrix[2][1] +
                        confusion_matrix[2][3] + confusion_matrix[3][1] + confusion_matrix[3][2]).numpy()
        TPR = utils.TPR(confusion_matrix[0][0],
                        confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]).numpy()
        FTPR_list.append(round(FPR.item(),2))
        FTPR_list.append(round(TPR.item(),2))

    return max_accuracy, max_precise, max_recall, max_f1_score,FTPR_list