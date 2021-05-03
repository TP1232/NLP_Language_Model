"""
Created on: 2021-02-06
Author: duytinvo
"""
import torch
from utils.core_nns import RNNModel
from utils.data_utils import Txtfile, Data2tensor
from utils.data_utils import SOS, EOS, UNK
from utils.data_utils import SaveloadHP
import random
import numpy as np

class LMInference:
    def __init__(self, arg_file="./results/lm.args", model_file="./results/lm.m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args, self.model = self.load_model(arg_file, model_file)

    def load_model(self, arg_file="./results/lm.args", model_file="./results/lm.m"):
        """
        Inputs:
            arg_file: the argument file (*.args)
            model_file: the pretrained model file
        Outputs:
            args: argument dict
            model: a pytorch model instance
        """
        args, model = None, None
        #######################
        # YOUR CODE STARTS HERE

# =============================================================================
# 
        args_load = SaveloadHP.load(arg_file)
        # model = torch.load(model_file)
        args = vars(args_load)
        print(args)
        rnn_type = args.get('model')
        ntoken = len(args.get('vocab').w2i)
        ninp = 16
        # ninp = args.get('ninp')
        nhid = args.get('nhid')
        nlayers = args.get('nlayers')
        dropout = args.get('dropout')
        tie_weights = args.get('tied')
        bidirect = args.get('bidirect')
        model = RNNModel(rnn_type=rnn_type, ntoken=ntoken, ninp=ninp, nhid=nhid, nlayers=nlayers,dropout=dropout, tie_weights=tie_weights, bidirect=bidirect)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        # batch_size = args.get('batch_size')

        # model2 = RNNModel(args.get()
        # ntoken = len(vocab.w2i)
        
        # for param in model.parameters():
        #     print(param)

        
# =============================================================================



        # YOUR CODE ENDS HERE
        #######################
        return args, model

    def generate(self, max_len=20):
        """
        Inputs:
            max_len: max length of a generated document
        Outputs:
             the text form of a generated document
        """
        doc = [SOS]
        #######################
        # YOUR CODE STARTS HERE

# =============================================================================
        # print(random_word)

        
	sos_idx = self.args.vocab.w2i[SOS]
        sos_tensor = Data2tensor.idx2tensor([sos_idx], self.device).reshape(1, -1)
        hidden = self.model.init_hidden(sos_tensor.size(0))
        doc = [SOS]
        for i in range(max_len):
            score, hidden = self.model(sos_tensor, hidden)
            pred_prob, pred_idx = self.model.inference(score, k=1)
            pred_wd = self.args.vocab.i2w[pred_idx.item()]
            if pred_wd == EOS:
                break
            doc += [pred_wd]
        # print(" ".join(doc))
# =============================================================================



        # YOUR CODE ENDS HERE
        #######################
        doc += [EOS]
        return " ".join(doc)

    def recommend(self, context="", topk=5):
        """
        Inputs:
            context: the text form of given context
            topk: number of recommended tokens
        Outputs:
            A list form of recommended words and their probabilities
                e,g, [('i', 0.044447630643844604),
                     ('it', 0.027285737916827202),
                     ("don't", 0.026111900806427002),
                     ('will', 0.023868300020694733),
                     ('had', 0.02248169668018818)]
        """
        rec_wds, rec_probs = [], []
        #######################
        # YOUR CODE STARTS HERE
# =============================================================================
# 

        hidden = self.model.init_hidden(1)
        dict_train = self.args.get(('vocab')).w2i
        dict_train_2 = self.args.get(('vocab')).i2w

        # output.to(torch.int64)
        # context = 'i went to school'
        words = context.split(' ')
        
        x = torch.tensor([[dict_train[w] for w in words[0:]]])
        
        # output, hidden = model(x, hidden)
        # print(output, hidden)
        
        output,hidden = self.model.forward(x,hidden)
        label_prob, label_pred = self.model.inference(output,k = topk)
        # print(label_prob, label_pred)
        
        last_pred = np.squeeze(label_pred.detach().numpy())

        word_index = np.random.choice(last_pred.size)
        
        wds,lst = [],[]
        for i in range(0,topk-1):
            wds = [dict_train_2[x] for x in  last_pred[i]]
            lst.append(wds)
        # print(lst)
    
       
        rec_probs_arr = np.squeeze(label_prob.detach().numpy())
        
        """ geting max probability index in suggested words tuple"""
        rec_probs_max = np.sum(rec_probs_arr,axis=1)
        rec_probs_max_id = np.where(rec_probs_max == rec_probs_max.max())[0][0]
        
        rec_wds, rec_probs = lst[rec_probs_max_id],rec_probs_arr[rec_probs_max_id]
# =============================================================================
        # YOUR CODE ENDS HERE
        #######################
        return list(zip(rec_wds, rec_probs))


if __name__ == '__main__':
    arg_file = "./results/lm.args"
    model_file = "./results/lm.m"
    lm_inference = LMInference(arg_file, model_file)

    max_len = 500
    doc = lm_inference.generate(max_len=max_len)
    print("Random doc: {}".format(doc)) 
    context = "i went to school"
    topk = 5
    rec_toks = lm_inference.recommend(context=context, topk=topk)
    print("Recommended words of {} is:".format(context))
    for wd, prob in rec_toks:
        print("\t- {} (p={})".format(wd, prob))
    pass
