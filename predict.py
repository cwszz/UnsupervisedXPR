import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from sklearn.model_selection import *
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from collections import OrderedDict
from DictMatching.Loss import MoCoLoss
from DictMatching.unsupMoco import MoCo
from utilsWord.test_args import getArgs
from utilsWord.tools import seed_everything
from utilsWord.sentence_process import load_words_mapping,WordWithContextDatasetWW,load_word2context_from_tsv

args = getArgs()
device_id = 0
seed_everything(args.seed)  # 固定随机种子
# tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
if args.distributed:
    device = torch.device('cuda', args.local_rank)
else:
    device = torch.device('cuda:{}'.format(str(device_id)))
    torch.cuda.set_device(device_id)
# device = torch.device('cpu')
lossFunc = MoCoLoss().to(device)


def test_model(model, val_loader, adv_loader=None):  # 验证
    model.eval()
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        first_src_examples,first_trg_examples = None,None
        second_src_examples,second_trg_examples = None,None
        adv_src_examples, adv_trg_examples = None, None
        for step, batch in enumerate(tk):
            batch_src = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 0]
            batch_trg = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 1]
            if args.distributed:
                first_src = model.module.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.module.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            else:
                first_src = model.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.encoder_q(*batch_trg,sample_num=args.dev_sample_num)      
            first_src_examples = first_src if first_src_examples is None else torch.cat([first_src_examples,first_src],dim=0)
            first_trg_examples = first_trg if first_trg_examples is None else torch.cat([first_trg_examples,first_trg],dim=0)
           
        first_src_examples = torch.nn.functional.normalize(first_src_examples,dim=1)
        first_trg_examples = torch.nn.functional.normalize(first_trg_examples,dim=1)
        first_st_sim_matrix = torch.mm(first_src_examples,first_trg_examples.T)/math.sqrt(first_src_examples.size(-1))/0.1
        second_st_sim_matrix = torch.mm(first_trg_examples,first_src_examples.T)/math.sqrt(first_trg_examples.size(-1))/0.1
        
        if adv_loader:
            adv_tk = tqdm(adv_loader, total=len(val_loader), position=0, leave=True)
            for step, batch in enumerate(adv_tk):
                adv_batch_src = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 0]
                adv_batch_trg = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 1]
                if args.distributed:
                    adv_src = model.module.encoder_q(*adv_batch_src,sample_num=args.dev_sample_num)
                    adv_trg = model.module.encoder_q(*adv_batch_trg,sample_num=args.dev_sample_num)
                else:
                    adv_src = model.encoder_q(*adv_batch_src,sample_num=args.dev_sample_num)
                    adv_trg = model.encoder_q(*adv_batch_trg,sample_num=args.dev_sample_num)
                adv_src_examples = adv_src if adv_src_examples is None else torch.cat([adv_src_examples,adv_src],dim=0)
                adv_trg_examples = adv_trg if adv_trg_examples is None else torch.cat([adv_trg_examples,adv_trg],dim=0)

            adv_src_examples = torch.nn.functional.normalize(adv_src_examples,dim=1)
            adv_trg_examples = torch.nn.functional.normalize(adv_trg_examples,dim=1)
            matrix = torch.mul(torch.mm(first_src_examples,adv_trg_examples.T), (1 - torch.eye(first_src_examples.size()[0]).to(device)))
            first_st_sim_matrix = torch.cat([first_st_sim_matrix,matrix],dim=1)
            matrix = torch.mul(torch.mm(first_trg_examples,adv_src_examples.T),(1 - torch.eye(first_trg_examples.size()[0]).to(device))) 
            second_st_sim_matrix = torch.cat([second_st_sim_matrix,matrix],dim=1)
        
        first_st_sim_matrix = F.softmax(first_st_sim_matrix,dim=1)
        second_st_sim_matrix = F.softmax(second_st_sim_matrix,dim=1)
        label = torch.LongTensor(list(range(first_st_sim_matrix.size(0)))).to(first_src_examples.device)
        st_acc = torch.argmax(first_st_sim_matrix, dim=1)    # [B]
        ts_acc = torch.argmax(second_st_sim_matrix, dim=1)
        acc = (st_acc == label).long().sum().item() / st_acc.size(0)
        acc += (ts_acc == label).long().sum().item() / ts_acc.size(0)
    return acc / 2


def test_model_single_encoder(model, val_loader):
    model.eval()

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        first_src_examples,first_trg_examples = None,None
        second_src_examples,second_trg_examples = None,None
        for step, batch in enumerate(tk):
            batch_src = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 0]
            batch_trg = [tensors.to(device) for i,tensors in enumerate(batch) if i % 2 == 1]
            if args.distributed:
                first_src = model.module.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.module.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.module.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                second_trg = model.module.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            else:
                first_src = model.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                first_trg = model.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
                second_src = model.encoder_q(*batch_src,sample_num=args.dev_sample_num)
                second_trg = model.encoder_q(*batch_trg,sample_num=args.dev_sample_num)
            first_src_examples = first_src if first_src_examples is None else torch.cat([first_src_examples,first_src],dim=0)
            first_trg_examples = first_trg if first_trg_examples is None else torch.cat([first_trg_examples,first_trg],dim=0)
            second_src_examples = second_src if second_src_examples is None else torch.cat([second_src_examples,second_src],dim=0)
            second_trg_examples = second_trg if second_trg_examples is None else torch.cat([second_trg_examples,second_trg],dim=0)
        
        first_src_examples = torch.nn.functional.normalize(first_src_examples,dim=1)
        first_trg_examples = torch.nn.functional.normalize(first_trg_examples,dim=1)
        second_src_examples = torch.nn.functional.normalize(second_src_examples,dim=1)
        second_trg_examples = torch.nn.functional.normalize(second_trg_examples,dim=1)
        first_st_sim_matrix = F.softmax(torch.mm(first_src_examples,first_trg_examples.T)/math.sqrt(first_src_examples.size(-1))/0.1,dim=1)
        second_st_sim_matrix = F.softmax(torch.mm(second_trg_examples,second_src_examples.T)/math.sqrt(second_trg_examples.size(-1))/0.1,dim=1)
        label = torch.LongTensor(list(range(first_st_sim_matrix.size(0)))).to(first_src_examples.device)
        st_acc = torch.argmax(first_st_sim_matrix, dim=1)    # [B]
        ts_acc = torch.argmax(second_st_sim_matrix, dim=1)
        acc = (st_acc == label).long().sum().item() / st_acc.size(0)
        acc += (ts_acc == label).long().sum().item() / ts_acc.size(0)
    return acc / 2
"""
模型训练数据准备
"""
if args.distributed:
    dist.init_process_group(backend='nccl')
"""
模型训练预测
"""
if __name__ == '__main__':
    test_folder = 'dev' if args.test_dev else 'test'
    dataset_path = args.dataset_path
    args.test_phrase_path = dataset_path + "{}/{}-en-{}-32-phrase.txt".format(test_folder,test_folder,args.test_lg)
    args.src_context_path = dataset_path + "sentences/en-{}-phrase-sentences.32.tsv".format(args.test_lg)
    args.trg_context_path = dataset_path + "sentences/{}-phrase-sentences.32.tsv".format(args.test_lg)
    # adv_src_context_path = "./adv_data/adv_sentences/en-{}-phrase-sentences.32.tsv".format(args.test_lg)
    # adv_trg_context_path = "./adv_data/adv_sentences/{}-phrase-sentences.32.tsv".format(args.test_lg)
    queue_length = int(args.queue_length)
    para_T = args.T_para 
    test_phrase_pairs = load_words_mapping(args.test_phrase_path)
    en_word2context = load_word2context_from_tsv(args.src_context_path,args.dev_all_sentence_num)
    lg_word2context = load_word2context_from_tsv(args.trg_context_path,args.dev_all_sentence_num)
    # adv_en_word2context = load_word2context_from_tsv(adv_src_context_path, args.dev_all_sentence_num)
    # adv_lg_word2context = load_word2context_from_tsv(adv_trg_context_path, args.dev_all_sentence_num)
    unsup = args.unsupervised
    best_acc = 0
    
    test_dataset = WordWithContextDatasetWW(test_phrase_pairs, en_word2context, lg_word2context, sampleNum=args.dev_sample_num,
        max_len=args.sentence_max_len)
    # adv_dataset = WordWithContextDatasetWW(test_phrase_pairs,adv_en_word2context, adv_lg_word2context, sampleNum=args.dev_sample_num,
        # max_len=args.sentence_max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate, shuffle=False,num_workers=16)
    # adv_loader = DataLoader(adv_dataset,  batch_size=args.eval_batch_size, collate_fn=adv_dataset.collate, shuffle=False,num_workers=16)
    config = AutoConfig.from_pretrained('xlm-roberta-base')
    # c = torch.load('./model/pytorch_model.bin')
    model = MoCo(config=config,args=args,K=queue_length,T=para_T,m=args.momentum).to(device)
    if not unsup:
        # t = torch.load('./adv_result/4-fr-32-true-0-0.06-1-100-0.999-0-dev_qq-layer_12/best.pt')
        t = torch.load(args.load_model_path, map_location={'cuda:1':'cuda:0'}) #
        k = OrderedDict()
        for each_k in t.keys():
            k[each_k.replace('module.','')] = t[each_k]
        model.load_state_dict(k)
        # model = torch.load(args.load_model_path)
        model.to(device)
            
        val_acc = test_model(model, test_loader, None)
        # val_acc = test_model_single_encoder(model, test_loader)
        print("src-lg: " + args.lg  +" trg-lg: " + args.test_lg + " acc:", val_acc)
