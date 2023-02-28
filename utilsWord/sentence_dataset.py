'''word with context dataset of whole words'''
import random
import string
import torch
import numpy
import csv
from collections import defaultdict
from transformers import XLMRobertaTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
logging.basicConfig(level=logging.ERROR)

def my_pad_sequence(sequences, batch_first=False, padding_value=0.0,max_len=512):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    ori_max_len = max_len
    max_len = max(max([s.size(0) for s in sequences]),max_len)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor[:,:ori_max_len, ...]

def load_word2context_from_tsv(tsv_path, sentence_num):
    print("Loading tsv from %s ..." % tsv_path)
    word2context = defaultdict(list)
    with open(tsv_path) as fp:
        rows = [l.strip() for l in fp]
    for row in rows:
        row = row.split("\t")
        word = row[0]
        sentences = row[2:2+sentence_num]
        tokenized_sentence = sentences
        word2context[word.lower()] = tokenized_sentence
    return word2context

def load_word2context_from_csv(csv_path):
    print("Loading csv from %s ..." % csv_path)
    word2context = {}
    with open(csv_path) as fp:
        csv_reader = csv.reader(fp)
        rows = list(csv_reader)
        for row in rows[1:]:
            word = row[0].lower()
            sentences = row[2:]
            tokenized_sentences = [s.strip().split() for s in sentences]
            word2context[word] = tokenized_sentences
    return word2context


def load_words_mapping(path):
    '''from en to lg'''
    print(f'Loading words mapping from {path}')
    f = open(path)
    src_words, trg_words = [], []
    for eachline in f.readlines():
        word1 = eachline.split('\t')[0].strip()
        word2 = eachline.split('\t')[1].strip()
        w1, w2 = word1.lower(), word2.lower()
        src_words.append(w1)
        trg_words.append(w2)

    return src_words, trg_words


class ExampleDataset(Dataset):
    
    def __init__(self, words, example_sentence, max_len=256, model_name='xlm-roberta-base', sampleNum=8):
        self.vocab = XLMRobertaTokenizer.from_pretrained(model_name)
        self.vocab.deprecation_warnings['sequence-length-is-longer-than-the-specified-maximum'] = True
        self.max_len = max_len
        self.mask_id = self.vocab.mask_token_id
        self.sampleNum = sampleNum  
        self.dataset = []
        for w1 in words:
            if w1 in example_sentence:
                sentence_bag,notfound = self.convert(example_sentence[w1], w1)
                self.dataset.append(w1, sentence_bag)
        print(f'[!] collect {len(self.dataset)} samples')
        print("没找到共" +str(notfound))
        
    def convert(self, bag, word):
        rest = []
        cnt = 0
        for sentence in bag:
            # lower case
            lower_sent = sentence.lower().replace('\u3000','')
            if lower_sent.find(word) == -1:
                for i,eachchar in enumerate(lower_sent):
                    if(eachchar in string.punctuation):
                        lower_sent = lower_sent.replace(eachchar,' ')
                for i,eachchar in enumerate(word):
                    if(eachchar in string.punctuation):
                        word = word.replace(eachchar,' ')
                if(lower_sent.find(word) == -1):
                    cnt += 1
                    continue
            idx = lower_sent.find(word)
            before_word = sentence[:idx]
            after_word = sentence[idx+len(word):]
            rest.append((before_word, word, after_word))
        return rest, cnt
    
    def str2indices(self, s):
        return self.vocab.encode(s,add_special_tokens=False,)
    
    def _length_cut(self, before, word, after):
        '''return tensor and the index of the word in it, and the length of the word'''
        max_len = self.max_len - 4 # CLS ... EOS word EOS ... EOS
        length_before, length_word, length_after = len(before), len(word), len(after)
        while length_before + length_word + length_after > max_len:
            if length_before > length_after:
                before = before[1:]
            else:
                after = after[:-1]
            length_before, length_after = len(before), len(after)

        assert len(before) + len(word) + len(after) <= max_len
        if self.prepend_bos:
            word = [self.vocab.eos_token_id] + word
        if self.append_eos:
            word = word + [self.vocab.eos_token_id]
        indices = [self.vocab.bos_token_id] + before + word + after + [self.vocab.eos_token_id]
        assert len(indices) <= max_len + 4
        index = len(before) + 1
        return torch.LongTensor(indices), index, len(word)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        word, sentence_bag = self.dataset[index]
        if self.sampleNum == 0:
            word_id = [self.vocab.bos_token_id] + [self.mask_id] * len(self.str2indices(word)) + [self.vocab.eos_token_id]
            return torch.LongTensor(word_id)
        # random choice 8 sentece from the bag
        sentences = random.sample(sentence_bag,self.sampleNum)
        sentence_subset_index_w1 = []
        subset_index_w1,subset_leng_w1 = [],[]
        word = sentences[0][1]
        
        for each_sentence in sentences:
            s_before_w1, s_word_w1, s_after_w1 = self.str2indices(each_sentence), [self.mask_id] * len(self.str2indices(word)), self.str2indices(each_sentence[2])
            indices_w1, w1_index, w1_length = self._length_cut(s_before_w1, s_word_w1, s_after_w1)
            sentence_subset_index_w1.append(indices_w1)
            subset_index_w1.append(w1_index)
            subset_leng_w1.append(w1_length)
       
        sentences_indexs1 = torch.tensor(subset_index_w1)
        sentences_length1 = torch.tensor(subset_leng_w1)
        return (sentence_subset_index_w1,sentences_indexs1,sentences_length1, word)

    def generate_mask(self, ids):
        attn_mask_index = (ids != self.vocab.pad_token_id).long().nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def generate_type(self,ids,idx,leng):
        type_ids = torch.zeros_like(ids)
        new_idx = torch.cat(idx,dim=0).view(-1)
        new_leng = torch.cat(leng,dim=0).view(-1)
        for i,(each_idx,each_len) in enumerate(zip(new_idx,new_leng)):
            type_ids[i][each_idx.item():each_idx.item()+each_len.item()] = 1
        return type_ids
        

    def collate(self, sample_tuples):
        if self.sampleNum == 0:
            w1_words = [i[0] for i in sample_tuples]
            w1_words = pad_sequence(w1_words, batch_first=True, padding_value=self.vocab.pad_token_id)
            w1_mask = self.generate_mask(w1_words)
            return w1_words, w1_mask

        indices_w1 = []
        for i in sample_tuples:
            indices_w1 += i[0][0]
        w1_index = [i[0][1] for i in sample_tuples]
        w1_length = [i[0][2] for i in sample_tuples]
        w1_words = [i[0][3] for i in sample_tuples]
        
        # indices_w1 B * [ExampleNum, L]
        # [B, S]
        indices_w1 = my_pad_sequence(indices_w1, batch_first=True, padding_value=self.vocab.pad_token_id,max_len=self.max_len)
        w1_mask = self.generate_mask(indices_w1)
        w1_type = self.generate_type(indices_w1,w1_index,w1_length)
        
        # [B*sampleNum]
        w1_index = torch.cat(w1_index,dim=0)

        return  indices_w1, w1_mask, w1_index, w1_length, w1_type
        

    



        
