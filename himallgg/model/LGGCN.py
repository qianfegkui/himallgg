import torch
import torch.nn as nn

from .ContextualEncoder import ContextualEncoder
from .EdgeAtt import EdgeAtt
from .GCN import GCN, SGCN
from .Classifier import Classifier
from .functions import batch_graphify,batch_graphify1
import himallgg
import torch
from torch.nn import TransformerEncoderLayer
import torch.nn as nn
import torch.nn.functional as F
from .Fusion import *
from transformers import BertTokenizerFast, BertModel

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class CrossModalAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossModalAttention, self).__init__()
        self.d_model = d_model
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y, mask=None):
        '''
        x: [batch_size, seq_len_x, d_model]
        y: [batch_size, seq_len_y, d_model]
        mask: [batch_size, seq_len_x, seq_len_y]
        '''
        q = self.q_layer(x)  # [batch_size, seq_len_x, d_model]
        k = self.k_layer(y)  # [batch_size, seq_len_y, d_model]
        v = self.v_layer(y)  # [batch_size, seq_len_y, d_model]
        
        # calculate attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # [batch_size, seq_len_x, seq_len_y]
        
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len_x, seq_len_y]
        
        # apply attention weights to values
        attn_output = torch.bmm(attn_weights, v)  # [batch_size, seq_len_x, d_model]
        
        # apply dropout and residual connection
        attn_output = self.dropout(attn_output)
        output = attn_output + x  # [batch_size, seq_len_x, d_model]
        
        return output


log = himallgg.utils.get_logger()


class LGGCN(nn.Module):

    def __init__(self, args):
        super(LGGCN, self).__init__()
        u_dim = 1024
        uA_dim = 1582
        uV_dim = 342
        g_dim = 160
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 6

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn = ContextualEncoder(u_dim, g_dim, args)
        self.rnn_A = ContextualEncoder(uA_dim, g_dim, args)
        self.rnn_V = ContextualEncoder(uV_dim, g_dim, args)

        self.cross_attention = CrossModalAttention(g_dim)
        self.edge_att = EdgeAtt(g_dim, args)
       
        self.gcn = GCN(g_dim*3, h1_dim, h2_dim, args)
        self.edge_att_all = EdgeAtt(g_dim*3, args)
        self.clf = Classifier(h2_dim+g_dim*3, hc_dim, tag_size, args)
        self.clf_T = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_A = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_V = Classifier(g_dim, hc_dim, tag_size, args)

        self.gcn1 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn2 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn3 = SGCN(g_dim, h1_dim, g_dim, args)
        self.att = MultiHeadedAttention(10,g_dim)
        self.args = args
        

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        self.edge_type_to_idx1 = {'00': 0,  '01': 1, '10': 2,  '11': 3}

        log.debug(self.edge_type_to_idx)

    def tokenize(self, data):
        TEXT_length = max(data['text_len_tensor'].tolist())
        ad = []
        sd = []
        for paragraph in data['sentence']:
            for text in paragraph:
                tokened = self.tokenizer(text)
                input_ids = tokened['input_ids']
                mask = tokened['attention_mask']
                if len(input_ids) < TEXT_length:
                    pad_len = (TEXT_length - len(input_ids))
                    input_ids += [0] * pad_len
                    mask += [0] * pad_len
            ad.append(input_ids[:TEXT_length])
            sd.append(mask[:TEXT_length])
        return torch.tensor(ad),torch.tensor(sd)

    def get_rep(self, data):
 
        node_features_T = self.rnn(data["text_len_tensor"], data["text_tensor"]) # [batch_size, mx_len, D_g]
        node_features_A = self.rnn_A(data["text_len_tensor"], data["audio_tensor"]) # [batch_size, mx_len, D_g]
        node_features_V = self.rnn_V(data["text_len_tensor"], data["visual_tensor"]) # [batch_size, mx_len, D_g]

        node_features = torch.cat((node_features_T, node_features_A, node_features_V), 2)


        features_T, edge_index_T, edge_norm_T, edge_type_T, edge_index_lengths_T = batch_graphify1(
            node_features_T, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx1, self.edge_att, self.device)
        
        features_A, edge_index_A, edge_norm_A, edge_type_A, edge_index_lengths_A = batch_graphify1(
            node_features_A, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx1, self.edge_att, self.device)
        
        features_V, edge_index_V, edge_norm_V, edge_type_V, edge_index_lengths = batch_graphify1(
            node_features_V, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx1, self.edge_att, self.device)  

        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att_all, self.device)
        

        graph_out_T = self.gcn1(features_T, edge_index_T, edge_norm_T, edge_type_T)
        graph_out_A = self.gcn2(features_A, edge_index_A, edge_norm_A, edge_type_A)
        graph_out_V = self.gcn3(features_V, edge_index_V, edge_norm_V, edge_type_V)


        fea1 = self.att(graph_out_A,graph_out_T, graph_out_A)
        fea2 = self.att(graph_out_V,graph_out_T, graph_out_V)
     

        features_graph = torch.cat([graph_out_T,fea1.squeeze(1),fea2.squeeze(1)],dim=-1)
    
        graph_out = self.gcn(features_graph, edge_index, edge_norm, edge_type)
        

        return graph_out, features, graph_out_T, graph_out_A, graph_out_V
    

    def forward(self, data):
        graph_out, features, graph_out_T, graph_out_A, graph_out_V = self.get_rep(data)

        out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        score = self.clf.get_prob1(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        score_T = self.clf_T.get_prob1(graph_out_T, data["text_len_tensor"])
        score_A = self.clf_A.get_prob1(graph_out_A, data["text_len_tensor"])
        score_V = self.clf_V.get_prob1(graph_out_V, data["text_len_tensor"])
        scores = score+0.7*score_T+0.2*score_V+0.1*score_A
        log_prob = F.log_softmax(scores, dim=-1)
        y_hat = torch.argmax(log_prob, dim=-1)


        return y_hat 

    def get_loss(self, data):
        graph_out, features, graph_out_T, graph_out_A, graph_out_V = self.get_rep(data)
        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])
        
        loss_T = self.clf_T.get_loss(graph_out_T,data["label_tensor"], data["text_len_tensor"])
        loss_A = self.clf_A.get_loss(graph_out_A,data["label_tensor"], data["text_len_tensor"])
        loss_V = self.clf_V.get_loss(graph_out_V,data["label_tensor"], data["text_len_tensor"])


        return loss + 0.7*loss_T + 0.2*loss_V + 0.1*loss_A
