import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
import torch_sparse
from copy import deepcopy
import numpy as np

# Class GCN
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.modules.loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class AdaGCL(BaseModel):
    def __init__(self, data_handler):
        super(AdaGCL, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.cl_weight = configs['model']['cl_weight']
        self.ib_weight = configs['model']['ib_weight']
        self.temperature = configs['model']['temperature']
        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.is_training = True
        self.final_embeds = None

    def set_denoiseNet(self, denoiseNet):
        self.denoiseNet = denoiseNet

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def _propagate(self, adj, embeds, flag=True):
        if flag:
            return t.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)

    def forward(self, adj):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]

    def forward_(self):
        iniEmbeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embedsLst = [iniEmbeds]
        count = 0
        for i in range(self.layer_num):
            with t.no_grad():
                adj = self.denoiseNet.denoise_generate(x=embedsLst[-1], layer=count)
            embeds = self._propagate(adj, embedsLst[-1])
            embedsLst.append(embeds)
            count += 1
        mainEmbeds = sum(embedsLst)
        return mainEmbeds

    def loss_graphcl(self, x1, x2, users, items):
        T = self.temperature
        user_embeddings1, item_embeddings1 = t.split(x1, [self.user_num, self.item_num], dim=0)
        user_embeddings2, item_embeddings2 = t.split(x2, [self.user_num, self.item_num], dim=0)
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)
        all_embs1 = t.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = t.cat([user_embs2, item_embs2], dim=0)
        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)
        sim_matrix = t.einsum('ik,jk->ij', all_embs1, all_embs2) / t.einsum('i,j->ij', all_embs1_abs, all_embs2_abs)
        sim_matrix = t.exp(sim_matrix / T)
        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - t.log(loss)
        return loss

    def cal_loss_cl(self, batch_data, generated_adj):
        self.is_training = True
        ancs, poss, negs = batch_data
        out1_u, out1_i = self.forward(generated_adj)
        out1 = t.concat([out1_u, out1_i])
        out2 = self.forward_()
        loss = self.loss_graphcl(out1, out2, ancs, poss).mean() * self.cl_weight
        losses = {'cl_loss': loss}
        return loss, losses, out1, out2

    def cal_loss_ib(self, batch_data, generated_adj, out1_old, out2_old):
        self.is_training = True
        ancs, poss, negs = batch_data
        out1_u, out1_i = self.forward(generated_adj)
        out1 = t.concat([out1_u, out1_i])
        out2 = self.forward_()
        loss_ib = self.loss_graphcl(out1, out1_old.detach(), ancs, poss) + self.loss_graphcl(out2, out2_old.detach(), ancs, poss)
        loss = loss_ib.mean() * self.ib_weight
        losses = {'ib_loss': loss}
        return loss, losses

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj)
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.stack(
            [torch.mm(inp, self.weight) for inp in torch.unbind(input, dim=0)],
            dim=0)
        output = torch.stack(
            [torch.spmm(adj, sup) for sup in torch.unbind(support, dim=0)],
            dim=0)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class GraphDecoder(nn.Module):
    def __init__(self, zdim, dropout, gdc='ip'):
        super(GraphDecoder, self).__init__()
        self.dropout = dropout
        self.gdc = gdc
        self.zdim = zdim
        self.rk_lgt = Parameter(torch.FloatTensor(torch.Size([1, zdim])))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        torch.nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        assert self.zdim == z.shape[2], 'zdim not compatible!'
        rk = torch.sigmoid(self.rk_lgt).pow(.5)
        if self.gdc == 'bp':
            z = z.mul(rk.view(1, 1, self.zdim))
        adj_lgt = torch.bmm(z, torch.transpose(z, 1, 2))
        if self.gdc == 'ip':
            adj = torch.sigmoid(adj_lgt)
        elif self.gdc == 'bp':
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())
        if not self.training:
            adj = torch.mean(adj, dim=0, keepdim=True)
        return adj, z, rk.pow(2)

class VGAE(nn.Module):
    def __init__(self):
        super(VGAE, self).__init__()
        hidden = configs['model']['embedding_size']
        self.K = configs['model']['K']
        self.J = configs['model']['J']
        self.ndim = configs['model']['edim']
        self.encsto = configs['model']['encsto']
        self.gc1 = GraphConvolution(hidden, hidden, 0.0, act=F.relu)  # 修正拼写错误：elf -> self
        self.gce = GraphConvolution(self.ndim, hidden, 0.0, act=F.relu)
        self.gc2 = GraphConvolution(hidden, hidden, 0.0, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden, hidden, 0.0, act=lambda x: x)
        self.dc = GraphDecoder(hidden, dropout=0.0)
        self.reweight = pow( (self.ndim + hidden) / (hidden + hidden), 0.5 )
        #self.reweight = ((self.ndim + hidden) / (hidden + hidden)) ​**​ 0.5
        self.ndist = tdist.Bernoulli(t.tensor([.5], device=self.device))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')

    def set_adagcl(self, adagcl):
        self.reg_weight = configs['model']['reg_weight']
        self.adagcl = adagcl

    def _propagate(self, adj, embeds, flag=True):
        if flag:
            return t.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)

    def encode(self, x, adj):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, f'Expected input to be 3D, got {x.dim()}D'
        hiddenx = self.gc1(x, adj)
        if self.ndim >= 1:
            e = self.ndist.sample(torch.Size([self.K + self.J, x.shape[1], self.ndim]))
            e = torch.squeeze(e, -1)
            e = e.mul(self.reweight)
            hiddene = self.gce(e, adj)
        else:
            print("no randomness.")
            hiddene = torch.zeros(self.K + self.J, hiddenx.shape[1], hiddenx.shape[2], device=self.device)
        hidden1 = hiddenx + hiddene.mean(0, keepdim=True)
        p_signal = hiddenx.pow(2.).mean()
        p_noise = hiddene.pow(2.).mean([-2, -1])
        snr = (p_signal / p_noise)
        mu = self.gc2(hidden1, adj)
        EncSto = (self.encsto == 'full')
        hidden_sd = EncSto * hidden1 + (1 - EncSto) * hiddenx
        logvar = self.gc3(hidden_sd, adj)
        mu = mu.squeeze(0)
        logvar = logvar.squeeze(0)
        return mu, logvar, snr

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps

    def forward_encoder(self, x, adj):
        mu, logvar, snr = self.encode(x, adj)
        gaussian_noise = t.randn(x_mean.shape).cuda()
        x = gaussian_noise * logvar + mu
        emb_mu = mu[self.K:, :]
        emb_logvar = logvar[self.K:, :]
        assert len(emb_mu.shape) == len(emb_logvar.shape), 'mu and logvar are not equi-dimension.'
        z, eps = self.reparameterize(emb_mu, emb_logvar)
        adj_, z_scaled, rk = self.dc(z)
        return adj_, mu, logvar, z, z_scaled, eps, rk, snr, x

    def loss_function(self, preds, mu, logvar, emb, eps):
        SMALL = 1e-6
        std = torch.exp(0.5 * logvar)
        J, N, zdim = emb.shape
        K = mu.shape[0] - J
        mu_mix, mu_emb = mu[:K, :], mu[K:, :]
        std_mix, std_emb = std[:K, :], std[K:, :]
        preds = torch.clamp(preds, min=SMALL, max=1-SMALL)
        log_prior_ker = torch.sum(- 0.5 * emb.pow(2), dim=[1,2]).mean()
        Z = emb.view(J, 1, N, zdim)
        mu_mix = mu_mix.view(1, K, N, zdim)
        std_mix = std_mix.view(1, K, N, zdim)
        log_post_ker_JK = - torch.sum(
            0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2,-1]
        )
        log_post_ker_JK += - torch.sum(
            (std_mix + SMALL).log(), dim=[-2,-1]
        )
        log_post_ker_J = - torch.sum(
            0.5 * eps.pow(2), dim=[-2,-1]
        )
        log_post_ker_J += - torch.sum(
            (std_emb + SMALL).log(), dim = [-2,-1]
        )
        log_post_ker_J = log_post_ker_J.view(-1,1)
        log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)
        log_post_ker -= np.log(K + 1.) / J
        log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()
        return  log_prior_ker, log_posterior_ker

    def cal_loss_vgae(self, data, batch_data):
        users, items, neg_items = batch_data
        adj_, x_mean, x_std, z, z_scaled, eps, rk, snr, x = self.forward_encoder(data)
        loss_prior, loss_post = self.loss_function(
            recovered, 
            mu=x_mean, 
            logvar=x_std, 
            emb=z, 
            eps=eps
        )
        x_user, x_item = t.split(x, [configs['data']['user_num'], configs['data']['item_num']], dim=0)
        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))
        loss_edge_pos = self.bceloss( edge_pos_pred, t.ones(edge_pos_pred.shape).cuda() )
        loss_edge_neg = self.bceloss( edge_neg_pred, t.zeros(edge_neg_pred.shape).cuda() )
        loss_rec = loss_edge_pos + loss_edge_neg
        WU = np.min([epoch/200., 1.])
        kl_divergence = (loss_post - loss_prior)*WU
        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        bprLoss = cal_bpr_loss(ancEmbeds, posEmbeds, negEmbeds) / ancEmbeds.shape[0]
        beta = 0.1
        loss = (loss_rec + WU * kl_divergence.mean() + bprLoss).mean()
        losses = {'generate_loss':loss}
        return loss, losses

    def vgae_generate(self, data, edge_index, adj):
        x, _, _ = self.forward_encoder(data)
        edge_pred = self.sigmoid(self.decoder(x[edge_index[0]] * x[edge_index[1]]))
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        edge_pred = edge_pred[:, 0]
        mask = ((edge_pred + 0.5).floor()).type(t.bool)
        newVals = vals[mask]
        newVals = newVals / (newVals.shape[0] / edgeNum[0])
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        hidden = configs['model']['embedding_size']
        self.edge_weights = []
        self.nblayers = []
        self.selflayers = []
        self.attentions = []
        self.attentions.append([])
        self.attentions.append([])
        self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.attentions_0 = nn.Sequential(nn.Linear( 2 * hidden, 1))
        self.attentions_1 = nn.Sequential(nn.Linear( 2 * hidden, 1))

    def set_adagcl(self, adagcl):
        self.user_embeds = adagcl.user_embeds
        self.item_embeds = adagcl.item_embeds
        self.user_num = adagcl.user_num
        self.item_num = adagcl.item_num
        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        self.features = t.concat([self.user_embeds, self.item_embeds]).cuda()
        self.set_fea_adj(self.user_num+self.item_num, adagcl.adj)

    def get_attention(self, input1, input2, layer=0):
        if layer == 0:
            nb_layer = self.nblayers_0
            selflayer = self.selflayers_0
        if layer == 1:
            nb_layer = self.nblayers_1
            selflayer = self.selflayers_1
        input1 = nb_layer(input1)
        input2 = selflayer(input2)
        input10 = t.concat([input1, input2], axis=1)
        if layer == 0:
            weight10 = self.attentions_0(input10)
        if layer == 1:
            weight10 = self.attentions_1(input10)
        return weight10

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        gamma = configs['model']['gamma']
        zeta = configs['model']['zeta']
        if training:
            debug_var = 1e-7
            bias = 0.0
            np_random = np.random.uniform(low=debug_var, high=1.0-debug_var, size=np.shape(log_alpha.cpu().detach().numpy()))
            random_noise = bias + t.tensor(np_random)
            gate_inputs = t.log(random_noise) - t.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.cuda() + log_alpha) / beta
            gate_inputs = t.sigmoid(gate_inputs)
        else:
            gate_inputs = t.sigmoid(log_alpha)
        stretched_values = gate_inputs * (zeta-gamma) +gamma
        cliped = t.clamp(stretched_values, 0.0, 1.0)
        return cliped.float()

    def denoise_generate(self, x, layer=0):
        f1_features = x[self.row, :]
        f2_features = x[self.col, :]
        weight = self.get_attention(f1_features, f2_features, layer)
        mask = self.hard_concrete_sample(weight, training=False)
        mask = t.squeeze(mask)
        adj = t.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape)
        ind = deepcopy(adj._indices())
        row = ind[0, :]
        col = ind[1, :]
        rowsum = t.sparse.sum(adj, dim=-1).to_dense()
        d_inv_sqrt = t.reshape(t.pow(rowsum, -0.5), [-1])
        d_inv_sqrt = t.clamp(d_inv_sqrt, 0.0, 10.0)
        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = t.mul(adj._values(), row_inv_sqrt)
        values = t.mul(values, col_inv_sqrt)
        support = t.sparse.FloatTensor(adj._indices(), values, adj.shape)
        return support

    def l0_norm(self, log_alpha, beta):
        gamma = configs['model']['gamma']
        zeta = configs['model']['zeta']
        gamma = t.tensor(gamma)
        zeta = t.tensor(zeta)
        reg_per_weight = t.sigmoid(log_alpha - beta * t.log(-gamma/zeta))
        return t.mean(reg_per_weight)

    def set_fea_adj(self, nodes, adj):
        self.node_size = nodes
        self.adj_mat = adj
        ind = deepcopy(adj._indices())
        self.row = ind[0, :]
        self.col = ind[1, :]

    def call(self, inputs, training=None):
        if training:
            temperature = inputs
        else:
            temperature = 1.0
        self.maskes = []
        x = self.features.detach()
        layer_index = 0
        embedsLst = [self.features.detach()]
        for i in range(self.layer_num):
            xs = []
            f1_features = x[self.row, :]
            f2_features = x[self.col, :]
            weight = self.get_attention(f1_features, f2_features, layer=layer_index)
            mask = self.hard_concrete_sample(weight, temperature, training)
            self.edge_weights.append(weight)
            self.maskes.append(mask)
            mask = t.squeeze(mask)
            adj = t.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape).coalesce()
            ind = deepcopy(adj._indices())
            row = ind[0, :]
            col = ind[1, :]
            rowsum = t.sparse.sum(adj, dim=-1).to_dense() + 1e-6
            d_inv_sqrt = t.reshape(t.pow(rowsum, -0.5), [-1])
            d_inv_sqrt = t.clamp(d_inv_sqrt, 0.0, 10.0)
            row_inv_sqrt = d_inv_sqrt[row]
            col_inv_sqrt = d_inv_sqrt[col]
            values = t.mul(adj.values(), row_inv_sqrt)
            values = t.mul(values, col_inv_sqrt)
            support = t.sparse.FloatTensor(adj._indices(), values, adj.shape).coalesce()
            nextx = self._propagate(support, x, False)
            xs.append(nextx)
            x = xs[0]
            embedsLst.append(x)
            layer_index += 1
        return sum(embedsLst)

    def lossl0(self, temperature):
        l0_loss = t.zeros([]).cuda()
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)
        self.edge_weights = []
        return l0_loss

    def cal_loss_denoise(self, batch_data, temperature):
        x = self.call(temperature, True)
        x_user, x_item = t.split(x, [self.user_num, self.item_num], dim=0)
        users, items, neg_items = batch_data
        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        bprLoss = cal_bpr_loss(ancEmbeds, posEmbeds, negEmbeds) / ancEmbeds.shape[0]
        lossl0 = self.lossl0(temperature) * configs['model']['lambda0']
        loss = bprLoss + lossl0
        losses = {'denoise_loss':loss}
        return loss, losses

    def _propagate(self, adj, embeds, flag=True):
        if flag:
            return t.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)
