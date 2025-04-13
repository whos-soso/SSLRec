import torch as t
import torch

from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
import torch_sparse
from copy import deepcopy
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
# from layers import GraphConvolution
import torch.distributions as tdist


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
		self.nblayers_2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.nblayers_3 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.nblayers_4 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		
		self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.selflayers_2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.selflayers_3 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.selflayers_4 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		

		self.attentions_0 = nn.Sequential(nn.Linear( 2 * hidden, 1))
		self.attentions_1 = nn.Sequential(nn.Linear( 2 * hidden, 1))
		self.attentions_2 = nn.Sequential(nn.Linear( 2 * hidden, 1))
		self.attentions_3 = nn.Sequential(nn.Linear( 2 * hidden, 1))
		self.attentions_4 = nn.Sequential(nn.Linear( 2 * hidden, 1))

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
		nb_layer = self.nblayers_1
		selflayer = self.selflayers_1

		input1 = nb_layer(input1)
		input2 = selflayer(input2)

		input10 = t.concat([input1, input2], axis=1)

		
		weight10 = self.attentions_0(input10)
		
		
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




class GraphDecoder(nn.Module):
    def __init__(self, zdim, dropout, gdc='ip'):
        super(GraphDecoder, self).__init__()
        self.dropout = dropout
        self.gdc = gdc
        self.zdim = zdim
        self.rk_lgt = Parameter(t.FloatTensor(1, zdim))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        t.nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        rk = torch.sigmoid(self.rk_lgt).pow(0.5)
        z = z.mul(rk.view(1, 1, self.zdim))
        adj_lgt = torch.bmm(z, z.transpose(1, 2))
        adj = torch.sigmoid(adj_lgt)
        if not self.training:
            adj = adj.mean(dim=0, keepdim=True)
        return adj, z, rk.pow(2)

class GraphConvolution(Module):
    """
    GCN layer, based on https://arxiv.org/abs/1609.02907
    that allows MIMO
    """

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
        """
        if the input features are a matrix -- excute regular GCN,
        if the input features are of shape [K, N, Z] -- excute MIMO GCN with shared weights.
        """
        # An alternative to derive XW (line 32 to 35)
        # W = self.weight.view(
        #         [1, self.in_features, self.out_features]
        #         ).expand([input.shape[0], -1, -1])
        # support = torch.bmm(input, W)

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
        self.rk_lgt = Parameter(t.FloatTensor(1, zdim))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        t.nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        rk = torch.sigmoid(self.rk_lgt).pow(0.5)
        z = z.mul(rk.view(1, 1, self.zdim))
        adj_lgt = torch.bmm(z, z.transpose(1, 2))
        adj = torch.sigmoid(adj_lgt)
        if not self.training:
            adj = adj.mean(dim=0, keepdim=True)
        return adj, z, rk.pow(2)

class VGAE(nn.Module):
    def __init__(self):
        super(VGAE, self).__init__()
        
        hidden = configs['model']['embedding_size']
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.reg_weight = configs['model']['reg_weight']
        self.beta = 0.1

        # Encoder: SIG-VAE
        self.ndim = 16
        self.gc1 = GraphConvolution(hidden, hidden, 0.0, act=F.relu)
        self.gce = GraphConvolution(self.ndim, hidden, 0.0, act=F.relu)
        self.gc2 = GraphConvolution(hidden, hidden, 0.0, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden, hidden, 0.0, act=lambda x: x)
        self.dc = GraphDecoder(hidden, dropout=0.0)
        self.reweight = ((self.ndim + hidden) / (hidden + hidden))**0.5
        self.K = configs['model']['K']
        self.J = configs['model']['J']
        self.device = 'cuda'
        self.ndist = tdist.Bernoulli(t.tensor([.5], device=self.device))

        # Decoder and loss
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')

    def set_adagcl(self, adagcl):
        self.adagcl = adagcl

    def _propagate(self, adj, embeds, flag=True):
        return t.spmm(adj, embeds) if flag else torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)

    def encode(self, x, adj):
        hiddenx = self.gc1(x, adj)
        e = self.ndist.sample((self.K + self.J, x.shape[1], self.ndim)).squeeze(-1).to(self.device)
        e = e.mul(self.reweight)
        hiddene = self.gce(e, adj)
        hidden1 = hiddenx + hiddene
        mu = self.gc2(hidden1, adj)
        logvar = self.gc3(hiddenx, adj)  # deterministic logvar (semi)
        return mu[self.K:], logvar[self.K:]

    def reparameterize(self, mu, logvar):
        std = t.exp(logvar / 2.)
        eps = t.randn(mu.shape).cuda()
        return eps * std + mu

    def forward_encoder(self, adj):
        self.is_training = True
        x_u, x_i = self.adagcl.forward(adj)
        x = t.concat([x_u.detach(), x_i.detach()])
        mu, logvar = self.encode(x.unsqueeze(0), adj)
        z = self.reparameterize(mu, logvar)
        return z.squeeze(0), mu.squeeze(0), logvar.squeeze(0)

    def cal_loss_vgae(self, data, batch_data):
        users, items, neg_items = batch_data
        x, x_mean, x_std = self.forward_encoder(data)

        x_user, x_item = t.split(x, [self.user_num, self.item_num], dim=0)

        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        loss_rec = self.bceloss(edge_pos_pred, t.ones_like(edge_pos_pred)) + \
                   self.bceloss(edge_neg_pred, t.zeros_like(edge_neg_pred))

        kl_divergence = -0.5 * (1 + 2 * t.log(x_std + 1e-10) - x_mean**2 - x_std**2).sum(dim=1)

        bprLoss = cal_bpr_loss(x_user[users], x_item[items], x_item[neg_items]) / users.shape[0]

        loss = (loss_rec.mean() + self.beta * kl_divergence.mean() + bprLoss).mean()
        return loss, {'generate_loss': loss}

    def vgae_generate(self, data, edge_index, adj):
        x, _, _ = self.forward_encoder(data)
        edge_pred = self.sigmoid(self.decoder(x[edge_index[0]] * x[edge_index[1]]))[:, 0]

        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((edge_pred + 0.5).floor()).type(t.bool)

        newVals = vals[mask]
        newVals = newVals / (newVals.shape[0] / edgeNum[0])
        newIdxs = idxs[:, mask]

        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)
