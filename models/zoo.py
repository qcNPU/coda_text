import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
# from .tit import load_clip_to_cpu,tokenize
import numpy as np
import transformers
from transformers import CLIPProcessor, CLIPModel
import copy
from .tit import load_clip_to_cpu,tokenize


# Our method!
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init 编码层的初始化，为每个编码层都初始化一个prompt（p），key（k），attention（a）矩阵
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)#prompt是3维，k、a是2维
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)#setattr是Python的一个内置函数，用于给对象设置属性值。
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):#计算向量v在向量u上的投影
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3#输入张量vv是三维的（即有三个维度），则将其重塑为二维张量。这里将原始形状存储在shape_2d中，以便之后恢复原始形状
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)#todo 这里不用deepcopy，shape_2d会变吗
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))#per task
        s = int(self.task_count * pt)#start
        f = int((self.task_count + 1) * pt)#final
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):#对指定范围内的向量进行正交化处理，这个过程确保得到的向量彼此正交，并归一化每个向量。
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) #最后包装为parameter

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')#获取出每一层中的P、K、A矩阵
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim  1.1 先用x*A
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)#x_querry：10,77； A：10,768
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)# 1.2 1.1的结果归一化得到q，再用q*K，得到权重
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p) # 1.3 P_ （22,8,768），用权重*P

            # select prompts，e_p_length是prompt长度
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()






class ClipZoo(nn.Module):  # ViTZoo本身就是一个model，里面包含预训练的vit model和初始化待训练了的prompt module
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None):
        super(ClipZoo, self).__init__()

        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder  在这里加载预训练模型
        if pt:
            # zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
            #                               num_heads=12, ckpt_layer=0,
            #                               drop_path_rate=0
            #                               )
            # from timm.models import vit_base_patch16_224
            # load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            # del load_dict['head.weight']
            # del load_dict['head.bias']
            # zoo_model.load_state_dict(load_dict)
            #
            # path = '/home/qc/CODA-Prompt-main/pretrained/clip_vit_base_patch16'
            # self.model = CLIPModel.from_pretrained(path)
            # dict1 = self.model.state_dict()

            clip = load_clip_to_cpu('/home/qc/pretrained_model/ViT-L-14.pt')
            # clip = load_clip_to_cpu('/home/qc/pretrained_model/ViT-B-16.pt')

        if self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

        # feature encoder changes if transformer vs resnet
        # self.feat = zoo_model
        self.clip = clip

    # pen: get penultimate features
    def forward(self, x, targets,classnames,pen=False, train=False):

        if self.prompt is not None:
            with torch.no_grad():  # 提取image feature
                img_fea = self.clip.encode_image(x)
                labels = targets - self.task_id*10
                q_text = [f'a photo of {classnames[label].replace("_", " ")}' for label in labels]
                q_text = tokenize(q_text).cuda()
                q = self.clip.encode_text(q_text)
                text = [f'a photo of {cname.replace("_", " ")}' for cname in classnames]
                text = tokenize(text).cuda()  # (10,77)
                # text = self.clip.encode_text(text)
                # text = self.clip.token_embedding(text)[:, 0, :]
                text = text.repeat(x.size(0), 1)
                # q和text使用一个class，把所有class都走一遍，最后得到总体的概率
                # q, _ = self.feat(x)  # （22,197,768）
                # q = q[:, 0, :]  # （22,768）
            txt_fea = self.clip.encode_text(text, prompt=self.prompt, q=q, train=train,task_id=self.task_id)#160,768
            # out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train,
            #                              task_id=self.task_id)  # out:(20,197,768)
            # out = out[:, 0, :]  # (20,768)
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        # out = out.view(out.size(0), -1)

        # normalized features
        image_features = F.normalize(img_fea,dim=-1)#64,768
        text_features = F.normalize(txt_fea,dim=-1)#640,768
        text_features  = text_features.view(x.size(0),-1,768)
        image_features = image_features.unsqueeze(1)
        logit_scale = self.clip.logit_scale.exp()
        # logits_per_text = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale *  (image_features * text_features).sum(-1)
        out1 = logits_per_text

        prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
        if self.prompt is not None and train:
            return out1, prompt_loss
        else:
            return out1

def clip_pt(out_dim,  prompt_flag='None', prompt_param=None):
    return ClipZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)


class ViTZoo(nn.Module):  # ViTZoo本身就是一个model，里面包含预训练的vit model和初始化待训练了的prompt module
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder  在这里加载预训练模型
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                          num_heads=12, ckpt_layer=0,
                                          drop_path_rate=0
                                          )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()  # 加载预训练模型参数vit/B-16
            del load_dict['head.weight']
            del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier todo 这里为什么重复定义了self.last
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False):#x：16,3224,224

        if self.prompt is not None:
            with torch.no_grad():  # 提取image feature
                q, _ = self.feat(x)  # （16,197,768）
                q = q[:, 0, :]  # （16,768）
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train,
                                         task_id=self.task_id)  # out:(16,197,768)
            out = out[:, 0, :]  # (16,768)
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)#(16,768)
        if not pen:
            out = self.last(out)#(16,100)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out


def vit_pt_imnet(out_dim, block_division=None, prompt_flag='None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}', p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    def _init_smart(self, emb_d, prompt_param):

        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0, 1]
        self.e_layers = [2, 3, 4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self, f'e_k_{l}')  # 0 based indexing here
            p = getattr(self, f'e_p_{l}')  # 0 based indexing here

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)

            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:, task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry), -1, -1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:, k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]

            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
            else:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, :, i:, :].reshape((B, -1, self.emb_d))

        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length / 2)
            p = getattr(self, f'g_p_{l}')  # 0 based indexing here
            P_ = p.expand(len(x_querry), -1, -1)
            Gk = P_[:, :j, :]
            Gv = P_[:, j:, :]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0, 1, 2, 3, 4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p