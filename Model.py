import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TransformerBlock(torch.nn.Module):
    def __init__(self,emb_dim,num_heads,drop_out=0.1):
        super().__init__()
        self.rms_norm=RMSNorm(emb_dim)
        self.attenion=torch.nn.MultiheadAttention(emb_dim,num_heads,drop_out)
        self.feed_forward=torch.nn.Sequential(
            torch.nn.Linear(emb_dim,4*emb_dim),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(4*emb_dim,emb_dim),
            torch.nn.Dropout(drop_out),
        )
    def forward(self,tensor):
        copy_tensor=tensor
        tensor=self.rms_norm(tensor)
        tensor,_=self.attenion(tensor,tensor,tensor)
        tensor+=copy_tensor
        copy_tensor=tensor
        tensor=self.feed_forward(tensor)
        tensor+=copy_tensor
        return tensor

#模型参数
emb_size=256    #字符嵌入维度
heads=32         #多头注意力头数
dict_size=60000  #字典大小
max_length=100    #文本生成字符长度

class MainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=torch.nn.Embedding(num_embeddings=1114112,embedding_dim=emb_size)
        self.font_block1=TransformerBlock(emb_size,heads)
        self.font_block2 = TransformerBlock(emb_size, heads)
        self.font_block3 = TransformerBlock(emb_size, heads)
        self.font_block4 = TransformerBlock(emb_size, heads)
        self.back_block1 = TransformerBlock(emb_size, heads)
        self.back_block2 = TransformerBlock(emb_size, heads)
        self.back_block3 = TransformerBlock(emb_size, heads)
        self.back_block4 = TransformerBlock(emb_size, heads)
        self.back_block5 = TransformerBlock(emb_size, heads)
        self.back_block6 = TransformerBlock(emb_size, heads)
        self.back_block7 = TransformerBlock(emb_size, heads)
        self.back_block8 = TransformerBlock(emb_size, heads)
        self.output_layer=torch.nn.Linear(emb_size,dict_size)
    def forward(self,input,target):
        input=self.emb(input)
        target=self.emb(target)
        target=self.font_block1(target)
        target = self.font_block2(target)
        target = self.font_block3(target)
        target = self.font_block4(target)
        target=(target*input).sum(dim=0).unsqueeze(0)
        target=self.back_block1(target)
        target = self.back_block2(target)
        target = self.back_block3(target)
        target = self.back_block4(target)
        target = self.back_block5(target)
        target = self.back_block6(target)
        target = self.back_block7(target)
        target = self.back_block8(target)
        target=torch.flatten(target)
        target=self.output_layer(target)
        return target


