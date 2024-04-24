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

class TransformerBlocks(torch.nn.Module):
    def __init__(self,emb_dim,num_heads,feed_forward_size,blocks_num=4,drop_out=0.1):
        super().__init__()
        self.blocks_num=blocks_num
        self.rms_norm=RMSNorm(emb_dim)
        self.attenion=torch.nn.MultiheadAttention(emb_dim,num_heads,drop_out)
        self.feed_forward=torch.nn.Sequential(
            torch.nn.Linear(emb_dim,feed_forward_size),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(feed_forward_size,emb_dim),
        )
    def forward(self,tensor):
        for i in range(self.blocks_num):
            copy_tensor=tensor
            tensor=self.rms_norm(tensor)
            tensor,_=self.attenion(tensor,tensor,tensor)
            tensor=self.feed_forward(tensor)
            tensor+=copy_tensor
        return tensor

#模型参数
emb_size=256     #字符嵌入维度
heads=32         #多头注意力头数
dict_size=60000  #字典大小
max_length=100    #文本生成字符长度

class MainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_input=torch.nn.Embedding(num_embeddings=1114112,embedding_dim=emb_size)
        self.emb_target = torch.nn.Embedding(num_embeddings=1114112, embedding_dim=emb_size)
        self.input_blocks=TransformerBlocks(emb_dim=emb_size,num_heads=heads,feed_forward_size=4*emb_size)
        self.target_blocks_font = TransformerBlocks(emb_dim=emb_size, num_heads=heads, feed_forward_size=4 * emb_size)
        self.target_blocks_back = TransformerBlocks(emb_dim=emb_size, num_heads=heads, feed_forward_size=4 * emb_size)
        self.output_layer=torch.nn.Linear(emb_size,dict_size)
    def forward(self,input,target):
        input=self.emb_input(input)
        target=self.emb_target(target)
        input=self.input_blocks(input)
        target=self.target_blocks_font(target)
        target=(input*target).sum(0).unsqueeze(0)
        target=self.target_blocks_back(target)
        target=torch.flatten(target)
        target=self.output_layer(target)
        return target


