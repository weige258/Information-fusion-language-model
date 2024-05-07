from main import train,generation,model
import torch
num=0
for e in range(5):
    for line in open("样例_内科5000-6000.csv",encoding="gbk"):
        try:
            a,b,c,d=line.split(",")
            train(c,d)
            generation(c)
            num+=1
            if num%500==0:
                torch.save(obj=model, f="model.pth")
            else:
                continue
        except:
            continue
    torch.save(obj=model,f="model.pth")

