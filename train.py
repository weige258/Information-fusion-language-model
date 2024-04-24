from main import train,generation,model
import torch

for e in range(5):
    for line in open("样例_内科5000-6000.csv"):
        try:
            a,b,c,d=line.split(",")
            train(c,d)
            print(d)
            generation(c)
        except:
            continue
    torch.save(obj=model,f="model.pth")

