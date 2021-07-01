from train import train
from validate import validate
from test import test
if __name__ == "__main__":
    inp = int(input("0 - Train\n1 - Validade\n2 - Test with input\n"))
    if(inp == 0):
        train()
    if(inp == 1):
        validate()
    if(inp == 2):
        test()
    