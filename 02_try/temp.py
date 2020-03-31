#
import torch
def test_str_enumerate() :
    labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
    int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
    print(int_to_char)


def test_dict() :
    labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
    labels_map = dict([(labels[i], i) for i in range(len(labels))])
    print(labels_map)
    return


def pytorch_version() :
    print(torch.__version__)


def test_torch_log_softmax() :

    T = 5  # Input sequence length
    C = 5  # Number of classes (including blank)
    N = 5  # Batch size
    S = 30  # Target sequence length of longest target in batch
    S_min = 10
    torch.manual_seed(1234)

    input = torch.randn(T, N, C)    # .log_softmax(2).detach().requires_grad_()
    # print("input : ", input)
    out1 = input.log_softmax(2)
    # print("input : ", input)
    out2 = torch.nn.functional.log_softmax(input)    # 默认执行 dim = 0 悲剧
    out_dim_0 = torch.nn.functional.log_softmax(input, dim=0)
    out_dim_1 = torch.nn.functional.log_softmax(input, dim=1)
    out_dim_2 = torch.nn.functional.log_softmax(input, dim=2)
    # out_dim_3 = torch.nn.functional.log_softmax(input, dim=3)    # error
    # print("out1 : ", out1)
    print("out2 : ", out2)
    print("out_dim_0 : ", out_dim_0)
    # print("out_dim_1 : ", out_dim_1)
    # print("out_dim_2 : ", out_dim_2)
    # print("out_dim_3 : ", out_dim_3)

def print_device() :
    input = torch.randn(2, 2)
    print(input.device)


def test_cuda() :
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    a = torch.ones([20, 20]).to(device)
    b = torch.ones([20, 20]).to(device)
    c = a + b
    print("c.device : ", c.device)
    print("c : ", c)


def test_torchaudio() :
    import torchaudio


def test_set() :
    ids = list(range(0, 5))
    batch_size = 3
    bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    print(bins)
    la = [0, 1, 2]
    print(la[2:5])


def test_shuffle() :
    import numpy as np
    la = [i for i in range(43)]
    b = 10
    lb = [la[i:i+b] for i in range(0, len(la), b)]
    print(lb)
    np.random.shuffle(lb)
    print(type(lb))
    print(lb)


if __name__ == "__main__" :
    # test_str_enumerate()
    # test_dict()
    # pytorch_version()
    # test_torch_log_softmax()
    # print_device()
    # test_cuda()
    # test_set()
    test_shuffle()
