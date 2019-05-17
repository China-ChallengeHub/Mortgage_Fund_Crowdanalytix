import argparse
def get_args(args_string=None):
    parser = argparse.ArgumentParser(description='Walmart Replenishment ISD rnn')
    parser.add_argument('--save-dir', type=str, help='main folder for saving models')
    #rnn
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--size-s', type=int, default=128,
                        help='rnn hidden state size (default: 128)')
    parser.add_argument('--rnn-layers', type=int, default=1,
                        help='rnn layers (default: 1)')
    parser.add_argument('--seq-len', type=int, default=14,
                        help='warm_up/prediction window lenght, note that the effective size of the seq_len is going to be 2x')


    print(args_string)
    args = parser.parse_args(args=args_string)
    return args
