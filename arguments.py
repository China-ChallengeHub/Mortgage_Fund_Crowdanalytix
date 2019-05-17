import argparse
def get_args(args_string=None):
    parser = argparse.ArgumentParser(description='Mortgage FUNDED probability calcuation')
    parser.add_argument('--save-dir', type=str, help='main folder for saving models')
    #xgb
    parser.add_argument('--nrounds', type=float, default=1e-3)

    print(args_string)
    args = parser.parse_args(args=args_string)
    return args
