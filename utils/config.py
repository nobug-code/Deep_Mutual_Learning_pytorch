import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of deep mutual learing models.')
    parser.add_argument('--model', type=str, default='MUTUAL', choices=['MUTUAL'])
    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--dataroot', type=str, default='/home/nkim/data', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_networks', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model_name', type=str, default='mobilenet', choices=['mobilenet'])
    parser.add_argument('--model_number', type=int, default=2)


    return check_args(parser.parse_args())

def check_args(args):

    try:
        assert args.epochs >= 1
    except:
        print("Number of epochs must be larger than or equal to one")

    try:
        assert args.batch_size >= 1
    except:
        print("Number of batch_size must be larger than or equal to one")

    return args