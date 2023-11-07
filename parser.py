import argparse

parser = argparse.ArgumentParser(description='choose dataset')

parser.add_argument('--dataset', type=str, default='supervised',
                    help='choose supervised or unsupervised contrastive approach')
parser.add_argument('--seed', type=int, default=2023,
                    help='random seed')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size including duplicate sequences')
parser.add_argument('--epoch', type=int, default=1,
                    help='epoch num')
parser.add_argument('--print_interval', type=int, default=1,
                    help='interval to print train loss')

args = parser.parse_args()
