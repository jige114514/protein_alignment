import argparse

parser = argparse.ArgumentParser(description='choose dataset')

parser.add_argument('--dataset', type=str, default='supervised',
                    help='choose supervised or unsupervised contrastive approach')
args = parser.parse_args()
