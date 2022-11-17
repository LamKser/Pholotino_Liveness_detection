from argparse import ArgumentParser 
from model import RunModel
import torch

parser = ArgumentParser(description='Run VGG19')

parser.add_argument('--train-path', default='dataset/train', type=str, metavar='--TRP',
                  help='Training data path (default: "dataset/train")')
parser.add_argument('--val-path', default='dataset/val', type=str, metavar='--VP',
                  help='Validation data path (default: "dataset/val")')
parser.add_argument('--test-path', default='dataset/test', type=str, metavar='--TEP',
                  help='Test data path (default: "dataset/test")')
parser.add_argument('--num-class', default=2, type=int, metavar='--NC',
                   help='Number of class ')
parser.add_argument('--batch-size', default=10, type=int, metavar='--BS',
                   help='Batch size (default: 10)')
parser.add_argument('--lr', default=1e-5, type=float, metavar='--LR',
                   help='Learning rate (default: 1e-5)')
parser.add_argument('--weight-decay', default=0.0, type=float, metavar='--WD',
                   help='weight decay (default: 0.0)')
parser.add_argument('--momentum', default=0.0, type=float, metavar='--MO',
                   help='momentum (default: 0.0)')                  
parser.add_argument('--step-size', default=25, type=int, metavar='--SS',
                   help='Step size (default: 25)')
parser.add_argument('--gamma', default=0.1, type=float, metavar='--GA',
                   help='Gamma (default: 0.1)')
parser.add_argument('--pretrained', default=False, type=bool, metavar='--PRE',
                   help='Pretrained (ImageNet) (default: False)')
parser.add_argument('--epochs', default=100, type=int, metavar='--E',
                   help='Epochs (default: 100)')


args = parser.parse_args()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = RunModel(device=device, 
                   train_path=args.train_path, val_path=args.val_path, test_path=args.test_path, batch_size=args.batch_size,
                   lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                   step_size=args.step_size, gamma=args.gamma,
                   num_class=args.num_class, pretrained=args.pretrained)
    run.train(args.epochs)

