from argparse import ArgumentParser 
from model import RunModel
import torch
import matplotlib.pyplot as plt
import os

parser = ArgumentParser(description='Run VGG19')

parser.add_argument('--train-path', default=None, type=str, metavar='--TRP',
                  help='Training data path (default: None)')
parser.add_argument('--val-path', default=None, type=str, metavar='--VP',
                  help='Validation data path (default: None)')
parser.add_argument('--test-path', default=None, type=str, metavar='--TEP',
                  help='Test data path (default: None)')
parser.add_argument('--save-path', default=None, type=str, metavar='--SP',
                  help='Save weight path (default: None)') 
parser.add_argument('--weight-file', default=None, type=str, metavar='--Weight',
                  help='Weight file (default: None)')
parser.add_argument('--csv-file', default=None, type=str, metavar='--CSV',
                   help='Save score to csv (default: None)')
parser.add_argument('--num-class', default=1, type=int, metavar='--NC',
                   help='Number of class (default: 2) ')
parser.add_argument('--batch-size', default=10, type=int, metavar='--BS',
                   help='Batch size (default: 10)')
parser.add_argument('--lr', default=1e-5, type=float, metavar='--LR',
                   help='Learning rate (default: 1e-3)')
parser.add_argument('--weight-decay', default=0.0, type=float, metavar='--WD',
                   help='weight decay (default: 0.0)')
parser.add_argument('--momentum', default=0.0, type=float, metavar='--MO',
                   help='momentum (default: 0.0)')    
parser.add_argument('--is-scheduler', default=True, type=bool, metavar='--S',
                   help='Is scheduler (default: False)')              
parser.add_argument('--step-size', default=25, type=int, metavar='--SS',
                   help='Step size (default: 25)')
parser.add_argument('--gamma', default=0.1, type=float, metavar='--GA',
                   help='Gamma (default: 0.1)')
parser.add_argument('--pretrained', default=False, type=bool, metavar='--PRE',
                   help='Pretrained (ImageNet) (default: False)')
parser.add_argument('--epochs', default=100, type=int, metavar='--E',
                   help='Epochs (default: 100)')

parser.add_argument('--mode', default='train', type=str, metavar='--M',
                   help='Choose mode for running model (default: train)')

args = parser.parse_args()
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = RunModel(device=device, 
                   train_path=args.train_path, val_path=args.val_path, test_path=args.test_path, batch_size=args.batch_size,
                   lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                   is_scheduler=args.is_scheduler, step_size=args.step_size, gamma=args.gamma,
                   num_class=args.num_class, pretrained=args.pretrained)

    if args.mode == 'train':
        train_acc, train_loss, val_acc, val_loss = run.train(args.epochs, args.save_path, args.weight_file)
    
    elif args.mode == 'test':
        run.test(args.csv_file, os.path.join(args.save_path, args.weight_file))
    elif args.mode == 'test_video':
        pass
