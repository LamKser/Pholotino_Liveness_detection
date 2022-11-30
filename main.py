from argparse import ArgumentParser
from model import RunModel
import torch
import os

parser = ArgumentParser(description='Run VGG19')

parser.add_argument('--validation', default=False, type=bool,
                    help='Validate model (default: "False")')
parser.add_argument('--name', default='vgg19', type=str,
                    help='Model name (default: "vgg19")')
parser.add_argument('--train-path', default='train', type=str,
                    help='Training data path (default: "train")')
parser.add_argument('--val-path', default='val', type=str,
                    help='Validation data path (default: "val")')
# parser.add_argument('--test-path', default='test', type=str,
                    # help='Test data path (default: "test")')
parser.add_argument('--test-video-path', default='videos', type=str,
                    help='Test video data path (default: "videos")')
parser.add_argument('--save-path', default='weight', type=str,
                    help='Save weight path (default: "weight")')
parser.add_argument('--weight-file', default='model.pt', type=str,
                    help='Weight file (default: "model.pt")')
parser.add_argument('--csv-file', default='Result.csv', type=str,
                    help='Save score to csv (default: "Result.csv")')
parser.add_argument('--csv-predict', default='Predict.csv', type=str,
                    help='Save predict score to csv (default: "Predict.csv")')
parser.add_argument('--num-class', default=2, type=int,
                    help='Number of class (default: 2) ')
parser.add_argument('--batch-size', default=16, type=int,
                    help='Batch size (default: 16)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Learning rate (default: 1e-3)')
parser.add_argument('--weight-decay', default=0.0, type=float,
                    help='weight decay (default: 0.0)')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='momentum (default: 0.0)')
parser.add_argument('--is-scheduler', default=True, type=bool,
                    help='Is scheduler (default: True)')
parser.add_argument('--step-size', default=25, type=int,
                    help='Step size (default: 25)')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma (default: 0.1)')
parser.add_argument('--pretrained', default=False, type=bool,
                    help='Pretrained (ImageNet) (default: False)')
parser.add_argument('--epochs', default=100, type=int,
                    help='Epochs (default: 100)')
parser.add_argument('--mode', default='train', type=str,
                    help='Choose mode for running model (default: train)')
parser.add_argument('--logger-path', default='runs', type=str,
                    help='Logger path (default: runs')
parser.add_argument('--continue-train', default=True, type=bool,
                    help='Continue train model (default: False)')
args = parser.parse_args()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = RunModel(device=device, name=args.name,
                   train_path=args.train_path, val_path=args.val_path, test_path=args.test_path, test_video_path=args.test_video_path, batch_size=args.batch_size,
                   lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                   is_scheduler=args.is_scheduler, step_size=args.step_size, gamma=args.gamma,
                   num_class=args.num_class, pretrained=args.pretrained)

    if args.mode == 'train':
        run.train(args.epochs, args.save_path, args.weight_file, args.logger_path, args.validation, args.continue_train)

    elif args.mode == 'test':
        run.test(args.csv_file, os.path.join(args.save_path, args.weight_file))

    elif args.mode == 'test_video':
        run.test_video(args.csv_predict, os.path.join(args.save_path, args.weight_file))
