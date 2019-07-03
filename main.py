from utils.config import parse_args
from utils.data_loader import get_data_loader
from models.mutual import MUTUAL

def main(args):

    model = None

    if args.model =='MUTUAL':
        model = MUTUAL(args)

    train_loader, test_loader = get_data_loader(args)

    if args.is_train == 'True':
        model.train(train_loader)
    else:
        model.evaluate(train_loader)

if __name__ == '__main__':
    args = parse_args()
    main(args)

#$ git remote set-url origin  https://github.com/namgil/Deep_Mutual_Learning_pytorch.git
