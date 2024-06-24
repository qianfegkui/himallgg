import argparse
import torch
import himallgg

from sklearn import metrics
from tqdm import tqdm

log = himallgg.utils.get_logger()

def main(args):
    himallgg.utils.set_seed(args.seed)
    log.debug("Loading data from '%s'." % args.data)
    data = himallgg.utils.load_pkl(args.data)
    log.info("Loaded data.")
    print(data.keys())
    testset = himallgg.Dataset(data["test"], args.batch_size)
    model_file = "./save/IEMOCAP/model1diantest.pt"
    model = himallgg.LGGCN(args).to(args.device)
    pred = himallgg.Prediction(testset, model, args)
    ckpt = torch.load(model_file)
    pred.load_ckpt(ckpt)
    test_f1 = pred.evaluate()
    log.info("[Test set] [f1 {:.4f}]".format(test_f1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prediction.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")

    parser.add_argument("--device", type=str, default="cpu",
                        help="Computing device.")

    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")

    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size.")

    parser.add_argument("--class_weight", action="store_true",
                        help="Use class weights in nll loss.")

    # Model parameters
    parser.add_argument("--drop_rate", type=float, default=0.4,
                        help="Dropout rate.")

    parser.add_argument("--wp", type=int, default=10,
                        help="Past context window size. Set wp to -1 to use all the past context.")

    parser.add_argument("--wf", type=int, default=10,
                        help="Future context window size. Set wp to -1 to use all the future context.")

    parser.add_argument("--n_speakers", type=int, default=2,
                        help="Number of directions.")

    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Hidden size of two layer GCN.")

    parser.add_argument("--rnn", type=str, default="lstm",
                        choices=["lstm", "gru"], help="Type of RNN cell.")

    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")

    parser.add_argument("--scale_para", type=float, default=-0.5,
                        help="attention scale para.")

    args = parser.parse_args()
    log.debug(args)

    main(args)