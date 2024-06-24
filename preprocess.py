import argparse

from tqdm import tqdm
import pickle

import himallgg

log = himallgg.utils.get_logger()


def split():
    himallgg.utils.set_seed(args.seed)

    video_ids, video_speakers, video_labels, video_text, \
        video_audio, video_visual, video_sentence, trainVids, \
        test_vids = pickle.load(open('data/iemocap/IEMOCAP_features1.pkl', 'rb'), encoding='latin1')

    [speakers, labels, roberta1, roberta2, roberta3, roberta4, sentences, train_ids, test_ids, valid_ids]\
    = pickle.load(open('data/iemocap/iemocap_features_roberta.pkl','rb'))

    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    trainVids = list(trainVids)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(himallgg.Sample(vid, video_speakers[vid], video_labels[vid],
                                 video_text[vid], video_audio[vid], video_visual[vid],
                                 video_sentence[vid]))
        # print(train)
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(himallgg.Sample(vid, video_speakers[vid], video_labels[vid],
                               video_text[vid], video_audio[vid], video_visual[vid],
                               video_sentence[vid]))
    for vid in tqdm(test_vids, desc="test"):
        test.append(himallgg.Sample(vid, video_speakers[vid], video_labels[vid],
                                video_text[vid], video_audio[vid], video_visual[vid],
                                video_sentence[vid]))
    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test


def main(args):
    train, dev, test = split()
    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))
    data = {"train": train, "dev": dev, "test": test}
    himallgg.utils.save_pkl(data, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["iemocap", "avec", "meld"],
                        help="Dataset name.")
    parser.add_argument("--seed", type=int, default=1024,
                        help="Random seed.")
    args = parser.parse_args()

    main(args)
