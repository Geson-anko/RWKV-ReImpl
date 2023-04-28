"""Train the sentencepiece tokenizer on the training data."""
import sentencepiece as spm
from argparse import ArgumentParser
import os


def get_argument_parser() -> ArgumentParser:
    """Create arguments for training the tokenizer."""
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_prefix", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--character_coverage", type=float, default=1.0)
    parser.add_argument("--model_type", type=str, default="unigram")
    parser.add_argument("--pad_id", type=int, default=0)
    parser.add_argument("--unk_id", type=int, default=1)
    parser.add_argument("--bos_id", type=int, default=2)
    parser.add_argument("--eos_id", type=int, default=3)

    return parser


def main():
    """Train the tokenizer."""
    print("Working on:",os.getcwd())
    args = get_argument_parser().parse_args()

    spm.SentencePieceTrainer.Train(
        input=os.path.abspath(args.input),
        model_prefix=os.path.abspath(args.model_prefix),
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        pad_id=args.pad_id,
        unk_id=args.unk_id,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
    )


if __name__ == "__main__":
    main()