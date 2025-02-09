import argparse
import time
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from Levenshtein import distance as levenshtein_distance


class TextDataset(Dataset):
    def __init__(self, sequences, word_to_idx, unk_idx):
        self.data = []
        for sentence in sequences:
            indices = [word_to_idx.get(word, unk_idx) for word in sentence]
            self.data.append((indices[:-1], indices[1:]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tuple(torch.tensor(x) for x in self.data[idx])


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden


def cleanup_sentence(sentence, token_level="word"):
    if token_level == "word":
        cleaned = (
            sentence.replace(" ,", ",")
            .replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ;", ";")
            .replace(" : ", ": ")
            .replace("( ", "(")
            .replace(" )", ")")
            .replace(" ' s", "'s")
            .replace(" ' t ", "'t ")
            .replace(" ' ve", "'ve")
            .replace(" ' re", "'re")
            .replace(" ' ll", "'ll")
            .replace("”", '"')
            .replace("‘", "'")
            .replace(" '", "'")
            .replace("' ", "'")
            .replace(' "', '"')
            .replace('" ', '"')
        )
    else:
        cleaned = re.sub(r"\s+([.,!?])", r"\1", sentence)

    return cleaned


def find_closest_sentence(generated, candidates):
    min_dist = torch.inf
    closest = ""
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        dist = levenshtein_distance(generated, candidate)
        if dist < min_dist:
            min_dist = dist
            closest = candidate
    return min_dist, closest


def tokenize_line(line, token_level="word", tokenize_punctuation=True):
    if token_level == "char":
        return list(line)
    else:
        # Add spaces to tokenize punctuation as own tokens.
        # At generation time, the cleanup function takes care
        # of removing the extra spaces.
        return (
            line.replace(",", " , ")
            .replace(".", " . ")
            .replace("!", " ! ")
            .replace("?", " ? ")
            .replace(";", " ; ")
            .replace(":", " : ")
            .replace("(", " ( ")
            .replace(")", " ) ")
            .replace('"', ' " ')
            .split()
        )


def generate_text(
    model,
    word_to_idx,
    idx_to_word,
    device,
    token_level="word",
    seed="",
    max_length=50,
    temperature=0.7,
    trim=True,
    trim_delimiters=".!?",
    ban_eos=False,
    scale_eos=1,
    xtc_threshold=0,
    xtc_probability=0,
):
    pad_idx = word_to_idx["<pad>"]
    unk_idx = word_to_idx["<unk>"]
    eos_idx = word_to_idx["<eos>"]
    bos_idx = word_to_idx["<bos>"]

    seed_tokens = tokenize_line(seed, token_level=token_level)
    seed_tokens = ["<bos>"] + seed_tokens
    indices = [word_to_idx.get(w, word_to_idx["<unk>"]) for w in seed_tokens]
    input_seq = torch.tensor([indices], device=device)
    hidden = None
    generated = []

    with torch.no_grad():
        if len(indices) > 1:
            _, hidden = model(input_seq[:, :-1], hidden)
            input_seq = input_seq[:, -1:]

        for _ in range(max_length):
            # Evaluate the model
            logits, hidden = model(input_seq, hidden)

            # Temperature
            logits = logits[0, -1] / temperature

            # Ban <unk>, <pad>
            logits[unk_idx] = -torch.inf
            logits[pad_idx] = -torch.inf

            probs = torch.softmax(logits, dim=-1)

            # Scale or ban <eos>
            if ban_eos:
                logits[eos_idx] = -torch.inf
            elif scale_eos != 1:
                logits[eos_idx] += torch.log(torch.tensor(scale_eos))

            probs = torch.softmax(logits, dim=-1)

            # XTC
            if torch.rand(1) < xtc_probability:
                cutoff = probs.max() * xtc_threshold
                logits[probs > cutoff] = -torch.inf
                probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, 1).item()

            if next_token == eos_idx:
                break

            generated.append(idx_to_word[next_token])
            input_seq = torch.tensor([[next_token]], device=device)

    is_cut_off = len(generated) == max_length

    full_sentence = seed_tokens + generated
    full_sentence = [t for t in full_sentence if t not in ["<bos>", "<eos>"]]

    if token_level == "char":
        full_sentence = "".join(full_sentence).strip()
    else:
        full_sentence = " ".join(full_sentence).strip()

    if is_cut_off and trim:
        last_delimiter_pos = max([full_sentence.rfind(d) for d in trim_delimiters])
        if last_delimiter_pos > -1:
            full_sentence = full_sentence[: last_delimiter_pos + 1]

    return full_sentence


def main():
    parser = argparse.ArgumentParser(description="Word-level LSTM Text Generator")
    parser.add_argument("mode", choices=["train", "generate"])
    parser.add_argument("--input", help="Input text file")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument(
        "--token-level",
        choices=["word", "char"],
        default="word",
        help="Tokenization level (default: word)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", help="Seed text for generation", default="")
    parser.add_argument(
        "--length",
        type=int,
        default=50,
        help="Maximum number of words/characters per generated sentence",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Do not trim incomplete sentences",
    )
    parser.add_argument(
        "--num-sentences", type=int, default=1, help="Number of sentences to generate"
    )
    parser.add_argument(
        "--scale-eos", type=float, default=1, help="Scale the eos probability"
    )
    parser.add_argument(
        "--ban-eos", action="store_true", help="Ban the EOS token during inference"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=1,
        help="XTC sampler threshold (1=disable)",
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=0,
        help="XTC sampler probabilty (0=disable)",
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=None,
        help="Embedding dimension (default depends on token-level)",
    )
    parser.add_argument(
        "--hid-dim",
        type=int,
        default=None,
        help="Hidden dimension (default depends on token-level)",
    )
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=0,
        help="Maximum vocabulary size (0=disable)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=-1,
        help="Maximum sequence length (-1=disable, 0=default (depends on token-level))",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save model every N epochs (0=disable)",
    )
    parser.add_argument(
        "--include-data", action="store_true", help="Store training data with the model"
    )
    parser.add_argument(
        "--compare-file", help="File with texts for similarity comparison"
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=10,
        help="Minimum Levenshtein distance to training data",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Disable output post-processing"
    )
    args = parser.parse_args()

    # Defaults depending on token_level
    if args.token_level == "word":
        default_emb = 256
        default_hid = 512
        default_seq_len = 50  # Words
    else:
        default_emb = 128
        default_hid = 128
        default_seq_len = 200  # Characters

    args.emb_dim = args.emb_dim or default_emb
    args.hid_dim = args.hid_dim or default_hid
    if args.max_seq_len == 0:
        args.max_seq_len = default_seq_len

    if args.mode == "train":
        sentences = []
        sequences = []
        with open(args.input) as file:
            for line in file:
                sentence = line.strip()
                if not sentence:
                    continue
                sentences.append(sentence)

                tokens = tokenize_line(sentence, token_level=args.token_level)

                tokens = tokens[: args.max_seq_len]
                sequences.append(["<bos>"] + tokens + ["<eos>"])

        # Sort tokens by frequency
        token_counts = defaultdict(int)
        for sequence in sequences:
            for token in sequence[1:-1]:  # Skip <bos> and <eos>
                token_counts[token] += 1
        vocab = [
            token
            for (token, count) in sorted(token_counts.items(), key=lambda x: -x[1])
        ]

        if args.max_vocab:
            vocab = vocab[: args.max_vocab]
        vocab += ["<pad>", "<unk>", "<bos>", "<eos>"]
        print(f"Vocab size (including special tokens): {len(vocab)}")
        # print(vocab)

        word_to_idx = {w: i for i, w in enumerate(vocab)}
        idx_to_word = {i: w for i, w in enumerate(vocab)}
        pad_idx = word_to_idx["<pad>"]

        dataset = TextDataset(sequences, word_to_idx, word_to_idx["<unk>"])
        dataloader = DataLoader(
            dataset,
            args.batch,
            shuffle=True,
            collate_fn=lambda b: tuple(pad_sequence(x, True, pad_idx) for x in zip(*b)),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMModel(
            len(vocab), args.emb_dim, args.hid_dim, args.num_layers, pad_idx
        ).to(device)
        opt = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        start_time = time.time()
        for epoch in range(args.epochs):
            model.train()
            opt.zero_grad()

            total_loss = 0
            for i, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
                loss = loss / args.accum_steps
                loss.backward()

                if (i + 1) % args.accum_steps == 0 or (i + 1) == len(dataloader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()

                total_loss += loss.item() * args.accum_steps

            current_loss = total_loss / len(dataloader)
            checkpoint = {
                "epoch": epoch + 1,
                "loss": current_loss,
                "state_dict": model.state_dict(),
                "word_to_idx": word_to_idx,
                "token_level": args.token_level,
                "training_data": [],
                "config": (
                    len(vocab),
                    args.emb_dim,
                    args.hid_dim,
                    args.num_layers,
                    pad_idx,
                ),
            }
            if args.include_data:
                checkpoint["training_data"] = sentences

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(checkpoint, f"{args.model}.epoch{epoch+1}")

            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{args.epochs} | Loss: {current_loss:.4f} | Time: {epoch_time:.1f}s"
            )

            for i in range(3):
                example = generate_text(
                    model=model,
                    word_to_idx=word_to_idx,
                    idx_to_word=idx_to_word,
                    max_length=args.length,
                    token_level=args.token_level,
                    device=device,
                    trim=not args.no_trim,
                )
                if not args.no_cleanup:
                    example = cleanup_sentence(example, args.token_level)
                print(f"Example: {example}")
            print()

            if (epoch + 1) == args.epochs:
                checkpoint["epoch"] = "final"
                torch.save(checkpoint, args.model)
            start_time = time.time()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.model, map_location=device, weights_only=True)

        word_to_idx = checkpoint["word_to_idx"]
        token_level = checkpoint["token_level"]
        idx_to_word = {i: w for w, i in word_to_idx.items()}
        model = LSTMModel(*checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        training_data = []
        if args.min_distance and args.compare_file:
            training_data = open(args.compare_file).read().split("\n")
        elif args.min_distance and "training_data" in checkpoint:
            training_data = checkpoint["training_data"]

        sentences = []
        for _ in range(args.num_sentences):
            generated = generate_text(
                model,
                word_to_idx,
                idx_to_word,
                device,
                seed=args.seed,
                max_length=args.length,
                trim=not args.no_trim,
                token_level=token_level,
                temperature=args.temperature,
                scale_eos=args.scale_eos,
                ban_eos=args.ban_eos,
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
            )

            # Filter sentences that are too similar to the training data
            if args.min_distance and training_data:
                min_dist, closest = find_closest_sentence(generated, training_data)
                if min_dist < args.min_distance:
                    continue

            if not args.no_cleanup:
                generated = cleanup_sentence(generated, token_level)
            sentences.append(generated)

        print("\n".join(filter(lambda x: x, sentences)))


if __name__ == "__main__":
    main()
