import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import os
from tqdm import tqdm


# GPT와 관련된 모듈들
class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(
            self.config.d_hidn, eps=self.config.layer_norm_epsilon
        )
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(
            self.config.d_hidn, eps=self.config.layer_norm_epsilon
        )

    def forward(self, dec_inputs, self_attn_mask):
        self_att_outputs, self_attn_prob = self.self_attn(
            dec_inputs, dec_inputs, dec_inputs, self_attn_mask
        )
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        ffn_outputs = self.pos_ffn(self_att_outputs)
        ffn_outputs = self.layer_norm3(self_att_outputs + ffn_outputs)
        return ffn_outputs, self_attn_prob


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(
            get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_hidn)
        )
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList(
            [DecoderLayer(self.config) for _ in range(self.config.n_layer)]
        )

    def forward(self, dec_inputs):
        positions = (
            torch.arange(
                dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype
            )
            .expand(dec_inputs.size(0), dec_inputs.size(1))
            .contiguous()
            + 1
        )
        pos_mask = dec_inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        self_attn_probs = []
        for layer in self.layers:
            dec_outputs, self_attn_prob = layer(dec_outputs, dec_self_attn_mask)
            self_attn_probs.append(self_attn_prob)
        return dec_outputs, self_attn_probs


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = Decoder(self.config)

    def forward(self, dec_inputs):
        dec_outputs, dec_self_attn_probs = self.decoder(dec_inputs)
        return dec_outputs, dec_self_attn_probs

    def save(self, epoch, loss, path):
        torch.save(
            {"epoch": epoch, "loss": loss, "state_dict": self.state_dict()}, path
        )

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]


class GPTPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt = GPT(self.config)
        self.projection_lm = nn.Linear(
            self.config.d_hidn, self.config.n_dec_vocab, bias=False
        )
        self.projection_lm.weight = self.gpt.decoder.dec_emb.weight

    def forward(self, dec_inputs):
        dec_outputs, dec_self_attn_probs = self.gpt(dec_inputs)
        logits_lm = self.projection_lm(dec_outputs)
        return logits_lm[:, :-1, :].contiguous(), dec_self_attn_probs


class PretrainDataSet(Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.sentences = []
        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1
        with open(infile, "r") as f:
            for i, line in enumerate(
                tqdm(f, total=line_cnt, desc="Make Pretrain Dataset", unit=" lines")
            ):
                instance = json.loads(line)
                self.sentences.append(
                    [vocab.piece_to_id(p) for p in instance["tokens"]]
                )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return (torch.tensor(self.sentences[item]), torch.tensor(item))


def pretrin_collate_fn(inputs):
    dec_inputs, item = list(zip(*inputs))
    dec_inputs = torch.nn.utils.rnn.pad_sequence(
        dec_inputs, batch_first=True, padding_value=0
    )
    batch = [dec_inputs, torch.stack(item, dim=0)]
    return batch


def train_epoch(config, epoch, model, criterion_lm, optimizer, train_loader):
    losses = []
    model.train()
    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, value in enumerate(train_loader):
            dec_inputs, _ = map(lambda v: v.to(config.device), value)
            labels_lm = dec_inputs[:, 1:].contiguous()
            optimizer.zero_grad()
            outputs = model(dec_inputs)
            logits_lm = outputs[0]
            loss_lm = criterion_lm(
                logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1)
            )
            loss = loss_lm
            loss_val = loss_lm.item()
            losses.append(loss_val)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)


def main():
    # Config 설정
    config = Config(
        {
            "n_dec_vocab": 8007,
            "n_dec_seq": 256,
            "n_layer": 6,
            "d_hidn": 256,
            "i_pad": 0,
            "d_ff": 1024,
            "n_head": 4,
            "d_head": 64,
            "dropout": 0.1,
            "layer_norm_epsilon": 1e-12,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
    )

    learning_rate = 5e-5
    n_epoch = 20

    # Pretrain 데이터 로드
    vocab = load_vocab()  # 사용할 SentencePiece 모델 로드
    dataset = PretrainDataSet(vocab, "<path_of_data>/chat_data.json")
    train_loader = DataLoader(
        dataset, batch_size=128, shuffle=True, collate_fn=pretrin_collate_fn
    )

    # 모델 초기화
    model = GPTPretrain(config)
    model.to(config.device)

    criterion_lm = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(n_epoch):
        loss = train_epoch(config, epoch, model, criterion_lm, optimizer, train_loader)
        model.save(epoch, loss, "chatbot_model.pth")


if __name__ == "__main__":
    main()
