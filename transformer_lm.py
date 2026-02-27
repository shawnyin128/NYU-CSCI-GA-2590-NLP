# models.py
import torch
import torch.nn as nn
import math
import numpy as np


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # normalize on hidden dim
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.arange(0, input_size, device=x.device, dtype=torch.long)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, n_heads):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_internal)
        self.k_proj = nn.Linear(d_model, d_internal)
        self.v_proj = nn.Linear(d_model, d_internal)
        self.o_proj = nn.Linear(d_internal, d_model)

        self.up_proj = nn.Linear(d_model, 4 * d_model)
        self.down_proj = nn.Linear(4 * d_model, d_model)

        self.input_layer_norm = RMSNorm(d_model)
        self.post_attention_norm = RMSNorm(d_model)

        torch.nn.init.kaiming_uniform_(self.q_proj.weight)
        torch.nn.init.kaiming_uniform_(self.k_proj.weight)
        torch.nn.init.kaiming_uniform_(self.v_proj.weight)
        torch.nn.init.kaiming_uniform_(self.o_proj.weight)
        torch.nn.init.kaiming_uniform_(self.up_proj.weight)
        torch.nn.init.kaiming_uniform_(self.down_proj.weight)

    def forward(self, input_vecs):
        N, D = input_vecs.size()
        res = input_vecs
        normed = self.input_layer_norm(input_vecs)
        q_state = self.q_proj(normed) # [N, d_internal]
        k_state = self.k_proj(normed) # [N, d_internal]
        v_state = self.v_proj(normed) # [N, d_internal]

        # split into multihead
        q_state = q_state.view(N, self.n_heads, self.head_dim).transpose(0, 1)
        k_state = k_state.view(N, self.n_heads, self.head_dim).transpose(0, 1)
        v_state = v_state.view(N, self.n_heads, self.head_dim).transpose(0, 1)

        attn_score = q_state @ k_state.transpose(-2, -1)
        attn_score = attn_score / math.sqrt(self.d_internal)

        # add causal mask
        causal_mask = torch.triu(torch.ones(N, N, device=input_vecs.device), diagonal=1).bool()
        attn_score = attn_score.masked_fill(causal_mask, float('-inf'))

        attn_weight = nn.functional.softmax(attn_score, dim=-1)
        attn_output = attn_weight @ v_state

        # merge multihead back
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, self.d_internal)

        output = self.o_proj(attn_output) + res # [N, d_model]

        res = output
        normed = self.post_attention_norm(output)
        up_state = self.up_proj(normed)
        up_state = nn.functional.silu(up_state)
        down_state = self.down_proj(up_state)
        output = down_state + res

        return output, attn_weight


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_index, vocab_size, num_positions, d_model, d_internal, n_heads, num_classes, num_layers):
        self.vocab_index = vocab_index
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=False)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_internal=d_internal, n_heads=n_heads) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        if indices.dim() == 2:
            indices = indices.squeeze(0)
        x = self.tok_emb(indices)
        x = self.pos_emb(x)  # [seq len, embedding dim]
        attn_weight_list = []
        for layer in self.layers:
            x, attn_weight = layer(x)  # [seq len, embedding dim], [n_heads, seq len, seq len]
            attn_weight_list.append(attn_weight.squeeze())
        logits = self.lm_head(x)
        prob = nn.functional.log_softmax(logits, dim=-1)
        return prob, attn_weight_list

    def get_next_char_log_probs(self, context):
        if len(context) == 0:
            context = " "

        with torch.no_grad():
            max_len = self.pos_emb.emb.num_embeddings
            device = self.tok_emb.weight.device
            indices = torch.tensor(
                [self.vocab_index.index_of(c) for c in context[-max_len:]],
                dtype=torch.long,
                device=device
            )
            log_probs, _ = self.forward(indices)
            return log_probs[-1].detach().cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0
        cur_context = context
        for c in next_chars:
            c_idx = self.vocab_index.index_of(c)
            if c_idx < 0:
                raise ValueError("Character %r is outside the vocabulary" % c)
            next_char_log_probs = self.get_next_char_log_probs(cur_context)
            total_log_prob += float(next_char_log_probs[c_idx])
            cur_context += c
        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    torch.manual_seed(42)
    np.random.seed(42)

    vocab_size = len(vocab_index)
    model = NeuralLanguageModel(
        vocab_index=vocab_index,
        vocab_size=vocab_size,
        num_positions=20,
        d_model=128,
        d_internal=128,
        n_heads=4,
        num_classes=vocab_size,
        num_layers=4
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.tok_emb = model.tok_emb.to(device)
    model.pos_emb = model.pos_emb.to(device)
    model.layers = model.layers.to(device)
    model.lm_head = model.lm_head.to(device)

    params = (
        list(model.tok_emb.parameters()) +
        list(model.pos_emb.parameters()) +
        list(model.layers.parameters()) +
        list(model.lm_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    loss_fcn = nn.NLLLoss()

    train_ids = [vocab_index.index_of(c) for c in train_text]
    chunk_len = 20
    sos_idx = vocab_index.index_of(" ")
    num_epochs = 10

    for epoch in range(num_epochs):
        starts = list(range(0, len(train_ids) - chunk_len + 1, chunk_len))
        np.random.shuffle(starts)
        total_loss = 0.0

        for s in starts:
            target_ids = train_ids[s:s + chunk_len]
            input_ids = [sos_idx] + target_ids[:-1]

            x = torch.tensor(input_ids, dtype=torch.long)
            y = torch.tensor(target_ids, dtype=torch.long)
            x = x.to(device)
            y = y.to(device)

            log_probs, _ = model.forward(x)
            loss = loss_fcn(log_probs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(starts))
        # print("Epoch %d/%d, avg train NLL: %.4f" % (epoch + 1, num_epochs, avg_loss))

    return model
