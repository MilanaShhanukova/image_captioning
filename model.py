import random
import torch, torch.nn as nn
import torch.nn.functional as F

class CaptionModel(nn.Module):
    def __init__(self, vocab_size, decoder_dim, embed_dim, cnn_feature_size=2048):
        super(self.__class__, self).__init__()

        # линейные для преобразования эмбеддинга картинки в начальные состояния h0 и c0
        self.init_h = nn.Linear(cnn_feature_size, decoder_dim)
        self.init_c = nn.Linear(cnn_feature_size, decoder_dim)

        # слой эмбеддинга для target caption
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)

        self.clf = nn.Linear(decoder_dim, vocab_size + 1)
        self.drop = nn.Dropout()

        self.lstm_cell = nn.LSTMCell(embed_dim, decoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size + 1)

    def forward(self, image_vectors, captions, teacher_forcing_ratio=0.75):
        h0, c0 = self.init_h(image_vectors), self.init_c(image_vectors)
        embedded_caption = self.drop(self.embed(captions))

        seq_length = len(captions[0]) - 1
        batch_size = captions.size(0)
        num_features = image_vectors.size(1)
        # fill all of the preds
        preds = torch.zeros(batch_size, seq_length, vocab_size + 1).to(device)

        # iterate through words
        for s in range(seq_length):
            teacher_force = random.random() < teacher_forcing_ratio
            if not teacher_force and s != 0:
                s = torch.tensor(top1).view(1, -1).to(device)
                word = self.embed(top1)
            else:
                word = embedded_caption[:, s]
            h0, c0 = self.lstm_cell(word, (h0, c0))

            output = self.fcn(self.drop(h0))
            preds[:, s] = output  # decision for each word

            # top1 predicted word
            top1 = output.argmax(-1)
        return preds

    def generate_caption(self, image, caption_prefix=(voc.word_to_idx["<START>"]), t=1, sample=True, max_len=10):
        assert isinstance(image, np.ndarray) and np.max(image) <= 1 \
               and np.min(image) >= 0 and image.shape[-1] == 3
        with torch.no_grad():
            image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)

            vectors_8x8, vectors_neck, logits = inception(image[None])
            caption_prefix = [caption_prefix]
            vectors_neck = vectors_neck.to(device)

            batch_size = vectors_neck.size(0)

            h0, c0 = self.init_h(vectors_neck), self.init_c(vectors_neck)

            word = torch.tensor(caption_prefix).view(1, -1).to(device)
            embeds = self.embed(word)
            result = []

            for i in range(max_len):
                h0, c0 = self.lstm_cell(embeds[:, 0], (h0, c0))

                output = self.fcn(self.drop(h0))
                output = output.view(batch_size, -1)

                predicted_word_idx = output.argmax(dim=1)
                result.append(predicted_word_idx.item())

                if voc.idx_to_word[predicted_word_idx.item()] == "<END>":
                    break

                embeds = self.embed(predicted_word_idx.view(1, -1))
            return voc.reverse_tokenizer(result)


class CaptionModelAttention(nn.Module):
    def __init__(self, vocab_size, decoder_dim, embed_dim, attention_dim, cnn_feature_size=2048):
        super(self.__class__, self).__init__()

        self.attention = Attention(decoder_dim, cnn_feature_size, attention_dim)

        # linear layers for initial image vectors h0 и c0
        self.init_h = nn.Linear(cnn_feature_size, decoder_dim)
        self.init_c = nn.Linear(cnn_feature_size, decoder_dim)

        # embedding layer for target caption
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)
        self.drop = nn.Dropout()

        # important that we use embedding dimension + decoder_dim in such mode with attention
        self.lstm_cell = nn.LSTMCell(embed_dim + cnn_feature_size, decoder_dim)

        # final classification layer
        self.fcn = nn.Linear(decoder_dim, vocab_size + 1)

    def forward(self, image_vectors, captions, teacher_forcing_ratio=0.75):
        h0, c0 = self.init_h(image_vectors), self.init_c(image_vectors)
        embedded_caption = self.drop(self.embed(captions))

        seq_length = len(captions[0]) - 1  # we disconnect the ending
        batch_size = captions.size(0)  # needed for prediction tensor

        # fill all of the preds
        preds = torch.zeros(batch_size, seq_length, vocab_size + 1).to(device)

        # iterate through words
        for s in range(seq_length):
            # teacher forcing choosing target word or predicted one
            teacher_force = random.random() < teacher_forcing_ratio
            if not teacher_force and s != 0:
                word = self.embed(top1)
                word = word.squeeze(0)
            else:
                word = embedded_caption[:, s]

            contexed = self.attention(image_vectors, h0)
            lstm_input = torch.cat((word, contexed), dim=1)

            h0, c0 = self.lstm_cell(lstm_input, (h0, c0))

            output = self.fcn(self.drop(h0))
            preds[:, s] = output  # decision for each word

            # top1 predicted word
            top1 = output.argmax(-1)
        return preds

    def generate_caption(self, image, caption_prefix=(voc.word_to_idx["<START>"]), t=1, sample=True, max_len=10):
        assert isinstance(image, np.ndarray) and np.max(image) <= 1 \
               and np.min(image) >= 0 and image.shape[-1] == 3

        with torch.no_grad():
            image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)

            vectors_8x8, vectors_neck, logits = inception(image[None])
            caption_prefix = [caption_prefix]
            vectors_neck = vectors_neck.to(device)

            batch_size = vectors_neck.size(0)

            h0, c0 = self.init_h(vectors_neck), self.init_c(vectors_neck)

            word = torch.tensor(caption_prefix).view(1, -1).to(device)
            embeds = self.embed(word)
            result = []

            for _ in range(max_len):
                h0, c0 = self.lstm_cell(embeds[:, 0], (h0, c0))

                output = self.fcn(self.drop(h0))
                output = output.view(batch_size, -1)

                predicted_word_idx = output.argmax(dim=1)
                result.append(predicted_word_idx.item())

                if voc.idx_to_word[predicted_word_idx.item()] == "<END>":
                    break

                embeds = self.embed(predicted_word_idx.view(1, -1))
            title = " ".join(voc.reverse_tokenizer(result))
            return title


class Attention(nn.Module):
    def __init__(self, dec_dim, pictures_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim

        self.W = nn.Linear(2048, attention_dim)
        self.U = nn.Linear(dec_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        w_hs = self.W(features.unsqueeze(1))
        u_ah = self.U(hidden_state)

        combined_states = torch.tanh(w_hs + u_ah.unsqueeze(1))
        attention_scores = self.A(combined_states).squeeze(2)

        alpha = F.softmax(attention_scores, dim=1)
        attention_weights = alpha * features
        attention_weights = attention_weights.unsqueeze(1).sum(dim=1)
        return attention_weights