def train(model, criterion, iterator, optimizer):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        image = batch["images"].to(device)
        captions = batch["caption"].to(device)

        optimizer.zero_grad()

        result = model(image, captions).view(-1, vocab_size + 1)
        target = captions[:, 1:].reshape(-1)  # disregard the first start token
        loss = criterion(result, target)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if i % 1000 == 0:
            wandb.log({"loss": loss})

    return epoch_loss / (i + 1)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss, bleu_total, meteor_total, rouge_uni_total, rouge_bi_total = 0, 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            image = batch["images"].to(device)
            captions = batch["caption"].to(device)

            result = model(image, captions[:, 0, :], teacher_forcing_ratio=0)

            predicted = result.view(-1, vocab_size + 1)
            target = captions[:, 0, 1:].reshape(-1)  # risregard the first start token

            loss = criterion(predicted, target)

            epoch_loss += loss.item()

            if i % 1000 == 0:
                wandb.log({"loss_validation": loss})

        result = result.argmax(dim=2)
        for references, title in zip(captions, result):
            references = [" ".join(voc.reverse_tokenizer(ref.tolist())) for ref in references]
            title = " ".join(voc.reverse_tokenizer(title.tolist()))
            bleu, meteor, rouge_uni, rouge_bi = metric_count(title, references)

            bleu_total += bleu
            meteor_total += meteor
            rouge_uni_total += rouge_uni
            rouge_bi_total += rouge_bi

        bleu_total /= 64
        meteor_total /= 64
        rouge_uni_total /= 64
        rouge_bi_total /= 64

        wandb.log({"bleu": bleu_total, "meteor": meteor_total, "rouge_unigrams": rouge_uni_total,
                   "rouge_bi_total": rouge_bi_total})

    return epoch_loss / (i + 1), bleu_total, meteor_total, rouge_uni_total, rouge_bi_total


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs