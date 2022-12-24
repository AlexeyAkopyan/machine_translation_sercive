from __future__ import print_function
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8.8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def beam_search(model, max_len, k, src_tokens, device, trg_tokens_start=None):
    model.eval()
    with torch.no_grad():
        if trg_tokens_start is None:
            trg_tokens_start = torch.LongTensor([[62517]])
        src_tokens = src_tokens.view(-1, 1)
        trg_pred_tensor = trg_tokens_start.view(-1, 1).T.to(device)

        softmax = torch.nn.Softmax(dim=-1)
        output = softmax(model(input_ids=src_tokens.T, decoder_input_ids=trg_pred_tensor).logits[:, -1])
        ones = torch.ones(k, 1).to(device)
        topk = torch.topk(output, k)
        trg_pred_tensor = torch.concat(((ones @ trg_pred_tensor.float()), topk.indices.T), dim=1).T
        src_tokens = src_tokens.float() @ ones.T
        probas = topk.values

        trg_tokens = []
        for i in range(max_len):
            output = softmax(model(input_ids=src_tokens.T.long(), decoder_input_ids=trg_pred_tensor.T.long()).logits[:, -1])
            topk = torch.topk(output, k, dim=-1)

            ones = torch.ones(k, 1).to(device)
            probas = (probas.unsqueeze(-1) @ ones.T * topk.values).view(1, k * k)
            probas, indices = torch.topk(probas, k)
            trg_pred_tensor = (trg_pred_tensor.unsqueeze(-1) @ ones.T).view(-1, k * k)
            trg_pred_tensor = torch.concat((trg_pred_tensor, topk.indices.view(1, k * k)), dim=0)
            trg_pred_tensor = trg_pred_tensor[:, indices[0]]

            stop_mask = trg_pred_tensor[-1] == 0
            if stop_mask.any():
                trg_tokens.extend(list(trg_pred_tensor[:, stop_mask].T.long()))
                trg_pred_tensor = trg_pred_tensor[:, ~stop_mask]
                src_tokens = src_tokens[:, ~stop_mask]
                probas = probas[:, ~stop_mask]
                k = trg_pred_tensor.size(1)
                if k == 0:
                    break
    return trg_tokens


def correct_n_word(model, n, max_len, k, src_tokens, device, trg_tokens_start=None):

    output = torch.nn.Softmax(dim=-1)(model(input_ids=src_tokens.view(1, -1),
                                     decoder_input_ids=trg_tokens_start.view(1, -1)[:, :-1]).logits[:, -1])
    curr_next_token = trg_tokens_start.view(1, -1)[:, -1].item()
    suggested_next_tokens = torch.topk(output, k + 1).indices[0]

    suggested_translations = []
    for next_token in [token for token in suggested_next_tokens if token != curr_next_token]:
        trg_tokens_start = torch.concat(
            (trg_tokens_start.view(1, -1)[:, :-1], torch.LongTensor([[next_token]]).to(device)
             ), dim=-1)
        suggested_translations.append(beam_search(
            model, max_len, k, src_tokens, device, trg_tokens_start=trg_tokens_start
        )[0])
    return suggested_translations


def translate(model, text, tokenizer, device, max_len):
    src_tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    trg_tokens = beam_search(model, max_len, 5, src_tokens, device)[0]
    return tokenizer.decode(trg_tokens[1:-1]), src_tokens, trg_tokens


def main():

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    device = torch.device("cuda")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en").to(device)
    max_len = 64
    n_beams = 5
    n_suggestions = 5
    src_text = input("Write sentence in Russian to translate: ")
    trg_text, src_token_ids, trg_token_ids = translate(model, src_text, tokenizer, device, max_len)
    print("Translation: ", trg_text)
    while True:
        print("Enter 1 to translate new sentence, 2 to correct just translated sentence, 3 to exit.")
        next_step = int(input("Your choice: "))
        if next_step == 1:
            src_text = input("Write sentence in Russian to translate:")
            trg_text, src_token_ids, trg_token_ids = translate(model, src_text, tokenizer, device, max_len)
            print("Translation: ", trg_text)
        elif next_step == 2:
            print(list(enumerate(tokenizer.convert_ids_to_tokens(trg_token_ids[1:-1]), start=1)))
            word_to_change = int(input("Select index of word to change: "))
            corrected = correct_n_word(model=model, n=n_suggestions, max_len=max_len, k=n_beams, device=device,
                                       src_tokens=src_token_ids, trg_tokens_start=trg_token_ids[:word_to_change + 1])
            print(list(enumerate([tokenizer.decode(sent[word_to_change]) for sent in corrected], start=1)))
            selected_suggestion = int(input("Select index of replacing word: ")) - 1
            trg_tokens_ids = corrected[selected_suggestion]
            print("Translation: ", tokenizer.decode(trg_tokens_ids[1:-1]))
        elif next_step == 3:
            break


if __name__ == '__main__':
    main()


