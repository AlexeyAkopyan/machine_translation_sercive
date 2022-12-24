from __future__ import print_function
import argparse
import logging
from typing import cast, Tuple, List

import torch
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from torch.nn import functional as F

from preprocess import pad_sequences, tokenize, detokenize
from src.preprocessing import data_utils
from train import load_checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from cli import translate, correct_n_word

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8.8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def start_handler(update, context) -> None:
    update.message.reply_text("Welcome to Machine Translation Bot! "
                              "The bot allows you to translate texts from Russian to English. "
                              "In addition, you can correct translation by selecting more "
                              "appropriate words in the target text. "
                              "To translate a text, just send it without any command. "
                              "To correct the last translated sentence, enter the /correct command.")


def translate_handler(update, context, **kwargs):
    message = update.message.reply_text("Translating ...")
    context.user_data["src_text"] = update.message.text
    logger.info(f"User: {update.effective_user['username']}. Source: {context.user_data['src_text']}")
    trg_text, src_token_ids, trg_token_ids = translate(text=context.user_data["src_text"], **kwargs)
    context.user_data["trg_text"] = trg_text
    context.user_data["src_token_ids"] = src_token_ids
    context.user_data["trg_token_ids"] = trg_token_ids
    context.bot.editMessageText(context.user_data["trg_text"], message_id=message.message_id,
                                chat_id=update.message.chat_id)
    logger.info(f"User: {update.effective_user['username']}. Target: {context.user_data['trg_text']}")


def correct_handler(update, context, tokenizer):
    if "trg_token_ids" not in context.user_data:
        update.message.reply_text("Nothing to correct ☹️. Please translate a sentence first")
        logger.info(f"User: {update.effective_user['username']}. Nothing to correct.")
    else:
        trg_tokens = tokenizer.convert_ids_to_tokens(context.user_data["trg_token_ids"][1:-1])
        context.user_data["token_choice"] = True
        update.message.reply_text("Please choose token to replace:", reply_markup=build_token_list(trg_tokens))


def build_token_list(trg_tokens) -> InlineKeyboardMarkup:

    return InlineKeyboardMarkup.from_column(
        [InlineKeyboardButton(token, callback_data=str(idx))
         for idx, token in enumerate(trg_tokens, start=1)]
    )


def build_available_replacement(replacement_tokens, selected_idx):
    return InlineKeyboardMarkup.from_column(
        [InlineKeyboardButton(token, callback_data=str(idx)) for idx, token in enumerate(replacement_tokens)]
    )


def list_tokens(update, context, tokenizer, **kwargs) -> None:
    if context.user_data["token_choice"]:
        query = update.callback_query
        query.answer()
        word_to_change = int(query.data)
        corrected = correct_n_word(src_tokens=context.user_data["src_token_ids"],
                                   trg_tokens_start=context.user_data["trg_token_ids"][:word_to_change + 1],
                                   **kwargs)
        context.user_data["corrected"] = corrected
        replacement_tokens = [tokenizer.decode(sent[word_to_change]) for sent in corrected]
        context.user_data["token_choice"] = False
        query.edit_message_text(
            text=f"Please select preferred replacement for token "
                 f"{tokenizer.convert_ids_to_tokens(context.user_data['trg_token_ids'][word_to_change].view(-1))[0]}.",
            reply_markup=build_available_replacement(replacement_tokens, word_to_change),
        )
    else:
        query = update.callback_query
        query.answer()
        selected = int(query.data)
        context.user_data["trg_token_ids"] = context.user_data["corrected"][selected]
        context.user_data["trg_text"] = tokenizer.decode(context.user_data["trg_token_ids"][1:-1])
        query.edit_message_text(text=context.user_data["trg_text"])
        logger.info(f"User: {update.effective_user['username']}. Corrected: {context.user_data['trg_text']}")


def main(cfg):
    # _, model = load_checkpoint(cfg.model_path, load_optimizer=False, load_scheduler=False)
    # device = torch.device("cuda") if cfg.use_cuda else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    device = torch.device("cuda")
    max_len = 64
    n_beams = 5
    n_suggestions = 5
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en").to(device)
    updater = Updater(cfg.telegram_token, use_context=True)
    # updater.dispatcher.add_handler(CommandHandler("translate", lambda u, c: telegram_translate(
    #         u, c, model=model, src_tokenizer_path=cfg.src_tokenizer_path, trg_tokenizer_path=cfg.trg_tokenizer_path,
    #         device=device, max_len=cfg.max_seq_len)))
    updater.dispatcher.add_handler(CommandHandler("start", start_handler))
    updater.dispatcher.add_handler(CommandHandler("correct", lambda u, c: correct_handler(u, c, tokenizer=tokenizer)))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, lambda u, c: translate_handler(
            u, c, model=model, tokenizer=tokenizer, device=device, max_len=max_len)))

    updater.dispatcher.add_handler(
        CallbackQueryHandler(lambda u, c: list_tokens(u, c, tokenizer=tokenizer, model=model, n=n_suggestions,
                                                      max_len=max_len, k=n_beams, device=device)))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation hyperparams')
    parser.add_argument('--telegram-token', required=True, type=str, help='Telegram bot token')
    # parser.add_argument('--model-path', required=True, type=str, help='Path to the test data')
    # parser.add_argument('--max-seq-len', type=int, default=100,
    #                     help='Maximum len of sentence to generate. Counted in subwords')
    # parser.add_argument('--src-tokenizer-path', required=True, type=str, help='Path to source tokenizer model')
    # parser.add_argument('--trg-tokenizer-path', required=True, type=str, help='Path to target tokenizer model')
    parser.add_argument('--use-cuda', type=bool, default=False)
    cfg = parser.parse_args()
    logger.info(cfg)
    main(cfg)
    logger.info('Terminated')
