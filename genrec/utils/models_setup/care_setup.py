from transformers import T5Config, T5ForConditionalGeneration

from genrec.models.LETTER.LETTER_CARE import LETTERCARE
from genrec.models.TIGER.TIGER_CARE import TIGERCARE


def create_care_tiger_model(vocab_size: int, model_config: dict) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=vocab_size,
        d_model=model_config["d_model"],
        d_kv=model_config["d_kv"],
        d_ff=model_config["d_ff"],
        num_layers=model_config["num_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        num_heads=model_config["num_heads"],
        dropout_rate=model_config["dropout_rate"],
        tie_word_embeddings=model_config["tie_word_embeddings"],
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    config.care_sid_length = int(model_config.get("care_sid_length", 4))
    config.care_query_numbers = model_config.get("care_query_numbers", [1, 1, 4, 4])
    config.care_alpha = float(model_config.get("care_alpha", 0.7))
    return TIGERCARE(config)


def create_care_letter_model(vocab_size: int, model_config: dict) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=vocab_size,
        d_model=model_config["d_model"],
        d_kv=model_config["d_kv"],
        d_ff=model_config["d_ff"],
        num_layers=model_config["num_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        num_heads=model_config["num_heads"],
        dropout_rate=model_config["dropout_rate"],
        tie_word_embeddings=model_config["tie_word_embeddings"],
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        tau=model_config["tau"],
    )
    config.care_sid_length = int(model_config.get("care_sid_length", 4))
    config.care_query_numbers = model_config.get("care_query_numbers", [1, 1, 4, 4])
    config.care_alpha = float(model_config.get("care_alpha", 0.7))
    return LETTERCARE(config)
