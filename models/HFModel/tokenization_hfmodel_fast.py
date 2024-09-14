"""Tokenization classes."""

from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_hfmodel import hfmodelTokenizer
from ...loggers.logging_colors import get_logger

logger = get_logger()

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class hfmodelTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" hfmodel tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = hfmodelTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",  # hack to enable padding
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        if not isinstance(self.backend_tokenizer.pre_tokenizer, pre_tokenizers.Sequence):
            raise ValueError(
                "The `backend_tokenizer` provided does not match the expected format. The hfmodel tokenizer has been"
                " heavily modified from transformers version 4.17.0. You need to convert the tokenizer you are using"
                " to be compatible with this version.The easiest way to do so is"
                ' `hfmodelTokenizerFast.from_pretrained("path_to_local_folder_or_hub_repo, from_slow=True)`. If you want'
                " to use your existing tokenizer, you will have to revert to a version prior to 4.17.0 of"
                " transformers."
            )
        self._wrap_decode_method_backend_tokenizer()

    # Very ugly hack to enable padding to have a correct decoding see https://github.com/huggingface/tokenizers/issues/872
    def _wrap_decode_method_backend_tokenizer(self):
        orig_decode_method = self.backend_tokenizer.decode

        ## define this as a local variable to avoid circular reference
        ## See: https://github.com/huggingface/transformers/issues/30930
        end_of_word_suffix = self.backend_tokenizer.model.end_of_word_suffix

        def new_decode_method(*args, **kwargs):
            text = orig_decode_method(*args, **kwargs)
            text = text.replace(end_of_word_suffix, " ").strip()
            return text

        self.backend_tokenizer.decode = new_decode_method

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A hfmodel sequence has the following format:

        - single sequence: `<|startoftext|> X <|endoftext|>`

        Pairs of sequences are not the expected use case, but they will be handled without a separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]

        if token_ids_1 is None:
            return bos_token + token_ids_0 + eos_token
        return bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed. hfmodel does not make use of token type ids, therefore a list of
        zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]

        if token_ids_1 is None:
            return len(bos_token + token_ids_0 + eos_token) * [0]
        return len(bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
