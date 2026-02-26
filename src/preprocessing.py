from transformers import DistilBertTokenizerFast
import torch

def get_tokenizer(model_name="distilbert-base-uncased"):
    """Initialize and return the fast tokenizer."""
    return DistilBertTokenizerFast.from_pretrained(model_name)

def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Tokenize examples and map answer character positions to token positions.
    Handles long contexts using a sliding window (stride).
    """
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride.
    # This results in one example potentially giving several features when a context is long.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a mapping from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mapping will help us compute the start and end indices
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that feature (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

if __name__ == "__main__":
    # Quick test with a sample
    from datasets import load_dataset
    dataset = load_dataset("squad", split="train[:5]")
    tokenizer = get_tokenizer()
    
    features = prepare_train_features(dataset[:5], tokenizer)
    
    print(f"Number of features generated: {len(features['input_ids'])}")
    for i in range(len(features['input_ids'])):
        start = features['start_positions'][i]
        end = features['end_positions'][i]
        input_ids = features['input_ids'][i]
        
        # Decode the span
        answer_tokens = input_ids[start:end+1]
        decoded_answer = tokenizer.decode(answer_tokens)
        
        print(f"\nFeature {i}:")
        print(f"Start pos: {start}, End pos: {end}")
        print(f"Decoded Answer: {decoded_answer}")
