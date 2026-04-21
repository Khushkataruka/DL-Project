from datasets import load_dataset, Audio, interleave_datasets


# Mapping from ISO-639 codes to the dataset's split names
LANG_CODE_TO_SPLIT = {
    "hi": "hindi",
    "bn": "bengali",
    "te": "telugu",
    "mr": "marathi",
    "gu": "gujarati",
    "as": "assamese",
    "kn": "kannada",
    "ml": "malayalam",
    "or": "odia",
    "pa": "punjabi",
    "ne": "nepali",
    "ta": "tamil",
    "ur": "urdu",
}


def get_processed_streaming_dataset(feature_extractor, tokenizer, target_languages=None):
    """Loads, filters, and prepares the IndicVoices-ST dataset in streaming mode.

    The dataset uses language names as splits (e.g., 'hindi', 'bengali'), not
    a single 'train' split. This function loads each target language split
    separately and interleaves them into a single stream.

    Args:
        feature_extractor: WhisperFeatureExtractor instance
        tokenizer: WhisperTokenizer instance
        target_languages: list of ISO-639 language codes to keep
                          (default: hi, bn, te, mr, gu)

    Returns:
        An IterableDataset with 'input_features' and 'labels' columns
    """
    if target_languages is None:
        # Hindi, Bengali, Telugu, Marathi, Gujarati
        target_languages = ["hi", "bn", "te", "mr", "gu"]

    # Resolve ISO codes to dataset split names
    split_names = []
    for code in target_languages:
        if code in LANG_CODE_TO_SPLIT:
            split_names.append(LANG_CODE_TO_SPLIT[code])
        else:
            print(f"WARNING: Unknown language code '{code}', skipping.")

    print(f"Loading IndicVoices-ST splits: {split_names}")

    # Load each language split as a separate streaming dataset
    datasets_list = []
    for split_name in split_names:
        ds = load_dataset(
            "ai4bharat/IndicVoices-ST",
            split=split_name,
            streaming=True,

        )
        ds = ds.cast_column("chunked_audio_filepath", Audio(sampling_rate=16000))
        datasets_list.append(ds)
        print(f"  ✓ Loaded split: {split_name}")

    # Interleave all language streams into one (round-robin by default)
    if len(datasets_list) == 1:
        combined_dataset = datasets_list[0]
    else:
        combined_dataset = interleave_datasets(datasets_list)

    # Filter out rows where audio or target string is missing/corrupted
    def is_valid_sample(batch):
        return batch.get("chunked_audio_filepath") is not None and batch.get("en_text") is not None
        
    combined_dataset = combined_dataset.filter(is_valid_sample)

    # ── Shuffle for better training (buffer_size controls memory usage) ──
    combined_dataset = combined_dataset.shuffle(seed=42, buffer_size=1000)

    # ── Prepare features ──
    def prepare_dataset(batch):
        audio = batch["chunked_audio_filepath"]

        # Extract log-Mel spectrogram features
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Encode the English translation as target labels
        batch["labels"] = tokenizer(batch["en_text"], max_length=448, truncation=True).input_ids
        return batch

    # remove_columns drops raw heavy columns after featurization
    # First, peek at available columns to build the remove list safely
    processed_dataset = combined_dataset.map(prepare_dataset)
    print("Dataset streaming pipeline ready.")

    return processed_dataset