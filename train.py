import torch
from transformers import (
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperProcessor,
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model

from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from data_preprocessing import get_processed_streaming_dataset

# 1. Initialize Whisper Processor
model_id = "openai/whisper-small"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="hindi", task="transcribe")
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# 2. Get Processed Streaming Dataset
processed_dataset = get_processed_streaming_dataset(feature_extractor, tokenizer)

# 3. Model & LoRA Config
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 4. Training Configuration
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-indic-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4, 
    learning_rate=1e-4,                 # 1e-3 was too high for LoRA — causes instability
    warmup_steps=200,                   # Gradual LR warmup prevents early divergence
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="no",                 # 'evaluation_strategy' is deprecated
    predict_with_generate=True,
    logging_steps=50,
    save_steps=1000,
    remove_unused_columns=False,        # REQUIRED for IterableDataset / streaming
    dataloader_pin_memory=False,        # Avoids issues with streaming datasets
    report_to="none",                   # Disable W&B / MLflow unless configured
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. Initialize Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=processed_dataset,
    data_collator=data_collator,
    processing_class=processor,         # Replaces deprecated `tokenizer=` parameter
)

print("Starting streaming training. This will take time depending on your GPU...")
trainer.train()

# 6. Save final artifacts
model.save_pretrained("./whisper-indic-finetuned-final")
processor.save_pretrained("./whisper-indic-finetuned-final")
print("Training complete! Model saved to ./whisper-indic-finetuned-final")