import os
import json
import torch
import librosa
import re
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

class IndicSpeechTranslator:
    def __init__(self, base_model_id="openai/whisper-small", lora_path="./whisper-indic-finetuned-final", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        print("Loading Whisper processor...")
        self.processor = WhisperProcessor.from_pretrained(lora_path)
        print("Loading base Whisper model + LoRA adapter...")
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model ready.")

    def process_audio_array(self, audio_array, sr=16000):
        # Audio array directly from datasets library (16khz)
        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        
        english_translation = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        raw_decoded = self.processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
        detected_lang = self._extract_lang_tag(raw_decoded)
        
        return {
            "detected_language": detected_lang,
            "english_translation": english_translation,
        }
        
    @staticmethod
    def _extract_lang_tag(text):
        match = re.search(r'<\|(.*?)\|>', text)
        return match.group(1) if match else "unknown"

if __name__ == "__main__":
    import evaluate
    language_splits = ["hindi", "bengali", "telugu", "marathi", "gujarati"]
    samples_per_lang = 3
    output_file = "streaming_eval_results.json"
    
    translator = IndicSpeechTranslator()
    results = []
    
    # Collect all predictions and ground truths for metrics
    all_predictions = []
    all_references = []
    
    # Load translation metrics safely
    try:
        metric_bleu = evaluate.load("sacrebleu")
        metric_wer = evaluate.load("wer")
    except Exception as e:
        print(f"Warning: Could not load metric packages natively ({e}). Ensure 'evaluate', 'sacrebleu', and 'jiwer' are installed.")
        metric_bleu, metric_wer = None, None
    
    for lang in language_splits:
        print(f"\nLoading '{lang}' streaming dataset for inference...")
        ds = load_dataset("ai4bharat/IndicVoices-ST", split=lang, streaming=True)
        ds = ds.cast_column("chunked_audio_filepath", Audio(sampling_rate=16000))
        
        print(f"Evaluating first {samples_per_lang} ground-truth samples from {lang}...")
        
        lang_count = 0
        for sample in ds:
            if lang_count >= samples_per_lang:
                break
                
            # Filter out corrupted rows
            if not sample.get("chunked_audio_filepath") or not sample.get("en_text"):
                continue
                
            lang_count += 1
            print(f"\n--- {lang.capitalize()} Sample {lang_count}/{samples_per_lang} ---")
            audio_data = sample["chunked_audio_filepath"]["array"]
            target_translation = sample["en_text"]
            
            # Run our fine-tuned LoRA prediction
            pred = translator.process_audio_array(audio_data)
            
            print(f"Detected Lang : {pred['detected_language']}")
            print(f"Actual (Truth): {target_translation}")
            print(f"Model Predict : {pred['english_translation']}")
            
            all_predictions.append(pred['english_translation'])
            all_references.append(target_translation)
            
            results.append({
                "sample_index": len(results) + 1,
                "true_language": lang,
                "detected_language": pred["detected_language"],
                "ground_truth": target_translation,
                "prediction": pred["english_translation"]
            })
            
    # Calculate Final Metrics
    print("\n" + "="*50)
    print("FINISHED ALL TRANSLATIONS. CALCULATING OVERALL METRICS...")
    
    final_metrics = {}
    if metric_bleu and metric_wer and len(all_predictions) > 0:
        # Compute SacreBLEU (needs references wrapped in lists)
        try:
            bleu_score = metric_bleu.compute(predictions=all_predictions, references=[[r] for r in all_references])
            final_metrics["sacre_bleu"] = bleu_score["score"]
            print(f"Overall SacreBLEU Score : {bleu_score['score']:.2f}")
        except Exception as e:
            print(f"Failed to calculate BLEU: {e}")
            
        # Compute Word Error Rate (WER)
        try:
            wer_score = metric_wer.compute(predictions=all_predictions, references=all_references)
            final_metrics["word_error_rate"] = wer_score
            print(f"Overall WER (Word Error Rate) : {wer_score:.4f}")
        except Exception as e:
            print(f"Failed to calculate WER: {e}")
    print("="*50)
            
    # Save everything to JSON
    output_data = {
        "final_metrics": final_metrics,
        "sample_translations": results
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nDone! Benchmark results and translations saved to {output_file}")
