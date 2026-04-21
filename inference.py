import torch
import librosa
import re
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
)
from peft import PeftModel


class IndicSpeechTranslator:
    """Translates Indic-language speech to English using a fine-tuned Whisper model.

    The Whisper model was trained on IndicVoices-ST 'indic2en' targets, so it
    directly outputs English text from Indic audio — no separate translation
    model is needed.
    """

    def __init__(
        self,
        base_model_id="openai/whisper-small",
        lora_path="./whisper-indic-finetuned-final",
        device=None,
    ):
        # Auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load processor (feature extractor + tokenizer)
        print("Loading Whisper processor...")
        self.processor = WhisperProcessor.from_pretrained(lora_path)

        # Load base model, then apply the LoRA adapter on top
        print("Loading base Whisper model + LoRA adapter...")
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.to(self.device)
        self.model.eval()

        print("Model ready.")

    def process_audio(self, audio_path):
        """Transcribe/translate a single audio file.
        Args:
            audio_path: Path to an audio file (wav, mp3, flac, etc.)
        Returns:
            dict with 'detected_language' and 'english_translation'
        """
        # 1. Load and resample audio to 16 kHz
        audio_array, sr = librosa.load(audio_path, sr=16000)

        # 2. Extract features
        inputs = self.processor(
            audio_array, sampling_rate=sr, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        # 3. Generate with Whisper (output is English, since trained on indic2en)
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        # 4. Decode
        english_translation = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        # Try to extract detected language tag from raw decode
        raw_decoded = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=False
        )[0]
        detected_lang = self._extract_lang_tag(raw_decoded)

        return {
            "detected_language": detected_lang,
            "english_translation": english_translation,
        }
        
    @staticmethod
    def _extract_lang_tag(text):
        """Extract language tag like <|hi|> from Whisper's raw output."""
        match = re.search(r'<\|(.*?)\|>', text)
        return match.group(1) if match else "unknown"


if __name__ == "__main__":
    import os
    import json
    
    test_dir = "test_audio_dir"
    output_file = "translations_output.json"
    
    if not os.path.exists(test_dir):
        print(f"Error: Please create a directory named '{test_dir}' and place your audio files (.wav, .mp3) there.")
        os.makedirs(test_dir, exist_ok=True)
    else:
        translator = IndicSpeechTranslator()
        
        valid_extensions = ('.wav', '.mp3', '.flac', '.m4a')
        audio_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_extensions)]
        
        if not audio_files:
            print(f"No valid audio files found in '{test_dir}'.")
        else:
            results_list = []
            print(f"\nFound {len(audio_files)} audio files. Starting processing...")
            
            for file_name in audio_files:
                file_path = os.path.join(test_dir, file_name)
                print(f"\nProcessing: {file_name}...")
                
                try:
                    result = translator.process_audio(file_path)
                    
                    data = {
                        "file_name": file_name,
                        "detected_language": result["detected_language"],
                        "english_translation": result["english_translation"]
                    }
                    results_list.append(data)
                    
                    print(f"  -> Lang: {result['detected_language']}")
                    print(f"  -> Text: {result['english_translation']}")
                    
                except Exception as e:
                    print(f"  -> Error processing {file_name}: {e}")
            
            # Save all to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_list, f, ensure_ascii=False, indent=4)
                
            print(f"\n=== DONE ===")
            print(f"Successfully processed {len(results_list)} files.")
            print(f"All translations have been saved to '{output_file}'.")