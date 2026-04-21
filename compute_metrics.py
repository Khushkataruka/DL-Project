import json
import warnings

# Suppress HuggingFace warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import evaluate
except ImportError:
    print("Dependencies missing! Run: pip install evaluate sacrebleu rouge_score jiwer bert_score")
    exit(1)

file_path = "streaming_eval_results.json"
output_file = "metrics_results.json"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Cannot find {file_path}")
    exit(1)

samples = data.get("sample_translations", [])

if not samples:
    print("No samples found in the JSON file.")
    exit()

print("Loading evaluation metrics natively...")
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
wer = evaluate.load("wer")
bertscore = evaluate.load("bertscore")

def compute_all(preds, refs):
    # BLEU
    try:
        b = bleu.compute(predictions=preds, references=[[r] for r in refs])
        bleu_score = b["score"]
    except:
        bleu_score = 0.0
        
    # ROUGE
    try:
        r = rouge.compute(predictions=preds, references=refs)
        rouge_score = r["rougeL"] * 100
    except:
        rouge_score = 0.0

    # WER
    try:
        w = wer.compute(predictions=preds, references=refs)
    except:
        w = 0.0
        
    # BERTScore
    try:
        bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")
        bert_f1 = sum(bert_res["f1"]) / len(bert_res["f1"]) * 100
    except Exception:
        bert_f1 = 0.0
        
    return {
        "sacre_bleu": bleu_score,
        "rouge_L": rouge_score,
        "wer": w,
        "bert_score_f1": bert_f1
    }

print("Computing overall metrics...")
all_preds = [s["prediction"] for s in samples]
all_refs = [s["ground_truth"] for s in samples]

overall_metrics = compute_all(all_preds, all_refs)

print("Computing language-wise metrics...")
langs = set([s["true_language"] for s in samples])
language_metrics = {}

for lang in sorted(langs):
    print(f"  -> Processing {lang}...")
    lang_preds = [s["prediction"] for s in samples if s["true_language"] == lang]
    lang_refs = [s["ground_truth"] for s in samples if s["true_language"] == lang]
    language_metrics[lang] = compute_all(lang_preds, lang_refs)

results_payload = {
    "overall": overall_metrics,
    "language_breakdown": language_metrics
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, ensure_ascii=False, indent=4)

print(f"\nSUCCESS! All detailed language-wise metrics have been securely stored in: {output_file}")
