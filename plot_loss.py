import json
import ast
import matplotlib.pyplot as plt

log_file = "STT_logs_remote/test_gpu_job_output_train_22050.log"
losses = []
epochs = []
steps = []

with open(log_file, "r") as f:
    for line in f:
        # Looking for lines like: {'loss': '17.87', ...}
        if line.startswith("{'loss'"):
            try:
                data = ast.literal_eval(line.strip())
                losses.append(float(data['loss']))
                # Use epoch as an approximation for step since it's steadily increasing
                epochs.append(float(data['epoch']))
                steps.append(len(losses) * 10) # assuming logging_steps=10
            except Exception as e:
                print(f"Error parsing line: {line.strip()} - {e}")

if not losses:
    print("No loss data found in the log file.")
else:
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='.', linestyle='-', color='b')
    plt.title('Training Loss over Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('STT_logs_remote/training_loss.png')
    print("Plot saved to STT_logs_remote/training_loss.png")
