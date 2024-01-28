from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from jiwer import wer
import unicodedata
from tqdm import tqdm


device = "cuda"
torch_dtype = torch.float16

model_id_A = "openai/whisper-small"
model_A = AutoModelForSpeechSeq2Seq.from_pretrained(model_id_A, torch_dtype=torch_dtype).to(device)
processor_A = AutoProcessor.from_pretrained(model_id_A)
pipeA = pipeline("automatic-speech-recognition", model=model_A, tokenizer=processor_A.tokenizer,
                 feature_extractor=processor_A.feature_extractor, max_new_tokens=128, chunk_length_s=15,
                 batch_size=1, torch_dtype=torch_dtype, device=device)

model_id_B = "openai/whisper-base"
model_B = AutoModelForSpeechSeq2Seq.from_pretrained(model_id_B, torch_dtype=torch_dtype).to(device)
processor_B = AutoProcessor.from_pretrained(model_id_B)
pipeB = pipeline("automatic-speech-recognition", model=model_B, tokenizer=processor_B.tokenizer,
                 feature_extractor=processor_B.feature_extractor, max_new_tokens=128, chunk_length_s=15,
                 batch_size=1, torch_dtype=torch_dtype, device=device)


def remove_punctuation(input_string):
    punctuation = u"।,"
    return ''.join(ch for ch in input_string if unicodedata.category(ch)[0] != 'P' and ch not in punctuation)

def calculate_errors(transcription, prediction):
    error_rate = wer(transcription, prediction)
    total_words = len(transcription.split())
    total_errors = int(round(error_rate * total_words))
    return total_errors


dataset = load_dataset("google/fleurs", "bn_in", split='test')


results_A = []
for sample in tqdm(dataset, desc="Processing with Model A"):
    audio = sample["audio"]
    original_transcription = remove_punctuation(sample["transcription"])
    total_words = len(original_transcription.split())
    result_A = pipeA(audio, generate_kwargs={"task": "transcribe", "language": "bengali"})
    prediction_A = remove_punctuation(result_A["text"])
    errors_A = calculate_errors(original_transcription, prediction_A)
    results_A.append((errors_A, total_words))


results_B = []
for sample in tqdm(dataset, desc="Processing with Model B"):
    audio = sample["audio"]
    result_B = pipeB(audio, generate_kwargs={"task": "transcribe", "language": "bengali"})
    prediction_B = remove_punctuation(result_B["text"])
    errors_B = calculate_errors(original_transcription, prediction_B)
    results_B.append(errors_B)


combined_results = []
for result_A, result_B in zip(results_A, results_B):
    errors_A, total_words = result_A
    errors_B = result_B
    combined_results.append(f"{errors_A}|{errors_B}|{total_words}")


with open("asr_evaluation_results_fleurs_combined.txt", "w") as file:
    for line in combined_results:
        file.write(line + "\n")

print("Evaluation completed. Combined results saved in 'asr_evaluation_results_fleurs_combined.txt'")
