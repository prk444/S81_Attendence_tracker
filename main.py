# Dynamic Prompting for Attendance Tracker Project with Token Logging

from transformers import pipeline
import tiktoken  # Install with: pip install tiktoken

# Initialize zero-shot classification pipeline with Top P (nucleus sampling)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    top_p=0.8  # Set Top P for nucleus sampling
)

# Dynamic data (could come from user input, database, or sensors)
classroom_name = "Class 10A"
date = "2025-08-28"
known_students = ["Alice", "Bob", "Carol", "David"]

# Construct a dynamic prompt using real-time/contextual information
attendance_prompt = (
    f"Today is {date}. This is a photo from {classroom_name}. "
    f"The registered students are: {', '.join(known_students)}. "
    "Identify and list the names of all students present for attendance."
)

# Possible attendance outcomes
candidate_labels = [
    "All students present",
    "Some students absent",
    "No students present",
    "Unable to identify students"
]

# Perform dynamic prompting classification
result = classifier(attendance_prompt, candidate_labels)

print("Dynamic Attendance Prompt:", attendance_prompt)
print("AI's Dynamic Prompting Response:", result)

# Token counting using tiktoken (OpenAI tokenizer, works for most transformer models)
encoding = tiktoken.get_encoding("cl100k_base")
prompt_tokens = len(encoding.encode(attendance_prompt))
label_tokens = sum(len(encoding.encode(label)) for label in candidate_labels)
total_tokens = prompt_tokens + label_tokens

print(f"Tokens used in prompt: {prompt_tokens}")
print(f"Tokens used in candidate labels: {label_tokens}")
print(f"Total tokens sent to AI: {total_tokens}")

# Explanation for video:
# Tokens are the basic units of text that AI models process, such as words or word pieces.
# Counting tokens helps you understand how much input/output is being processed by