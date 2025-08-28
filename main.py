# Dynamic Prompting for Attendance Tracker Project with Token Logging

from transformers import pipeline
import tiktoken  # Install with: pip install tiktoken

# Initialize zero-shot classification pipeline with Top P, Temperature, and Top K
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    top_p=0.8,
    temperature=0.7,
    top_k=50
)

# Dynamic data
classroom_name = "Class 10A"
date = "2025-08-28"
known_students = ["Alice", "Bob", "Carol", "David"]

# Structured output prompt
attendance_prompt = (
    f"Today is {date}. This is a photo from {classroom_name}. "
    f"The registered students are: {', '.join(known_students)}. "
    "Identify and list the names of all students present for attendance. "
    "Return the result as a JSON object with the following format: "
    '{"present": [list of present student names], "absent": [list of absent student names]}'
)

candidate_labels = [
    "All students present",
    "Some students absent",
    "No students present",
    "Unable to identify students"
]

result = classifier(attendance_prompt, candidate_labels)

print("Structured Output Attendance Prompt:", attendance_prompt)
print("AI's Structured Output Response:", result)

# Token counting using tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
prompt_tokens = len(encoding.encode(attendance_prompt))
label_tokens = sum(len(encoding.encode(label)) for label in candidate_labels)
total_tokens = prompt_tokens + label_tokens

print(f"Tokens used in prompt: {prompt_tokens}")
print(f"Tokens used in candidate labels: {label_tokens}")
print(f"Total tokens sent to AI: {total_tokens}")

# Explanation for video:
# Structured output in LLMs means instructing the AI to return results in a specific, machine-readable format (like JSON).
# This makes it easier for other programs to parse and use the AI's output automatically.