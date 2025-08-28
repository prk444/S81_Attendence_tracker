# One-Shot Prompting for Attendance Tracker Project

from transformers import pipeline

# Initialize zero-shot classification pipeline (used for demonstration)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# One-shot prompt for attendance tracking with a single example
attendance_prompt = (
    "Example: In this classroom photo, Alice, Bob, and Carol are present. "
    "Now, given a new classroom photo, identify and list the names of all students present for attendance."
)

# Possible attendance outcomes
candidate_labels = [
    "All students present",
    "Some students absent",
    "No students present",
    "Unable to identify students"
]

# Perform one-shot classification
result = classifier(attendance_prompt, candidate_labels)

print("One-Shot Attendance Prompt:", attendance_prompt)
print("AI's One-Shot Response:", result)

# Explanation:
# One-shot prompting means providing the AI with a single example of the task in the prompt.
# Here, we show the AI an example attendance extraction before asking it to perform the same task on a new input.
# This helps the AI better understand the expected output format for the task.