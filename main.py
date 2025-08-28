# Zero-Shot Prompting for Attendance Tracker Project

from transformers import pipeline

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Zero-shot prompt for attendance tracking
attendance_prompt = (
    "Given a classroom photo, identify and list the names of all students present for attendance."
)

# Possible attendance outcomes
candidate_labels = [
    "All students present",
    "Some students absent",
    "No students present",
    "Unable to identify students"
]

# Perform zero-shot classification
result = classifier(attendance_prompt, candidate_labels)

print("Zero-Shot Attendance Prompt:", attendance_prompt)
print("AI's Zero-Shot Response:", result)

# This demonstrates zero-shot prompting in Python for an automated attendance tracker.