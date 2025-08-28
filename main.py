# Dynamic Prompting for Attendance Tracker Project

from transformers import pipeline

# Initialize zero-shot classification pipeline (used for demonstration)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

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

# Explanation:
# Dynamic prompting means generating the prompt on-the-fly using current context or data.
# Here, the prompt is built using the classroom name, date, and the list of registered students,
# making the AI's task more specific and context-aware for improved attendance tracking.