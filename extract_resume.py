from docx import Document
import re

# Load the resume
doc = Document("Resume.docx")

# Extract all text
text = "\n".join([para.text for para in doc.paragraphs])

# Extract email
email = re.findall(r'\S+@\S+', text)
# Extract phone number (common patterns)
phone = re.findall(r'\+?\d[\d -]{8,}\d', text)

# Extract name (optional: first line or line containing 'Name:')
lines = text.split("\n")
name = lines[0]  # assuming first line is name
# or use regex if your resume has "Name: John Doe"
# name_match = re.search(r'Name[:\-]\s*(.*)', text)
# if name_match:
#     name = name_match.group(1)

print("Name:", name)
print("Email:", email)
print("Phone:", phone)


# python extract_resume.py (for running the file)