# Resume–Job Description Matching System

A Gemini-powered Python application that compares a candidate's resume with a job description and generates a structured JSON assessment.

## Features

- Resume–JD matching
- Match score from 0–100
- Top candidate strengths
- Missing skill detection
- Evidence-based reasoning
- Prompt injection protection
- Pydantic structured output validation
- Input validation
- JSON validation
- API retry handling
- Exponential backoff for temporary API failures
- Multiline resume and JD input

## Tech Stack

- Python
- Google Gemini API
- google-genai
- Pydantic
- python-dotenv

## Project Structure

```text
resume-job-matcher/
├── resume_job_matcher.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore