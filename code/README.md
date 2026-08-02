![Python](https://img.shields.io/badge/Python-3.12-blue)
![Hackathon](https://img.shields.io/badge/HackerRank-Orchestrate-green)
![License](https://img.shields.io/badge/License-MIT-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)


# AI-Powered WhatsApp Message Notification Router

> Intelligent notification prioritization system built for the HackerRank Orchestrate Hackathon 2026.

## Overview

The AI-Powered WhatsApp Message Notification Router is a modular message intelligence pipeline designed to reduce notification overload by automatically prioritizing incoming WhatsApp messages.

Instead of treating every notification equally, the system analyzes message content, sender reputation, user behavior, business metadata, historical interactions, images, and voice notes to determine the most appropriate notification action.

For every incoming message, the router predicts:

- **Action** → Notify, Digest, or Mute
- **Message Type** → Personal, Business Update, Promotion, Payment, Event, Scam, Spam, Greeting, Forward, etc.
- **Reason** explaining the routing decision
- **Confidence Score**
- **Supporting Evidence Message IDs**

The project combines deterministic business rules with AI-assisted reasoning to create explainable and personalized notification decisions.

---

# Key Features

- Intelligent notification routing
- Personalized user behavior modeling
- Historical message analysis
- Sender trust scoring
- Business trust evaluation
- Scam and phishing detection
- Spam filtering
- Promotion detection
- Urgency estimation
- Notification fatigue modeling
- Do Not Disturb (DND) support
- OCR-ready image processing
- Voice message processing
- Evidence-based decision generation
- Confidence calibration
- Explainable AI decisions

---

## 📊 Dataset Processing Pipeline

<p align="center">
  <img src="assets/CSV Dataset Processing Pipeline - visual selection.png"
       alt="SV Dataset Processing Pipeline"
       width="100%">
</p>

---

## Project Structure

```text
code
│
├── ai
│   └── gemini_router.py
│
├── context
│   ├── context_builder.py
│   └── index_builder.py
│
├── evaluation
│   └── evaluator.py
│
├── features
│   └── feature_engine.py
│
├── loaders
│   └── csv_loader.py
│
├── media
│   ├── image_parser.py
│   └── voice_parser.py
│
├── output
│   └── output_writer.py
│
├── retrieval
│   └── evidence_retriever.py
│
├── rules
│   └── rule_engine.py
│
├── utils
│   ├── confidence_calibrator.py
│   └── precompute_media.py
│
├── config.py
└── main.py
```

---

## Processing Pipeline

Every incoming message passes through the following stages:

1. Load structured datasets
2. Build fast lookup indexes
3. Construct contextual information
4. Extract behavioral and semantic features
5. Apply deterministic routing rules
6. Use Gemini for ambiguous scenarios (optional)
7. Calibrate confidence score
8. Retrieve supporting evidence
9. Generate final predictions
10. Export results to `dataset/output.csv`

---

## Datasets Used

The routing engine combines information from multiple structured datasets:

- Messages
- Users
- Groups
- Group Members
- Business Accounts
- Business History
- Message History
- Message Events
- Images
- Voice Notes
- Notification Summary

These datasets enable personalized routing decisions based on historical engagement, sender trust, business credibility, and contextual metadata.

---

## Core Decision Factors

The notification decision is influenced by multiple independent signals, including:

- Sender trust
- Business trust
- User engagement history
- Message urgency
- Promotion likelihood
- Scam probability
- Spam probability
- Historical interaction patterns
- Notification fatigue
- Do Not Disturb window
- OCR-derived image content
- Voice message context

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.12 |
| Data Processing | Pandas |
| Configuration | python-dotenv |
| Logging | Python Logging |
| AI | Google Gemini (optional) |
| Architecture | Modular Rule-Based Pipeline |




![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini_API-4285F4?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

--- 
## 🛠️ Development Tools

![Cursor](https://img.shields.io/badge/Cursor-AI-000000?style=for-the-badge)
![Antigravity](https://img.shields.io/badge/Antigravity-AI-6A5ACD?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

## Running the Project

## Clone the repository

```bash
git clone https://github.com/deepankarsingh106/hackerrank-orchestrate-august26.git
cd hackerrank-orchestrate-august26
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configure environment

Create a `.env` file (only if using Gemini):

```env
GEMINI_API_KEY=YOUR_API_KEY
```

## Run

```bash
python code/main.py
```

The predictions will be written to:

```text
dataset/output.csv
```

---

## Output Format

The generated CSV contains one prediction for every input message.

| Column | Description |
|---------|-------------|
| message_id | Unique message identifier |
| action | Notify / Digest / Mute |
| message_type | Predicted category |
| reason | Human-readable explanation |
| confidence | Prediction confidence |
| evidence_message_ids | Supporting historical messages |

---

## Engineering Highlights

- Modular architecture
- Explainable decision pipeline
- Separation of feature extraction and decision logic
- Deterministic rule engine
- AI-assisted fallback routing
- Configurable thresholds
- Production-style logging
- Extensible design for future ML integration

---

## Future Improvements

- Transformer-based message classification
- Vector search for semantic retrieval
- Advanced OCR pipeline
- Whisper-based voice transcription
- Learned confidence calibration
- Adaptive personalization
- Real-time deployment
- REST API with FastAPI
- Docker support
- Cloud-native deployment

---

## Author

**Deepankar Singh**

  - Pre-Final Year CSE Student at MNIT, Jaipur

---

## Acknowledgements

Developed as part of the **HackerRank Orchestrate Hackathon 2026**, focused on building an intelligent, explainable, and scalable notification routing system.
