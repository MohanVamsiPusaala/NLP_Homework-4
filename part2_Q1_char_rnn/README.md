# Part 2 – Q1: Character-Level RNN Language Model

### 👨‍🎓 Student Information
**Name:** Mohan Vamsi Pusaala  
**Student ID:** 700773458  
**Course:** CS5760 – Natural Language Processing  
**University:** University of Central Missouri  
**Semester:** Fall 2025

---

## Objective
Train a small character-level RNN to predict next character given previous characters.  
Provide training/validation loss curves and sample generations at multiple temperatures.

---

## Files included
- `q1_char_rnn.py` — training, sampling, and result saving (fully commented).  
- `requirements.txt` — dependencies.  
- `data/toy.txt` — small toy corpus (create this file; the script will auto-create it if missing).  
- `data/book.txt` — optional larger text (50–200 KB) — place here to train on bigger data.  
- `results/` — contains `loss_curve.png`, `gen_temp_*.txt`, `console_output.txt`, and saved model.

---

## How to run (VS Code / local)
1. Open `part2_Q1_char_rnn` folder in VS Code.  
2. (Optional) Create virtual env:
   ```bash
   python -m venv venv
   venv\Scripts\activate
