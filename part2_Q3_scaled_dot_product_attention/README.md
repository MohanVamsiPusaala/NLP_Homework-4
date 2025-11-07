# Part 2 – Q3: Scaled Dot-Product Attention

### 👨‍🎓 Student Information
**Name:** Mohan Vamsi Pusaala  
**Student ID:** 700773458  
**Course:** CS5760 – Natural Language Processing  
**University:** University of Central Missouri  
**Semester:** Fall 2025

---

## Objective
Implement the Scaled Dot-Product Attention function:
Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V
and demonstrate:
- the attention weight matrix,
- the output vectors,
- a softmax stability check (before and after scaling / max-subtraction).

---

## Files Included
- `q3_attention.py` — implementation and test (prints & saves results).  
- `requirements.txt` — dependencies.  
- `results/console_output.txt` — program output after running the script.

---

## How to run (locally in VS Code)
1. Open the folder `part2_Q3_scaled_dot_product_attention` in VS Code.  
2. Open Terminal.  
3. (Optional) Create environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
