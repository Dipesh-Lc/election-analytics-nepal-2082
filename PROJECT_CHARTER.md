# Election Analytics Nepal 2082 — Project Charter

---
## 1. Background & Context

**Nepal’s 2082 general election** is being conducted earlier than its scheduled cycle following the collapse of the previous government. The collapse was triggered by widespread **Gen-Z-led protests** against corruption, ban on social media, political stagnation, and perceived elite capture of the political system.

The protests marked a significant departure from routine political dissent, resulting in widespread mobilization of younger generations and a loss of legitimacy for traditional political parties that have dominated governance since 2047. A citizen-led interim government was subsequently formed with a mandate to conduct elections.

This election therefore represents a potential case of **generational political realignment**, rather than a routine electoral contest governed solely by historical inertia.

Nepal’s **2082 general election** is being conducted earlier than its scheduled cycle following the collapse of the previous government.

The collapse was triggered by widespread **Gen-Z-led protests** against corruption, political stagnation, restrictions on social media, and perceived elite capture of the political system. Attempts to suppress these protests resulted in violence, loss of life, and a sharp decline in public legitimacy of the ruling coalition.

A citizen-led interim government was subsequently formed with a mandate to conduct elections.

This election therefore represents a potential case of **generational political realignment**, rather than a routine electoral contest governed solely by historical inertia.

---

## 2. Project Objective

### Primary Objective — Proportional Representation (PR)

To estimate **district-level PR vote share** for political parties in Nepal’s 2082 election and aggregate these predictions to the national level, producing:

- National PR vote share forecasts
- PR seat allocation under Nepal’s electoral rules:
  - **3% national threshold**
  - **Sainte-Laguë method for 110 seats**

### Deferred Objective (Future Work)

- Direct constituency (FPTP) modeling for 165 seats, requiring candidate-level and constituency-specific datasets

---

## 3. Scope Definition

### In Scope
- Historical PR election data (2074, 2079)
- Census demographics (2078)
- Voter roll and turnout indicators (2082)
- Aggregate social media engagement (2082 / 2026)
- District-level modeling and national aggregation
- Rule-based PR seat simulation

### Out of Scope
- Individual voter behavior
- Real-time polling or surveys
- Targeted political messaging
- Candidate-level FPTP modeling (current iteration)

---

## 4. Modeling Framework Overview

The project adopts a **multi-layered PR modeling approach**:

1. **District-Level Vote Share Modeling**
   - Predict vote-share change rather than raw share
   - Incorporate demographic, turnout, and historical inertia features

2. **National Aggregation**
   - Turnout-weighted aggregation from districts to national totals

3. **Electoral Rule Enforcement**
   - Remove parties below the 3% national threshold
   - Renormalize eligible vote shares
   - Allocate seats using Sainte-Laguë divisors

4. **Scenario Overlay**
   - Consolidation of micro-party vote mass
   - Social media engagement tilt
   - Backlash penalties for negative sentiment
   - Shock-gated amplification for high-momentum new parties

This layered design separates **structural prediction** from **behavioral scenarios**.

---

## 5. Units of Analysis & Prediction Targets

### Proportional Representation Model
- **Unit:** District × Party × Election cycle
- **Target:** Vote share (%) via modeled change
- **Outputs:**
  - District-level vote share predictions
  - National PR vote share estimates
  - PR seat allocation (110 seats)

---

## 6. Data Inventory

| Dataset          | Level    | Years      | Purpose                          |
|------------------|----------|------------|----------------------------------|
| Election Results | District | 2074, 2079 | Historical PR voting behavior    |
| Census           | District | 2078       | Demographic features             |
| Voter Roll       | District | 2082       | Turnout and voter composition    |
| Social Media     | National | 2081–2082  | Mobilization & sentiment overlay |

---

## 7. Assumptions & Limitations

- Social media users are not representative of the electorate
- Online sentiment does not translate directly into votes
- New parties may exhibit dynamics not seen historically
- Models are exploratory and probabilistic, not deterministic

---

## 8. Ethical Considerations

This project is conducted strictly for analytical and educational purposes.

- No individual-level data is used
- No targeted persuasion or messaging is performed
- Social media data is aggregated and anonymized
- Results are not intended to influence voter behavior

---

## 9. Expected Outputs

- District-level PR vote share forecasts (2082)
- National PR vote share forecasts
- PR seat simulations using:
  - 3% national threshold
  - Sainte-Laguë allocation
- Scenario comparisons and sensitivity analysis
- Visualizations and narrative summaries

---

## 10. Project Status

- **PR modeling pipeline:** Complete
- **Scenario analysis:** Complete
- **FPTP modeling:** Planned (future iteration)
