# 1. Background & Context

**Nepal’s 2082 general election** is being conducted earlier than its scheduled cycle following the collapse of the previous government. The collapse was triggered by the gen-Z protests against corruption, political stagnation, and perceived elite capture of the political system.

The protests marked a significant departure from routine political dissent, resulting in widespread mobilization of younger voters and a loss of legitimacy for traditional political parties that have dominated governance since 2047. A citizen-led interim government was subsequently formed with a mandate to conduct elections.

This election therefore represents a potential case of generational political realignment rather than a routine electoral contest.


# 2. Project Objectives

## Objective A — Proportional Representation (PR)

To estimate proportional representation vote share for major political parties in the 2082 election by modeling district-level voting behavior and aggregating predictions to the national level.

## Objective B — Direct Constituency Seats

To estimate the probability of victory for parties and candidates in each of the 165 directly elected constituencies, accounting for historical voting patterns, demographic context, candidate characteristics, and national political momentum.


# 3. Scope Definition (What This Project Covers)

## In Scope

- Historical election data (2074, 2079)

- Census demographics (2078)

- Voter roll data (2082)

- Candidate-level attributes (where publicly available)

- Aggregate social media sentiment and narratives

- District- and constituency-level analysis


## Out of Scope

- Individual voter behavior

- Real-time polling

- Campaign finance analysis

- Targeted political messaging

- Causal claims about protest participation and voting


# 4. Modeling Framework Overview

Nepal’s mixed electoral system requires two distinct analytical approaches. Proportional representation outcomes depend on national vote share, while constituency outcomes depend on localized competition and candidate-specific effects.

For this reason, the project is structured around two separate but interconnected modeling pipelines:

- A district-based vote share model for proportional representation

- A constituency-level candidate-aware classification model for direct seats


# 5. Units of Analysis & Prediction Targets
   
## Proportional Representation Model

- **Unit:** District × Election Year

- **Target:** Party vote share (%)

- **Output:** Aggregated national vote share and simulated PR seat allocation


## Constituency Model

- **Unit:** Candidate × Constituency

- **Target:** Probability of winning a seat

- **Output:** Win probabilities and electoral volatility indicators


# 6. Data Inventory

| Dataset          | Level                   | Years      | Purpose                    |
| ---------------- | ----------------------- | ---------- | -------------------------- |
| Election Results | District / Constituency | 2074, 2079 | Historical voting behavior |
| Census           | District                | 2078       | Demographic features       |
| Voter Roll       | Constituency            | 2082       | New voter signals          |
| Candidate Data   | Constituency            | 2082       | Candidate effects          |
| Social Media     | National / District     | 2081–2082  | Sentiment & narratives     |


# 7. Assumptions & Limitations
- District boundaries are assumed to be consistent across datasets unless explicitly adjusted

- Social media users are not representative of the electorate

- Youth political sentiment does not translate directly into votes

- Candidate data may be incomplete or uneven across constituencies

- Models are probabilistic and exploratory, not predictive guarantees


# 8. Ethical Considerations
This project is intended for analytical and educational purposes only. It does not aim to influence voter behavior or support political campaigning.

The use of social media data is restricted to aggregate-level analysis, and no individual-level targeting or profiling is performed.

Given the violent nature of recent protests, care is taken to avoid sensationalism, normative judgment, or amplification of extremist narratives.


# 9. Expected Outputs

- District-level PR vote share forecasts

- National PR seat simulations

- Constituency win probability estimates

- Visualizations of electoral volatility

- Model explanations and narrative summaries

