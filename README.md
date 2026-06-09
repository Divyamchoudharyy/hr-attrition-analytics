# HR Attrition Analytics Dashboard

An end-to-end HR analytics project built on the IBM HR Employee Attrition dataset (1,470 employees). Covers data cleaning, exploratory analysis in Python, and an interactive browser-based dashboard — no frameworks required.

**Live Demo → [divyamchoudharyy.github.io/hr-attrition-analytics](https://divyamchoudharyy.github.io/hr-attrition-analytics)**

---

## Key Findings

| Finding | Detail |
|---|---|
| Overall attrition rate | 19.7% (290 of 1,470 employees) |
| #1 risk factor | Overtime — 27.9% vs 16.4% attrition (70% higher) |
| Highest-risk department | Human Resources at 24.2% |
| Satisfaction gap | Very low satisfaction: 26.4% leave vs 15.1% for very high |
| Age cohort most at risk | 26–35 (21.9%) — career-switching peak |
| Salary effect | Minimal — high earners leave at 17.4%, barely below average |

---

## Project Structure

```
hr-attrition-analytics/
├── hr_attrition.csv            # IBM HR dataset (1,470 records)
├── hr_attrition_eda.py         # Python EDA pipeline
├── index.html                  # Interactive dashboard (live on GitHub Pages)
├── attrition_analysis.png      # Matplotlib charts (4-panel)
└── correlation_heatmap.png     # Feature correlation heatmap
```

---

## EDA Pipeline (`hr_attrition_eda.py`)

- Data loading, null checks, dtype inspection
- Feature engineering — age groups, salary bands, tenure bands
- Group-by aggregations across 8 dimensions (dept, role, OT, satisfaction, age, salary, tenure, WLB)
- Correlation heatmap across 9 numeric features vs attrition
- 4-panel Matplotlib figure exported as PNG

**Run locally:**
```bash
pip install pandas numpy matplotlib seaborn
python hr_attrition_eda.py
```

---

## Dashboard (`index.html`)

Built with vanilla HTML, CSS, and Canvas API — no React, no build step, opens with a double-click.

**Charts included:**
- Animated KPI strip (5 metrics)
- Horizontal bar charts — department, job role, age group, salary band, work-life balance
- SVG line chart — satisfaction vs attrition trend
- Donut chart — overtime breakdown
- Stacked bar — tenure vs retention/attrition split
- Canvas scatter plot — age vs income, colored by attrition (hover tooltips)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data analysis | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | HTML, CSS, Canvas API |
| Hosting | GitHub Pages |

---

## Dataset

IBM HR Analytics Employee Attrition dataset — a standard HR analytics benchmark with 1,470 employee records and 23 features including demographics, compensation, satisfaction scores, and attrition status.

Source: [Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

*Built by Divyam Choudhary ·*
