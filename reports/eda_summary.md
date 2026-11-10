# Exploratory Data Analysis — Key Findings

## Dataset at a glance

- **7,043 customers** and **20 predictive features** (after removing the customer ID).
- **1,869 customers churned (26.5% churn rate)**.
- The target is imbalanced, so accuracy alone is misleading; recall, F1 and PR-AUC are the primary metrics.

## What separates churners from retained customers?

| Signal | Churned | Retained |
|---|---:|---:|
| Average tenure (months) | 18.0 | 37.6 |
| Median tenure (months) | 10 | 38 |
| Average monthly charges | $74.44 | $61.27 |

- Strongest **positive** correlations with churn: `MonthlyCharges` (+0.19), `AvgMonthlyCharges` (+0.19), `SeniorCitizen` (+0.15).
- Strongest **negative** correlations with churn: `tenure` (-0.35), `TotalCharges` (-0.20), `FamilySize` (-0.18).

## Contract type

| Contract | Customers | Churn rate |
|---|---:|---:|
| Month-to-month | 3,875 | 42.7% |
| One year | 1,473 | 11.3% |
| Two year | 1,695 | 2.8% |

Month-to-month customers churn at a far higher rate than customers on one- or two-year contracts. Contract commitment is the single strongest categorical retention lever in the dataset.

## Internet service

| Internet service | Customers | Churn rate |
|---|---:|---:|
| DSL | 2,421 | 19.0% |
| Fiber optic | 3,096 | 41.9% |
| No | 1,526 | 7.4% |

Fiber-optic customers churn more than DSL customers despite paying more, suggesting a perceived value or service-quality gap.

## Payment method

| Payment method | Customers | Churn rate |
|---|---:|---:|
| Electronic check | 2,365 | 45.3% |
| Mailed check | 1,612 | 19.1% |
| Bank transfer (automatic) | 1,544 | 16.7% |
| Credit card (automatic) | 1,522 | 15.2% |

Electronic-check customers churn the most; automatic payment methods are associated with materially lower churn.

## Tenure

| Tenure group | Customers | Churn rate |
|---|---:|---:|
| 0-12 months | 2,186 | 47.4% |
| 13-24 months | 1,024 | 28.7% |
| 25-48 months | 1,594 | 20.4% |
| 49-72 months | 2,239 | 9.5% |

Churn risk is concentrated in the first year. Customers who survive the first 12 months become progressively more loyal.

## Services

| Services subscribed | Customers | Churn rate |
|---|---:|---:|
| 0 | 80 | 43.8% |
| 1 | 1,701 | 21.1% |
| 2 | 1,188 | 32.8% |
| 3 | 965 | 36.5% |
| 4 | 922 | 31.3% |
| 5 | 908 | 25.6% |
| 6 | 676 | 22.5% |
| 7 | 395 | 12.4% |
| 8 | 208 | 5.3% |

Customers with more services and more support add-ons churn less, supporting bundle-based retention offers.

## Figures

All figures are saved in `reports/figures/`:

1. `01_churn_distribution.png` — target imbalance
2. `02_churn_by_contract.png` — contract-level churn
3. `03_tenure_analysis.png` — tenure distributions and buckets
4. `04_monthly_charges_by_churn.png` — billing differences
5. `05_churn_by_internet_and_payment.png` — service/payment segments
6. `06_services_vs_churn.png` — service bundling effect
7. `07_correlation_heatmap.png` — numeric feature relationships
