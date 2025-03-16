# portfolio-performance-analytics
portfolio-performance-analytics (ppar) is a python package that produces holdings-based multi-period attribution, contribution, and benchmark-relative ex-post risk statistics.

[License](LICENSE)

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Implementation](#implementation)
- [Installation](#installation)
- [Usage](#usage)
- [Technical](#technical)
- [Enhancements](#enhancements)

---

## Description

portfolio-performance-analytics (ppar) is a python package that produces holdings-based multi-period attribution, contribution, and benchmark-relative ex-post risk statistics. It uses the Brinson-Fachler methodology for calculating attribution effects, and uses the Carino method for logarithmically-smoothing cumulative effects over multi-period time frames.

The inputs required to produce the analytics fall into three categories:
1. Periodic "classification-level" weights and returns for a portfolio and its benchmark.  A "classification" can be any category such as region, country, economic sector, industry, security, etc.  The weights and returns must satisfy the formula: *SumOf(weights * returns) = Total Return*. They will typically be beginning-of-period weights and period returns. (Required)
2. Classification items and descriptions. (Optional)
3. Mappings from the classification scheme of the weights and returns to a reporting classification. (Optional)

The input data may be provided directly as either:
1. Pandas DataFrames.
2. Polars DataFrames.
3. Python dictionaries (for Classifications and Mappings).
4. csv files.

For sample input data sources, please refer to the python script demo.py and the ppar/demo_data directory.  Once the input data has been provided, then the analytics may be requested using different calculation parameters, time-periods, and frequencies:
1. Daily (or for whatever data frequency is provided).
2. Monthly
3. Quarterly
4. Yearly

The outputs are represented by different views and charts.  See [Features](#features) below.  They may be delivered in different formats:
1. csv files
2. html strings
3. json strings
4. Pandas DataFrames
5. png files
6. Polars DataFrames
7. Python "great tables"
8. xml strings

---

## Features

The below sample outputs portray a large-cap alpha strategy that has achieved a high active return of 1737 bps over the benchmark.  In the total lines of the Economic Sector Attribution reports, you can see that this active return can be broken down into 359 bps in sector allocation and 1378 bps in selecting securities.  From the Risk Statistics report, you can see that this has been accomplished with a lower downside probabilty than the benchmark (29% vs 36%), and a higher annualized sharpe ratio than the benchmark (2.02 vs 1.27).  The largest contributor to active performance was in the Information Technology Sector.  Although the portfolio was slightly under-allocated in the Information Technology sector (by -0.05%), it did an excellent job of selecting securities for a total active contribution of 431 bps in the sector.

- **Attribution & Contribution**:
<img src="images/CumulativeAttributionByEconomicSector.jpg" alt="Cumulative Attribution by Economic Sector Table" width="100%" />
<br><br><br>
<img src="images/OverallAttributionByEconomicSector.jpg" alt="Overall Attribution by Economic Sector Table" width="100%" />
<br><br><br>
<img src="images/OverallAttributionBySecurity.jpg" alt="Overall Attribution by Security Table" width="100%" />
<br><br><br>
<img src="images/OverallAttributionByEconomicSector.png" alt="Overall Attribution by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/OverallContributionByEconomicSector.png" alt="Overall Contribution by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/SubPeriodAttributionEffectsByEconomicSector.png" alt="Sub-Period Attribution Effects by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/SubPeriodReturns.png" alt="Sub-Period Returns Chart" width="100%" />
<br><br><br>
<img src="images/ActiveContributionsByEconomicSector.png" alt="Active Contributions by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/TotalAttributionEffectsByEconomicSector.png" alt="Total Attribution Effects by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/CumulativeAttributionEffectsByEconomicSector.png" alt="Cumulative Attribution Effect by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/CumulativeReturns.png" alt="Cumulative Returns" width="100%" />
<br><br><br>

- **Ex-Post Risk Statistics**:
<img src="images/RiskStatistics.jpg" alt="Risk Statistics" width="100%" />
<br>

---

## Implementation
Typically, a user will develop their own "data source" functions that provide the data in one of the above formats.  The python script "demo.py" has sample data source functions.

Users can also develop their own "presentation layer" using the various output formats as the inputs to their presentation layer.

---

## Installation
pip install ppar

---

## Usage
python demo.py

---

## Technical
Being built on top of Polars dataframes, ppar is able to efficiently process large datasets through parallel processing, vectorization, lazy evaluation, and using Apache Arrow as its underlying data format.

---

## Enhancements
Future enhancements may include:
1. Break out the interaction (cross-product) effect.  It is currently included in the selection effect.
2. Break out the currency effect.
3. Break out the long and short sides of each sector.
4. Add additional multi-period smoothing algorithms (e.g. Menchero).
5. Support time-series of risk-free rates (as opposed to a single annual rate).
6. Calculate additional risk statistics.
