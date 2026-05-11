# MovieLens: Advanced Rating Prediction System
### HarvardX Data Science Professional Certificate - Capstone Project

![R](https://img.shields.io/badge/Language-R-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![OS](https://img.shields.io/badge/OS-Zorin%20OS-blueviolet.svg)

## Project Overview
This project focuses on building a high-performance recommendation algorithm using the **MovieLens 10M dataset**. By implementing a regularized linear model, the system accounts for multiple biases (user, movie, genre, and temporal effects) to predict ratings with high precision.

**Key Achievement:** Reached a final **RMSE of 0.86332**, successfully surpassing the HarvardX target threshold of **0.86490**.

## Key Technical Features
* **Scalable Modeling:** Implementation of a regularized linear model using an **iterative backfitting algorithm** to handle large-scale data within hardware constraints (6GB RAM).
* **Multi-dimensional Analysis:** Comprehensive Exploratory Data Analysis (EDA) investigating user "strictness", movie popularity, and genre specialization.
* **Methodological Rigor:** Strict separation between training sets and the **final holdout test set** to ensure zero data leakage and true model evaluation.
* **Academic Foundation:** Model design grounded in the seminal works of the Netflix Prize (Koren et al.) and modern Matrix Factorization principles.

## Tech Stack
* **Language:** R
* **IDE:** RStudio
* **Key Libraries:** `tidyverse`, `caret`, `data.table`, `lubridate`

### Computing Environment (Virtualized)
To ensure stability and performance during heavy data processing, the project was developed in a controlled environment:
* **Hypervisor:** Oracle VirtualBox 7.0
* **Guest OS:** Zorin OS 18.1 (Linux)
* **Dedicated Resources:** 6GB RAM and 4 Virtual CPUs
* **Optimization:** Used garbage collection (`gc()`) and pre-aggregation strategies to maintain system stability.
  
## Visualizing the Biases (Movie & User Effects)

To build an accurate recommendation engine, we must account for different types of systematic errors. Below are the two most significant biases identified during our Exploratory Data Analysis:

| **The Blockbuster Bias (Movie Effect)** | **The Critic Bias (User Effect)** |
|:---:|:---:|
| ![Blockbuster Bias](./blockbusters-bias.png) | ![User Effect](./user-effect.png) |
| *Distribution of movie-specific bias ($b_i$). Blockbusters skew the average upwards.* | *Distribution of user-specific bias ($b_u$). Shows "strict" vs "lenient" critics.* |

---

## Results & Performance
The model was built incrementally to measure the impact of each variable on the Root Mean Square Error (RMSE):

| Model Evolution | RMSE |
| :--- | :--- |
| **Baseline (Mean only)** | 1.05990 |
| + Movie Effect | 0.94374 |
| + User Effect | 0.86593 |
| **Final Model (Regularized + Time/Genre)** | **0.86332** |

## Repository Structure
This repository is organized to distinguish between the development phase and the final academic presentation:

* [MovieLens_project_report.pdf](./MovieLens_project_report.pdf) : **Final Publication.** The formal academic report detailing methodology, EDA, and results in a professional layout.
* [MovieLens_project_report.Rmd](./MovieLens_project_report.Rmd) : **The Refined Version.** The source R Markdown file used to generate the report. It is optimized for readability and clean presentation of the final model.
* [capstoneMovieLens.R](./capstoneMovieLens.R) : **The Laboratory.** A comprehensive, standalone R script containing the entire "under the hood" workflow. It includes data cleaning, extensive experimentation, and the complete modeling pipeline for direct execution.

---
**Author:** Fabian Hiernaux  
**Date:** May 2026  
*Submitted in partial fulfillment of the requirements for the HarvardX Professional Certificate in Data Science.*
