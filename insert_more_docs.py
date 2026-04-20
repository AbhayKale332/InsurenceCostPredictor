import nbformat

def insert_theory():
    nb_path = 'TrainingScripts/InsurenceHypothesis.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Prepare markdown snippets
    bg_prob_md = """## 2. Background and Objectives
**Background**: The healthcare and insurance industry generates vast amounts of data that remains underutilised in traditional actuarial workflows. Predicting medical insurance premiums is a canonical regression problem where a model learns the relationship between individual characteristics and the expected cost of healthcare coverage.

**Objectives**:
- Build a regression model to predict medical insurance charges.
- Perform comprehensive EDA to understand data distributions.
- Apply preprocessing and feature engineering for improved model performance.

## 3. Problem Statement
**Formal Definition**: Given a set of demographic and lifestyle features of an insurance applicant (age, sex, BMI, number of children, smoking status, residential region), predict the annual medical insurance charge (in USD) that the insurance company is likely to bill.

**Type of ML Problem**: Supervised Learning — Regression."""

    dataset_md = """## 4. Dataset Description
**Source**: The dataset is sourced from a publicly available insurance data repository. The file contains 1,338 rows and 6 features + 1 target.

**Features (Input Variables)**:
- `age`: Age of the insured (18 - 64)
- `sex`: Gender of the insured
- `bmi`: Body Mass Index
- `children`: No. of dependents covered
- `smoker`: Smoking status
- `region`: US residential region (NW, NE, SW, SE)

**Target Variable**:
- `charges`: Annual medical insurance cost billed by the company (Float, USD)."""

    preprocess_md = """## 5. Data Preprocessing
**5.1 Missing Values**: The dataset was inspected and confirmed to contain no missing values across all 1,338 records. No imputation was required.

**5.3 Outlier Detection**: A boxplot of the `charges` column revealed significant right-skew and upper outliers. These correspond to genuine high-cost patients."""

    new_cells = []
    
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and '1. Introduction' in cell.source:
            new_cells.append(cell)
            new_cells.append(nbformat.v4.new_markdown_cell(bg_prob_md))
        elif cell.cell_type == 'code' and 'pd.read_csv' in cell.source:
            new_cells.append(nbformat.v4.new_markdown_cell(dataset_md))
            new_cells.append(cell)
        elif cell.cell_type == 'code' and 'plt.boxplot(data[' in cell.source:
            new_cells.append(nbformat.v4.new_markdown_cell(preprocess_md))
            new_cells.append(cell)
        else:
            new_cells.append(cell)

    nb.cells = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("Theory documentation inserted successfully")

if __name__ == '__main__':
    insert_theory()
