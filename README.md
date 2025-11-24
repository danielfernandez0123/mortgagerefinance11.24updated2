# Optimal Mortgage Refinancing Calculator

A Streamlit dashboard implementation of the closed-form solution for optimal mortgage refinancing based on the NBER Working Paper 13487 by Agarwal, Driscoll, and Laibson (2007).

## 📚 Paper Reference

**Title:** Optimal Mortgage Refinancing: A Closed Form Solution  
**Authors:** Sumit Agarwal, John C. Driscoll, and David Laibson  
**Publication:** NBER Working Paper No. 13487 (October 2007)  
**URL:** http://www.nber.org/papers/w13487

## 🎯 Overview

This application implements the first closed-form optimal refinancing rule that tells homeowners exactly when to refinance their mortgages. Unlike previous approaches that required complex numerical methods, this calculator provides an exact solution using the Lambert W-function.

### Key Features

- **Exact Optimal Solution**: Implements Theorem 2 from the paper using the Lambert W-function
- **Square Root Approximation**: Provides a simpler second-order Taylor approximation
- **NPV Comparison**: Shows the suboptimal NPV break-even rule for comparison
- **Sensitivity Analysis**: Interactive charts showing how thresholds change with parameters
- **Welfare Loss Calculator**: Quantifies the cost of using suboptimal rules
- **Educational Content**: Detailed explanations tied to specific pages in the paper

## 🚀 Installation & Usage

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mortgage-refinancing-calculator.git
cd mortgage-refinancing-calculator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

### Deploy on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy directly from your forked repository

## 📊 The Mathematics

### Main Formula (Theorem 2, page 13)

The optimal refinancing threshold is:

```
x* = (1/ψ)[φ + W(-exp(-φ))]
```

Where:
- `ψ = √(2(ρ + λ))/σ`
- `φ = 1 + ψ(ρ + λ)κ/(M(1-τ))`
- `W(·)` is the Lambert W-function

### Decision Rule

Refinance when the current mortgage rate falls below the original rate by at least x* (in decimal form).

## 📈 Key Parameters

| Parameter | Symbol | Description | Typical Value |
|-----------|---------|-------------|---------------|
| Mortgage Value | M | Remaining principal balance | $250,000 |
| Original Rate | i₀ | Interest rate on current mortgage | 6% |
| Discount Rate | ρ | Real discount rate | 5% |
| Rate Volatility | σ | Annual standard deviation | 0.0109 |
| Tax Rate | τ | Marginal tax rate | 28% |
| Moving Probability | μ | Annual probability of relocating | 10% |
| Inflation | π | Expected inflation rate | 3% |

## 🔍 Key Findings from the Paper

1. **Optimal thresholds** typically range from 100 to 200 basis points
2. **NPV rule** (ignoring option value) leads to refinancing too early
3. **Welfare losses** from using NPV rule can exceed $85,000 on a $500,000 mortgage
4. **Validation** against Chen & Ling (1989) shows differences of less than 10 basis points

## 📁 Repository Structure

```
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── data/                # (Optional) Historical rate data
```

## ⚠️ Important Notes

### What This Calculator Does
- Provides the **optimal refinancing threshold** based on your specific parameters
- Shows the **difference** between optimal and common suboptimal rules
- Quantifies **welfare losses** from using simple NPV rules
- Offers **sensitivity analysis** to understand how thresholds change

### What This Calculator Doesn't Do
- This implements the paper's model which makes simplifying assumptions
- Does not account for cash-out refinancing or changes in mortgage size
- Assumes fixed-rate mortgages only
- Does not consider adjustable-rate mortgages (ARMs)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:
- Add historical mortgage rate data visualization
- Implement third-order approximation
- Add more sophisticated tax calculations
- Include adjustable-rate mortgage analysis

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Original paper authors: Sumit Agarwal, John C. Driscoll, and David Laibson
- National Bureau of Economic Research (NBER) for publishing the working paper
- The Streamlit team for the excellent dashboard framework

## 📧 Contact

For questions about the implementation (not the economic model itself), please open an issue in this repository.

---

**Disclaimer:** This calculator is for educational purposes. Always consult with financial professionals before making refinancing decisions.
