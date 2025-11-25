"""
Optimal Mortgage Refinancing Calculator
Based on: "Optimal Mortgage Refinancing: A Closed Form Solution"
By Sumit Agarwal, John C. Driscoll, and David Laibson
NBER Working Paper No. 13487 (October 2007)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import lambertw
import math

# Page configuration
st.set_page_config(
    page_title="Optimal Mortgage Refinancing Calculator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4788;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .formula-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .result-box {
        background-color: #e8f4ea;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">Optimal Mortgage Refinancing Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Based on the NBER Working Paper 13487 by Agarwal, Driscoll, and Laibson (2007)</div>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("📊 Input Parameters")
st.sidebar.markdown("---")

# Create tabs for different input sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Main Calculator", "📈 Sensitivity Analysis", "📖 Paper Explanation", "🔧 Additional Tools", "💰 Points Analysis"])
with st.sidebar:
    st.subheader("Mortgage Information")
    M = st.number_input(
        "Remaining Mortgage Value ($)", 
        min_value=10000, 
        max_value=5000000, 
        value=250000,
        step=10000,
        help="The remaining principal balance on your mortgage (M in the paper)"
    )
    
    i0 = st.number_input(
        "Original Mortgage Rate (%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=6.0,
        step=0.1,
        help="The interest rate on your current mortgage (i₀ in the paper)"
    ) / 100
    
    st.subheader("Economic Parameters")
    rho = st.number_input(
        "Real Discount Rate (%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=5.0,
        step=0.5,
        help="Your personal discount rate (ρ in the paper, page 17)"
    ) / 100
    
    sigma = st.number_input(
        "Interest Rate Volatility", 
        min_value=0.001, 
        max_value=0.05, 
        value=0.0109,
        step=0.001,
        format="%.4f",
        help="Annual standard deviation of mortgage rate (σ in the paper, calibrated on page 18)"
    )
    
    st.subheader("Tax & Cost Information")
    tau = st.number_input(
        "Marginal Tax Rate (%)", 
        min_value=0.0, 
        max_value=50.0, 
        value=28.0,
        step=1.0,
        help="Your marginal tax rate (τ in the paper)"
    ) / 100
    
    fixed_cost = st.number_input(
        "Fixed Refinancing Cost ($)", 
        min_value=0, 
        max_value=20000, 
        value=2000,
        step=100,
        help="Fixed costs like inspection, title insurance, lawyers (page 17)"
    )
    
    points = st.number_input(
        "Points (%)", 
        min_value=0.0, 
        max_value=5.0, 
        value=1.0,
        step=0.1,
        help="Points charged as percentage of mortgage"
    ) / 100
    
    st.subheader("Prepayment Parameters")
    mu = st.number_input(
        "Annual Probability of Moving (%)", 
        min_value=0.0, 
        max_value=50.0, 
        value=10.0,
        step=1.0,
        help="Annual probability of relocating (μ in the paper)"
    ) / 100
    
    pi = st.number_input(
        "Expected Inflation Rate (%)", 
        min_value=0.0, 
        max_value=10.0, 
        value=3.0,
        step=0.5,
        help="Expected inflation rate (π in the paper)"
    ) / 100
    
    Gamma = st.number_input(
        "Remaining Mortgage Years", 
        min_value=1, 
        max_value=30, 
        value=25,
        help="Years remaining on mortgage (Γ in the paper)"
    )

# Calculate derived parameters
def calculate_lambda(mu, i0, Gamma, pi):
    """Calculate λ (lambda) as per page 19 and Appendix C of the paper"""
    if i0 * Gamma < 100:  # Prevent overflow
        lambda_val = mu + i0 / (np.exp(i0 * Gamma) - 1) + pi
    else:
        lambda_val = mu + pi  # Simplified for very large values
    return lambda_val

def calculate_kappa(M, points, fixed_cost, tau):
    """Calculate κ(M) - tax-adjusted refinancing cost (Appendix A)"""
    # Simplified version - full formula in Appendix A
    kappa = fixed_cost + points * M
    return kappa

def calculate_optimal_threshold(M, rho, lambda_val, sigma, kappa, tau):
    """
    Calculate the optimal refinancing threshold x* using Lambert W function
    As per Theorem 2 (page 13) and equation (12)
    """
    # Calculate ψ (psi) as per equation in Theorem 2
    psi = np.sqrt(2 * (rho + lambda_val)) / sigma
    
    # Calculate φ (phi) as per equation in Theorem 2
    C_M = kappa / (1 - tau)  # Normalized refinancing cost
    phi = 1 + psi * (rho + lambda_val) * C_M / M
    
    # Calculate x* using Lambert W function (equation 12)
    # x* = (1/ψ)[φ + W(-exp(-φ))]
    try:
        w_arg = -np.exp(-phi)
        w_val = np.real(lambertw(w_arg, k=0))
        x_star = (1 / psi) * (phi + w_val)
    except:
        x_star = np.nan
    
    return x_star, psi, phi, C_M

def calculate_square_root_approximation(M, rho, lambda_val, sigma, kappa, tau):
    """
    Calculate the square root approximation (second-order Taylor expansion)
    As per Section 2.3 (page 16-17)
    """
    # Square root rule approximation
    sqrt_term = sigma * np.sqrt(kappa / (M * (1 - tau))) * np.sqrt(2 * (rho + lambda_val))
    return -sqrt_term

def calculate_npv_threshold(M, rho, lambda_val, kappa, tau):
    """
    Calculate the NPV break-even threshold
    As per Definition 3 (page 16)
    """
    C_M = kappa / (1 - tau)
    x_npv = -(rho + lambda_val) * C_M / M
    return x_npv

# Main calculations
lambda_val = calculate_lambda(mu, i0, Gamma, pi)
kappa = calculate_kappa(M, points, fixed_cost, tau)
x_star, psi, phi, C_M = calculate_optimal_threshold(M, rho, lambda_val, sigma, kappa, tau)
x_star_sqrt = calculate_square_root_approximation(M, rho, lambda_val, sigma, kappa, tau)
x_npv = calculate_npv_threshold(M, rho, lambda_val, kappa, tau)

# Convert to basis points for display
x_star_bp = -x_star * 10000 if not np.isnan(x_star) else np.nan
x_star_sqrt_bp = -x_star_sqrt * 10000
x_npv_bp = -x_npv * 10000

with tab1:
    st.header("📊 Optimal Refinancing Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Exact Optimal Threshold", 
            f"{x_star_bp:.0f} bps" if not np.isnan(x_star_bp) else "N/A",
            help="Refinance when current rate is this many basis points below original rate (Theorem 2, page 13)"
        )
    
    with col2:
        st.metric(
            "Square Root Approximation", 
            f"{x_star_sqrt_bp:.0f} bps",
            f"{x_star_sqrt_bp - x_star_bp:.0f} bps difference" if not np.isnan(x_star_bp) else "N/A",
            help="Second-order Taylor approximation (Section 2.3, page 16-17)"
        )
    
    with col3:
        st.metric(
            "NPV Break-even Threshold", 
            f"{x_npv_bp:.0f} bps",
            f"{x_npv_bp - x_star_bp:.0f} bps difference" if not np.isnan(x_star_bp) else "N/A",
            help="Simple NPV rule ignoring option value (Definition 3, page 16)"
        )
    
    # Detailed breakdown
    st.markdown("---")
    st.subheader("📐 Solution Components (Theorem 2, page 13)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Parameters")
        st.markdown(f"""
        <div class="formula-box">
        <b>ψ (psi)</b> = √(2(ρ + λ))/σ<br>
        ψ = √(2({rho:.3f} + {lambda_val:.3f}))/{sigma:.4f}<br>
        <b>ψ = {psi:.4f}</b>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="formula-box">
        <b>φ (phi)</b> = 1 + ψ(ρ + λ)κ/(M(1-τ))<br>
        φ = 1 + {psi:.4f}×({rho:.3f} + {lambda_val:.3f})×{kappa:.0f}/({M:.0f}×(1-{tau:.2f}))<br>
        <b>φ = {phi:.4f}</b>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Derived Values")
        st.markdown(f"""
        <div class="formula-box">
        <b>λ (lambda)</b> = μ + i₀/(e^(i₀Γ) - 1) + π<br>
        λ = {mu:.3f} + {i0:.3f}/(e^({i0:.3f}×{Gamma}) - 1) + {pi:.3f}<br>
        <b>λ = {lambda_val:.4f}</b><br>
        <i>(Equation from page 19, Appendix C)</i>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="formula-box">
        <b>C(M)</b> = κ(M)/(1-τ)<br>
        C(M) = {kappa:.0f}/(1-{tau:.2f})<br>
        <b>C(M) = ${C_M:.0f}</b><br>
        <i>(Normalized refinancing cost, page 8)</i>
        </div>
        """, unsafe_allow_html=True)
    
    # Final formula
    st.markdown("### 🎯 Optimal Refinancing Rule (Equation 12)")
    st.markdown(f"""
    <div class="result-box">
    <h4>x* = (1/ψ)[φ + W(-exp(-φ))]</h4>
    <p>x* = (1/{psi:.4f})[{phi:.4f} + W(-exp(-{phi:.4f}))]</p>
    <p><b>x* = {x_star:.6f}</b></p>
    <p>Converting to basis points: <b>{x_star_bp:.0f} basis points</b></p>
    <br>
    <p><b>Decision Rule:</b> Refinance when the current mortgage rate falls <b>{x_star_bp:.0f} basis points</b> below your original rate of {i0*100:.1f}%</p>
    <p><b>Refinance at or below:</b> {(i0 - abs(x_star))*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📈 Sensitivity Analysis")
    
    # Choose parameter for sensitivity analysis
    param_choice = st.selectbox(
        "Select parameter to analyze:",
        ["Mortgage Size (M)", "Interest Rate Volatility (σ)", "Discount Rate (ρ)", 
         "Tax Rate (τ)", "Probability of Moving (μ)", "Refinancing Costs"]
    )
    
    # Generate sensitivity data
    if param_choice == "Mortgage Size (M)":
        M_range = np.linspace(100000, 1000000, 50)
        thresholds = []
        sqrt_approx = []
        npv_thresholds = []
        
        for M_test in M_range:
            kappa_test = calculate_kappa(M_test, points, fixed_cost, tau)
            x_test, _, _, _ = calculate_optimal_threshold(M_test, rho, lambda_val, sigma, kappa_test, tau)
            x_sqrt_test = calculate_square_root_approximation(M_test, rho, lambda_val, sigma, kappa_test, tau)
            x_npv_test = calculate_npv_threshold(M_test, rho, lambda_val, kappa_test, tau)
            
            thresholds.append(-x_test * 10000 if not np.isnan(x_test) else 0)
            sqrt_approx.append(-x_sqrt_test * 10000)
            npv_thresholds.append(-x_npv_test * 10000)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=M_range/1000, y=thresholds, mode='lines', name='Exact Optimal', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=M_range/1000, y=sqrt_approx, mode='lines', name='Square Root Approx', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=M_range/1000, y=npv_thresholds, mode='lines', name='NPV Rule', line=dict(dash='dot')))
        
        fig.update_layout(
            title="Refinancing Threshold vs Mortgage Size (Table 1, page 20)",
            xaxis_title="Mortgage Size ($1000s)",
            yaxis_title="Refinancing Threshold (basis points)",
            height=500,
            hovermode='x unified'
        )
        
    elif param_choice == "Interest Rate Volatility (σ)":
        sigma_range = np.linspace(0.005, 0.025, 50)
        thresholds = []
        sqrt_approx = []
        npv_thresholds = []
        
        for sigma_test in sigma_range:
            x_test, _, _, _ = calculate_optimal_threshold(M, rho, lambda_val, sigma_test, kappa, tau)
            x_sqrt_test = calculate_square_root_approximation(M, rho, lambda_val, sigma_test, kappa, tau)
            x_npv_test = calculate_npv_threshold(M, rho, lambda_val, kappa, tau)
            
            thresholds.append(-x_test * 10000 if not np.isnan(x_test) else 0)
            sqrt_approx.append(-x_sqrt_test * 10000)
            npv_thresholds.append(-x_npv_test * 10000)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sigma_range, y=thresholds, mode='lines', name='Exact Optimal', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=sigma_range, y=sqrt_approx, mode='lines', name='Square Root Approx', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=sigma_range, y=npv_thresholds, mode='lines', name='NPV Rule', line=dict(dash='dot')))
        
        fig.update_layout(
            title="Refinancing Threshold vs Interest Rate Volatility",
            xaxis_title="Interest Rate Volatility (σ)",
            yaxis_title="Refinancing Threshold (basis points)",
            height=500,
            hovermode='x unified'
        )
    
    elif param_choice == "Tax Rate (τ)":
        tau_range = np.array([0, 0.10, 0.15, 0.25, 0.28, 0.33, 0.35])
        thresholds = []
        
        for tau_test in tau_range:
            kappa_test = calculate_kappa(M, points, fixed_cost, tau_test)
            x_test, _, _, _ = calculate_optimal_threshold(M, rho, lambda_val, sigma, kappa_test, tau_test)
            thresholds.append(-x_test * 10000 if not np.isnan(x_test) else 0)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tau_range*100, y=thresholds, text=[f"{t:.0f} bps" for t in thresholds], textposition='outside'))
        
        fig.update_layout(
            title="Refinancing Threshold vs Tax Rate (Table 2, page 20)",
            xaxis_title="Marginal Tax Rate (%)",
            yaxis_title="Refinancing Threshold (basis points)",
            height=500
        )
    
    elif param_choice == "Probability of Moving (μ)":
        mu_range = np.linspace(0.05, 0.25, 50)
        thresholds = []
        
        for mu_test in mu_range:
            lambda_test = calculate_lambda(mu_test, i0, Gamma, pi)
            x_test, _, _, _ = calculate_optimal_threshold(M, rho, lambda_test, sigma, kappa, tau)
            thresholds.append(-x_test * 10000 if not np.isnan(x_test) else 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=1/mu_range, y=thresholds, mode='lines', line=dict(width=3)))
        
        fig.update_layout(
            title="Refinancing Threshold vs Expected Time to Move (Table 3, page 21)",
            xaxis_title="Expected Years Until Move (1/μ)",
            yaxis_title="Refinancing Threshold (basis points)",
            height=500
        )
    
    elif param_choice == "Discount Rate (ρ)":
        rho_range = np.linspace(0.02, 0.10, 50)
        thresholds = []
        
        for rho_test in rho_range:
            x_test, _, _, _ = calculate_optimal_threshold(M, rho_test, lambda_val, sigma, kappa, tau)
            thresholds.append(-x_test * 10000 if not np.isnan(x_test) else 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rho_range*100, y=thresholds, mode='lines', line=dict(width=3)))
        
        fig.update_layout(
            title="Refinancing Threshold vs Discount Rate",
            xaxis_title="Real Discount Rate (%)",
            yaxis_title="Refinancing Threshold (basis points)",
            height=500
        )
    
    else:  # Refinancing Costs
        cost_range = np.linspace(500, 5000, 50)
        thresholds = []
        sqrt_approx = []
        npv_thresholds = []
        
        for cost_test in cost_range:
            kappa_test = calculate_kappa(M, points, cost_test, tau)
            x_test, _, _, _ = calculate_optimal_threshold(M, rho, lambda_val, sigma, kappa_test, tau)
            x_sqrt_test = calculate_square_root_approximation(M, rho, lambda_val, sigma, kappa_test, tau)
            x_npv_test = calculate_npv_threshold(M, rho, lambda_val, kappa_test, tau)
            
            thresholds.append(-x_test * 10000 if not np.isnan(x_test) else 0)
            sqrt_approx.append(-x_sqrt_test * 10000)
            npv_thresholds.append(-x_npv_test * 10000)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cost_range, y=thresholds, mode='lines', name='Exact Optimal', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=cost_range, y=sqrt_approx, mode='lines', name='Square Root Approx', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=cost_range, y=npv_thresholds, mode='lines', name='NPV Rule', line=dict(dash='dot')))
        
        fig.update_layout(
            title="Refinancing Threshold vs Fixed Costs (Related to Table 4, page 22)",
            xaxis_title="Fixed Refinancing Cost ($)",
            yaxis_title="Refinancing Threshold (basis points)",
            height=500,
            hovermode='x unified'
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show comparison table similar to paper
    st.markdown("---")
    st.subheader("📊 Comparison with Paper Results")
    
    if param_choice == "Mortgage Size (M)":
        st.markdown("Compare with **Table 1** (page 20):")
        comparison_data = {
            'Mortgage': ['$1,000,000', '$500,000', '$250,000', '$100,000'],
            'Paper (Exact)': [107, 118, 139, 193],
            'Paper (2nd order)': [97, 106, 123, 163],
            'Paper (NPV)': [27, 33, 44, 76]
        }
        st.dataframe(pd.DataFrame(comparison_data))
    
    elif param_choice == "Tax Rate (τ)":
        st.markdown("Compare with **Table 2** (page 20) for $250,000 mortgage:")
        comparison_data = {
            'Tax Rate': ['0%', '10%', '15%', '25%', '28%', '33%', '35%'],
            'Paper Results': [124, 129, 131, 137, 139, 143, 145]
        }
        st.dataframe(pd.DataFrame(comparison_data))

with tab3:
    st.header("📖 Paper Explanation & Key Concepts")
    
    st.markdown("""
    ### 📑 Paper Overview
    
    This calculator implements the **first closed-form optimal refinancing rule** derived by Agarwal, Driscoll, and Laibson (2007).
    
    ### 🔑 Key Innovation
    
    Previous research required numerical methods to solve complex partial differential equations. This paper provides an exact, 
    closed-form solution that can be calculated on a simple calculator.
    
    ### 📐 The Main Formula (Theorem 2, page 13)
    
    The optimal refinancing threshold is:
    """)
    
    st.latex(r"x^* = \frac{1}{\psi}[\phi + W(-\exp(-\phi))]")
    
    st.markdown("""
    Where:
    - **x*** is the interest rate differential at which you should refinance
    - **W(·)** is the Lambert W-function
    - **ψ** and **φ** are parameters based on your specific situation
    
    ### 💡 Economic Intuition
    
    The optimal rule balances three key factors:
    
    1. **Interest Savings**: Lower rate saves money on future payments
    2. **Refinancing Costs**: Upfront costs must be recouped
    3. **Option Value**: Value of waiting for rates to potentially fall further
    
    ### 📊 Key Findings (Section 3, pages 17-22)
    
    - Optimal thresholds typically range from **100 to 200 basis points**
    - Smaller mortgages require larger rate drops to justify refinancing
    - Higher volatility increases the value of waiting
    - Tax deductibility of interest affects the optimal threshold
    
    ### ⚠️ Common Mistakes (Section 5, pages 24-28)
    
    The paper shows that most financial advisors recommend the **NPV rule**, which:
    - Ignores the option value of waiting
    - Can lead to refinancing too early
    - Results in expected losses of **$85,000+ on a $500,000 mortgage**
    
    ### 📈 Parameter Calibration (Section 3)
    
    The paper calibrates parameters using historical data:
    - **σ = 0.0109**: Based on 30-year mortgage rate volatility (1971-2004)
    - **ρ = 5%**: Typical real discount rate
    - **τ = 28%**: Common marginal tax rate
    - **μ = 10%**: Annual probability of moving
    """)
    
    # Add comparison with Chen and Ling
    st.markdown("---")
    st.subheader("🔄 Validation (Section 4, pages 22-24)")
    
    st.markdown("""
    The paper validates their closed-form solution against Chen and Ling (1989), who used numerical methods:
    
    | Refinancing Cost | Chen & Ling | This Paper | Difference |
    |-----------------|-------------|------------|------------|
    | 4.24 points | 228 bps | 218 bps | 10 bps |
    | 5.51 points | 256 bps | 255 bps | 1 bp |
    
    The close agreement validates the simplifying assumptions used to derive the closed-form solution.
    """)

with tab4:
    st.header("🔧 Additional Analysis Tools")
    
    tool_choice = st.selectbox(
        "Select Analysis Tool:",
        ["Welfare Loss Calculator", "Break-even Analysis", "Historical Rate Comparison"]
    )
    
    if tool_choice == "Welfare Loss Calculator":
        st.subheader("💰 Welfare Loss from Suboptimal Rules (Section 5, pages 25-28)")
        
        st.markdown("""
        The paper derives the expected loss from using the NPV rule instead of the optimal rule (Proposition 4, page 25-26).
        """)
        
        # Calculate welfare loss
        loss_npv = (np.exp(psi * abs(x_star)) / (psi * (rho + lambda_val))) * M if not np.isnan(x_star) else 0
        loss_sqrt = 0  # Would need more complex calculation
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="warning-box">
            <h4>Loss from NPV Rule</h4>
            <p>Expected Loss: <b>${loss_npv:,.0f}</b></p>
            <p>As % of Mortgage: <b>{(loss_npv/M)*100:.1f}%</b></p>
            <p><i>Based on equation (25), page 27</i></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Show table from paper
            st.markdown("**Table 6** from paper (page 28):")
            loss_data = {
                'Mortgage': ['$1M', '$500K', '$250K', '$100K'],
                'NPV Loss (%)': [16.3, 17.4, 19.6, 26.8]
            }
            st.dataframe(pd.DataFrame(loss_data))
    
    elif tool_choice == "Break-even Analysis":
        st.subheader("📊 NPV Break-even Analysis")
        
        st.markdown("""
        This tool shows when you'll recoup your refinancing costs under different scenarios.
        This is the simple NPV rule that **ignores option value**.
        """)
        
        current_rate = st.number_input("Current Market Rate (%)", 0.0, 20.0, 4.5, 0.1) / 100
        rate_diff = i0 - current_rate
        
        if rate_diff > 0:
            annual_savings = M * rate_diff * (1 - tau)
            payback_period = kappa / annual_savings if annual_savings > 0 else float('inf')
            
            st.markdown(f"""
            <div class="result-box">
            <h4>NPV Break-even Analysis</h4>
            <p>Rate Reduction: <b>{rate_diff*10000:.0f} basis points</b></p>
            <p>Annual Interest Savings: <b>${annual_savings:,.0f}</b></p>
            <p>Refinancing Cost: <b>${kappa:,.0f}</b></p>
            <p>Payback Period: <b>{payback_period:.1f} years</b></p>
            <br>
            <p><b>Note:</b> This ignores the option value of waiting!</p>
            <p>Optimal threshold suggests waiting until rates drop <b>{x_star_bp:.0f} bps</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Current rate must be lower than original rate for refinancing to make sense.")
    
    else:  # Historical Rate Comparison
        st.subheader("📈 Historical Context")
        
        st.markdown("""
        ### Historical 30-Year Mortgage Rates
        
        The paper uses data from 1971-2004 to calibrate σ = 0.0109.
        
        Key periods mentioned in the paper:
        - **1980s-1990s**: Generally falling rates, many failed to refinance despite deep in-the-money options
        - **1996-2003**: Over 1/3 of borrowers refinanced too early
        
        ### Current Application
        
        Your current situation:
        """)
        
        st.markdown(f"""
        - Original Rate: **{i0*100:.1f}%**
        - Optimal Refinancing Threshold: **{x_star_bp:.0f} basis points**
        - Refinance when rates reach: **{(i0 - abs(x_star))*100:.2f}%**
        
        Remember: The optimal threshold accounts for:
        1. Direct costs of refinancing
        2. The value of waiting for potentially better rates
        3. The probability you might move before capturing full benefits
        """)
with tab5:
      st.header("💰 Points vs Lender Credit Analysis")

      st.markdown("""
      This table shows what the optimal refinancing rate would be for different closing cost
  scenarios.
      The optimal rate is calculated as: Original Rate - Optimal Rate Drop
      """)

      # Calculate the base optimal rate (at current closing costs)
      base_optimal_rate = i0 - abs(x_star)  # x_star is negative, so we use abs()

      st.markdown(f"""
      **Current Parameters:**
      - Original Mortgage Rate: {i0*100:.2f}%
      - Current Optimal Rate Drop: {x_star_bp:.0f} basis points
      - Current Optimal Rate (Par): {base_optimal_rate*100:.3f}%
      - Current Closing Costs: ${kappa:,.0f}
      """)

      st.markdown("---")

      # Generate closing cost range in $500 increments
      cost_increments = []
      cost = 0
      max_cost = kappa * 4  # Up to 4x current closing costs

      while cost <= max_cost:
          cost_increments.append(cost)
          cost += 500

      # Calculate optimal rates for each closing cost level
      results = []

      for closing_cost in cost_increments:
          # Calculate points as percentage of loan
          points_percent = 0  # Start with 0% points
          fixed_cost_temp = closing_cost - (points_percent * M)

          # Ensure fixed costs are non-negative
          if fixed_cost_temp < 0:
              fixed_cost_temp = 0
              points_percent = closing_cost / M

          # Recalculate kappa with new closing cost
          temp_kappa = closing_cost

          # Calculate the optimal threshold with this closing cost
          temp_x_star, _, _, _ = calculate_optimal_threshold(M, rho, lambda_val, sigma,
  temp_kappa, tau)

          # Calculate the optimal rate for this closing cost
          optimal_rate = i0 - abs(temp_x_star)  # Subtract the rate drop

          results.append({
              'Closing Costs ($)': closing_cost,
              'Optimal Rate (%)': optimal_rate * 100
          })

      # Create DataFrame
      df_results = pd.DataFrame(results)

      # Display as a formatted table
      st.dataframe(
          df_results.style.format({
              'Closing Costs ($)': '${:,.0f}',
              'Optimal Rate (%)': '{:.3f}%'
          }),
          use_container_width=True,
          height=600  # Make it scrollable
      )

      # Add some analysis
      st.markdown("---")
      st.subheader("Analysis")

      # Find key points
      zero_cost_rate = df_results[df_results['Closing Costs ($)'] == 0]['Optimal Rate (%)'].values[0]
      current_cost_idx = 0  # Default value
      current_cost_rounded = int(kappa/500)*500
      if current_cost_rounded in df_results['Closing Costs ($)'].values:
          current_cost_idx = df_results[df_results['Closing Costs ($)'] ==
  current_cost_rounded].index[0]

      col1, col2, col3 = st.columns(3)

      with col1:
          st.metric(
              "Zero Cost Rate",
              f"{zero_cost_rate:.3f}%",
              help="Rate needed if closing costs were $0"
          )

      with col2:
          st.metric(
              "Current Cost Rate",
              f"{base_optimal_rate*100:.3f}%",
              help=f"Rate needed at ${kappa:,.0f} closing costs"
          )

      with col3:
          rate_per_1000 = (df_results.iloc[2]['Optimal Rate (%)'] - df_results.iloc[0]['Optimal Rate (%)']) / 1000 if len(df_results) > 2 else 0
          st.metric(
              "Rate per $1,000",
              f"{rate_per_1000:.3f}%",
              help="How much rate increases per $1,000 in closing costs"
          )

      # Download button
      csv = df_results.to_csv(index=False)
      st.download_button(
          label="Download Table as CSV",
          data=csv,
          file_name="closing_costs_to_optimal_rate.csv",
          mime="text/csv"
      )

      # Add explanation
      st.markdown("---")
      st.info("""
      **How to use this table:**
      1. Find your expected closing costs in the left column
      2. The right column shows the rate that would trigger optimal refinancing
      3. If market rates are below this optimal rate, consider refinancing
      4. If market rates are above this optimal rate, wait

      **Example:** If closing costs are $5,000 and the table shows 4.500%,
      then you should refinance when rates drop to 4.500% or below.
      """)

      # Helper functions for ENPV calculation (from imp file)
      def payment(principal, monthly_rate, n_months):
          """Level payment on an amortizing loan."""
          if monthly_rate == 0:
              return principal / n_months
          denom = 1.0 - (1.0 + monthly_rate) ** (-n_months)
          return principal * monthly_rate / denom

      def calculate_enpv_benefit(current_balance, current_rate, new_rate, remaining_years, new_term_years,
                                closing_costs, invest_rate, discount_rate, cpr, finance_costs_in_loan=True):
          """Calculate ENPV benefit using the imp file methodology"""
          n_old = int(round(remaining_years * 12))
          n_new = int(round(new_term_years * 12))
          horizon = max(n_old, n_new)

          r_old = current_rate / 12.0
          r_new = new_rate / 12.0
          r_inv = invest_rate / 12.0
          r_disc = discount_rate / 12.0

          old_principal = current_balance
          new_principal = current_balance + closing_costs if finance_costs_in_loan else current_balance

          # Monthly payments
          pmt_old = payment(old_principal, r_old, n_old)
          pmt_new = payment(new_principal, r_new, n_new)

          bal_old = old_principal
          bal_new = new_principal
          inv_bal = 0.0

          net_gain_pv = []

          # Build cash flows
          for t in range(1, horizon + 1):
              # Old loan
              if t <= n_old and bal_old > 0:
                  interest_old = r_old * bal_old
                  principal_old = pmt_old - interest_old
                  bal_old = max(0.0, bal_old - principal_old)
                  p_old_t = pmt_old
              else:
                  p_old_t = 0.0
                  bal_old = 0.0

              # New loan
              if t <= n_new and bal_new > 0:
                  interest_new = r_new * bal_new
                  principal_new = pmt_new - interest_new
                  bal_new = max(0.0, bal_new - principal_new)
                  p_new_t = pmt_new
              else:
                  p_new_t = 0.0
                  bal_new = 0.0

              # Payment savings and investment
              pmt_sav_t = p_old_t - p_new_t
              inv_bal = inv_bal * (1.0 + r_inv) + pmt_sav_t

              # Total advantage and present value
              balance_adv = bal_old - bal_new
              total_adv = inv_bal + balance_adv

              # Present value
              pv_factor = 1.0 / ((1.0 + r_disc) ** t)
              net_gain_pv.append(total_adv * pv_factor)

          # Calculate ENPV with mortality
          SMM = 1 - (1 - cpr)**(1/12)
          survival = 1.0
          enpv = 0.0

          for t in range(min(360, len(net_gain_pv))):
              mortality_t = survival * SMM
              enpv += net_gain_pv[t] * mortality_t
              survival = survival * (1 - SMM)

          return enpv

# Second Table - User Input Section
      st.markdown("---")
      st.subheader("📊 Compare Your Actual Quotes")

      st.markdown("""
      Enter your actual lender quotes below to see how they compare to the optimal rates.
      """)

      # Create empty dataframe for user input
      input_data = pd.DataFrame({
          'Closing Costs ($)': [0] * 10,  # Start with 10 empty rows
          'Actual Rate Offered (%)': [0.0] * 10,
          'Model Optimal Rate (%)': [0.0] * 10,
          'Difference (%)': [0.0] * 10,
          'Net Benefit ($)': [0.0] * 10,
          'ENPV Benefit ($)': [0.0] * 10
      })

      # Create editable dataframe
      edited_df = st.data_editor(
          input_data,
          column_config={
              'Closing Costs ($)': st.column_config.NumberColumn(
                  'Closing Costs ($)',
                  help="Enter the total closing costs quoted",
                  format="$%.0f",
                  min_value=0,
                  step=500
              ),
              'Actual Rate Offered (%)': st.column_config.NumberColumn(
                  'Actual Rate Offered (%)',
                  help="Enter the rate the lender is offering",
                  format="%.3f",
                  min_value=0.0,
                  max_value=20.0,
                  step=0.125
              ),
              'Model Optimal Rate (%)': st.column_config.NumberColumn(
                  'Model Optimal Rate (%)',
                  help="The optimal rate calculated by the model",
                  disabled=True,
                  format="%.3f"
              ),
              'Difference (%)': st.column_config.NumberColumn(
                  'Difference (%)',
                  help="Actual minus Optimal (negative = good deal)",
                  disabled=True,
                  format="%.3f"
              ),
              'Net Benefit ($)': st.column_config.NumberColumn(
                  'Net Benefit ($)',
                  help="Net benefit = (-x·M)/(ρ+λ) - C(M)",
                  disabled=True,
                  format="$%.2f"
              ),
              'ENPV Benefit ($)': st.column_config.NumberColumn(
                  'ENPV Benefit ($)',
                  help="Expected NPV using detailed cash flow model with prepayment",
                  disabled=True,
                  format="$%.2f"
              )
          },
          num_rows="dynamic",
          hide_index=True,
          use_container_width=True
      )

      # Calculate optimal rates for entered closing costs
      for idx in edited_df.index:
          if edited_df.loc[idx, 'Closing Costs ($)'] > 0:
              # Get the entered closing cost
              closing_cost = edited_df.loc[idx, 'Closing Costs ($)']

              # Calculate the optimal threshold for this closing cost
              temp_x_star, _, _, _ = calculate_optimal_threshold(M, rho, lambda_val, sigma, closing_cost, tau)

              # Calculate the optimal rate
              optimal_rate = i0 - abs(temp_x_star)

              # Update model optimal rate
              edited_df.loc[idx, 'Model Optimal Rate (%)'] = optimal_rate * 100

              # Calculate difference if actual rate is entered
              if edited_df.loc[idx, 'Actual Rate Offered (%)'] > 0:
                  difference = edited_df.loc[idx, 'Actual Rate Offered (%)'] - (optimal_rate * 100)
                  edited_df.loc[idx, 'Difference (%)'] = difference

                  # Calculate Net Benefit using the corrected formula
                  # x is negative when the new rate is lower than the original rate
                  x = (edited_df.loc[idx, 'Actual Rate Offered (%)'] / 100) - i0
                  C_M = closing_cost
                  # net_benefit = ((-x * M) / (rho + lambda_val)) - C_M
                  net_benefit = ((-x * M) / (rho + lambda_val)) - C_M
                  edited_df.loc[idx, 'Net Benefit ($)'] = net_benefit

                  # Calculate ENPV Benefit
                  # Use mu (probability of moving) as CPR, not the full lambda
                  cpr_for_calc = mu  # Just the moving probability
                  # If points were specified in closing costs, extract them
                  points_amount = points * M  # Use the sidebar points value
                  fixed_fees = closing_cost - points_amount

                  enpv_benefit = calculate_enpv_benefit(
                      current_balance=M,
                      current_rate=i0,
                      new_rate=edited_df.loc[idx, 'Actual Rate Offered (%)'] / 100,
                      remaining_years=Gamma,
                      new_term_years=30,  # Assuming 30-year refi
                      closing_costs=closing_cost,
                      invest_rate=rho,  # Using discount rate as investment rate
                      discount_rate=rho,
                      cpr=cpr_for_calc,
                      finance_costs_in_loan=True
                  )
                  edited_df.loc[idx, 'ENPV Benefit ($)'] = enpv_benefit

      # Display with color coding
      def highlight_difference(val):
          """Color code the difference column"""
          if isinstance(val, (int, float)):
              if val < 0:
                  return 'background-color: lightgreen'
              elif val > 0:
                  return 'background-color: lightcoral'
          return ''

      def highlight_benefit(val):
          """Color code the benefit columns"""
          if isinstance(val, (int, float)):
              if val > 0:
                  return 'background-color: lightgreen'
              elif val < 0:
                  return 'background-color: lightcoral'
          return ''

      # Apply styling - note we need to handle multiple columns
      style = edited_df.style
      style = style.applymap(highlight_difference, subset=['Difference (%)'])
      style = style.applymap(highlight_benefit, subset=['Net Benefit ($)', 'ENPV Benefit ($)'])

      st.dataframe(style, use_container_width=True)

      # Debug section - show calculation details for populated rows
      st.markdown("---")
      st.subheader("📐 Calculation Details")

      # Find rows with both closing costs and actual rates entered
      populated_rows = edited_df[(edited_df['Closing Costs ($)'] > 0) & (edited_df['Actual Rate Offered (%)'] > 0)]

      if len(populated_rows) > 0:
          # Show calculation for each populated row
          for idx in populated_rows.index:
              row_num = idx + 1  # Display as 1-based row number
              closing_cost = edited_df.loc[idx, 'Closing Costs ($)']
              actual_rate = edited_df.loc[idx, 'Actual Rate Offered (%)'] / 100

              # Calculate all components
              x = actual_rate - i0  # x is negative when offered rate < original rate
              C_M = closing_cost
              net_benefit = ((-x * M) / (rho + lambda_val)) - C_M
              enpv_benefit = edited_df.loc[idx, 'ENPV Benefit ($)']

              st.markdown(f"**Row {row_num} Calculation:**")

              col1, col2 = st.columns(2)

              with col1:
                  st.markdown(f"""
                  <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; font-family: monospace;'>
                  <b>Simple Net Benefit Formula:</b><br>
                  Net Benefit = (-x·M)/(ρ+λ) - C(M)<br><br>

                  Where:<br>
                  - ρ = {rho:.4f} ({rho*100:.1f}%)<br>
                  - λ = {lambda_val:.4f}<br>
                  - x = {actual_rate:.4f} - {i0:.4f} = {x:.4f}<br>
                  - M = ${M:,.0f}<br>
                  - C(M) = ${C_M:,.0f}<br><br>

                  Result: <b>${net_benefit:,.2f}</b>
                  </div>
                  """, unsafe_allow_html=True)

              with col2:
                  st.markdown(f"""
                  <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; font-family: monospace;'>
                  <b>ENPV (Detailed Model):</b><br>
                  Uses actual cash flows with:<br><br>

                  - CPR = {mu*100:.1f}% (moving prob)<br>
                  - Old rate = {i0*100:.2f}%<br>
                  - New rate = {actual_rate*100:.2f}%<br>
                  - Remaining term = {Gamma} years<br>
                  - New term = 30 years<br>
                  - Investment rate = {rho*100:.1f}%<br><br>

                  Result: <b>${enpv_benefit:,.2f}</b>
                  </div>
                  """, unsafe_allow_html=True)

              st.markdown("")  # Add spacing between rows
      else:
          st.info("Enter closing costs and actual rates in the table above to see calculation details.")

      # Summary of entered quotes
      active_quotes = edited_df[(edited_df['Closing Costs ($)'] > 0) & (edited_df['Actual Rate Offered (%)'] > 0)]

      if len(active_quotes) > 0:
          st.markdown("### Quote Analysis")
          col1, col2, col3, col4 = st.columns(4)

          with col1:
              best_idx = active_quotes['Difference (%)'].idxmin()
              best_rate = active_quotes.loc[best_idx, 'Actual Rate Offered (%)']
              best_diff = active_quotes.loc[best_idx, 'Difference (%)']
              st.metric("Best Quote", f"{best_rate:.3f}%", f"{best_diff:+.3f}%")

          with col2:
              good_deals = len(active_quotes[active_quotes['Difference (%)'] < 0])
              st.metric("Good Deals", f"{good_deals} of {len(active_quotes)}")

          with col3:
              best_enpv_idx = active_quotes['ENPV Benefit ($)'].idxmax()
              best_enpv = active_quotes.loc[best_enpv_idx, 'ENPV Benefit ($)']
              st.metric("Best ENPV", f"${best_enpv:,.0f}")

          with col4:
              avg_diff = active_quotes['Difference (%)'].mean()
              st.metric("Avg Difference", f"{avg_diff:+.3f}%")

      # Download button for comparison table
      csv2 = edited_df.to_csv(index=False)
      st.download_button(
          label="Download Comparison Table as CSV",
          data=csv2,
          file_name="rate_comparison_analysis.csv",
          mime="text/csv",
          key="download_comparison"  # Unique key to avoid conflict with first download button
      )


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><b>Reference:</b> Agarwal, S., Driscoll, J. C., & Laibson, D. (2007). 
"Optimal Mortgage Refinancing: A Closed Form Solution" NBER Working Paper No. 13487</p>
<p>Calculator implementation for educational purposes</p>
</div>
""", unsafe_allow_html=True)
