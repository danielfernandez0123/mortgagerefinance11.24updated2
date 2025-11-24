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
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Main Calculator", "📈 Sensitivity Analysis", "📖 Paper Explanation", "🔧 Additional Tools"])

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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><b>Reference:</b> Agarwal, S., Driscoll, J. C., & Laibson, D. (2007). 
"Optimal Mortgage Refinancing: A Closed Form Solution" NBER Working Paper No. 13487</p>
<p>Calculator implementation for educational purposes</p>
</div>
""", unsafe_allow_html=True)
