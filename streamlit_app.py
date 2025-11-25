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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Main Calculator", "📈 Sensitivity Analysis", "📖 Paper Explanation", "🔧 Additional Tools", "💰 Points Analysis", "📊 ENPV Analysis"])
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
      st.header("💰 Points vs Lender Credit Analysis with Net Gain Comparison")

      st.markdown("""
      Enter your lender quotes below and select which ones to compare using the detailed cash flow model.
      The charts will show month-by-month comparisons with full calculation details on hover.
      """)

      # Input parameters for calculations
      st.subheader("📊 Analysis Parameters")
      col1p, col2p, col3p, col4p = st.columns(4)

      with col1p:
          comp_invest_rate = st.number_input(
              "Investment Rate (%)",
              min_value=0.0,
              max_value=20.0,
              value=rho*100,
              step=0.5,
              key="comp_invest",
              help="Annual return on invested payment savings"
          ) / 100

      with col2p:
          comp_new_term = st.number_input(
              "New Loan Term (years)",
              min_value=15,
              max_value=30,
              value=30,
              step=5,
              key="comp_term",
              help="Term for new refinanced loans"
          )

      with col3p:
          comp_finance_costs = st.checkbox(
              "Finance costs in loan",
              value=True,
              key="comp_finance",
              help="Roll closing costs into the new loan"
          )

      with col4p:
          comp_include_taxes = st.checkbox(
              "Include tax effects",
              value=True,
              key="comp_taxes",
              help="Account for mortgage interest deduction"
          )

      st.markdown("---")

      # Table showing optimal rates vs closing costs
      st.subheader("📋 Optimal Rates by Closing Cost")

      # Generate closing cost range
      cost_increments = []
      cost = 0
      max_cost = kappa * 4

      while cost <= max_cost:
          cost_increments.append(cost)
          cost += 500

      # Calculate optimal rates
      results = []
      for closing_cost in cost_increments:
          temp_x_star, _, _, _ = calculate_optimal_threshold(M, rho, lambda_val, sigma, closing_cost, tau)
          optimal_rate = i0 - abs(temp_x_star)
          results.append({
              'Closing Costs ($)': closing_cost,
              'Optimal Rate (%)': optimal_rate * 100
          })

      df_results = pd.DataFrame(results)

      # Display table
      st.dataframe(
          df_results.style.format({
              'Closing Costs ($)': '${:,.0f}',
              'Optimal Rate (%)': '{:.3f}%'
          }),
          use_container_width=True,
          height=300
      )

      st.markdown("---")

      # User input section with checkboxes
      st.subheader("📊 Compare Your Actual Quotes")

      # Create dataframe with checkbox column
      input_data = pd.DataFrame({
          'Select': [False] * 10,
          'Closing Costs ($)': [0] * 10,
          'Actual Rate Offered (%)': [0.0] * 10,
          'Model Optimal Rate (%)': [0.0] * 10,
          'Difference (%)': [0.0] * 10,
          'Simple Net Benefit ($)': [0.0] * 10
      })

      # Create editable dataframe
      edited_df = st.data_editor(
          input_data,
          column_config={
              'Select': st.column_config.CheckboxColumn(
                  'Select',
                  help="Check to include in comparison charts",
                  default=False,
                  width="small"
              ),
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
              'Simple Net Benefit ($)': st.column_config.NumberColumn(
                  'Simple Net Benefit ($)',
                  help="Net benefit = (-x·M·(1-τ))/(ρ+λ) - C(M)",
                  disabled=True,
                  format="$%.2f"
              )
          },
          num_rows="dynamic",
          hide_index=True,
          use_container_width=True
      )

      # Calculate optimal rates and simple net benefit
      for idx in edited_df.index:
          if edited_df.loc[idx, 'Closing Costs ($)'] > 0:
              closing_cost = edited_df.loc[idx, 'Closing Costs ($)']

              # Calculate optimal threshold
              temp_x_star, _, _, _ = calculate_optimal_threshold(M, rho, lambda_val, sigma, closing_cost, tau)
              optimal_rate = i0 - abs(temp_x_star)
              edited_df.loc[idx, 'Model Optimal Rate (%)'] = optimal_rate * 100

              # Calculate difference and net benefit if rate is entered
              if edited_df.loc[idx, 'Actual Rate Offered (%)'] > 0:
                  difference = edited_df.loc[idx, 'Actual Rate Offered (%)'] - (optimal_rate * 100)
                  edited_df.loc[idx, 'Difference (%)'] = difference

                  # Simple net benefit
                  x = (edited_df.loc[idx, 'Actual Rate Offered (%)'] / 100) - i0
                  net_benefit = ((-x * M * (1 - tau)) / (rho + lambda_val)) - closing_cost
                  edited_df.loc[idx, 'Simple Net Benefit ($)'] = net_benefit

      # Helper function for detailed calculations
      def payment(principal, monthly_rate, n_months):
          """Level payment on an amortizing loan."""
          if monthly_rate == 0:
              return principal / n_months
          denom = 1.0 - (1.0 + monthly_rate) ** (-n_months)
          return principal * monthly_rate / denom

      def compute_scenario_history(rate, closing_costs, label):
          """Compute full history for one refinance scenario"""
          n_old = int(round(Gamma * 12))
          n_new = int(round(comp_new_term * 12))
          horizon = max(n_old, n_new)
          gamma_month = n_old

          r_old = i0 / 12.0
          r_new = rate / 12.0
          r_inv = comp_invest_rate / 12.0

          old_principal = M
          if comp_finance_costs:
              new_principal = M + closing_costs
          else:
              new_principal = M

          pmt_old = payment(old_principal, r_old, n_old)
          pmt_new = payment(new_principal, r_new, n_new)

          bal_old = old_principal
          bal_new = new_principal
          inv_bal = 0.0
          opt1_sav = 0.0
          opt2_sav = 0.0
          inv_bal_at_gamma = None

          history = []

          for t in range(1, horizon + 1):
              # Old loan
              if t <= n_old and bal_old > 0:
                  interest_old = r_old * bal_old
                  principal_old = pmt_old - interest_old
                  bal_old = max(0.0, bal_old - principal_old)
                  if comp_include_taxes:
                      p_old_t = pmt_old - (interest_old * tau)
                      p_old_t_nominal = pmt_old
                      tax_benefit_old = interest_old * tau
                  else:
                      p_old_t = pmt_old
                      p_old_t_nominal = pmt_old
                      tax_benefit_old = 0
              else:
                  p_old_t = 0.0
                  p_old_t_nominal = 0.0
                  bal_old = 0.0
                  interest_old = 0.0
                  tax_benefit_old = 0.0

              # New loan
              if t <= n_new and bal_new > 0:
                  interest_new = r_new * bal_new
                  principal_new = pmt_new - interest_new
                  bal_new = max(0.0, bal_new - principal_new)
                  if comp_include_taxes:
                      p_new_t = pmt_new - (interest_new * tau)
                      p_new_t_nominal = pmt_new
                      tax_benefit_new = interest_new * tau
                  else:
                      p_new_t = pmt_new
                      p_new_t_nominal = pmt_new
                      tax_benefit_new = 0
              else:
                  p_new_t = 0.0
                  p_new_t_nominal = 0.0
                  bal_new = 0.0
                  interest_new = 0.0
                  tax_benefit_new = 0.0

              # Payment savings
              pmt_sav_t = p_old_t - p_new_t

              # Calculate total advantage based on gamma
              if t < gamma_month:
                  inv_bal = inv_bal * (1.0 + r_inv) + pmt_sav_t
                  balance_adv = bal_old - bal_new
                  total_adv = inv_bal + balance_adv

                  # Components for hover
                  calculation_parts = {
                      'inv_bal_prev': inv_bal - pmt_sav_t,
                      'inv_interest': (inv_bal - pmt_sav_t) * r_inv,
                      'pmt_sav': pmt_sav_t,
                      'inv_bal': inv_bal,
                      'bal_old': bal_old,
                      'bal_new': bal_new,
                      'balance_adv': balance_adv,
                      'total_adv': total_adv,
                      'formula': f"({inv_bal:.2f} + {balance_adv:.2f})"
                  }

              elif t == gamma_month:
                  inv_bal = inv_bal * (1.0 + r_inv) + pmt_sav_t
                  inv_bal_at_gamma = inv_bal
                  opt2_sav = inv_bal_at_gamma
                  opt1_sav = 0.0
                  balance_adv = bal_old - bal_new
                  total_adv = inv_bal + balance_adv

                  calculation_parts = {
                      'inv_bal_at_gamma': inv_bal_at_gamma,
                      'bal_new': bal_new,
                      'total_adv': total_adv,
                      'formula': f"Gamma point: {inv_bal_at_gamma:.2f} + {balance_adv:.2f}"
                  }

              else:
                  # After gamma
                  opt1_sav_prev = opt1_sav
                  opt1_sav = opt1_sav * (1.0 + r_inv) + pmt_old

                  opt2_sav_prev = opt2_sav
                  opt2_sav = opt2_sav * (1.0 + r_inv)

                  total_adv = (opt2_sav - bal_new) - opt1_sav

                  calculation_parts = {
                      'opt1_sav_prev': opt1_sav_prev,
                      'opt1_interest': opt1_sav_prev * r_inv,
                      'pmt_old': pmt_old,
                      'opt1_sav': opt1_sav,
                      'opt2_sav_prev': opt2_sav_prev,
                      'opt2_interest': opt2_sav_prev * r_inv,
                      'opt2_sav': opt2_sav,
                      'bal_new': bal_new,
                      'total_adv': total_adv,
                      'formula': f"({opt2_sav:.2f} - {bal_new:.2f}) - {opt1_sav:.2f}"
                  }

              rec = {
                  "month": t,
                  "p_old": p_old_t,
                  "p_old_nominal": p_old_t_nominal,
                  "p_new": p_new_t,
                  "p_new_nominal": p_new_t_nominal,
                  "pmt_sav_t": pmt_sav_t,
                  "inv_bal": inv_bal if t <= gamma_month else opt2_sav,
                  "opt1_sav": opt1_sav,
                  "bal_old": bal_old,
                  "bal_new": bal_new,
                  "total_adv": total_adv,
                  "interest_old": interest_old,
                  "interest_new": interest_new,
                  "tax_benefit_old": tax_benefit_old,
                  "tax_benefit_new": tax_benefit_new,
                  "calculation_parts": calculation_parts,
                  "label": label
              }
              history.append(rec)

          return history, pmt_old, pmt_new, gamma_month

      # Get selected rows
      selected_rows = edited_df[edited_df['Select'] & (edited_df['Closing Costs ($)'] > 0) & (edited_df['Actual Rate Offered (%)'] > 0)]

      if len(selected_rows) > 0:
          st.markdown("---")
          st.subheader("📊 Net Gain Comparison Chart")

          # Compute histories for all selected scenarios
          all_histories = []
          colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

          for idx, (row_idx, row) in enumerate(selected_rows.iterrows()):
              rate = row['Actual Rate Offered (%)'] / 100
              costs = row['Closing Costs ($)']
              label = f"Rate: {row['Actual Rate Offered (%)']:.3f}%, Costs: ${costs:,.0f}"

              history, _, _, gamma = compute_scenario_history(rate, costs, label)
              all_histories.append((history, colors[idx % len(colors)], label))

          # Create the Net Gain chart
          fig1 = go.Figure()

          for history, color, label in all_histories:
              months = [rec["month"] for rec in history]
              net_gains = [rec["total_adv"] for rec in history]

              # Create hover text with calculation details
              hover_texts = []
              for rec in history:
                  t = rec["month"]
                  parts = rec["calculation_parts"]

                  if t < gamma:
                      hover_text = f"""<b>Month {t} - {label}</b><br>
                      <b>Net Gain Calculation:</b><br>
                      Investment Balance + Balance Advantage<br>
                      = {parts['inv_bal']:.2f} + {parts['balance_adv']:.2f}<br>
                      = <b>${rec['total_adv']:.2f}</b><br><br>

                      <b>Investment Balance Detail:</b><br>
                      Previous Balance: ${parts['inv_bal_prev']:.2f}<br>
                      Interest Earned: ${parts['inv_interest']:.2f}<br>
                      Payment Savings: ${parts['pmt_sav']:.2f}<br>
                      New Balance: ${parts['inv_bal']:.2f}<br><br>

                      <b>Loan Balances:</b><br>
                      Old Loan: ${parts['bal_old']:.2f}<br>
                      New Loan: ${parts['bal_new']:.2f}<br>
                      Difference: ${parts['balance_adv']:.2f}"""

                  elif t == gamma:
                      hover_text = f"""<b>Month {t} - GAMMA POINT - {label}</b><br>
                      <b>Net Gain: ${rec['total_adv']:.2f}</b><br>
                      Investment at Gamma: ${parts['inv_bal_at_gamma']:.2f}<br>
                      Remaining New Balance: ${parts['bal_new']:.2f}"""

                  else:
                      hover_text = f"""<b>Month {t} - POST GAMMA - {label}</b><br>
                      <b>Net Gain Calculation:</b><br>
                      (Option 2 - New Balance) - Option 1<br>
                      = ({parts['opt2_sav']:.2f} - {parts['bal_new']:.2f}) - {parts['opt1_sav']:.2f}<br>
                      = <b>${rec['total_adv']:.2f}</b><br><br>

                      <b>Option 1 (No Refi) Detail:</b><br>
                      Previous: ${parts['opt1_sav_prev']:.2f}<br>
                      Interest: ${parts['opt1_interest']:.2f}<br>
                      Old Payment: ${parts['pmt_old']:.2f}<br>
                      New Total: ${parts['opt1_sav']:.2f}<br><br>

                      <b>Option 2 (Did Refi) Detail:</b><br>
                      Previous: ${parts['opt2_sav_prev']:.2f}<br>
                      Interest: ${parts['opt2_interest']:.2f}<br>
                      New Total: ${parts['opt2_sav']:.2f}<br>
                      Remaining Loan: ${parts['bal_new']:.2f}"""

                  hover_texts.append(hover_text)

              fig1.add_trace(go.Scatter(
                  x=months,
                  y=net_gains,
                  mode='lines+markers',
                  name=label,
                  line=dict(width=2, color=color),
                  marker=dict(size=4),
                  hovertemplate='%{text}<extra></extra>',
                  text=hover_texts
              ))

          # Add reference lines
          fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
          fig1.add_vline(x=gamma, line_dash="dash", line_color="red", opacity=0.5,
                        annotation_text=f"Gamma ({gamma} mo)")

          fig1.update_layout(
              title="Net Gain Comparison - Hover for Detailed Calculations",
              xaxis_title="Month",
              yaxis_title="Net Gain ($)",
              height=600,
              hovermode='closest'
          )

          st.plotly_chart(fig1, use_container_width=True)

          # Create component breakdown chart
          st.markdown("---")
          st.subheader("📊 Component Breakdown")

          # Show component charts for the first selected scenario
          first_history = all_histories[0][0]
          first_label = all_histories[0][2]

          months = [rec["month"] for rec in first_history]

          # Extract component data
          inv_bals = []
          opt1_savs = []
          bal_olds = [rec["bal_old"] for rec in first_history]
          bal_news = [rec["bal_new"] for rec in first_history]

          for rec in first_history:
              if rec["month"] <= gamma:
                  inv_bals.append(rec["inv_bal"])
                  opt1_savs.append(0)
              else:
                  inv_bals.append(rec["calculation_parts"]["opt2_sav"])
                  opt1_savs.append(rec["opt1_sav"])

          fig2 = go.Figure()

          # Add traces for each component
          fig2.add_trace(go.Scatter(
              x=months,
              y=bal_olds,
              mode='lines',
              name='Old Loan Balance',
              line=dict(width=2, color='darkblue'),
              stackgroup='one'
          ))

          fig2.add_trace(go.Scatter(
              x=months,
              y=bal_news,
              mode='lines',
              name='New Loan Balance',
              line=dict(width=2, color='darkred'),
              stackgroup='two'
          ))

          fig2.add_trace(go.Scatter(
              x=months,
              y=inv_bals,
              mode='lines',
              name='Investment/Option 2 Savings',
              line=dict(width=2, color='green'),
              stackgroup='three'
          ))

          fig2.add_trace(go.Scatter(
              x=months,
              y=opt1_savs,
              mode='lines',
              name='Option 1 Savings (Post-Gamma)',
              line=dict(width=2, color='orange'),
              stackgroup='four'
          ))

          # Add gamma line
          fig2.add_vline(x=gamma, line_dash="dash", line_color="red", opacity=0.5,
                        annotation_text=f"Gamma ({gamma} mo)")

          fig2.update_layout(
              title=f"Component Breakdown - {first_label}",
              xaxis_title="Month",
              yaxis_title="Amount ($)",
              height=600,
              hovermode='x unified'
          )

          st.plotly_chart(fig2, use_container_width=True)

          # Summary statistics
          st.markdown("---")
          st.subheader("📊 Summary Statistics")

          summary_data = []
          for history, _, label in all_histories:
              final_gain = history[-1]["total_adv"]
              max_gain = max(rec["total_adv"] for rec in history)
              breakeven = next((rec["month"] for rec in history if rec["total_adv"] >= 0), None)

              summary_data.append({
                  'Scenario': label,
                  'Final Net Gain': f"${final_gain:,.2f}",
                  'Max Net Gain': f"${max_gain:,.2f}",
                  'Breakeven Month': f"{breakeven} months" if breakeven else "Never"
              })

          st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

      else:
          st.info("Select one or more rows using the checkboxes to see the comparison charts.")

with tab6:
      st.header("📊 ENPV Analysis - Detailed Cash Flow Model")

      st.markdown("""
      This analysis uses the detailed cash flow model from the imp file to calculate Expected Net Present Value (ENPV)
      with mortality-weighted prepayment probabilities.
      """)

      # Input parameters for ENPV calculation
      st.subheader("📈 ENPV Model Parameters")

      col1, col2, col3, col4 = st.columns(4)

      with col1:
          enpv_new_rate = st.number_input(
              "New Rate (%)",
              min_value=0.0,
              max_value=20.0,
              value=5.0,
              step=0.125,
              help="The refinance rate you're considering"
          ) / 100

      with col2:
          enpv_invest_rate = st.number_input(
              "Investment Rate (%)",
              min_value=0.0,
              max_value=20.0,
              value=rho*100,
              step=0.5,
              help="Annual return on invested payment savings"
          ) / 100

      with col3:
          enpv_discount_rate = st.number_input(
              "Discount Rate (%)",
              min_value=0.0,
              max_value=20.0,
              value=rho*100,
              step=0.5,
              help="Rate for present value calculations"
          ) / 100

      with col4:
          enpv_cpr = st.number_input(
              "CPR (%)",
              min_value=0.0,
              max_value=100.0,
              value=mu*100,
              step=1.0,
              help="Conditional Prepayment Rate"
          ) / 100

      col1b, col2b, col3b, col4b = st.columns(4)

      with col1b:
          enpv_new_term = st.number_input(
              "New Loan Term (years)",
              min_value=15,
              max_value=30,
              value=30,
              step=5,
              help="Term for new refinanced loan"
          )

      with col2b:
          enpv_closing_costs = st.number_input(
              "Total Closing Costs ($)",
              min_value=0,
              max_value=50000,
              value=int(kappa),
              step=500,
              help="Total refinancing costs"
          )

      with col3b:
          enpv_finance_costs = st.checkbox(
              "Finance costs in loan",
              value=True,
              help="Roll closing costs into the new loan"
          )

      with col4b:
          include_taxes = st.checkbox(
              "Include tax effects",
              value=True,
              help="Account for mortgage interest deduction"
          )

      # Helper functions
      def payment(principal, monthly_rate, n_months):
          """Level payment on an amortizing loan."""
          if monthly_rate == 0:
              return principal / n_months
          denom = 1.0 - (1.0 + monthly_rate) ** (-n_months)
          return principal * monthly_rate / denom

      def compute_enpv_full(current_balance, current_rate, new_rate, remaining_years_old, new_term_years,
                           closing_costs, finance_costs_in_loan, invest_rate, discount_rate, cpr, tau_rate, include_tax):
          """Full ENPV calculation matching the imp file"""
          n_old = int(round(remaining_years_old * 12))
          n_new = int(round(new_term_years * 12))
          horizon = max(n_old, n_new)
          gamma_month = n_old

          r_old = current_rate / 12.0
          r_new = new_rate / 12.0
          r_inv = invest_rate / 12.0
          r_disc = discount_rate / 12.0

          old_principal = current_balance
          if finance_costs_in_loan:
              new_principal = current_balance + closing_costs
          else:
              new_principal = current_balance

          # Monthly payments
          pmt_old = payment(old_principal, r_old, n_old)
          pmt_new = payment(new_principal, r_new, n_new)

          bal_old = old_principal
          bal_new = new_principal
          cum_sav = 0.0
          inv_bal = 0.0

          history = []
          opt1_sav = 0.0
          opt2_sav = 0.0
          inv_bal_at_gamma = None

          for t in range(1, horizon + 1):
              # Old loan
              if t <= n_old and bal_old > 0:
                  interest_old = r_old * bal_old
                  principal_old = pmt_old - interest_old
                  bal_old = max(0.0, bal_old - principal_old)
                  if include_tax:
                      p_old_t = pmt_old - (interest_old * tau_rate)
                  else:
                      p_old_t = pmt_old
              else:
                  p_old_t = 0.0
                  bal_old = 0.0

              # New loan
              if t <= n_new and bal_new > 0:
                  interest_new = r_new * bal_new
                  principal_new = pmt_new - interest_new
                  bal_new = max(0.0, bal_new - principal_new)
                  if include_tax:
                      p_new_t = pmt_new - (interest_new * tau_rate)
                  else:
                      p_new_t = pmt_new
              else:
                  p_new_t = 0.0
                  bal_new = 0.0

              # Payment savings
              pmt_sav_t = p_old_t - p_new_t
              cum_sav += pmt_sav_t

              # Investment account and total advantage
              if t < gamma_month:
                  inv_bal = inv_bal * (1.0 + r_inv) + pmt_sav_t
                  balance_adv = bal_old - bal_new
                  total_adv = inv_bal + balance_adv
              elif t == gamma_month:
                  inv_bal = inv_bal * (1.0 + r_inv) + pmt_sav_t
                  inv_bal_at_gamma = inv_bal
                  opt2_sav = inv_bal_at_gamma
                  opt1_sav = 0.0
                  balance_adv = bal_old - bal_new
                  total_adv = inv_bal + balance_adv
              else:
                  # After gamma: new logic
                  opt1_sav = opt1_sav * (1.0 + r_inv) + pmt_old
                  opt2_sav = opt2_sav * (1.0 + r_inv)
                  total_adv = (opt2_sav - bal_new) - opt1_sav

              rec = {
                  "month": t,
                  "p_old": p_old_t,
                  "p_new": p_new_t,
                  "pmt_sav_t": pmt_sav_t,
                  "cum_sav": cum_sav,
                  "inv_bal": inv_bal if t <= gamma_month else opt2_sav,
                  "bal_old": bal_old,
                  "bal_new": bal_new,
                  "balance_adv": bal_old - bal_new,
                  "total_adv": total_adv,
              }
              history.append(rec)

          return history, pmt_old, pmt_new, gamma_month

      # Run calculation
      history, pmt_old_calc, pmt_new_calc, gamma_month = compute_enpv_full(
          current_balance=M,
          current_rate=i0,
          new_rate=enpv_new_rate,
          remaining_years_old=Gamma,
          new_term_years=enpv_new_term,
          closing_costs=enpv_closing_costs,
          finance_costs_in_loan=enpv_finance_costs,
          invest_rate=enpv_invest_rate,
          discount_rate=enpv_discount_rate,
          cpr=enpv_cpr,
          tau_rate=tau if include_taxes else 0,
          include_tax=include_taxes
      )

      # Calculate NPV and ENPV
      months = [rec["month"] for rec in history]
      net_gain_fv = [rec["total_adv"] for rec in history]
      net_gain_pv = [gain / ((1.0 + enpv_discount_rate/12) ** t) for gain, t in zip(net_gain_fv, months)]

      # Calculate ENPV with mortality
      SMM = 1 - (1 - enpv_cpr)**(1/12)
      survival = 1.0
      mortality = []
      npv_times_mortality = []

      # Extend to 360 months
      last_pv = net_gain_pv[-1] if net_gain_pv else 0
      while len(net_gain_pv) < 360:
          net_gain_pv.append(last_pv)

      for t in range(360):
          m_t = survival * SMM
          mortality.append(m_t)
          npv_times_mort = net_gain_pv[t] * m_t
          npv_times_mortality.append(npv_times_mort)
          survival = survival * (1 - SMM)

      # Add remaining survival to month 360
      if survival > 0.001:
          mortality[-1] += survival
          npv_times_mortality[-1] = net_gain_pv[-1] * mortality[-1]

      ENPV = sum(npv_times_mortality)

      # Display results
      st.markdown("---")
      st.subheader("📊 Results Summary")

      col1r, col2r, col3r, col4r = st.columns(4)

      with col1r:
          st.metric("Old Payment", f"${pmt_old_calc:,.2f}")

      with col2r:
          st.metric("New Payment", f"${pmt_new_calc:,.2f}")

      with col3r:
          monthly_savings = pmt_old_calc - pmt_new_calc
          st.metric("Monthly Savings", f"${monthly_savings:,.2f}")

      with col4r:
          st.metric("**ENPV**", f"**${ENPV:,.2f}**",
                   "Good deal!" if ENPV > 0 else "Consider waiting")

      # Additional metrics
      st.markdown("---")
      col1m, col2m, col3m = st.columns(3)

      with col1m:
          st.metric("SMM (monthly)", f"{SMM*100:.5f}%")

      with col2m:
          # Find break-even month
          breakeven_month = None
          for rec in history:
              if rec["total_adv"] >= 0:
                  breakeven_month = rec["month"]
                  break
          if breakeven_month:
              st.metric("Break-even", f"{breakeven_month} months ({breakeven_month/12:.1f} years)")
          else:
              st.metric("Break-even", "Never")

      with col3m:
          st.metric("Gamma (Γ)", f"{gamma_month} months ({gamma_month/12:.1f} years)")

      # Full table
      st.markdown("---")
      st.subheader("📋 Full 360-Month ENPV Table")

      # Create full dataframe
      table_data = []
      for t in range(360):
          if t < len(history):
              rec = history[t]
              row = {
                  'Month': t + 1,
                  'NPV ($)': net_gain_pv[t],
                  'Mortality': mortality[t],
                  'NPV × Mortality ($)': npv_times_mortality[t],
                  'Cumulative ENPV ($)': sum(npv_times_mortality[:t+1])
              }
          else:
              row = {
                  'Month': t + 1,
                  'NPV ($)': net_gain_pv[t],
                  'Mortality': mortality[t],
                  'NPV × Mortality ($)': npv_times_mortality[t],
                  'Cumulative ENPV ($)': sum(npv_times_mortality[:t+1])
              }
          table_data.append(row)

      df_enpv = pd.DataFrame(table_data)

      # Display with formatting
      st.dataframe(
          df_enpv.style.format({
              'Month': '{:d}',
              'NPV ($)': '${:,.2f}',
              'Mortality': '{:.8f}',
              'NPV × Mortality ($)': '${:,.2f}',
              'Cumulative ENPV ($)': '${:,.2f}'
          }),
          use_container_width=True,
          height=400
      )

      # Download button
      csv_enpv = df_enpv.to_csv(index=False)
      st.download_button(
          label="Download Full ENPV Table",
          data=csv_enpv,
          file_name="enpv_full_analysis.csv",
          mime="text/csv"
      )

      # Charts
      st.markdown("---")
      st.subheader("📈 Visualizations")

      # Chart 1: Net Benefit (Future Value)
      months_display = [rec["month"] for rec in history]
      net_gain_fv_display = [rec["total_adv"] for rec in history]

      fig1 = go.Figure()
      fig1.add_trace(go.Scatter(
          x=months_display,
          y=net_gain_fv_display,
          mode='lines',
          name='Net Benefit',
          line=dict(width=2, color='blue')
      ))
      fig1.add_hline(y=0, line_dash="dash", line_color="gray")

      # Add gamma line
      fig1.add_vline(x=gamma_month, line_dash="dash", line_color="red",
                    annotation_text=f"Gamma ({gamma_month} months)")

      fig1.update_layout(
          title="Net Benefit of Refinancing (Future Value)",
          xaxis_title="Month",
          yaxis_title="Net Gain (FV $)",
          height=500,
          hovermode='x unified'
      )

      st.plotly_chart(fig1, use_container_width=True)

      # Chart 2: Net Benefit (Present Value)
      fig2 = go.Figure()
      fig2.add_trace(go.Scatter(
          x=months_display,
          y=net_gain_pv[:len(months_display)],
          mode='lines',
          name='Net Benefit (PV)',
          line=dict(width=2, color='green')
      ))
      fig2.add_hline(y=0, line_dash="dash", line_color="gray")

      # Add gamma line
      fig2.add_vline(x=gamma_month, line_dash="dash", line_color="red",
                    annotation_text=f"Gamma ({gamma_month} months)")

      fig2.update_layout(
          title="Net Benefit of Refinancing (Present Value)",
          xaxis_title="Month",
          yaxis_title="Net Gain (PV $)",
          height=500,
          hovermode='x unified'
      )

      st.plotly_chart(fig2, use_container_width=True)

      # Explanation
      st.markdown("---")
      st.info("""
      **Understanding ENPV Analysis:**

      - **ENPV** = Expected Net Present Value, accounting for the probability of prepayment each month
      - **Mortality** = Probability of prepaying in that specific month (based on CPR)
      - **Gamma (Γ)** = Month when the original loan would be paid off
      - **Post-Gamma Logic**: After gamma, the model compares having the old payment to invest vs. continuing with the new loan
      - **Break-even**: The month when cumulative benefits exceed refinancing costs

      The ENPV gives the expected value of refinancing, weighted by how long you're likely to keep the mortgage.
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
