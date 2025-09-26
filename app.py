import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
import re
from datetime import datetime, timedelta
from io import StringIO

st.set_page_config(layout="wide")

# File upload moved to sidebar
st.sidebar.subheader("Gamma Exposure Chart")
try:
    with open("quotedata.csv", "r") as f:
        lines = f.read().splitlines()
except FileNotFoundError:
    st.error("'quotedata.csv' not found and no file uploaded.")
    st.stop()

# Check structure
if len(lines) < 4:
    st.error("CSV missing rows.")
    st.stop()

# --- Summary ---
name_line = lines[1]
quote_line = lines[2]

name_match = re.match(r"^(.*?),Last:\s*([\d.]+),Change:\s*([-.\d]+)", name_line)
quote_match = re.match(r'^"Date:\s*(.*?)",Bid:\s*([\d.]+),Ask:\s*([\d.]+)', quote_line)

summary_rendered = False
if name_match and quote_match:
    name, last, change = name_match.groups()
    raw_date, bid, ask = quote_match.groups()
    date = raw_date.split(' EDT')[0]
    last = float(last)
    change = float(change)
    bid = float(bid)
    ask = float(ask)
    color = "green" if change >= 0 else "red"
    st.markdown(f"""
    <table style='width:100%; font-size:18px;'>
        <thead><tr><th>Symbol</th><th>Last</th><th>Bid</th><th>Ask</th></tr></thead>
        <tbody><tr>
            <td><b>{name}</b></td>
            <td style='color:{color}'><b>{last:.2f} ({change:+.2f})</b></td>
            <td>{bid}</td><td>{ask}</td>
        </tr></tbody>
        <tfoot><tr><td colspan='4' style='font-size:14px; color:gray;'>As of {date}</td></tr></tfoot>
    </table>
    """, unsafe_allow_html=True)
    summary_rendered = True

# --- DataFrame ---
data_str = "\n".join(lines[3:])
df = pd.read_csv(StringIO(data_str))

# Rename and parse
df["Expiration Date"] = pd.to_datetime(df["Expiration Date"], errors="coerce")
df = df.dropna(subset=["Expiration Date", "Strike"])
df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
df = df.dropna(subset=["Strike"])

today = pd.Timestamp.today().normalize()
df["DTE"] = (df["Expiration Date"] - today).dt.days
df = df[df["DTE"] >= 0]

# Use known column positions
call_gamma = pd.to_numeric(df.iloc[:, 9], errors="coerce")
call_oi = pd.to_numeric(df.iloc[:, 10], errors="coerce")
call_volume = pd.to_numeric(df.iloc[:, 11], errors="coerce")
put_gamma = pd.to_numeric(df.iloc[:, 20], errors="coerce")
put_oi = pd.to_numeric(df.iloc[:, 21], errors="coerce")
put_volume = pd.to_numeric(df.iloc[:, 22], errors="coerce")

df["call_gamma_expo"] = call_gamma * call_oi * 100
df["put_gamma_expo"] = put_gamma * put_oi * 100 * -1
df["call_oi"] = call_oi
df["put_oi"] = put_oi
df["call_volume"] = call_volume
df["put_volume"] = put_volume

# Now let's use some functions similar to those in gex.py to calculate the gamma profile
# Black-Scholes European-Options Gamma
def calc_gamma_ex(S, K, vol, T, r, q, opt_type, OI):
    """Calculate gamma exposure for a specific option"""
    if T <= 0 or vol <= 0:
        return 0

    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T) 

    if opt_type == 'call':
        gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma 
    else:  # Gamma is same for calls and puts
        gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma

def is_third_friday(d):
    """Check if date is the third Friday of the month"""
    return d.weekday() == 4 and 15 <= d.day <= 21

def get_gamma_profile(df, spot_price, strike_range, r=0, q=0, strike_levels=None):
    """Calculate gamma profile across a range of price levels"""
    from_strike = spot_price * strike_range[0]/100
    to_strike = spot_price * strike_range[1]/100
    
    # Use provided strike levels if available, otherwise use linspace
    if strike_levels is not None:
        # Use actual strike prices from the data, but extend range slightly for smooth curves
        unique_strikes = np.sort(strike_levels.unique())
        min_strike = unique_strikes.min()
        max_strike = unique_strikes.max()
        
        # Create a denser grid that includes all actual strikes plus interpolated points
        padding = (max_strike - min_strike) * 0.1
        extended_range = np.linspace(min_strike - padding, max_strike + padding, 80)
        
        # Combine actual strikes with extended range and sort
        all_levels = np.concatenate([unique_strikes, extended_range])
        levels = np.sort(np.unique(all_levels))
    else:
        levels = np.linspace(from_strike, to_strike, 60)
    
    today_date = datetime.now().date()
    
    # Calculate DTE in years for Black-Scholes
    df.loc[:, 'daysTillExp'] = [(x - today).days/365 for x in df["Expiration Date"]]
    # For 0DTE options, setting minimum DTE to avoid division by zero
    df.loc[:, 'daysTillExp'] = df['daysTillExp'].apply(lambda x: max(x, 1/262))
    
    # Handle potential NaN values
    df.loc[:, 'daysTillExp'] = df['daysTillExp'].fillna(1/262)
    
    next_expiry = df['Expiration Date'].min()
    
    df['IsThirdFriday'] = [is_third_friday(x) for x in df["Expiration Date"]]
    third_fridays = df.loc[df['IsThirdFriday'] == True]
    
    if len(third_fridays) > 0:
        next_monthly_exp = third_fridays['Expiration Date'].min()
    else:
        next_monthly_exp = next_expiry
    
    total_gamma = []
    total_gamma_ex_next = []
    total_gamma_ex_fri = []
    
    # For each spot level, calculate gamma exposure
    for level in levels:
        # Use user-provided volatility estimate
        avg_vol = avg_volatility
        
        df.loc[:, 'callGammaEx'] = df.apply(lambda row: calc_gamma_ex(
            level, row['Strike'], avg_vol, 
            row['daysTillExp'], r, q, "call", row['call_oi']), axis=1)
            
        df.loc[:, 'putGammaEx'] = df.apply(lambda row: calc_gamma_ex(
            level, row['Strike'], avg_vol, 
            row['daysTillExp'], r, q, "put", row['put_oi']), axis=1)
            
        total_gamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())
        
        ex_next = df.loc[df['Expiration Date'] != next_expiry]
        total_gamma_ex_next.append(ex_next['callGammaEx'].sum() - ex_next['putGammaEx'].sum())
        
        ex_fri = df.loc[df['Expiration Date'] != next_monthly_exp]
        total_gamma_ex_fri.append(ex_fri['callGammaEx'].sum() - ex_fri['putGammaEx'].sum())
    
    # Convert to billions
    total_gamma = np.array(total_gamma) / 10**9
    total_gamma_ex_next = np.array(total_gamma_ex_next) / 10**9
    total_gamma_ex_fri = np.array(total_gamma_ex_fri) / 10**9
    
    # Find Gamma Flip Point (where gamma crosses zero)
    zero_cross_idx = np.where(np.diff(np.sign(total_gamma)))[0]
    
    if len(zero_cross_idx) > 0:
        neg_gamma = total_gamma[zero_cross_idx]
        pos_gamma = total_gamma[zero_cross_idx+1]
        neg_strike = levels[zero_cross_idx]
        pos_strike = levels[zero_cross_idx+1]
        
        zero_gamma = pos_strike - ((pos_strike - neg_strike) * pos_gamma/(pos_gamma-neg_gamma))
        zero_gamma = zero_gamma[0]
    else:
        zero_gamma = None
    
    return levels, total_gamma, total_gamma_ex_next, total_gamma_ex_fri, zero_gamma

# --- Filters ---
max_dte = int(df["DTE"].max())
days_ahead = st.sidebar.slider("Max DTE (days)", 0, max_dte, min(7, max_dte))
df = df[df["DTE"] <= days_ahead]

spot_price = last if summary_rendered else df["Strike"].median()
strike_range = st.sidebar.slider("Strike range (± around spot)", 0, 200, 50)
lo, hi = spot_price - strike_range / 2, spot_price + strike_range / 2
df = df[(df["Strike"] >= lo) & (df["Strike"] <= hi)]

# Sidebar for risk-free rate, dividend yield, and volatility
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.267, min_value=0.0, max_value=10.0, step=0.001) / 100
dividend_yield = st.sidebar.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, max_value=10.0, step=0.25) / 100
avg_volatility = st.sidebar.number_input("Average Volatility", value=0.17, min_value=0.01, max_value=2.0, step=0.01)

grouped = df.groupby(["DTE", "Strike"]).agg({
    "call_gamma_expo": "sum",
    "put_gamma_expo": "sum",
    "call_oi": "sum",
    "put_oi": "sum",
    "call_volume": "sum",
    "put_volume": "sum"
}).reset_index()

if grouped.empty:
    st.warning("No data in selected range.")
    st.stop()

bar_mode = st.sidebar.radio("Bar Mode", ["Stacked", "Grouped (side-by-side)"], index=0)
bar_mode_val = "stack" if bar_mode == "Stacked" else "relative"

# File uploader in sidebar
# uploaded_file = st.sidebar.file_uploader("Upload a CSV file of options chain data from https://www.cboe.com/delayed_quotes/spy/quote_table", type=["csv"])
# if uploaded_file:
#    lines = uploaded_file.read().decode("utf-8").splitlines()

grouped["abs_total"] = grouped["call_gamma_expo"].abs() + grouped["put_gamma_expo"].abs()
sorted_dtes = grouped.groupby("DTE")["abs_total"].sum().sort_values(ascending=False).index.tolist()

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

# Create custom hover template
def format_number(num):
    if abs(num) >= 1e9:
        return f"{num/1e9:.3f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.3f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.0f}K"
    else:
        return f"{num:.0f}"

# Create the figure
fig = go.Figure()

# Calculate gamma profile from gex.py
try:
    # We need to ensure our strike range covers the full range of available strikes
    # Use the actual min/max strikes from the filtered data for better alignment
    if len(grouped) > 0:
        actual_lo = grouped["Strike"].min()
        actual_hi = grouped["Strike"].max()
        # Add some padding to the range
        strike_padding = (actual_hi - actual_lo) * 0.1
        from_strike = max(lo, actual_lo - strike_padding)
        to_strike = min(hi, actual_hi + strike_padding)
        percent_range = (from_strike/spot_price*100, to_strike/spot_price*100)
        
        levels, total_gamma, total_gamma_ex_next, total_gamma_ex_fri, zero_gamma = get_gamma_profile(
            df, spot_price, percent_range, risk_free_rate, dividend_yield, grouped["Strike"])
    else:
        # If no data, create dummy values to avoid errors
        levels = np.linspace(lo, hi, 60)
        total_gamma = np.zeros_like(levels)
        total_gamma_ex_next = np.zeros_like(levels)
        total_gamma_ex_fri = np.zeros_like(levels)
        zero_gamma = None
        st.warning("Insufficient data for gamma profile calculation.")
except Exception as e:
    st.error(f"Error calculating gamma profile: {str(e)}")
    # Create dummy values to avoid errors
    levels = np.linspace(lo, hi, 60)
    total_gamma = np.zeros_like(levels)
    total_gamma_ex_next = np.zeros_like(levels)
    total_gamma_ex_fri = np.zeros_like(levels)
    zero_gamma = None

# Unified coordinate system calculation
def calculate_unified_axis_bounds(bar_data_min, bar_data_max, gamma_data_min, gamma_data_max, padding_factor=1.2):
    """
    Calculate unified axis bounds that ensure zero alignment between primary and secondary axes.

    Returns:
        tuple: (primary_min, primary_max, secondary_min, secondary_max)
    """
    # Calculate the natural ranges for both datasets
    bar_range = abs(bar_data_max - bar_data_min)
    gamma_range = abs(gamma_data_max - gamma_data_min)

    # Add padding to both ranges
    bar_padded_min = bar_data_min - (bar_range * (padding_factor - 1) / 2)
    bar_padded_max = bar_data_max + (bar_range * (padding_factor - 1) / 2)

    gamma_padded_min = gamma_data_min - (gamma_range * (padding_factor - 1) / 2)
    gamma_padded_max = gamma_data_max + (gamma_range * (padding_factor - 1) / 2)

    # Ensure zero is included in both ranges
    bar_padded_min = min(bar_padded_min, 0)
    bar_padded_max = max(bar_padded_max, 0)
    gamma_padded_min = min(gamma_padded_min, 0)
    gamma_padded_max = max(gamma_padded_max, 0)

    # Calculate the zero position as a fraction of each range
    bar_total_range = bar_padded_max - bar_padded_min
    gamma_total_range = gamma_padded_max - gamma_padded_min

    bar_zero_fraction = abs(bar_padded_min) / bar_total_range if bar_total_range != 0 else 0.5
    gamma_zero_fraction = abs(gamma_padded_min) / gamma_total_range if gamma_total_range != 0 else 0.5

    # Adjust ranges so zero is at the same relative position
    target_zero_fraction = max(bar_zero_fraction, gamma_zero_fraction)

    # Recalculate ranges with aligned zero position
    if target_zero_fraction > 0 and target_zero_fraction < 1:
        # For bar axis
        if bar_zero_fraction != target_zero_fraction:
            bar_left_range = bar_total_range * target_zero_fraction
            bar_right_range = bar_total_range * (1 - target_zero_fraction)
            bar_unified_min = -bar_left_range
            bar_unified_max = bar_right_range
        else:
            bar_unified_min = bar_padded_min
            bar_unified_max = bar_padded_max

        # For gamma axis
        if gamma_zero_fraction != target_zero_fraction:
            gamma_left_range = gamma_total_range * target_zero_fraction
            gamma_right_range = gamma_total_range * (1 - target_zero_fraction)
            gamma_unified_min = -gamma_left_range
            gamma_unified_max = gamma_right_range
        else:
            gamma_unified_min = gamma_padded_min
            gamma_unified_max = gamma_padded_max
    else:
        # Fallback to original ranges if calculation fails
        bar_unified_min, bar_unified_max = bar_padded_min, bar_padded_max
        gamma_unified_min, gamma_unified_max = gamma_padded_min, gamma_padded_max

    # Make both axes symmetric around zero for perfect visual centering
    bar_max_abs = max(abs(bar_unified_min), abs(bar_unified_max))
    gamma_max_abs = max(abs(gamma_unified_min), abs(gamma_unified_max))

    bar_unified_min = -bar_max_abs
    bar_unified_max = bar_max_abs
    gamma_unified_min = -gamma_max_abs
    gamma_unified_max = gamma_max_abs

    return bar_unified_min, bar_unified_max, gamma_unified_min, gamma_unified_max

# Calculate unified axis bounds for proper zero alignment
if len(grouped) > 0 and len(total_gamma) > 0:
    bar_data_min = grouped["put_gamma_expo"].min()
    bar_data_max = grouped["call_gamma_expo"].max()
    gamma_data_min = min(total_gamma.min(), total_gamma_ex_next.min(), total_gamma_ex_fri.min())
    gamma_data_max = max(total_gamma.max(), total_gamma_ex_next.max(), total_gamma_ex_fri.max())

    bar_x_min, bar_x_max, gamma_x_min, gamma_x_max = calculate_unified_axis_bounds(
        bar_data_min, bar_data_max, gamma_data_min, gamma_data_max
    )
else:
    # Fallback values
    bar_x_min, bar_x_max = -1000, 1000
    gamma_x_min, gamma_x_max = -1, 1

# Add background color areas based on gamma flip point
if zero_gamma is not None:
    
    # Background for PRIMARY x-axis (bar chart area) - GREEN above flip
    fig.add_shape(
        type="rect",
        x0=bar_x_min, x1=bar_x_max,
        y0=zero_gamma, y1=hi + (hi-lo)*0.05,
        fillcolor="rgba(0, 255, 0, 0.05)",
        line=dict(width=0),
        layer="below",
        xref="x"  # Primary x-axis
    )
    
    # Background for PRIMARY x-axis (bar chart area) - RED below flip  
    fig.add_shape(
        type="rect",
        x0=bar_x_min, x1=bar_x_max,
        y0=lo - (hi-lo)*0.05, y1=zero_gamma,
        fillcolor="rgba(255, 0, 0, 0.05)",
        line=dict(width=0),
        layer="below",
        xref="x"  # Primary x-axis
    )
    
    # Background for SECONDARY x-axis (gamma profile area) - GREEN above flip
    fig.add_shape(
        type="rect",
        x0=gamma_x_min, x1=gamma_x_max,
        y0=zero_gamma, y1=hi + (hi-lo)*0.05,
        fillcolor="rgba(0, 255, 0, 0.05)",
        line=dict(width=0),
        layer="below",
        xref="x2"  # Secondary x-axis
    )
    
    # Background for SECONDARY x-axis (gamma profile area) - RED below flip
    fig.add_shape(
        type="rect",
        x0=gamma_x_min, x1=gamma_x_max,
        y0=lo - (hi-lo)*0.05, y1=zero_gamma,
        fillcolor="rgba(255, 0, 0, 0.05)",
        line=dict(width=0),
        layer="below",
        xref="x2"  # Secondary x-axis
    )

# Add the bar charts from app.py (adjusted to appear behind line charts)
for i, dte in enumerate(sorted_dtes):
    color = colors[i % len(colors)]
    sub = grouped[grouped["DTE"] == dte]
    
    # Calculate net gamma exposure for hover text
    sub["net_gamma"] = sub["call_gamma_expo"] + sub["put_gamma_expo"]
    
    # Create hover template for puts
    put_hovertemplate = (
        "Strike: %{y:,.2f}<br>" +
        "Expiration: %{customdata[0]}<br>" +
        "• Net Gamma Exposure: %{customdata[1]}<br>" +
        "Call Open Interest: %{customdata[2]}<br>" +
        "Put Open Interest: %{customdata[3]}<br>" +
        "Call Volume: %{customdata[4]}<br>" +
        "Put Volume: %{customdata[5]}<br>" +
        "<extra></extra>"
    )
    
    # Create custom data for hover template
    # Get the expiration date for this DTE
    expiration_date = ""
    try:
        exp_date_row = df[df["DTE"] == dte].iloc[0]
        if isinstance(exp_date_row["Expiration Date"], pd.Timestamp):
            expiration_date = exp_date_row["Expiration Date"].strftime("%B %d, %Y")
        else:
            expiration_date = str(exp_date_row["Expiration Date"])
    except (IndexError, KeyError):
        expiration_date = f"DTE: {dte}"
    
    # Use the same expiration date for all rows with this DTE
    expiration_dates = [expiration_date] * len(sub)
    
    customdata = list(zip(
        expiration_dates,
        [format_number(val) for val in sub["net_gamma"]],
        [format_number(val) for val in sub["call_oi"]],
        [format_number(val) for val in sub["put_oi"]],
        [format_number(val) for val in sub["call_volume"]],
        [format_number(val) for val in sub["put_volume"]]
    ))
    
    fig.add_trace(go.Bar(
        x=sub["put_gamma_expo"],
        y=sub["Strike"],
        orientation="h",
        marker_color=color,
        name=f"{dte} DTE",
        legendgroup=f"{dte} DTE",
        showlegend=True,
        width=0.7,
        hovertemplate=put_hovertemplate,
        customdata=customdata
    ))
    
    fig.add_trace(go.Bar(
        x=sub["call_gamma_expo"],
        y=sub["Strike"],
        orientation="h",
        marker_color=color,
        name=f"{dte} DTE",
        legendgroup=f"{dte} DTE",
        showlegend=False,
        width=0.7,
        hovertemplate=put_hovertemplate,
        customdata=customdata
    ))

# Add gamma profile as line charts (now added after bars to appear on top)
fig.add_trace(go.Scatter(
    y=levels,
    x=total_gamma,  # Note: x and y are swapped from gex.py since we're using horizontal orientation
    mode='lines',
    name='Gamma Profile (All Expiries)',
    line=dict(color='blue', width=3),
    yaxis='y',
    xaxis='x2',
    legendgroup='Gamma Profile',
    hovertemplate="Price: %{y:,.2f}<br>Gamma: %{x:,.3f}B<extra></extra>"
))

fig.add_trace(go.Scatter(
    y=levels,
    x=total_gamma_ex_next,
    mode='lines',
    name='Ex-Next Expiry',
    line=dict(color='orange', width=2, dash='dash'),
    yaxis='y',
    xaxis='x2',
    legendgroup='Gamma Profile',
    hovertemplate="Price: %{y:,.2f}<br>Gamma: %{x:,.3f}B<extra></extra>"
))

fig.add_trace(go.Scatter(
    y=levels,
    x=total_gamma_ex_fri,
    mode='lines',
    name='Ex-Next Monthly',
    line=dict(color='purple', width=2, dash='dot'),
    yaxis='y',
    xaxis='x2',
    legendgroup='Gamma Profile',
    hovertemplate="Price: %{y:,.2f}<br>Gamma: %{x:,.3f}B<extra></extra>"
))

# Vertical line at x=0 is handled by the secondary axis zeroline

# Add horizontal line at current spot price
fig.add_shape(
    type="line",
    x0=0, x1=1,               # full width of chart (0 = left, 1 = right)
    y0=spot_price, y1=spot_price,
    xref="paper", yref="y",   # x in "paper" (0–1), y in data coordinates
    line=dict(color="green", width=3, dash="dot")
)

fig.add_shape(type="line",
              x0=grouped["put_gamma_expo"].min(),
              x1=grouped["call_gamma_expo"].max(),
              y0=spot_price,
              y1=spot_price,
              line=dict(color="green", width=3, dash="dot"))

# Add spot price annotation on the left side
fig.add_annotation(
    x=bar_x_min,  # Use left edge of unified axis range
    y=spot_price,
    text="Spot price",
    showarrow=False,
    xanchor="left",
    yshift=10,
    font=dict(color="lightgreen", size=14),
    bgcolor="rgba(0.1,0.1,0.2, 0.0)"
)

# Add gamma flip point annotation if it exists
if zero_gamma is not None:
    fig.add_shape(
        type="line",
        x0=0, x1=1,               
        y0=zero_gamma, y1=zero_gamma,
        xref="paper", yref="y",   
        line=dict(color="red", width=2, dash="dot")
    )
    
    fig.add_annotation(
        x=bar_x_max,  # Use right edge of unified axis range
        y=zero_gamma,
        text="Gamma Flip Point",
        showarrow=False,
        xanchor="right",  # Anchor to the right side
        yshift=-20,
        font=dict(color="red", size=14),
        bgcolor="rgba(0.1,0.1,0.2, 0.0)"
    )
    
    # Display gamma flip info
    flip_diff = ((zero_gamma / spot_price) - 1) * 100
    st.sidebar.markdown(f"**Gamma Flip Point**: {zero_gamma:.2f} ({flip_diff:.2f}% from spot)")
    
    if zero_gamma > spot_price:
        st.sidebar.info(f"Market is in negative gamma territory below {zero_gamma:.2f}. This typically leads to increased volatility when the market moves downward.")
    else:
        st.sidebar.info(f"Market is in positive gamma territory above {zero_gamma:.2f}. This typically leads to increased volatility when the market moves upward.")

# Update layout with unified axis configuration for proper zero alignment
fig.update_layout(
    barmode=bar_mode_val,
    xaxis_title="Gamma Exposure",
    yaxis_title="Strike Price",
    yaxis=dict(
        range=[levels.min() * 0.995, levels.max() * 1.005] if len(levels) > 0 else None,
        showgrid=True,
        gridcolor="rgba(0.3,0.3,0.3.1.0)",
        tickfont=dict(size=16)
    ),
    xaxis=dict(
        range=[bar_x_min, bar_x_max],  # Use unified primary axis range
        showgrid=True,
        gridcolor="rgba(0.1,0.1,0.1.1.0)",
        tickfont=dict(size=14),
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="rgba(255, 218, 3, 0.6)",
    ),
    xaxis2=dict(
        title="Gamma Profile (billions $ / 1% move)",
        range=[gamma_x_min, gamma_x_max],  # Use unified secondary axis range
        overlaying="x",
        side="top",
        showgrid=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="rgba(255, 218, 3, 0.6)",
    ),
    height=max(800, len(grouped) * 15 + 400),  # Dynamic height: minimum 800px, scale with data points
    legend=dict(
        orientation="v",  # Vertical layout - one row per entry
        yanchor="bottom",
        y=0.05,  # Position from bottom of chart
        xanchor="right",
        x=0.98,  # Right side
        bgcolor="rgba(200, 200, 200, 0.12)",  # Works in both light/dark modes
        bordercolor="rgba(100, 100, 100, 0.8)",
        borderwidth=1,
        itemsizing="constant",  # Keep item sizes consistent
        font=dict(size=12),
        tracegroupgap=5  # Add a small gap between legend groups
    )
)

st.plotly_chart(fig, use_container_width=True)

# Add gamma profile explanation
# st.markdown("""
# ### Chart Explanation
# - **Bar chart**: Shows gamma exposure at each strike price, with puts (negative gamma) on the left and calls (positive gamma) on the right.
# - **Line charts**: Show the gamma profile (gamma exposure across different price levels):
#   - **Blue line**: All expiries
#   - **Orange line**: Excluding the next expiry
#   - **Purple line**: Excluding the next monthly expiry
# - **Green dotted line**: Current spot price
# - **Red dotted line**: Gamma flip point (where dealer gamma exposure changes from negative to positive)
# - **Light green area**: Positive gamma region (typically less volatile when market moves in this direction)
# - **Light red area**: Negative gamma region (typically more volatile when market moves in this direction)
# """)

try:
    # Calculate net gamma at each strike
    strike_gamma = grouped.groupby("Strike").agg({
        "call_gamma_expo": "sum",
        "put_gamma_expo": "sum",
        "call_oi": "sum",
        "put_oi": "sum"
    }).reset_index()
    
    strike_gamma["net_gamma"] = strike_gamma["call_gamma_expo"] + strike_gamma["put_gamma_expo"]
    strike_gamma["call_put_ratio"] = strike_gamma["call_oi"] / strike_gamma["put_oi"].replace(0, 1)  # Avoid div by zero
    
    # Find strikes with highest positive and negative gamma
    high_pos_gamma = strike_gamma.nlargest(3, "net_gamma")
    high_neg_gamma = strike_gamma.nsmallest(3, "net_gamma")
    
    # Find strikes with high call/put imbalance
    high_call_strikes = strike_gamma.nlargest(3, "call_put_ratio")
    high_put_strikes = strike_gamma.nsmallest(3, "call_put_ratio")
    
    # Simple strategy recommendations based on gamma profile
    st.markdown("#### Call Strategy Recommendation")
    
    long_call_targets = []
    
    # Above gamma flip, look for high positive gamma as potential support levels
    if zero_gamma is not None:
        above_flip = strike_gamma[strike_gamma["Strike"] > zero_gamma]
        if not above_flip.empty:
            resistance_levels = above_flip.nlargest(2, "net_gamma")
            for _, row in resistance_levels.iterrows():
                long_call_targets.append({
                    "strike": row["Strike"],
                    "rationale": f"High positive gamma at {row['Strike']:.2f} indicates potential dealer hedging support",
                    "net_gamma": row["net_gamma"]
                })
    
    # Below current price but above zero gamma is often a good target for long calls
    if zero_gamma is not None and zero_gamma < spot_price:
        target_zone = strike_gamma[(strike_gamma["Strike"] < spot_price) & 
                                 (strike_gamma["Strike"] > zero_gamma)]
        if not target_zone.empty:
            for _, row in target_zone.nlargest(1, "net_gamma").iterrows():
                long_call_targets.append({
                    "strike": row["Strike"], 
                    "rationale": f"Between zero gamma ({zero_gamma:.2f}) and spot price ({spot_price:.2f})",
                    "net_gamma": row["net_gamma"]
                })
    
    # Look at call/put imbalance for potential interest
    for _, row in high_call_strikes.iterrows():
        if row["call_put_ratio"] > 1.5:  # Significant call bias
            long_call_targets.append({
                "strike": row["Strike"],
                "rationale": f"High call/put ratio ({row['call_put_ratio']:.2f}) suggests bullish sentiment",
                "net_gamma": row["net_gamma"]
            })
    
    # Display long call targets
    if long_call_targets:
        # Sort by strike price for cleaner display
        long_call_targets = sorted(long_call_targets, key=lambda x: x["strike"])
        for i, target in enumerate(long_call_targets[:3], 1):  # Limit to top 3
            gamma_color = "green" if target["net_gamma"] > 0 else "red"
            st.markdown(f"""
            **Target {i}: Strike {target['strike']:.2f}**
            - *Rationale*: {target['rationale']}
            - *Net Gamma*: <span style='color:{gamma_color}'>{format_number(target['net_gamma'])}</span>
            """, unsafe_allow_html=True)
    else:
        st.markdown("No clear long call opportunities identified in the current gamma profile.")
    
    # Put Opportunities
    st.markdown("#### Put Strategy Recommendation")
    
    short_put_targets = []
    
    # Below gamma flip but above major support levels
    if zero_gamma is not None:
        below_flip = strike_gamma[strike_gamma["Strike"] < zero_gamma]
        if not below_flip.empty:
            support_levels = below_flip.nlargest(2, "net_gamma")
            for _, row in support_levels.iterrows():
                distance_from_spot = (spot_price - row["Strike"]) / spot_price * 100
                if distance_from_spot > 0 and distance_from_spot < 15:  # Within reasonable range of spot
                    short_put_targets.append({
                        "strike": row["Strike"],
                        "rationale": f"Support level {distance_from_spot:.1f}% below spot price with positive gamma",
                        "net_gamma": row["net_gamma"]
                    })
    
    # Areas with high put OI but not excessive negative gamma might be good for selling puts
    high_put_oi_reasonable_gamma = strike_gamma[
        (strike_gamma["put_oi"] > strike_gamma["put_oi"].median() * 1.5) & 
        (strike_gamma["net_gamma"] > strike_gamma["net_gamma"].min() * 0.5)
    ]
    
    for _, row in high_put_oi_reasonable_gamma.nlargest(2, "put_oi").iterrows():
        distance_from_spot = (spot_price - row["Strike"]) / spot_price * 100
        if distance_from_spot > 0:  # Only below current price
            short_put_targets.append({
                "strike": row["Strike"],
                "rationale": f"High put OI with manageable gamma exposure, {distance_from_spot:.1f}% below spot",
                "net_gamma": row["net_gamma"],
                "put_oi": row["put_oi"]
            })
    
    # Display short put targets
    if short_put_targets:
        # Sort by strike price for cleaner display
        short_put_targets = sorted(short_put_targets, key=lambda x: x["strike"], reverse=True)
        for i, target in enumerate(short_put_targets[:3], 1):  # Limit to top 3
            gamma_color = "green" if target["net_gamma"] > 0 else "red"
            st.markdown(f"""
            **Target {i}: Strike {target['strike']:.2f}**
            - *Rationale*: {target['rationale']}
            - *Net Gamma*: <span style='color:{gamma_color}'>{format_number(target['net_gamma'])}</span>
            """, unsafe_allow_html=True)
    else:
        st.markdown("No clear short put opportunities identified in the current gamma profile.")
    
    # Market Structure Analysis
    st.markdown("### Overall Market Structure Analysis")
    
    # Determine overall gamma environment
    overall_gamma = strike_gamma["net_gamma"].sum()
    gamma_color = "green" if overall_gamma > 0 else "red"
    
    st.markdown(f"""
    **Net Market Gamma: <span style='color:{gamma_color}'>{format_number(overall_gamma)}</span>**
    
    The market is currently in a **{'positive' if overall_gamma > 0 else 'negative'}** gamma environment.
    """, unsafe_allow_html=True)
    
    # Analysis based on gamma flip point
    if zero_gamma is not None:
        flip_diff = ((zero_gamma / spot_price) - 1) * 100
        flip_direction = "above" if zero_gamma > spot_price else "below"
        
        st.markdown(f"""
        **Gamma Flip Point Analysis:**
        - Current flip point is at {zero_gamma:.2f}, which is {abs(flip_diff):.2f}% {flip_direction} the spot price
        - {'Market is in negative gamma territory below the flip point' if zero_gamma > spot_price else 'Market is in positive gamma territory above the flip point'}
        - {'This structure typically creates higher volatility on downward moves' if zero_gamma > spot_price else 'This structure typically creates higher volatility on upward moves'}
        """)
        
        # Trading strategy suggestion based on flip point
        if zero_gamma > spot_price:
            st.markdown("""
            **Strategy Implication:** Consider options strategies that benefit from increased downside volatility, 
            such as long puts or put spreads, while being cautious with short put positions.
            """)
        else:
            st.markdown("""
            **Strategy Implication:** Consider options strategies that benefit from increased upside volatility,
            such as long calls or call spreads, while being cautious with short call positions near the upper resistance levels.
            """)
    
except Exception as e:
    st.error(f"Error generating market analysis: {str(e)}")

