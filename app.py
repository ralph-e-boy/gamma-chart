
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import datetime
from io import StringIO

st.set_page_config(layout="wide")

# File upload in sidebar
st.sidebar.subheader("Gamma Exposure Chart")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file of options chain data from https://www.cboe.com/delayed_quotes/spy/quote_table", type=["csv"])

# Get data from uploaded file or default file
if uploaded_file:
    lines = uploaded_file.read().decode("utf-8").splitlines()
else:
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
from io import StringIO
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

# --- Filters ---
max_dte = int(df["DTE"].max())
days_ahead = st.sidebar.slider("Max DTE (days)", 0, max_dte, min(7, max_dte))
df = df[df["DTE"] <= days_ahead]

spot_price = last if summary_rendered else df["Strike"].median()
strike_range = st.sidebar.slider("Strike range (± around spot)", 0, 200, 50)
lo, hi = spot_price - strike_range / 2, spot_price + strike_range / 2
df = df[(df["Strike"] >= lo) & (df["Strike"] <= hi)]

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

bar_mode = st.sidebar.radio("Bar Mode", ["Stacked", "Grouped (side-by-side)"], index=1)
bar_mode_val = "stack" if bar_mode == "Stacked" else "group"

# File uploader has been moved to the top of the script

grouped["abs_total"] = grouped["call_gamma_expo"].abs() + grouped["put_gamma_expo"].abs()
sorted_dtes = grouped.groupby("DTE")["abs_total"].sum().sort_values(ascending=False).index.tolist()

# Use the specified color palette
def get_dte_color(dte, max_dte):
    # Specified color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    
    # Special case for 0 DTE
    if dte == 0:
        return "#d62728"  # Red from the palette
    
    # For other DTEs, assign colors in order from the palette
    color_index = min(dte - 1, len(colors) - 1)
    return colors[color_index]

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

fig = go.Figure()
# Sort DTEs from smallest to largest for legend ordering
sorted_dtes_asc = sorted(sorted_dtes)
max_dte_value = max(sorted_dtes_asc) if sorted_dtes_asc else 1
for i, dte in enumerate(sorted_dtes_asc):
    color = get_dte_color(dte, max_dte_value)
    sub = grouped[grouped["DTE"] == dte]
    
    # Calculate net gamma exposure for hover text
    sub = sub.copy()  # Create a copy to avoid the SettingWithCopyWarning
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
        width=0.5,  # Further reduced width to prevent overlap
        hovertemplate=put_hovertemplate,
        customdata=customdata,
        offset=0  # Ensure proper alignment
    ))
    
    fig.add_trace(go.Bar(
        x=sub["call_gamma_expo"],
        y=sub["Strike"],
        orientation="h",
        marker_color=color,
        name=f"{dte} DTE",
        legendgroup=f"{dte} DTE",
        showlegend=False,
        width=0.5,  # Further reduced width to prevent overlap
        hovertemplate=put_hovertemplate,
        customdata=customdata,
        offset=0  # Ensure proper alignment
    ))

# Note: These shapes are now added in the update_layout shapes list

fig.add_annotation(
    x=grouped["put_gamma_expo"].min(),
    y=spot_price,
    text="Spot price",
    showarrow=False,
    xanchor="left",
    yshift=10,
    font=dict(color="green", size=14),
    bgcolor="rgba(255,255,255,0.7)"
)

# Calculate y-axis tick values (every 5 points)
min_strike = grouped["Strike"].min()
max_strike = grouped["Strike"].max()
y_tick_start = 5 * (min_strike // 5)  # Round down to nearest 5
y_tick_end = 5 * (max_strike // 5 + 1)  # Round up to nearest 5
y_ticks = list(range(int(y_tick_start), int(y_tick_end) + 5, 5))

fig.update_layout(
    barmode="group",  # Force grouped mode to prevent overlap
    xaxis_title="Gamma Exposure",
    yaxis_title="Strike Price",
    yaxis=dict(
        autorange=True, 
        showgrid=True, 
        gridcolor="#666666",  # Darker gray for horizontal grid lines
        tickmode="array",
        tickvals=y_ticks,
        dtick=5,  # Grid lines every 5 points
        fixedrange=False,  # Allow zooming on y-axis
    ),
    xaxis=dict(
        showgrid=False,  # Remove vertical grid marks
        fixedrange=False,  # Allow zooming on x-axis
    ),
    height=1200,  # Further increased height for better spacing
    bargap=0.4,   # Further increased gap between bars
    bargroupgap=0.25,  # Increased gap between bar groups
    uniformtext=dict(mode="hide", minsize=10),  # Ensure text is readable
    legend=dict(
        orientation="v",  # Vertical legend
        yanchor="bottom",
        y=0.02,  # Position at bottom
        xanchor="right",
        x=0.98,  # Position at right
        bgcolor="rgba(50,50,50,0.9)",  # Dark gray background
        bordercolor="gray",
        borderwidth=1,
        itemsizing="constant",
        font=dict(color="white")  # White text for better readability
    ),
    margin=dict(l=50, r=50, t=80, b=50),  # Add more margin for better spacing
    plot_bgcolor='rgba(30,30,30,0.3)',  # Slightly lighter than black for plot background
    paper_bgcolor='rgba(30,30,30,0.3)',  # Slightly lighter than black for paper background
    font=dict(color='white'),  # White text for all labels
)

# Add a border around the entire chart
fig.update_layout(
    shapes=[
        # Add existing shapes
        dict(type="line", x0=0, x1=0,
             y0=grouped["Strike"].min() - 5, y1=grouped["Strike"].max() + 5,
             line=dict(color="lightgray", width=2)),
        dict(type="line",
             x0=grouped["put_gamma_expo"].min(),
             x1=grouped["call_gamma_expo"].max(),
             y0=spot_price,
             y1=spot_price,
             line=dict(color="green", width=2, dash="dot")),
        # Add border around the chart
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#999999", width=2),
            fillcolor="rgba(0,0,0,0)",
        )
    ]
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('''\n\n> Tip: Enable dark mode in your Streamlit settings for best visual contrast.''')
