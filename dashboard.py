import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import os

# ✅ Dataset path
DATA_FILE = r"C:\Users\Kavya R\Desktop\INT82\Data_Quality\cleaned_sales_data.csv"

# ✅ Load and clean dataset
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Ensure necessary columns exist
        required_columns = ["Sales", "Discount", "Profit", "Sub Category", "Days"]
        for col in required_columns:
            if col not in df:
                df[col] = 0  # Create missing columns
        df.fillna(0, inplace=True)  # Handle NaN
        return df
    else:
        return pd.DataFrame(columns=["Sales", "Discount", "Profit", "Sub Category", "Days"])

# ✅ Load initial dataset
df = load_data()

# ✅ Function to get top correlation pair
def get_top_correlation_pair(data):
    correlation_matrix = data[["Sales", "Discount", "Profit"]].corr()
    correlation_matrix.values[[0, 1, 2], [0, 1, 2]] = 0  # Remove diagonal
    return correlation_matrix.abs().unstack().idxmax()

# ✅ Initialize Dash app
app = dash.Dash(__name__)

# ✅ Dashboard Layout
app.layout = html.Div(children=[
    html.H1("📊 Data Quality Monitoring Dashboard", style={"textAlign": "center"}),

    # 🔹 Input Form
    html.Div([
        html.Label("Sales:"),
        dcc.Input(id="sales", type="number", placeholder="Enter Sales"),

        html.Label("Discount:"),
        dcc.Input(id="discount", type="number", placeholder="Enter Discount"),

        html.Label("Profit:"),
        dcc.Input(id="profit", type="number", placeholder="Enter Profit"),

        html.Label("Sub Category:"),
        dcc.Input(id="sub_category", type="text", placeholder="Enter Category"),

        html.Button("Add Record", id="submit", n_clicks=0),
        html.Hr()
    ], style={"width": "300px", "margin": "auto"}),

    # 🔹 Color Variable Dropdown
    html.Label("Select Color Variable:"),
    dcc.Dropdown(
        id="color_variable",
        options=[{"label": col, "value": col} for col in ["Sub Category", "Discount", "Profit"]],
        value="Sub Category",
        clearable=False
    ),

    # 🔹 Graphs
    dcc.Graph(id="heatmap"),
    dcc.Graph(id="scatter"),
    dcc.Graph(id="boxplot"),
    dcc.Graph(id="histogram")
])

# ✅ Callback for Graph Updates
@app.callback(
    [Output("heatmap", "figure"),
     Output("scatter", "figure"),
     Output("boxplot", "figure"),
     Output("histogram", "figure")],
    [Input("submit", "n_clicks"),
     Input("color_variable", "value")],
    [State("sales", "value"),
     State("discount", "value"),
     State("profit", "value"),
     State("sub_category", "value")]
)
def update_dashboard(n_clicks, color_variable, sales, discount, profit, sub_category):
    global df

    # 🔸 Reload dataset
    df = load_data()

    # 🔸 Add new record if provided
    if n_clicks > 0 and all([sales, discount, profit]):
        new_record = pd.DataFrame({
            "Sales": [sales],
            "Discount": [discount],
            "Profit": [profit],
            "Sub Category": [sub_category if sub_category else "Unknown"],
            "Days": [df["Days"].max() + 1 if not df.empty else 1]
        })
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

    # 🔸 Handle empty dataset gracefully
    if df.empty or len(df) < 2:
        return [px.scatter(title="❌ Not Enough Data") for _ in range(4)]

    # 🔸 Heatmap
    heatmap = px.imshow(
        df[["Sales", "Discount", "Profit"]].corr(),
        color_continuous_scale="RdBu",
        title="📊 Correlation Heatmap"
    )

    # 🔸 Scatter Plot
    top_pair = get_top_correlation_pair(df)
    scatter = px.scatter(
        df, x=top_pair[0], y=top_pair[1], color=color_variable,
        title=f"📈 {top_pair[0]} vs {top_pair[1]} (Colored by {color_variable})"
    )

    # 🔸 Boxplot
    boxplot = px.box(
        df, y="Sales", title="📦 Sales Outlier Detection", points="all"
    )

    # 🔸 Histogram
    histogram = px.histogram(
        df, x="Sales", title="📊 Sales Distribution", nbins=50
    )

    return heatmap, scatter, boxplot, histogram

# 🚀 Run the App
if __name__ == "__main__":
    app.run_server(debug=True)
