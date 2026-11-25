import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import plotly.express as px

data_dir = "datasets" if os.path.exists("datasets") else os.path.join("DV_Lab", "datasets")
out_dir = "outputs" if os.path.exists("outputs") else (os.path.join("DV_Lab", "outputs") if os.path.exists("DV_Lab") else "outputs")
os.makedirs(out_dir, exist_ok=True)

orders_df = pd.read_csv(os.path.join(data_dir, "superstore_sample.csv"))
cust_df = pd.read_csv(os.path.join(data_dir, "Dim_Customer.csv"))
df = orders_df.merge(cust_df, on="Customer_ID", how="left")

num_cols = ["Quantity", "Unit_Price", "Discount", "Sales", "Cost", "Profit"]
corr_matrix = df[num_cols].corr().round(3)
print("Correlation Matrix:")
print(corr_matrix)

cat_sub = (
    df.groupby(["Category", "Sub_Category"], as_index=False)[["Sales", "Profit"]]
    .sum()
    .sort_values("Sales", ascending=False)
)
print("\nMultilevel Category -> Sub-Category Sales:")
print(cat_sub.round(2).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    center=0.0,
    linewidths=0.6,
    ax=ax[0],
)
ax[0].set_title("Correlation Matrix Heatmap (Superstore Metrics)")

treemap_labels = [
    f"{row['Category']}\n{row['Sub_Category']}\n${row['Sales']:,.0f}"
    for _, row in cat_sub.iterrows()
]
cat_palette = {
    "Technology": "teal",
    "Furniture": "slateblue",
    "Office Supplies": "darkorange",
}
treemap_colors = [cat_palette[c] for c in cat_sub["Category"]]

squarify.plot(
    sizes=cat_sub["Sales"],
    label=treemap_labels,
    color=treemap_colors,
    alpha=0.88,
    edgecolor="white",
    linewidth=2,
    text_kwargs={"fontsize": 8.5, "color": "white", "fontweight": "bold"},
    ax=ax[1],
)
ax[1].set_title("Tree Map: Category > Sub-Category Sales")
ax[1].axis("off")

plt.tight_layout()
plt.show()

tree_fig = px.treemap(
    df,
    path=[px.Constant("All Categories"), "Category", "Sub_Category", "Loyalty_Tier"],
    values="Sales",
    color="Profit",
    color_continuous_scale="Tealrose",
    title="Interactive Multilevel Tree Map (Category > Sub-Category > Loyalty Tier)",
)
html_path = os.path.join(out_dir, "DV_Exp7_Interactive_Treemap.html")
tree_fig.write_html(html_path, include_plotlyjs="cdn")
print("Interactive Treemap saved to:", html_path)
