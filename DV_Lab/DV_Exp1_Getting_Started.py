import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "datasets" if os.path.exists("datasets") else os.path.join("DV_Lab", "datasets")

orders_df = pd.read_csv(os.path.join(data_dir, "superstore_sample.csv"))
customers_df = pd.read_csv(os.path.join(data_dir, "Dim_Customer.csv"))
iris_df = pd.read_csv(os.path.join(data_dir, "iris_sample.csv"))

merged_df = orders_df.merge(customers_df, on="Customer_ID", how="left")
merged_df["Order_Date"] = pd.to_datetime(merged_df["Order_Date"])
merged_df["Year_Month"] = merged_df["Order_Date"].dt.to_period("M").astype(str)

print("Joined Retail Table Preview (Orders + Dim_Customer):")
print(
    merged_df[["Order_ID", "Order_Date", "Customer_Name", "Loyalty_Tier", "Category", "Sales", "Profit"]]
    .head(7)
    .to_string(index=False)
)

tier_summary = merged_df.groupby("Loyalty_Tier", as_index=False)[["Sales", "Profit", "Quantity"]].sum()
print("\nSales Summary by Customer Loyalty Tier:")
print(tier_summary.round(2).to_string(index=False))

print("\nIris Species Feature Averages:")
print(iris_df.groupby("species").mean(numeric_only=True).round(2))

cat_sales = merged_df.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
monthly_sales = merged_df.groupby("Year_Month", as_index=False)["Sales"].sum().sort_values("Year_Month")

fig, panels = plt.subplots(1, 2, figsize=(13, 5))

panels[0].bar(cat_sales["Category"], cat_sales["Sales"], color=["teal", "slateblue", "goldenrod"], edgecolor="black")
for idx, val in enumerate(cat_sales["Sales"]):
    panels[0].text(idx, val + 1200, f"${val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
panels[0].set_title("Category Sales Overview (Bar View)")
panels[0].set_xlabel("Product Category")
panels[0].set_ylabel("Total Sales ($)")
panels[0].grid(True, axis="y", linestyle=":", alpha=0.5)

panels[1].plot(monthly_sales["Year_Month"], monthly_sales["Sales"], marker="s", color="indigo", linewidth=2)
panels[1].set_title("Monthly Revenue Trajectory (Line View)")
panels[1].set_xlabel("Month")
panels[1].set_ylabel("Sales ($)")
panels[1].tick_params(axis="x", rotation=60)
panels[1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()
