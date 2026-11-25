# Power BI & Tableau Reference Guide (DV Lab Experiments 1–8)

This companion guide documents the exact Power BI Desktop / Tableau steps, Power Query (M) transformations, and DAX calculated formulas corresponding to `DV_Exp1` through `DV_Exp8` using the CSV files in `datasets/`.

---

## Experiment 1: Getting Started with Power BI / Tableau
- **Dataset:** `datasets/superstore_sample.csv`, `datasets/Dim_Customer.csv`, and `datasets/iris_sample.csv`
- **Power BI Steps:**
  1. `Home` -> `Get Data` -> `Text/CSV` -> Load `superstore_sample.csv` and `Dim_Customer.csv` (link `Customer_ID` in Model view).
  2. Create a **Table** visual with `Order_ID`, `Customer_Name`, `Loyalty_Tier`, `Category`, `Sales`, `Profit`.
  3. Create a **Clustered Bar Chart** (`Axis`: `Category`, `Values`: `Sum of Sales`).
  4. Create a **Line Chart** (`X-axis`: `Order_Date` by Month, `Y-axis`: `Sum of Sales`).

## Experiment 2: Data Cleaning and Transformation (Power Query)
- **Dataset:** `datasets/raw_sales_dirty.csv`
- **Power Query M Script (Advanced Editor):**
```powerquery
let
    Source = Csv.Document(File.Contents("datasets/raw_sales_dirty.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    TrimmedColumns = Table.TransformColumnNames(PromotedHeaders, Text.Trim),
    RenamedColumns = Table.RenameColumns(TrimmedColumns, {
        {"Order Id", "Order_ID"},
        {"Cust Name", "Customer_Name"},
        {"Order_Dt", "Order_Date"},
        {"Region_Name", "Region"},
        {"unit_price", "Unit_Price"},
        {"Qty", "Quantity"},
        {"Disc %", "Discount_Pct"}
    }),
    CleanedRegion = Table.TransformColumns(RenamedColumns, {{"Region", each Text.Proper(Text.Trim(_)), type text}}),
    RemovedDuplicates = Table.Distinct(CleanedRegion, {"Order_ID"}),
    TypedTable = Table.TransformColumnTypes(RemovedDuplicates, {{"Quantity", Int64.Type}, {"Unit_Price", type number}, {"Discount_Pct", type number}}),
    AddedNetRevenue = Table.AddColumn(TypedTable, "Net_Revenue", each [Quantity] * [Unit_Price] * (1 - [Discount_Pct] / 100), type number)
in
    AddedNetRevenue
```

## Experiment 3: Bar and Line Charts for Series Data
- **Dataset:** `datasets/superstore_sample.csv`
- **Power BI / Tableau Steps:**
  - Bar chart of `Sales` and `Profit` by `Sub_Category` with **Data Labels** turned ON.
  - Line chart of monthly `Sales` split by `Category` in `Legend`.
  - Add `Profit`, `Quantity`, and `Profit Margin %` into the **Tooltips** field well.

## Experiment 4: Interactive Dashboards with Filters and Slicers
- **Dataset:** `datasets/superstore_sample.csv`
- **DAX Measures:**
```dax
Total Sales = SUM(superstore_sample[Sales])
Total Profit = SUM(superstore_sample[Profit])
Profit Margin % = DIVIDE([Total Profit], [Total Sales], 0)
Total Orders = DISTINCTCOUNT(superstore_sample[Order_ID])
```
- **Slicers:** Add Slicers for `Region`, `Category`, and `Order_Date` (Between slider) and verify cross-filtering across KPI cards, bar chart, and monthly line chart.

## Experiment 5: Geo-Visualization with Maps
- **Datasets:** `datasets/country_geo_data.csv` and `datasets/city_geo_data.csv`
- **Power BI / Tableau Steps:**
  - **Filled Map (Choropleth):** `Location` = `Country`, `Color saturation` = `Sum of Sales`.
  - **Symbol / Bubble Map:** `Latitude` & `Longitude` from `city_geo_data.csv`, `Bubble size` = `Sales`, `Legend` = `State_Country`.

## Experiment 6: Pie and Donut Charts for Category Proportion
- **Dataset:** `datasets/olympic_medals.csv`
- **Power BI / Tableau Steps:**
  - **Pie Chart:** `Legend` = `Country`, `Values` = `Total`.
  - **Donut Chart:** `Legend` = `Continent` / `Country`, `Values` = `Gold`, `Silver`, `Bronze` with percentage data labels enabled.

## Experiment 7: Heatmaps and Tree Maps
- **Dataset:** `datasets/superstore_sample.csv`
- **Power BI / Tableau Steps:**
  - **Treemap:** `Category` -> `Sub_Category` in `Group`/`Details`, `Values` = `Sum of Sales`.
  - **Matrix Heatmap:** Rows = `Category` / `Sub_Category`, Columns = `Region`, Values = `Sum of Sales` with Conditional Formatting -> Background color scale.

## Experiment 8: Time Series Forecasting and Trend Lines
- **Dataset:** `datasets/monthly_sales_stock.csv`
- **Power BI Analytics Panel / Tableau Analytics Pane:**
  - Plot `Date` vs `Monthly_Sales` on a Line Chart.
  - Enable **Trend line** in the Analytics pane.
  - Enable **Forecast** (`Forecast length` = 6 months, `Seasonality` = 12, `Confidence interval` = 95%).
  - Add a **Moving Average** DAX measure:
```dax
Sales 3M Moving Avg =
AVERAGEX(
    DATESINPERIOD(monthly_sales_stock[Date], LASTDATE(monthly_sales_stock[Date]), -3, MONTH),
    CALCULATE(SUM(monthly_sales_stock[Monthly_Sales]))
)
```
