# Sales Analytics Dashboard

## 📊 Description
Power BI Dashboard for comprehensive sales analytics including revenue tracking, customer analysis, product performance, and regional insights.

## 🗂️ Project Structure
```
Sales_Analytics_Dashboard/
├── Sales_Analytics_Dashboard.pbip
├── Sales_Analytics_Dashboard.Report/      # Report visuals definition
└── Sales_Analytics_Dashboard.SemanticModel/ # Data model (tables, measures, relationships)
```

## 📈 Report Pages
| Page | Description |
|------|-------------|
| Site Mapping | Executive dashboard with main KPIs |
| Cust Mapping | Customer analysis and segmentation |
| Letter Mockup | Product performance and categories |

## 📐 Data Model

### Fact Table
- **FactSales** - Sales transactions with 16 columns including SalesAmount, Quantity, Profit, Discount

### Dimension Tables
| Table | Description |
|-------|-------------|
| DimCustomer | Customer information (Name, Email, LoyaltyTier, Gender) |
| DimProduct | Product catalog (Name, Category, SubCategory, Color, Size) |
| DimDate | Date dimension (Year, Month, Quarter, Day, IsWeekend, IsHoliday) |
| DimGeography | Geographic data (Country, Region, City) |
| DimEmployee | Employee information (Name, Role, HireDate) |

### Key Measures
| Measure | Description |
|---------|-------------|
| Total Revenue | Sum of sales amount |
| Total Profit | Sum of profit |
| Total Cost | Sum of total cost |
| Profit Margin % | Profit percentage over revenue |
| Revenue YoY Growth % | Year over year revenue growth |
| Revenue MoM % | Month over month revenue change |
| Total Orders | Count of distinct orders |
| Active Customers | Count of unique customers |
| Avg Order Value | Average revenue per order |
| Customer Lifetime Value | Average revenue per customer |
| Products Sold | Count of unique products sold |
| Total Quantity | Sum of units sold |
| Revenue YTD | Year to date revenue |
| Profit YTD | Year to date profit |
| Revenue LY | Last year revenue (same period) |
| Profit LY | Last year profit (same period) |
| Avg Daily Revenue | Average revenue per day |
| New Customers | Count of new customers in period |
| Revenue per Employee | Revenue divided by employees |
| Product Rank | Ranking of products by revenue |
| Revenue % of Total | Percentage contribution to total |

## 🔧 Requirements
- Power BI Desktop (with PBIP format support)
- Access to configured data sources

## 🚀 How to Use
1. Clone this repository
2. Open `Sales_Analytics_Dashboard.pbip` in Power BI Desktop
3. Configure data source connections if needed
4. Refresh data

## 👤 Author
[Your Name]

## 📅 Last Updated
January 2026
