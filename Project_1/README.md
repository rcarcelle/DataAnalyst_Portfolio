# COVID-19 Data Analysis
This project aims to clean and analyze the COVID-19 dataset from Johns Hopkins University to gain insights into the trends and patterns of COVID-19 cases, deaths, and recoveries. The analysis will also explore the impact of various factors such as demographics and healthcare systems on the spread and severity of the disease. The project ultimately aims to provide valuable information and insights to aid in the global fight against COVID-19.

## Getting Started
To run this project, you will need to download the Johns Hopkins CoronaVirus dataset, which is available on their [GitHub repository](https://github.com/CSSEGISandData/COVID-19). You will need to download the following two files:

- ['time_series_covid19_confirmed_global.csv'](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv)

- ['time_series_covid19_deaths_global.csv'](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv)

## Prerequisites
This project requires Python 3 and the following packages:

- 'pandas'
- 'numpy'
- 'matplotlib'
- 'seaborn'
- 'plotly'
- 'calmap'
- 'folium'
- 'wget'

You can install the required packages using pip by running the following command:
```bash
pip install -r requirements.txt
```
## Running the Analysis
To run the analysis, simply run the EDA_visualization_COVID-19.py script. This will perform data cleaning and preprocessing to ensure accuracy and consistency. The script will then generate various visualizations and conduct statistical analyses to gain insights into the trends and patterns of COVID-19 cases, deaths, and recoveries.

## Authors
Rafael Carcellé

## Acknowledgments
- The Johns Hopkins University Center for Systems Science and Engineering for providing the COVID-19 dataset.
- The open-source community for developing the necessary tools and libraries.
