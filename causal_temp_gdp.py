# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_partregress, plot_partregress_grid

# %%
temp_df = pd.read_csv('./data/average_surface_temperature.csv')
gdp_df = pd.read_csv('./data/gdp_per_capita_worldbank.csv')

# %%
common_rename_columns = {
    'Entity': 'country',
    'Code': 'country_code',
    'Year': 'year'
}

temp_specific_rename = {
    'Day': 'day',
    'Average surface temperature': 'daily_average_surface_temperature'
}

gdp_specific_rename = {
    'GDP per capita, PPP (constant 2021 international $)': 'gdp'
}

temp_df.rename(columns={**common_rename_columns, **temp_specific_rename}, inplace=True)
gdp_df.rename(columns={**common_rename_columns, **gdp_specific_rename}, inplace=True)

# %%
def filter_valid_country_codes(df, code_col, length=3):
    return df[df[code_col].str.len() == length]

def create_avplot(yvar, xvar, other_vars, df, title, filename,
                  figsize=(10, 6), dpi=300):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plot_partregress(yvar, xvar, other_vars, data=df, ax=ax, obs_labels=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.show() 

# %%
temp_df = filter_valid_country_codes(temp_df, 'country_code')
gdp_df = filter_valid_country_codes(gdp_df, 'country_code')
# %%
temp_df = temp_df.groupby(['country_code', 'year']).agg(
    average_annual_temperature=('daily_average_surface_temperature', 'mean')
).reset_index()

gdp_df['ln_gdp'] = np.log(gdp_df['gdp'])
# %%
temp_gdp_df = pd.merge(gdp_df, temp_df, on=['country_code', 'year'], how='inner')
temp_gdp_df
# %%
# Regressão 1: ln_gdp ~ temp (robusto)
modelo1 = smf.ols(formula='ln_gdp ~ average_annual_temperature', data=temp_gdp_df[temp_gdp_df['year'] == 2020]).fit(cov_type='HC1')
print("\nRegressão 1: ln_gdp ~ temp (2020)")
print(modelo1.summary())

# %%
# a) avplot para modelo1 (só temp)
create_avplot(
    yvar='ln_gdp',
    xvar='average_annual_temperature',
    other_vars=[],
    df=temp_gdp_df[temp_gdp_df['year'] == 2020],
    title='Added Variable Plot - Temperatura',
    filename='./graphs/gdp_temperatura_python.png'
)

# %%
industry_df = pd.read_csv("./data/industry_share_of_total_emplyoment.csv")

industry_specific_rename = {
    'Employment in industry (% of total employment) (modeled ILO estimate)': 'industry_level'
}

industry_df.rename(columns={**common_rename_columns, **industry_specific_rename}, inplace=True)
industry_df = filter_valid_country_codes(industry_df, 'country_code')

# %%

educ_df = pd.read_csv("./data/mean_years_of_schooling_long_run.csv")

educ_specific_rename = {
    'Combined - average years of education for 15-64 years male and female youth and adults': 'educ_level'
}

educ_df.rename(columns={**common_rename_columns, **educ_specific_rename}, inplace=True)
educ_df = filter_valid_country_codes(educ_df, 'country_code')
# %%
rule_law_df = pd.read_csv("./data/rule_of_law_index.csv")

rule_law_specific_rename = {
    'Rule of Law index (central estimate, aggregate: average)': 'law_level'
}

rule_law_df.rename(columns={**common_rename_columns, **rule_law_specific_rename}, inplace=True)
rule_law_df = filter_valid_country_codes(rule_law_df, 'country_code')
# %%
df_confounders = temp_gdp_df.copy()

for df in [industry_df, educ_df, rule_law_df]:
    df_confounders = pd.merge(df_confounders, df.drop(columns='country'), on=['country_code', 'year'], how='inner')

# %%
# Regressão 2: ln_gdp ~ temp + educ + rule_law + industry (robusto)
modelo2 = smf.ols(formula='ln_gdp ~ average_annual_temperature + educ_level + law_level + industry_level', 
                  data=df_confounders[df_confounders['year'] == 2020]).fit(cov_type='HC1')
print("\nRegressão 2: ln_gdp ~ temp + educ + law + industry")
print(modelo2.summary())

# %%
create_avplot(
    yvar='ln_gdp',
    xvar='average_annual_temperature',
    other_vars=['educ_level', 'law_level', 'industry_level'],
    df=df_confounders[df_confounders['year'] == 2020],
    title='Added Variable Plot - Temperatura (Condicionado)',
    filename='./graphs/gdp_temperatura_condicional_python.png'
)

# %%
fig = plt.figure(figsize=(10, 8), dpi=300)
plot_partregress_grid(modelo2, fig=fig)
fig.tight_layout()
fig.savefig('./graphs/gdp_controles_python.png')
plt.show() 