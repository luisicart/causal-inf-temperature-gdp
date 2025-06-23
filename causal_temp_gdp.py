# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_partregress, plot_partregress_grid
# %%
plt.rcParams.update({
        'axes.grid': False,  
        'axes.linewidth': 1.5,  
        'axes.edgecolor': 'black',  
        'axes.facecolor': 'white',  
        'figure.facecolor': 'white',  
        'axes.spines.top': False,  
        'axes.spines.right': False
    })
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
selected_year = 2020
temp_gdp_df = temp_gdp_df[temp_gdp_df['year'] == selected_year]

# %%
top_5_countries = temp_gdp_df.nlargest(5, 'gdp')['country_code'].tolist()
bottom_5_countries = temp_gdp_df.nsmallest(4, 'gdp')['country_code'].tolist()

top_and_bottom_countries = top_5_countries + bottom_5_countries
# %%
plt.figure(figsize=(10, 6), dpi=300)
sns.scatterplot(
    data=temp_gdp_df,
    x='average_annual_temperature',
    y='ln_gdp',
    size='ln_gdp',
    hue='ln_gdp',
    palette='RdBu'
)

for i, row in temp_gdp_df.iterrows():
    if row['country_code'] in top_and_bottom_countries:
        plt.text(
            row['average_annual_temperature'],
            row['ln_gdp'] + 0.05,  
            row['country'],   
            fontsize=8,
            ha='center',
            va='bottom'
        )

sns.regplot(
    data=temp_gdp_df,
    x='average_annual_temperature',
    y='ln_gdp',
    ci=95,
    scatter=False,
    color='black',
    line_kws={'label': 'Regressão Linear'}
)
plt.title(f'Relação entre Temperatura e PIB per Capita (ln) - {selected_year}')
plt.xlabel('Temperatura Anual Média (°C)')
plt.ylabel('PIB per Capita (ln)')
plt.savefig('./graphs/gdp_vs_temperatura.png')
# %%
# ln_gdp ~ temp
reg_model_temp_gdp = smf.ols(formula='ln_gdp ~ average_annual_temperature', data=temp_gdp_df).fit(cov_type='HC1')
print(f"\nRegressão: ln_gdp ~ temp {selected_year}")
print(reg_model_temp_gdp.summary())
# %%
ate_simple = reg_model_temp_gdp.params['average_annual_temperature']
print(f"\nATE (modelo simples, sem controles): {ate_simple:.4f}")
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

df_confounders = df_confounders[df_confounders['year'] == selected_year]
# %%
# ln_gdp ~ temp + educ + rule_law + industry 
reg_model_confounders = smf.ols(formula='ln_gdp ~ average_annual_temperature + educ_level + law_level + industry_level', 
                  data=df_confounders).fit(cov_type='HC1')
print(f"\nRegressão: ln_gdp ~ temp + educ + law + industry {selected_year}")
print(reg_model_confounders.summary())

# %%
ate_adjusted = reg_model_confounders.params['average_annual_temperature']
print(f"\nATE (modelo com controles - confounders): {ate_adjusted:.4f}")
# %%
fig = plt.figure(figsize=(10, 6), dpi=300)
ax = plt.gca()
plot_partregress(
        endog='ln_gdp', 
        exog_i='average_annual_temperature', 
        exog_others=['educ_level', 'law_level', 'industry_level'], 
        data=df_confounders, 
        obs_labels=False,
        ax=ax)

ax.set_title('Added Variable Plot - Temperatura Condicional')
fig.tight_layout()
fig.savefig('./graphs/gdp_temperatura_condicional.png')
plt.show()
# %%
fig = plt.figure(figsize=(10, 8), dpi=300)
axes = plot_partregress_grid(reg_model_confounders, fig=fig)
fig.suptitle('Gráficos de Regressão Parcial - Condicionais')
fig.tight_layout()
fig.savefig('./graphs/gdp_confounders.png')
plt.show()