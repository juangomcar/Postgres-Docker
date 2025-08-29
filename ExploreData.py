#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


# # 1a) conexión con puerto y bloque

# In[8]:


import psycopg2

conn = None
cur = None

try:
    # Conexión a la BD
    conn = psycopg2.connect(
        host="localhost",
        port=5433,   # puerto externo correcto (5433:5432)
        database="bigdatatools1",
        user="psqluser",
        password="psqlpass"
    )

    cur = conn.cursor()
    cur.execute("SELECT version();")
    ver = cur.fetchone()
    print("Conexión exitosa. Versión de PostgreSQL:", ver[0])

except Exception as e:
    print(f"Error al conectar a la base de datos: {e}")

finally:
    if cur is not None:
        cur.close()
    if conn is not None:
        conn.close()
        print("Conexión cerrada correctamente.")


# **Challenge**: Use the proper port and use try-except-finally block(s)

# ## Example: Demographic Data

# In[9]:


population=pd.read_parquet('World-Population-Estimates.parquet')
population.info()


# In[10]:


population.drop(columns=['Unnamed: 95'], inplace=True)
population.describe()


# Show the indicators in the parquet file, _select only those relevant for the analysis_

# In[12]:


print("Indicators:", population['Indicator Name'].unique()[:20])  # primeros 20


# In[11]:


relevant_indicators=['Life expectancy at birth, total (years)', 'Population, total', 'Fertility rate, total (births per woman)', 'Birth rate, crude (per 1,000 people)' , 'Mortality rate, neonatal (per 1,000 live births)' ]
population_data=population[population['Indicator Name'].isin(relevant_indicators)]


# **Filtering**: Use only data between 1994 and 2024

# In[13]:


ind_cols=[col for col in population_data.columns if not col.isdigit()]
year_cols=[col for col in population_data.columns if col.isdigit() and 1991 <= int(col) <=2024]

population_data=population_data[ind_cols+year_cols]
print(f"Indicators: {population_data['Indicator Name'].unique()}")
population_data.info()


# In[14]:


population_data


# ## Example Per Capita Gross Domestic Product PPP data

# In[15]:


gdp=pd.read_csv('GDP.PCAP.PP.CD_DS2_en_csv_v2_37774.csv', sep=';')
gdp.info()


# In[16]:


ind_cols=[col for col in gdp.columns if not col.isdigit()]
year_cols=[col for col in gdp.columns if col.isdigit() and 1991 <= int(col) <=2024]

gdp_data=gdp[ind_cols+year_cols]
gdp_data.info()


# In[17]:


gdp_data


# ## Example: Load GDP and Demographic datasets into the SQL database
# 
# Instead of traditional INSERT operations, we will use a COPY strategy according to performance considerations

# In[20]:


import io

id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
# 1. Reshape population_data
pop_long = population_data.melt(
    id_vars=id_vars,
    var_name='Year',
    value_name='Value'
)

# 2. Reshape gdp_data
gdp_long = gdp_data.melt(
    id_vars=id_vars,
    var_name='Year',
    value_name='Value'
)

# 3. Combine the two long DataFrames
combined_data = pd.concat([pop_long, gdp_long], ignore_index=True)

# 4. Clean the data for database insertion
combined_data.dropna(subset=['Value'], inplace=True)  # Drop rows where the value is missing
combined_data['Year'] = combined_data['Year'].astype(int)  # Convert Year column to integer

# Rename columns for SQL friendliness (lowercase, no spaces)
combined_data.rename(columns={
    'Country Name': 'country_name', 'Country Code': 'country_code',
    'Indicator Name': 'indicator_name', 'Indicator Code': 'indicator_code',
    'Year': 'year', 'Value': 'value'
}, inplace=True)

print("Reshaped and combined data preview:")
print(combined_data.head())
print(f"\nTotal rows to insert: {len(combined_data)}")

# 5. Batch load into PostgreSQL using COPY command for high performance
table_name = 'country_indicators'

# Reconnect to the database
conn = psycopg2.connect(host="localhost", port=5433, database="bigdatatools1", user="psqluser", password="psqlpass")
cur = conn.cursor()

# Create table schema. Using TEXT is flexible, NUMERIC is good for values.
cur.execute(f"""
DROP TABLE IF EXISTS {table_name};
CREATE TABLE {table_name} (
    id SERIAL PRIMARY KEY, country_name TEXT, country_code VARCHAR(3),
    indicator_name TEXT, indicator_code TEXT, year INTEGER, value NUMERIC
);""")
conn.commit()
print(f"Table '{table_name}' created successfully.")

# Use an in-memory buffer (StringIO) to prepare data for COPY
buffer = io.StringIO()
combined_data[['country_name', 'country_code', 'indicator_name', 'indicator_code', 'year', 'value']].to_csv(buffer, header=False, index=False)
buffer.seek(0)  # Rewinds the buffer to the beginning

# Use copy_expert to load data using the CSV format, which correctly handles quoted values.
# This is more robust than copy_from when data fields might contain the separator character (comma).
try:
    # The SQL command specifies the columns and that the format is CSV, which pandas.to_csv() produces.
    sql_copy_command = f"""
        COPY {table_name} (country_name, country_code, indicator_name, indicator_code, year, value)
        FROM STDIN WITH (FORMAT CSV, HEADER FALSE)
    """
    cur.copy_expert(sql=sql_copy_command, file=buffer)
    conn.commit()
    print("Data loaded successfully using COPY.")
except (Exception, psycopg2.DatabaseError) as error:
    print(f"Error: {error}")
    conn.rollback()
finally:
    cur.execute(f"SELECT COUNT(*) FROM {table_name};")  # Verify the load
    print(f"Verification: {cur.fetchone()[0]} rows were inserted into '{table_name}'.")
    cur.close()
    conn.close()


# **Challenge**: perform the same operation using SQL inserts and Pandas, without optimizations

# ## Analyzing the Evolution of GDP for big-sized economies
# To study the evolution of the overall GDP (PPP) for different countries. We need to query the data and **pivot** it, so that _each indicator becomes its own column_. This query calculates the total GDP PPP by fetching both GDP per capita and population for each country/year, pivoting them into the same row, and then multiplying them.

# In[21]:


# Ensure the SQLAlchemy engine is available
db_url = "postgresql+psycopg2://psqluser:psqlpass@localhost:5433/bigdatatools1"
engine = create_engine(db_url)

# Define the list of countries/regions to compare
countries_to_compare = ('United States', 'Japan', 'China', 'Germany', 'United Kingdom', 'European Union')


query = """
        SELECT
            year,
            country_name,
            (MAX(CASE WHEN indicator_name = 'GDP per capita PPP (current international $)' THEN value END) *
             MAX(CASE WHEN indicator_name = 'Population, total' THEN value END)) AS total_gdp_ppp
        FROM
            country_indicators
        WHERE
            country_name IN %(countries)s
          AND indicator_name IN ('GDP per capita PPP (current international $)', 'Population, total')
        GROUP BY
            country_name, year
        ORDER BY
            country_name, year;
        """

# Execute the query using pandas and SQLAlchemy
gdp_comparison_df = pd.read_sql_query(query, engine, params={'countries': countries_to_compare})

print("Total GDP (PPP) Data for Major Economies:")
print(gdp_comparison_df.head())

# Visualize the comparison using a line plot with Matplotlib
fig, ax = plt.subplots(figsize=(14, 8))

# To plot multiple lines from a long-format DataFrame, we loop through each country
# and plot its data on the same axes.
for country in countries_to_compare:
    country_df = gdp_comparison_df[gdp_comparison_df['country_name'] == country]
    ax.plot(country_df['year'], country_df['total_gdp_ppp'], marker='o', linestyle='-', label=country)

ax.set_title('Total GDP (PPP) Evolution: USA, Japan, China, Germany, UK & EU', fontsize=18)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Total GDP (PPP, current international $)', fontsize=14)
ax.legend(title='Country/Region')
ax.grid(True)
plt.show()


# ## Analyzing GDP vs. Life Expectancy for the US
# To study the evolution of two different indicators for the same country, we need to query the data and **pivot** it, so that _each indicator becomes its own column_. The most efficient way to do this is with a conditional aggregation query in SQL.
# 
# The SQL query uses then MAX(CASE WHEN...) to pivot the rows into columns. It selects the year and creates two new columns: one for the GDP value and one for the Life Expectancy value.

# In[22]:


# To use SQLAlchemy, we first create an engine that manages connections to the database.
# This is a more robust way to interact with databases in Python applications.
db_url = "postgresql+psycopg2://psqluser:psqlpass@localhost:5433/bigdatatools1"
engine = create_engine(db_url)


query = """
        SELECT
            year, MAX (CASE WHEN indicator_name = 'GDP per capita PPP (current international $)' THEN value END) AS gdp_per_capita, MAX (CASE WHEN indicator_name = 'Life expectancy at birth, total (years)' THEN value END) AS life_expectancy
        FROM
            country_indicators
        WHERE
            country_name = 'United States'
          AND indicator_name IN (
            'GDP per capita PPP (current international $)'
            , 'Life expectancy at birth, total (years)'
            )
        GROUP BY
            year
        ORDER BY
            year; \
        """

# Execute the query and load the results into a pandas DataFrame
us_evolution_df = pd.read_sql_query(query, engine)

print("Data for US GDP vs. Life Expectancy:")
print(us_evolution_df.head())


# In[23]:


# To compare the time series for GDP and Life Expectancy, we'll use a plot with two y-axes.
# This allows us to see both trends on the same chart, even though their scales are very different.
fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.set_title('Time Series of GDP and Life Expectancy in the United States', fontsize=16)
ax1.set_xlabel('Year', fontsize=12)

# Configure the primary (left) y-axis for GDP
color1 = 'tab:blue'
ax1.set_ylabel('GDP per Capita (current international $)', color=color1, fontsize=12)
ax1.plot(us_evolution_df['year'], us_evolution_df['gdp_per_capita'], color=color1, marker='o', label='GDP per Capita')
ax1.tick_params(axis='y', labelcolor=color1)

# Create a secondary (right) y-axis that shares the same x-axis
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Life Expectancy at Birth (years)', color=color2, fontsize=12)
ax2.plot(us_evolution_df['year'], us_evolution_df['life_expectancy'], color=color2, marker='x', label='Life Expectancy')
ax2.tick_params(axis='y', labelcolor=color2)

# Add a unified legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

fig.tight_layout() # Adjust plot to prevent labels from overlapping
plt.show()


# ## Example: Load Economic Blocks Table
# 
# Load country classification data into a new SQL table, using a pure Python approach
# 

# In[25]:


cont_blocks_df = pd.read_csv('country_classification.csv', sep=';')
cont_blocks_df.rename(columns={
    'Country Code': 'country_code',
    'bloc': 'economic_bloc' # Renaming for clarity and to avoid potential reserved words
}, inplace=True)

# Use the SQLAlchemy engine to load the DataFrame into the database
db_url = "postgresql+psycopg2://psqluser:psqlpass@localhost:5433/bigdatatools1"
engine = create_engine(db_url)
table_name = 'country_classification'
cont_blocks_df.to_sql(table_name, engine, if_exists='replace', index=False)

print(f"Table '{table_name}' created and loaded with {len(cont_blocks_df)} rows.")


# ## Compare Overall GDP: US vs EU using a JOIN
# We will use a JOIN with our country_classification table to calculate the total GDP for the EU
# by summing the GDP of its member states. We then compare this to the total GDP of the US.

# In[26]:


from matplotlib.ticker import FuncFormatter

# Ensure the SQLAlchemy engine is available
db_url = "postgresql+psycopg2://psqluser:psqlpass@localhost:5433/bigdatatools1"
engine = create_engine(db_url)

# The query first calculates the total GDP for each BRICS country, then sums them up by year.
# It then calculates the total GDP for the US.
# Finally, it combines these two results using UNION ALL.
query = """
        WITH brics_country_gdp AS (
            -- Step 1: Calculate total GDP for each individual BRICS country using a JOIN
            SELECT ci.year,
                   ci.country_name,
                   (MAX(CASE
                            WHEN ci.indicator_name = 'GDP per capita PPP (current international $)' THEN ci.value END) *
                    MAX(CASE WHEN ci.indicator_name = 'Population, total' THEN ci.value END)) AS total_gdp
            FROM country_indicators ci
                     JOIN country_classification cc ON ci.country_code = cc.country_code
            WHERE cc.economic_bloc = 'BRICS'
              AND ci.indicator_name IN ('GDP per capita PPP (current international $)', 'Population, total')
            GROUP BY ci.year, ci.country_name
            -- Ensure both population and gdp_per_capita exist for the calculation
            HAVING MAX(CASE
                           WHEN ci.indicator_name = 'GDP per capita PPP (current international $)'
                               THEN ci.value END) IS NOT NULL
               AND MAX(CASE WHEN ci.indicator_name = 'Population, total' THEN ci.value END) IS NOT NULL),
             brics_total_gdp AS (
                 -- Step 2: Sum the GDP of all EU countries for each year
                 SELECT
            year, 'BRICS (Calculated)' AS entity, SUM (total_gdp) AS total_gdp_ppp
        FROM brics_country_gdp
        GROUP BY year
            ),
            us_total_gdp AS (
        -- Step 3: Calculate total GDP for the United States
        SELECT
            year, 'United States' AS entity, (MAX (CASE WHEN indicator_name = 'GDP per capita PPP (current international $)' THEN value END) *
            MAX (CASE WHEN indicator_name = 'Population, total' THEN value END)) AS total_gdp_ppp
        FROM country_indicators
        WHERE
            country_name = 'United States'
          AND indicator_name IN ('GDP per capita PPP (current international $)'
            , 'Population, total')
        GROUP BY year
            )
-- Step 4: Combine the results
        SELECT *
        FROM brics_total_gdp
        UNION ALL
        SELECT *
        FROM us_total_gdp
        ORDER BY entity, year; \
        """

# Execute the query
us_brics_gdp_df = pd.read_sql_query(query, engine)

# For a bar plot, it's best to compare a few specific years
years_for_plot = [1995, 2005, 2015, 2021]  # Choosing some representative years
plot_df = us_brics_gdp_df[us_brics_gdp_df['year'].isin(years_for_plot)]

# Pivot the data to make it suitable for a grouped bar plot and use pandas plotting
pivot_df = plot_df.pivot(index='year', columns='entity', values='total_gdp_ppp')
ax = pivot_df.plot(kind='bar', figsize=(12, 8), width=0.8, colormap='viridis')

ax.set_title('GDP Comparison: United States vs. BRICS (Calculated)', fontsize=18)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Total GDP (PPP, current international $)', fontsize=14)
ax.tick_params(axis='x', rotation=0)  # Keep year labels horizontal


# Format y-axis to be more readable (in trillions)
def trillions(x, pos):
    'The two args are the value and tick position'
    return f'${x * 1e-12:1.1f}T'

formatter = FuncFormatter(trillions)
ax.yaxis.set_major_formatter(formatter)

ax.legend(title='Entity')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# **Challenge**: Include ASEAN and European Union in the Analysis
# 

# # 1.b)  Comparación COPY vs INSERT fila por fila
# 
# 
# Función que usa el método INSERT fila por fila para cargar el contenido del DataFrame combined_data en la tabla country_indicators del PostgreSQL

# In[31]:


def insert_row_by_row(df, conn):
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO country_indicators 
            (country_name, country_code, indicator_name, indicator_code, year, value)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            row['country_name'], 
            row['country_code'], 
            row['indicator_name'], 
            row['indicator_code'], 
            int(row['year']), 
            float(row['value']) if pd.notnull(row['value']) else None
        ))
    conn.commit()
    cur.close()


# # Lo mismo pero con la función que implementa el metodo COPY

# In[37]:


def copy_from_csv(df, conn):
    cur = conn.cursor()
    buffer = io.StringIO()
    # Reordenar columnas al mismo orden de la tabla
    df[['country_name', 'country_code', 'indicator_name', 'indicator_code', 'year', 'value']] \
        .to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)
    cur.copy_from(buffer, 'country_indicators',
                  sep="\t",
                  columns=('country_name', 'country_code', 'indicator_name', 'indicator_code', 'year', 'value'))
    conn.commit()
    cur.close()


# # Comparación

# In[38]:


import time
import psycopg2

# 🚦 1. Vaciar la tabla
conn = psycopg2.connect(
    host="localhost", port=5433,
    database="bigdatatools1",
    user="psqluser", password="psqlpass"
)
cur = conn.cursor()
cur.execute("TRUNCATE country_indicators;")
conn.commit()
cur.close()
conn.close()

# 🚦 2. Medir INSERT fila por fila
conn = psycopg2.connect(
    host="localhost", port=5433,
    database="bigdatatools1",
    user="psqluser", password="psqlpass"
)
start = time.time()
insert_row_by_row(combined_data, conn)
end = time.time()
conn.close()
print("⏱ Tiempo INSERT fila por fila:", round(end - start, 2), "segundos")

# 🚦 3. Vaciar la tabla otra vez
conn = psycopg2.connect(
    host="localhost", port=5433,
    database="bigdatatools1",
    user="psqluser", password="psqlpass"
)
cur = conn.cursor()
cur.execute("TRUNCATE country_indicators;")
conn.commit()
cur.close()
conn.close()

# 🚦 4. Medir COPY
conn = psycopg2.connect(
    host="localhost", port=5433,
    database="bigdatatools1",
    user="psqluser", password="psqlpass"
)
start = time.time()
copy_from_csv(combined_data, conn)
end = time.time()
conn.close()
print("⏱ Tiempo COPY:", round(end - start, 2), "segundos")


# ### Conclusión
# 
# Al comparar los dos métodos de inserción en PostgreSQL se observa lo siguiente:
# 
# - **INSERT fila por fila**: tardó 16.6 segundos en cargar los datos.
# - **COPY**: tardó únicamente 0.26 segundos para la misma cantidad de datos.
# 
# Esto demuestra que el método **COPY** es mucho más eficiente que los INSERT individuales, ya que aprovecha la carga masiva en bloque, reduciendo la sobrecarga de transacciones y llamadas al motor de base de datos.  
# En la práctica, **COPY es la estrategia recomendada para cargas grandes de datos**.
# 

# # 1c) Análisis de crecimiento del PIB por bloques económicos
# 
# En este punto se incluyen los países de la Unión Europea (UE) y de la ASEAN en el análisis de crecimiento del PIB.  
# Adicionalmente, se reemplaza el dato de Estados Unidos por el del bloque USMCA (Estados Unidos, México y Canadá).  
# 
# Se calculan las series temporales de PIB por bloque sumando el valor del PIB de los países que lo conforman.

# In[39]:


import matplotlib.pyplot as plt

# --- Definir bloques económicos ---
asean = [
    "Brunei Darussalam", "Cambodia", "Indonesia", "Lao PDR",
    "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam"
]

european_union = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Netherlands", "Poland", "Portugal", "Romania",
    "Slovak Republic", "Slovenia", "Spain", "Sweden"
]

usmca = ["United States", "Mexico", "Canada"]

# --- Filtrar solo el indicador PIB (ajusta el nombre exacto si es distinto en tu DF) ---
gdp_data = combined_data[combined_data["indicator_name"].str.contains("GDP", case=False)]

# --- Función para agrupar por bloque ---
def aggregate_block(df, countries, block_name):
    block_df = df[df["country_name"].isin(countries)]
    agg = block_df.groupby("year")["value"].sum().reset_index()
    agg["block"] = block_name
    return agg

# --- Calcular PIB por bloque ---
asean_gdp = aggregate_block(gdp_data, asean, "ASEAN")
eu_gdp = aggregate_block(gdp_data, european_union, "European Union")
usmca_gdp = aggregate_block(gdp_data, usmca, "USMCA")

# --- Combinar todos ---
blocks_gdp = pd.concat([asean_gdp, eu_gdp, usmca_gdp])

# --- Graficar ---
plt.figure(figsize=(10,6))
for block, data in blocks_gdp.groupby("block"):
    plt.plot(data["year"], data["value"], label=block)

plt.title("Crecimiento del PIB por Bloques Económicos")
plt.xlabel("Año")
plt.ylabel("PIB (US$ actuales o PPP)")
plt.legend()
plt.grid(True)
plt.show()


# # Punto 2 - Consultas SQL con JOIN (ASEAN vs Mercosur)
# 
# En este punto se construyen consultas SQL que permiten identificar, para el año 2019, 
# qué países cumplen simultáneamente con:
# - Esperanza de vida al nacer superior a 75 años.
# - PIB per cápita (PPP, dólares internacionales actuales) superior a 20.000.
# 
# Se realiza la consulta para el bloque ASEAN y luego para el bloque Mercosur, 
# comparando los resultados obtenidos.

# In[45]:


import pandas as pd
from sqlalchemy import create_engine

# =========================
# 1) Conexión (SQLAlchemy)
# =========================
engine = create_engine("postgresql+psycopg2://psqluser:psqlpass@localhost:5433/bigdatatools1")

# =========================
# 2) Parámetros del análisis
# =========================
YEAR = 2019
LIFE_MIN = 75
GDP_MIN = 20000

# Nombres EXACTOS de los indicadores (según lo que listaste)
LIFE_IND = "Life expectancy at birth, total (years)"
GDP_IND  = "GDP per capita PPP (current international $)"

# Listas de países por bloque
ASEAN = [
    "Brunei Darussalam","Cambodia","Indonesia",
    "Lao PDR","Malaysia","Myanmar",
    "Philippines","Singapore","Thailand","Vietnam"
]

MERCOSUR = ["Argentina","Brazil","Paraguay","Uruguay"]

def quote_list(xs):
    """Convierte ['A','B'] -> 'A','B' para el IN (...) de SQL."""
    return ", ".join(f"'{x}'" for x in xs)

# =========================
# 3) Plantilla de consulta
# =========================
def make_query(countries_list):
    countries_sql = quote_list(countries_list)
    q = f"""
    WITH life AS (
        SELECT country_name, year, value::float AS life_expectancy
        FROM country_indicators
        WHERE indicator_name = '{LIFE_IND}' AND year = {YEAR}
    ),
    gdp AS (
        SELECT country_name, year, value::float AS gdp_per_capita
        FROM country_indicators
        WHERE indicator_name = '{GDP_IND}' AND year = {YEAR}
    )
    SELECT l.country_name, l.life_expectancy, g.gdp_per_capita
    FROM life l
    JOIN gdp g
      ON l.country_name = g.country_name AND l.year = g.year
    WHERE l.country_name IN ({countries_sql})
      AND l.life_expectancy > {LIFE_MIN}
      AND g.gdp_per_capita > {GDP_MIN}
    ORDER BY g.gdp_per_capita DESC;
    """
    return q

query_asean    = make_query(ASEAN)
query_mercosur = make_query(MERCOSUR)

# =========================
# 4) Ejecutar consultas
# =========================
asean_df    = pd.read_sql_query(query_asean, engine)
mercosur_df = pd.read_sql_query(query_mercosur, engine)

print("Resultados ASEAN:")
print(asean_df if not asean_df.empty else pd.DataFrame(columns=["country_name","life_expectancy","gdp_per_capita"]))

print("\nResultados Mercosur:")
print(mercosur_df if not mercosur_df.empty else pd.DataFrame(columns=["country_name","life_expectancy","gdp_per_capita"]))

# =========================
# 5) Comparación en una sola tabla
# =========================
asean_df["block"] = "ASEAN"
mercosur_df["block"] = "Mercosur"

comparison_df = pd.concat([asean_df, mercosur_df], ignore_index=True)

print("\nTabla comparativa ASEAN vs Mercosur:")
print(comparison_df if not comparison_df.empty else pd.DataFrame(columns=["country_name","life_expectancy","gdp_per_capita","block"]))


# # Resumen y visualización (ASEAN vs Mercosur)

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt

# Nos aseguramos de que las columnas numéricas estén en formato numérico
comparison_df["life_expectancy"] = pd.to_numeric(comparison_df["life_expectancy"], errors="coerce")
comparison_df["gdp_per_capita"]  = pd.to_numeric(comparison_df["gdp_per_capita"],  errors="coerce")

# 1) Resumen por bloque
summary = (
    comparison_df
    .groupby("block")
    .agg(
        countries=("country_name", "nunique"),
        life_expectancy_avg=("life_expectancy", "mean"),
        gdp_per_capita_avg=("gdp_per_capita", "mean"),
        gdp_per_capita_median=("gdp_per_capita", "median")
    )
    .round(2)
    .reset_index()
)

print("Resumen por bloque (ASEAN vs Mercosur):")
print(summary)

# 2) Barras – PIB per cápita por país
ax = comparison_df.sort_values("gdp_per_capita").plot(
    kind="barh",
    x="country_name",
    y="gdp_per_capita",
    figsize=(8, 5),
    legend=False
)
ax.set_title("PIB per cápita (PPP, intl $) — Países filtrados")
ax.set_xlabel("PIB per cápita")
ax.set_ylabel("País")
for i, v in enumerate(comparison_df.sort_values("gdp_per_capita")["gdp_per_capita"]):
    ax.text(v, i, f"  {v:,.0f}", va="center", fontsize=8)
plt.tight_layout()
plt.show()

# 3) Barras – Esperanza de vida por país
ax = comparison_df.sort_values("life_expectancy").plot(
    kind="barh",
    x="country_name",
    y="life_expectancy",
    figsize=(8, 5),
    legend=False
)
ax.set_title("Esperanza de vida al nacer — Países filtrados")
ax.set_xlabel("Años")
ax.set_ylabel("País")
for i, v in enumerate(comparison_df.sort_values("life_expectancy")["life_expectancy"]):
    ax.text(v, i, f"  {v:.2f}", va="center", fontsize=8)
plt.tight_layout()
plt.show()

# 4) Dispersión – Vida vs PIB per cápita, coloreado por bloque
fig, ax = plt.subplots(figsize=(7, 5))
for blk, sub in comparison_df.groupby("block"):
    ax.scatter(sub["gdp_per_capita"], sub["life_expectancy"], s=80, label=blk)
    # etiquetas de puntos
    for _, r in sub.iterrows():
        ax.annotate(r["country_name"], (r["gdp_per_capita"], r["life_expectancy"]),
                    xytext=(5, 2), textcoords="offset points", fontsize=8)

ax.set_title("Esperanza de vida vs PIB per cápita")
ax.set_xlabel("PIB per cápita (PPP, intl $)")
ax.set_ylabel("Esperanza de vida (años)")
ax.legend(title="Bloque")
plt.tight_layout()
plt.show()

