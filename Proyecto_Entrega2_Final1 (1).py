# Databricks notebook source
# MAGIC %md
# MAGIC #**PROYECTO ENTREGA 2**
# MAGIC **Universidad**: Pontificia Universidad Javeriana\
# MAGIC **Profesor**: John Corredor\
# MAGIC **Nombres**:     
# MAGIC >Daniel Ordoñez
# MAGIC
# MAGIC >Neyl Peñuela
# MAGIC
# MAGIC >Laura Salamanca
# MAGIC
# MAGIC
# MAGIC **Carrera**:     Ciencia de Datos\
# MAGIC **Materia**:     Procesamiento de datos a gran escala\
# MAGIC **Fecha:**    10/9/2023 \
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Exploración inicial, transformaciones y filtros de los Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC En esta sección se realizaran las exploraciones iniciales y las transformaciones necesarias de todos los siguientes datasets (_**Corrección primera entrega**_):
# MAGIC
# MAGIC - **NYPD Arrest Data (Year to Date)**: este conjunto de datos tiene indicadores sobre la capacidad
# MAGIC policial para atrapar y arrestar a los criminales y la seguridad de la ciudad.
# MAGIC - **NYCgov Poverty Measure Data**: este conjunto de datos nos ofrece distintas características de
# MAGIC las personas en condición de pobreza en el estado de Nueva York.
# MAGIC - **Motor Vehicle Collisions - Vehicles**: este conjunto de datos nos ofrece distintas características
# MAGIC de los accidentes y/o colisiones que se dieron en el estado de Nueva York.
# MAGIC - **2016 - 2017 Health Education**: este conjunto de datos nos ofrece características sobre los
# MAGIC colegios y los estudiantes de grados 9 a 12.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Importación de librerias

# COMMAND ----------

import pyspark
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import count, when, col
import pandas as pd

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Dataset de arrestos (_NYPD Arrest Data (Year to Date)_)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Carga de datos

# COMMAND ----------

#Se lee el archivo especificando la dirección de donde se va a tomar y el tipo al que pertenece
file_location = "dbfs:/FileStore/tables/NYPD_Arrest_Data__Year_to_Date__20231105.csv"
file_type = "csv"

# Se especifican las opciones de csv que utiliza databricks para lectura de archivos.
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# Se crea el dataframe a partir de lo descrito anteriormente
df_arrestos = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
#Se muestran los datos en su totalidad
display(df_arrestos)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Descripcion general

# COMMAND ----------

#Se presentan los nombres de las columnas 
df_arrestos.columns

# COMMAND ----------

# MAGIC %md
# MAGIC + debido a la falta de caracter explicativo de algunas de las columnas, se decide cambiar los nombres de estas por otros más faciles de identificar.

# COMMAND ----------

df_arrestos = df_arrestos.withColumnRenamed("PD_CD", "CODIGO_ESPECIFICO_DE_DELITO")
df_arrestos = df_arrestos.withColumnRenamed("PD_DESC", "DESCRIPCIÓN_CODIGO_ESPECIFICO")
df_arrestos = df_arrestos.withColumnRenamed("KY_CD", "CATEGORIA_SECUNDARIA_INTERNA")
df_arrestos = df_arrestos.withColumnRenamed("OFNS_DESC", "DESCRIPCIÓN_DEL_DELITO")
df_arrestos = df_arrestos.withColumnRenamed("LAW_CAT_CD", "NIVEL_DEL_DELITO")
df_arrestos = df_arrestos.withColumnRenamed("ARREST_BORO", "DISTRITO_DE_ARRESTO")
df_arrestos = df_arrestos.withColumnRenamed("ARREST_PRECINCT", "COMISARIA_DE_ARRESTO")
df_arrestos = df_arrestos.withColumnRenamed("JURISDICTION_CODE", "CODIGO_DE_JURISDICCIÓN_RESPONSABLE")

# COMMAND ----------

df_arrestos.columns

# COMMAND ----------

df_arrestos.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observaciones en el conjunto de datos**
# MAGIC >+ El schema muestra una ingesta correcta del conjunto de datos, cada variable cuenta con un tipo de dato razonable.
# MAGIC >+ Dada la naturaleza de los datos (registros que no cuentan con variables que representen medidas) , no se ve necesario hacer un describe para verificar las medidas de tendencia central del dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Manejo de nulos

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import count, when, col

#Funcion para contar los nulos de un dataframe
def conteo_nulos():
    conteo = df_arrestos.select([count(when(col(c).isNull(), c)).alias(c) for c in df_arrestos.columns])

    tabla = conteo.toPandas()
    
    #Para que al imprimir la tabla, la muestre completa
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    print(tabla)

conteo_nulos()

# COMMAND ----------

valores_unicos = df_arrestos.select("DESCRIPCIÓN_DEL_DELITO").distinct().rdd.flatMap(lambda x: x).collect()

for valor in valores_unicos:
    print(valor)


# COMMAND ----------

valores_unicos = df_arrestos.select("DESCRIPCIÓN_CODIGO_ESPECIFICO").distinct().rdd.flatMap(lambda x: x).collect()

for valor in valores_unicos:
    print(valor)

# COMMAND ----------

num_filas = df_arrestos.count()
print(f"El número de filas en el DataFrame es: {num_filas}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Observaciones del conteo de nulos**
# MAGIC > + Los nulos de "codigo_especifico_de_delito" son despreciables por el tamaño de los registros y lo que representa para la investigación, al ser un identificador secundario es posible obtener la misma calidad descriptiva tomando otro identificador como lo puede ser el codigo de ley. 
# MAGIC > +  Los nulos de "categoría_secundaria_interna" son despreciables por el tamaño de los registros, al ser un identificador secundario es posible obtener la misma calidad descriptiva tomando otro identificador como lo puede ser "Law_code" que es una forma más general de identificar el delito. 
# MAGIC > + Hay valores nulos que no se están teniendo en cuenta, dado que columnas como "Descripción_del_delito" ó "Descripción_codigo_especifico", cuentan con valores nulos que aparecen como '(null)'
# MAGIC > + Es importante eliminar los valores nulos de "NIVEL_DEL_DELITO" dado que al no poder contar con una forma de imputar los datos faltantes, puede que estos causen sesgos en el entrenamiento de modelos ML. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC  *Transformaciones:*
# MAGIC
# MAGIC >+ Transformación 1: Se completan los valores faltantes en la columna, se reemplazará cada descripción general con las encontradas en la página web del senado de nueva york.
# MAGIC >+ Transfromación 2: Se normaliza el formato en el que se encuentran los valores nulos de la columna "PD_DESC", pasando los valores '(null)' a NULL.
# MAGIC >+ Transfromación 3: Se llenan los espacios vacios de las columnas numericas de codigos con "0" para evitar perdida de información por columnas no relevantes
# MAGIC >+ Transformación 4: Se llenan los espacios vacioes de las columnas de descripción secundaria con "no_cuenta" para evitar perdida de información por columnas no relevantes
# MAGIC
# MAGIC >+ Filtro 1: Se eliminan los valores nulos de la columna "NIVEL_DEL_DELITO".
# MAGIC

# COMMAND ----------

#Transformación 1:
from pyspark.sql.functions import when, col
#Creación de un diccionario para reemplazo de valores
category_mapping = {
    "VTL04020BI": "Inclumplimiento de limpieza de placa",
    "VTL119204T": "Conducción bajo el efecto de alcohol",
    "RPA0076801": "Intento de desalojo ilegal",
    "PL 2650700": "Porte de arma sin serialización",
    "PL 265019I": "Posesión de armas agravado",
    "PL 2650110": "Posesión criminal de armas en cuarto grado",
    "PL 241051F": "Acoso a un inquilino en primer grado",
    "PL 2410202": "Acoso a un inquilino en regimen de alquiler en segundo grado",
    "PL 2410200": "Acoso a un inquilino en regimen de alquiler en segundo grado",
    "PL 2225500": "Venta de cannabis en segundo grado",
    "PL 2224000": "Posesión de cannabis en primer grado",
    "PL 2223500": "Posesión de cannabis en segundo grado",
    "PL 2223000": "Posesión de cannabis en tercer grado",
    "CPL5700600": "Aplicación de ley a criminal buscado",
    "PL 215401B": "Alteración de pruebas",
    "PL 1251401": "Homicidio vehicular agravado"
}
#Se mapean los items que se contienen al hacer la busqueda de descripción por codigo
for category, new_value in category_mapping.items():
    df_arrestos = df_arrestos.withColumn("DESCRIPCIÓN_DEL_DELITO ", when(col("LAW_CODE") == category, new_value).otherwise(col("DESCRIPCIÓN_DEL_DELITO")))
    
display(df_arrestos)

# COMMAND ----------

#Transformación 3:
from pyspark.sql.functions import when, col

# Reemplazar "(null)" con valores nulos en la columna "PD_DESC"
df_arrestos = df_arrestos.withColumn("DESCRIPCIÓN_CODIGO_ESPECIFICO", when(col("DESCRIPCIÓN_CODIGO_ESPECIFICO") == "(null)", None).otherwise(col("DESCRIPCIÓN_CODIGO_ESPECIFICO")))

# Muestra la columna "LAW_CODE" resultante
display(df_arrestos.select("DESCRIPCIÓN_CODIGO_ESPECIFICO"))

# COMMAND ----------

from pyspark.sql.functions import col

# Elimina los valores nulos de la columna "NIVEL_DEL_DELITO"
df_arrestos = df_arrestos.filter(col("NIVEL_DEL_DELITO").isNotNull())


# COMMAND ----------

from pyspark.sql.functions import col

# Reemplazar valores nulos en la columna "CODIGO_ESPECIFICO_DE_DELITO" con 0
# Reemplazar valore nulos en la columna "DESCRIPCIÓN_CODIGO_ESPECIFICO" con "no cuenta"
df_arrestos = df_arrestos.fillna(0)
df_arrestos = df_arrestos.fillna("no cuenta")
conteo_nulos()

# COMMAND ----------

# MAGIC %md
# MAGIC **Se verifican los cambios realizados a partír de consultas sql**

# COMMAND ----------

#Se crea una tabla temporal para verificar los resultados
temp_table_name = "Arrestos"
df_arrestos.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC --Se verifican las transformaciones en la columna "DESCRIPCIÓN_CODIGO_ESPECIFICO"--
# MAGIC SELECT count(*)
# MAGIC FROM Arrestos
# MAGIC WHERE `DESCRIPCIÓN_CODIGO_ESPECIFICO` IS NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC ###Manejo de duplicados

# COMMAND ----------

# Contar filas duplicadas
duplicados = df_arrestos.count() - df_arrestos.dropDuplicates().count()

print("Número de filas duplicadas:", duplicados)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Dataset de pobreza (_NYCgov Poverty Measure Data_)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Carga de datos

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/NYCgov_Poverty_Measure_Data__2018__20231106.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_pobreza = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df_pobreza)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Descripcion general

# COMMAND ----------

#se presentan los nombres de las columnas
df_pobreza.columns

# COMMAND ----------

#Se revisa el schema (tipo de datos de cada columna)
df_pobreza.printSchema()

# COMMAND ----------

#Se renombran las columnas con nombres poco faciles de identificar
df_pobreza = df_pobreza.withColumnRenamed("SERIALNO","NumSerieHogar")\
    .withColumnRenamed("SPORDER","NumOrdenPersonas")\
    .withColumnRenamed("PWGTP","PesoPersona")\
    .withColumnRenamed("WGTP","Sueldo12Meses")\
    .withColumnRenamed("AGEP","CategoriaEdad")\
    .withColumnRenamed("CIT","EstadoCiudadania")\
    .withColumnRenamed("REL","RelacionPrincipal")\
    .withColumnRenamed("SCH","InscripcionEscolar")\
    .withColumnRenamed("SCHG","Grado")\
    .withColumnRenamed("SCHL","NivelEducativo")\
    .withColumnRenamed("ESR","RegistroEstadoEmpleo")\
    .withColumnRenamed("LANX","SegundoIdiomaHogar")\
    .withColumnRenamed("ENG","HabilidadIngles")\
    .withColumnRenamed("MSP","EstadoMarital")\
    .withColumnRenamed("MAR","EstadoCivil")\
    .withColumnRenamed("WKW","HorasTrabajadasSemana")\
    .withColumnRenamed("WKHP","PesoVivienda")\
    .withColumnRenamed("DIS","RegistroDiscapacidad")\
    .withColumnRenamed("JWTR","MediTransporteTrabajo")\
    .withColumnRenamed("NP","NumPersonas")\
    .withColumnRenamed("TEN","TenenciaVivienda")\
    .withColumnRenamed("HHT","TipoHogar")
display(df_pobreza)

# COMMAND ----------

#Resumen estadistico del dataframe
import pandas as pd
# Realiza el resumen con describe()
resumen_describe = df_pobreza.describe()

# Convierte el resumen a un DataFrame de Pandas P
resumen_pandas = resumen_describe.toPandas()

# Transpone el DataFrame de Pandas para asegurar legibilidad
resumen_transpuesto = resumen_pandas.T

#Imprime el DataFrame resultante
resumen_transpuesto

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC El describe del dataset, el que indica distintas métricas estadisticas, nos ayudará a solucionar el problema de los nulos.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Manejo de nulos

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import count, when, col

#Funcion para contar los nulos de un dataframe
def conteo_nulos():
    conteo = df_pobreza.select([count(when(col(c).isNull(), c)).alias(c) for c in df_pobreza.columns])

    tabla = conteo.toPandas()
    
    #Para que al imprimir la tabla, la muestre completa
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    print(tabla)

conteo_nulos()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Las columnas que tienen nulos y su respectivo tratamiento será:
# MAGIC
# MAGIC - **NivelEducativo**: en este caso, el rango esta entre _(1,24)_, por lo que se toma la decisión de llenar los nulos con 0. Esto indicaría o que no hay registro o que no se tiene un nivel educativo. 
# MAGIC - **RegistroEstadoEmpleo**: en este caso se llenará con 0. Por el diccionario de datos tenemos que se da principalmente porque las personas tienen menos de 16 años, o sea, no hacen parte de la fuerza laboral. 
# MAGIC - **SegundoIdiomaHogar**: en este caso se llenará con 0. Por el diccionario de datos tenemos que se da principalmente porque las personas tienen menos de 5 años.
# MAGIC - **HabilidadIngles**: en este caso se llenará con 0. Por el diccionario de datos tenemos que se da principalmente porque las personas tienen menos de 5 años, por lo que es imposible clasificar a estas personas. 
# MAGIC - **EstadoMarital**: en este caso se llenará con 0. Por el diccionario de datos tenemos que se da principalmente porque las personas tienen menos de 15 años, por lo que se consideran que no tienen o no clasifican en ningun estado marital. 
# MAGIC - **HorasTrabajadasSemana**: en este caso se llenará con 0. Por el diccionario de datos tenemos que se da principalmente porque las personas tienen menos de 16 años, o sea, no hacen parte de la fuerza laboral. 
# MAGIC - **MediTransporteTrabajo**: se llenará con 0. A pesar de que es una clasificación de 1 a 12, no se tiene información concreta de cada una de las categorias. Sin embargo, para no eliminar los registros, se decide llenarlos con 0. 
# MAGIC - **EducAttain**: se llenará con 0, indicando que no tiene ningun logro, o no hay registro de ello. 
# MAGIC
# MAGIC En conclusión, todas las columnas con nulos se llenan de 0. En cada una, ese valor tiene un significado diferente. 

# COMMAND ----------

# Rellena los valores nulos con 0
df_pobreza = df_pobreza.na.fill(0)

# COMMAND ----------

conteo_nulos()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Se confirma que no hayan nulos en el dataset. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Duplicados

# COMMAND ----------

# Contar filas duplicadas
duplicados = df_pobreza.count() - df_pobreza.dropDuplicates().count()

print("Número de filas duplicadas:", duplicados)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC En este dataset no tenemos ningun problema en cuanto a registros duplicados. Por lo tanto, se culmina la limpieza de la base. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Filtros
# MAGIC

# COMMAND ----------

temp_table_name = "pobreza"
df_pobreza.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC En primer lugar, tenemos que un identificador para cada hogar/casa. Sin embargo, este no es único; pues hay varias personas que viven en ese hogar. Es de interes saber cuantas personas viven en cada casa, pues eso puede dar un indicio de pobreza.

# COMMAND ----------

filtro1_pobreza = spark.sql("""SELECT NumSerieHogar,
                COUNT(NumOrdenPersonas) AS total_personas_en_hogar
            FROM pobreza
            GROUP BY NumSerieHogar""")
       
display(filtro1_pobreza)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC No consideramos necesario realizar otro filtro sobre esta base de datos. Como los datos son tan únicos (referentes a cada persona), los filtros no tendrían tanto sentido para nuestro analisis (el cual es de forma general). Por ello no se realizan más filtos
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Dataset de colisiones (_Motor Vehicle Collisions - Vehicles_)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Carga de datos

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Motor_Vehicle_Collisions___Vehicles.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_colision = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df_colision)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Descripcion general

# COMMAND ----------

df_colision.columns

# COMMAND ----------

print(df_colision.dtypes)

# COMMAND ----------

df_colision.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Describe de los datasets:

# COMMAND ----------


#Resumen estadistico del dataframe
import pandas as pd
# Realiza el resumen con describe()
resumen_describe = df_colision.describe()

# Convierte el resumen a un DataFrame de Pandas P
resumen_pandas = resumen_describe.toPandas()

# Transpone el DataFrame de Pandas para asegurar legibilidad
resumen_transpuesto = resumen_pandas.T

#Imprime el DataFrame resultante
resumen_transpuesto


# COMMAND ----------

# MAGIC %md
# MAGIC ###Manejo de nulos:

# COMMAND ----------

def conteo_nulos():
    conteo = df_colision.select([count(when(col(c).isNull(), c)).alias(c) for c in df_colision.columns])

    tabla = conteo.toPandas()
    
    #Para que al imprimir la tabla, la muestre completa

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    print(tabla)

conteo_nulos()

# COMMAND ----------

license_null = df_colision.filter(
    (col("DRIVER_LICENSE_STATUS").isNull()) & (col("DRIVER_LICENSE_JURISDICTION").isNull())
)

print("La cantidad de siniestros viales registrados en este DB que tienen las columnas DRIVER_LICENSE_STATUS, DRIVER_LICENSE_JURISDICTION con valores nulos al mismo tiempo son: ", license_null.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Se denota que las variables más importantes son las relacionadas con la licencia del conductor; si se decide continuar con las filas que tienen los 3 valores en nulo podría entorpecer la investigación.

# COMMAND ----------

# Calcula el número de valores nulos en cada columna y crea la columna "null_count"
for c in df_colision.columns:
    df_colision = df_colision.withColumn(c + "_null_count", col(c).isNull().cast("int"))

# Calcula la suma de las columnas "_null_count" y crea una nueva columna "null_count" en df_colision
columnas_null_count = [col(c) for c in df_colision.columns if c.endswith("_null_count")]
df_colision = df_colision.withColumn("null_count", sum(columnas_null_count))

# Filtra las filas con "null_count" mayor que 4
df_colision = df_colision.filter(df_colision.null_count <= 4)

# Elimina las columnas "_null_count" que ya no son necesarias
df_colision = df_colision.drop(*[c for c in df_colision.columns if c.endswith("_null_count")])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Al seleccionar solo aquellos registros de siniestros viales que mostraban valores nulos en el registro del siniestro se acortó el dataframe significativamente, esto es necesario ya que son variables que se pueden establecer como variables objetivo. Después de esto, se muestra el tratado de los nulos de las demás columnas:
# MAGIC
# MAGIC - **VEHICLE_DAMAGE :** Al mostrarse que si esta columna está con un valor "null", los valores de los daños de los otros vehiculos estaran a su vez en "null", se entendería que en este caso aunque sea un vehiculo debió estar involucrado, entonces se cambía a "No_Damage".
# MAGIC - **VEHICLE_DAMAGE_1, VEHICLE_DAMAGE_2, VEHICLE_DAMAGE_3:** Al saber que para haber un accidente se necesita solo de un vehiculo, los valores nulos se reemplazaron por "No_involved".
# MAGIC - **PUBLIC_PROPERTY_DAMAGE_TYPE:** Muestra que casi en su totalidad está denotada como valores "null", por ende se decide quitar la columna en su totalidad.
# MAGIC - **VEHICLE_MAKE**: Los nulos se presentan como valores que faltaron poner en el reporte, por ende se pone "Indefinite".
# MAGIC - **VEHICLE_MODEL**: Se decide quitar la columna en su totalidad ya que está en una gran mayoría retratado por nulos, y a su vez no aporta mucha información. La información sobre el vehiculo ya la aportan otras columnas tal como Vehicle_Make
# MAGIC - **VEHICLE_YEAR**: Este valor no se puede automatizar, por ende se cambia por un 0 
# MAGIC - **TRAVEL_DIRECTION**: Los nulos en este caso se trabajan como direcciones y al haber nulos se cambia por "Unmarked", donde la dirección de la se provenia el accionante que produjo el accidente.
# MAGIC - **PRE_CRASH**: Los nulos se normalizan y se ponen como "Going straight ahead"
# MAGIC - **POINT_OF_IMPACT:** Los nulos se cambian por valores "Other"
# MAGIC - **CONTRIBUTING_FACTOR_1, CONTRIBUTING_FACTOR_2:** Los valores nulos se llevan a "Unspecified"
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Transformaciones:

# COMMAND ----------

# se decide quitar columnas que en su gran mayoria solo continen nulos
df_colision=df_colision.drop("PUBLIC_PROPERTY_DAMAGE_TYPE")\
                      .drop("VEHICLE_MODEL")


# COMMAND ----------

df_colision=df_colision.withColumn("VEHICLE_DAMAGE_2", when(col("VEHICLE_DAMAGE_2").isNull(), "No_involved").otherwise(col("VEHICLE_DAMAGE_2"))) \
    .withColumn("VEHICLE_DAMAGE_3", when(col("VEHICLE_DAMAGE_3").isNull(), "No_involved").otherwise(col("VEHICLE_DAMAGE_3"))) \
    .withColumn("VEHICLE_DAMAGE_1", when(col("VEHICLE_DAMAGE_1").isNull(), "No_involved").otherwise(col("VEHICLE_DAMAGE_1"))) \
    .withColumn("VEHICLE_DAMAGE", when(col("VEHICLE_DAMAGE").isNull(), "No_Damage").otherwise(col("VEHICLE_DAMAGE"))) \
    .withColumn("VEHICLE_MAKE", when(col("VEHICLE_MAKE").isNull(), "Indefinite").otherwise(col("VEHICLE_MAKE"))) \
    .withColumn("TRAVEL_DIRECTION", when(col("TRAVEL_DIRECTION").isNull(), "Unmarked").otherwise(col("TRAVEL_DIRECTION"))) \
    .withColumn("VEHICLE_YEAR", when(col("VEHICLE_YEAR").isNull(), 0).otherwise(col("VEHICLE_YEAR"))) \
    .withColumn("PRE_CRASH", when(col("PRE_CRASH").isNull(), "Going Straight Ahead").otherwise(col("PRE_CRASH"))) \
    .withColumn("POINT_OF_IMPACT", when(col("POINT_OF_IMPACT").isNull(), "Other").otherwise(col("POINT_OF_IMPACT"))) \
    .withColumn("CONTRIBUTING_FACTOR_1", when(col("CONTRIBUTING_FACTOR_1").isNull(), "Unspecified").otherwise(col("CONTRIBUTING_FACTOR_1"))) \
    .withColumn("CONTRIBUTING_FACTOR_2", when(col("CONTRIBUTING_FACTOR_2").isNull(), "Unspecified").otherwise(col("CONTRIBUTING_FACTOR_2")))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Duplicados

# COMMAND ----------

# Contar filas duplicadas
duplicados = df_colision.count() - df_colision.dropDuplicates().count()

print("Número de filas duplicadas en dataframe de siniestros viales:", duplicados)

# COMMAND ----------

# MAGIC %md
# MAGIC En este dataset no tenemos ningun problema en cuanto a registros duplicados. Por lo tanto, se culmina la limpieza de la base.

# COMMAND ----------

df_colision.count()

# COMMAND ----------

display(df_colision)

# COMMAND ----------

temp_table_name = "colisiones"
df_colision.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Dataset de educación (_2016 - 2017 Health Educations_)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Carga de datos

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/2016___2017_Health_Education_Report_20231107.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_education = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df_education)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Descripcion general

# COMMAND ----------

#se presentan los nombres de las columnas
df_education.columns

# COMMAND ----------

#Se revisa el schema (tipo de datos de cada columna)
df_education.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Revisando los registros, tenemos que en la mayoria de columnas hay caracteres 's', incluso en columnas de tipo integer. Se remplazara ese caracter por 0. 

# COMMAND ----------

for columna in df_education.columns:
    df_education = df_education.withColumn(columna, when(col(columna) == 's', '0').otherwise(col(columna)))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC De igual forma, vemos que la columna **%** y **% 1** esta como _string_, siendo realmente un _float_. Por lo tanto, se le quitara el caracter '%' a cada registro y se procederá a castear la columna

# COMMAND ----------

from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import col

columna = "%"
columna2 = "% 1"

# Carácter que deseas quitar
caracter_a_quitar = "%"

# Aplicar la transformación a la columna
df_education = df_education.withColumn(columna, regexp_replace(col(columna), caracter_a_quitar, ""))
df_education = df_education.withColumn(columna2, regexp_replace(col(columna2), caracter_a_quitar, ""))

df_education = df_education.withColumn(columna, col(columna).cast("float")).withColumn(columna2, col(columna2).cast("float"))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Despues de las anteriores tranformaciones, algunas columnas se cambiaron a _string_. Por esta razon, se hace el casteo manual de las respectivas columnas

# COMMAND ----------

df_education = df_education.withColumn("# of students in grades 9-12", col("# of students in grades 9-12").cast("Integer")).withColumn("# of students in grades 9-12 scheduled for at least one semester of health instruction", col("# of students in grades 9-12 scheduled for at least one semester of health instruction").cast("Integer")).withColumn("# of 16-17 June and August graduates", col("# of 16-17 June and August graduates").cast("Integer")).withColumn("# of 16-17 June and August graduates meeting high school health requirements", col("# of 16-17 June and August graduates meeting high school health requirements").cast("Integer"))

# COMMAND ----------

#Se revisa el schema (tipo de datos de cada columna)
df_education.printSchema()

# COMMAND ----------

#Resumen estadistico del dataframe
import pandas as pd
# Realiza el resumen con describe()
resumen_describe = df_education.describe()

# Convierte el resumen a un DataFrame de Pandas P
resumen_pandas = resumen_describe.toPandas()

# Transpone el DataFrame de Pandas para asegurar legibilidad
resumen_transpuesto = resumen_pandas.T

#Imprime el DataFrame resultante
resumen_transpuesto

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Manejo de nulos

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import count, when, col

#Funcion para contar los nulos de un dataframe
def conteo_nulos():
    conteo = df_education.select([count(when(col(c).isNull(), c)).alias(c) for c in df_education.columns])

    tabla = conteo.toPandas()
    
    #Para que al imprimir la tabla, la muestre completa
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    print(tabla)

conteo_nulos()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Se llenan los nulos con 0, pues son registros que no tienen ninguna información

# COMMAND ----------

# Rellena los valores nulos con 0
df_education = df_education.na.fill(0)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Se confirma que no hayan nulos en el dataset. 

# COMMAND ----------

conteo_nulos()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Duplicados

# COMMAND ----------

# Contar filas duplicadas
duplicados = df_education.count() - df_education.dropDuplicates().count()

print("Número de filas duplicadas:", duplicados)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC En este dataset no tenemos ningun problema en cuanto a registros duplicados. Por lo tanto, se culmina la limpieza de la base. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Filtros
# MAGIC

# COMMAND ----------

df_education = df_education.withColumn("Boro", when(df_education["School DBN"].substr(3, 1) == "X", "Bronx")
                               .when(df_education["School DBN"].substr(3, 1) == "K", "Brooklyn")
                               .when(df_education["School DBN"].substr(3, 1) == "M", "Manhattan")
                               .when(df_education["School DBN"].substr(3, 1) == "Q", "Queens")
                               .when(df_education["School DBN"].substr(3, 1) == "R", "Staten Island")
                               .otherwise("Desconocido"))


# COMMAND ----------

display(df_education)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Preguntas
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####¿Qué características (edad,raza,sexo) representan la mayor proporción de arrestos cometidos dentro del dataset?

# COMMAND ----------

from pyspark.sql import functions as F

# Contar arrestos por raza, sexo y edad
arrestos_por_raza_sexo_edad = df_arrestos.groupBy("PERP_RACE", "PERP_SEX", "AGE_GROUP").count()

# Calcular la proporción de arrestos por raza, sexo y edad
total_arrestos = df_arrestos.count()
arrestos_por_raza_sexo_edad = arrestos_por_raza_sexo_edad.withColumn("proporcion_arrestos", F.col("count") / total_arrestos)

# Encontrar la combinación de raza, sexo y edad con la mayor proporción de arrestos
combinacion_con_mas_arrestos = arrestos_por_raza_sexo_edad.orderBy(F.col("proporcion_arrestos").desc()).first()

print(f"Raza con mayor proporción de arrestos: {combinacion_con_mas_arrestos['PERP_RACE']}  Género con mayor proporción de arrestos: {combinacion_con_mas_arrestos['PERP_SEX']}  Edad con mayor proporción de arrestos: {combinacion_con_mas_arrestos['AGE_GROUP']}")

# COMMAND ----------

# MAGIC %md
# MAGIC *Observaciones*
# MAGIC >+ Es importante que el grupo racial con mayor numero de arrestos realizados son afroamericanos hombres, entre la edad de 25-44 Esto en un primer vistazo puede ser un indicador para hacer posteriores inferencias agregando otros indicadores como lo podría ser el nivel del delito.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####¿Qué nivel de infracción es el más recurrente?

# COMMAND ----------

from pyspark.sql.functions import desc

# Agrupa los datos por el nivel de delito y cuenta la frecuencia de cada nivel
nivel_delito_frecuencia = df_arrestos.groupBy("NIVEL_DEL_DELITO").count()

# Ordena en orden descendente por frecuencia para encontrar el nivel de delito más recurrente
nivel_delito_frecuencia = nivel_delito_frecuencia.orderBy(desc("count"))

# Obtiene el nivel de delito más recurrente
nivel_delito_mas_recurrente = nivel_delito_frecuencia.first()

# Imprime el resultado
print("El nivel de delito más recurrente es:", nivel_delito_mas_recurrente["NIVEL_DEL_DELITO"])

# Detén la sesión de Spark

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Cómo se puede ver; el nivel de delito más recurrente es "M" o "Misdemeanor", que corresponde a un nivel de delito medio. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####¿Cuales son los _Boroughs_ con mayor proporción de personas por hogar? ¿Hay alguna relación con el índice de pobreza?
# MAGIC
# MAGIC

# COMMAND ----------

#Se hace la consulta que indica el promedio de personas por cada hogar, así como el promedio del indice de pobreza. 
#Primero se saca la cantidad de personas por cada hogar, y luego si se maneja segun la cantidad distinta de cant_de_hogares
#El promedio se hace en la siguiente celda

resultado = spark.sql("""
        SELECT Boro,
            total_personas_en_hogar,
            COUNT(NumSerieHogar) AS cantidad_de_hogares,
            AVG(promedio_pobreza_hogar) AS prom_indice_pobreza
       FROM (
            SELECT Boro,
                NumSerieHogar,
                COUNT(NumOrdenPersonas) AS total_personas_en_hogar,
                AVG(NYCgov_Pov_Stat) AS promedio_pobreza_hogar
            FROM pobreza
            GROUP BY Boro, NumSerieHogar
       ) tabla_total_personas_en_hogar
       GROUP BY Boro, total_personas_en_hogar""")

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

promedio_por_boro = resultado.groupBy("Boro").agg({"cantidad_de_hogares": "sum", "total_personas_en_hogar": "avg","prom_indice_pobreza":"avg" }).withColumnRenamed("avg(total_personas_en_hogar)", "promedio_personas_por_hogar").withColumnRenamed("avg(prom_indice_pobreza)", "promedio_indice_pobreza")

df_pandas = promedio_por_boro.toPandas()

nombres_boro = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

df_pandas["Boro"] = df_pandas["Boro"].map(lambda x: nombres_boro[x - 1])

ax = df_pandas.plot(x="Boro",y="promedio_personas_por_hogar", kind="bar", figsize=(12, 6))

plt.title("Promedio de personas que viven en un hogar por cada borough")
plt.xlabel("Barrio")
plt.ylabel("Cantidad de personas")

for index, value in enumerate(df_pandas["promedio_personas_por_hogar"]):
    ax.text(index, value, f"{value:.2f}", ha="center", va="bottom")

plt.show()

# COMMAND ----------

ax = df_pandas.plot(x="Boro",y="promedio_indice_pobreza", kind="bar", figsize=(12, 6))

plt.title("Promedio de indice de pobrez por cada borough")
plt.xlabel("Barrio")
plt.ylabel("Indice de pobreza promedio")

for index, value in enumerate(df_pandas["promedio_indice_pobreza"]):
    ax.text(index, value, f"{value:.2f}", ha="center", va="bottom")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Hay que tener en cuenta que, el indice de pobreza entre mas cercano a 1, indica más pobreza. Entre más cercano a 2, indica menos pobreza. 

# COMMAND ----------

promedio_por_boro.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Segun el primer grafico, vemos los _boroughs_ con un mayor promedio de personas por cada hogar son:
# MAGIC 1. Queens
# MAGIC 2. Bronx
# MAGIC 3. Brooklyn
# MAGIC 4. Staten Island
# MAGIC 5. Manhattan
# MAGIC
# MAGIC Por parte del promedio del indice de pobreza en cada barrio tenemos, en su respectivo orden de más pobreza a menos, los siguientes _boroughs_:
# MAGIC 1. Brooklyn
# MAGIC 2. Queens
# MAGIC 3. Bronx - Staten Island
# MAGIC 4. Manhattan
# MAGIC
# MAGIC A partir de ello podemos decir que los dos _boroughs_ con mayor promedio de personas por hogar son **Queens** y **Bronx**. Ademas, aunque los indices de pobreza no son los peores, si tienen un valor considerablemente alto. Por lo que se puede decir que el hecho de tener más gente por hogar indica una gran posibilidad de que la pobreza incremente. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####¿Cual es la relación entre la Etnicidad y los ingresos de las personas?

# COMMAND ----------

resultado2 = spark.sql("""
    SELECT Ethnicity, AVG(NYCgov_Income) AS IngresoPromedio
    FROM pobreza
    GROUP BY Ethnicity
""")



# COMMAND ----------

resultado2.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  

df = resultado2.toPandas()

nombres_etnias = ["Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian ", "Hispanic, Any Race", "Other Race/Ethnic Group"]

df["Ethnicity"] = df["Ethnicity"].map(lambda x: nombres_etnias[x - 1])

df = df.sort_values("IngresoPromedio", ascending=False)

plt.figure(figsize=(10, 6))
ax = df.plot(x="Ethnicity",y="IngresoPromedio", kind="bar", figsize=(12, 6))
plt.xticks(rotation=45)  # Rotar etiquetas del eje x para mayor legibilidad

plt.title('Relación entre la etnicidad y los Ingresos')
plt.xlabel('Etnicidad')
plt.ylabel('Ingreso Promedio')

for index, value in enumerate(df["IngresoPromedio"]):
    ax.text(index, value, f"{value:.2f}", ha="center", va="bottom")

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC En este caso, vemos que las personas pertenecientes a _**Non-Hispanic White**_ , son las que reciben mejores ingresos, a comparación de los otros grupos etnicos. De segundas le sigue la variedad de grupos etnicos diferentes a los considerados. En tercer y cuarto lugar estan los **Asiaticos no hispanicos** y los **Negros no hispanicos**. Por último, los que menos ingresos reciben son los **Hispanicos**. 
# MAGIC
# MAGIC Aca vemos una relación marcada entre las razas y etnias junto con los ingresos recibidos por las personas. En Nueva York **hay una marcada discriminación salarial hacia los Hispanicos**, mientras que los demas grupos tienden a recibir mejores salarios. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####¿Hay alguna relacion entre el _borough_ y el porcentaje de personas graduadas con los requisitos de fundamentales de salud en secundaria?

# COMMAND ----------

temp_table_name = "educacion"
df_education.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

resultado3 = spark.sql("""SELECT Boro, avg(proporcion_cumplen_requisitos_salud) as prom_requisitos_cumplidos
    FROM (SELECT Boro,
       `# of 16-17 June and August graduates meeting high school health requirements`/`# of 16-17 June and August graduates` AS proporcion_cumplen_requisitos_salud
        FROM educacion
        WHERE `# of 16-17 June and August graduates` > 0) intermedia
    GROUP BY Boro""")

# COMMAND ----------

display(resultado3)

# COMMAND ----------

df = resultado3.toPandas()

df = df.sort_values("prom_requisitos_cumplidos", ascending=False)

plt.figure(figsize=(10, 6))
ax = df.plot(x="Boro",y="prom_requisitos_cumplidos", kind="bar", figsize=(12, 6))
plt.xticks(rotation=45)  # Rotar etiquetas del eje x para mayor legibilidad

plt.title('Promedio de estudiantes que cumplen los requisitos de salud en secundaria segun el borough')
plt.xlabel('Boro')
plt.ylabel('Promedio de estudiantes')

for index, value in enumerate(df["prom_requisitos_cumplidos"]):
    ax.text(index, value, f"{value:.4f}", ha="center", va="bottom")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Se puede observar con claridad que, casi el 100% de estudiantes que se graduaron cumplen con los requisitos de salud en secundaria, pues en la mayoría de _boroughs_ el promedio esta muy cercano al 100%. Sin embargo, el _borough_ que menor proporción de personas que cumplen el requisito es **Bronx**, pues cada 2 personas de 100, **NO** cumplen aquel requisito. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####¿Existe alguna relación entre el distrito del colegio y la proporción de estudiantes graduados de este en los años 2016-17?

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

# Agrupa por el distrito del colegio y calcula la cantidad promedio de estudiantes graduados
district_avg_graduation = df_education.groupBy('Boro').agg(avg('# of 16-17 June and August graduates').alias('AvgGraduatesCount'))

# Muestra la cantidad promedio de estudiantes graduados por distrito
district_avg_graduation.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC >+ Los resultados obtenidos muestran un menor numero de graduados en los distritos de: Queens, y Staten Island.
# MAGIC >+ Los distritos con menor numero de gruadados son: Brooklyn, Manhattan, y Bronx.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Se propone una ANOVA que responda a las siguientes hipotesis**
# MAGIC
# MAGIC >+ H0: No hay diferencias significativas en la proporción de estudiantes graduados entre los diferentes distritos 'Boro'.
# MAGIC >+ H1: Existen diferencias significativas en la proporción de estudiantes graduados entre al menos dos de los grupos de distritos 'Boro'.
# MAGIC

# COMMAND ----------

from scipy.stats import f_oneway
import numpy as np

# Selecciona los datos del DataFrame como un pandas DataFrame
df_pandas = df_education.select('Boro', '# of 16-17 June and August graduates').toPandas()

# Agrupa los datos por 'Boro' y crea un diccionario de listas
groups = {}
for name, group in df_pandas.groupby('Boro'):
    groups[name] = group['# of 16-17 June and August graduates']

# Realiza el ANOVA
f_statistic, p_value = f_oneway(*groups.values())

# Imprime el resultado
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observaciones de los resultados**
# MAGIC >+ Al obtener un P-valor tan bajo (menor a 0.05), podemos concluir que la hipotesis nula (H0) se rechaza, mostrando que si hay una diferencia significativa entre las proporciones de estudiantes graduados para cada distrito.
# MAGIC
# MAGIC > **Ambos resultados indican una diferencia significativa para la proporción de estudiantes graduados en el periodo determinado.**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### ¿Cuáles son las características predominantes de los accidentes de tráfico registrados en cada año? (En función de factores como el punto de impacto, el tipo de vehículo más involucrado y la causa previa al accidente)

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH accidentes_por_ano AS (
# MAGIC     SELECT YEAR(CRASH_DATE) AS crash_year
# MAGIC     FROM colisiones
# MAGIC ),
# MAGIC
# MAGIC puntos_impacto_por_ano AS (
# MAGIC     SELECT
# MAGIC         YEAR(CRASH_DATE) AS crash_year,
# MAGIC         POINT_OF_IMPACT,
# MAGIC         COUNT(*) AS count
# MAGIC     FROM colisiones
# MAGIC     GROUP BY YEAR(CRASH_DATE), POINT_OF_IMPACT
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     a.crash_year,
# MAGIC     COUNT(*) AS Accidentes_por_year,
# MAGIC     p.POINT_OF_IMPACT AS Punto_de_Impacto_Mas_Comun, 
# MAGIC     p.count as Reincidencia_impacto
# MAGIC FROM accidentes_por_ano a
# MAGIC LEFT JOIN (
# MAGIC     SELECT pi.*, 
# MAGIC            ROW_NUMBER() OVER (PARTITION BY pi.crash_year ORDER BY pi.count DESC) AS rn
# MAGIC     FROM puntos_impacto_por_ano pi
# MAGIC ) p ON a.crash_year = p.crash_year AND p.rn = 1
# MAGIC GROUP BY a.crash_year, p.POINT_OF_IMPACT, p.count
# MAGIC ORDER BY a.crash_year;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Causa previa al accidente más comun por año
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH accidentes_por_ano AS (
# MAGIC     SELECT YEAR(CRASH_DATE) AS crash_year
# MAGIC     FROM colisiones
# MAGIC ),
# MAGIC
# MAGIC causa_previa_mas_comun_por_ano AS (
# MAGIC     SELECT
# MAGIC         YEAR(CRASH_DATE) AS crash_year,
# MAGIC         PRE_CRASH,
# MAGIC         COUNT(*) AS count
# MAGIC     FROM colisiones
# MAGIC     GROUP BY YEAR(CRASH_DATE), PRE_CRASH
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     a.crash_year,
# MAGIC     COUNT(*) AS Accidentes_por_year,
# MAGIC     c.PRE_CRASH AS Causa_Previa_Mas_Comun, 
# MAGIC     c.count as Reincidencia_causa_previa
# MAGIC FROM accidentes_por_ano a
# MAGIC LEFT JOIN (
# MAGIC     SELECT cp.*, 
# MAGIC            ROW_NUMBER() OVER (PARTITION BY cp.crash_year ORDER BY cp.count DESC) AS rn
# MAGIC     FROM causa_previa_mas_comun_por_ano cp
# MAGIC ) c ON a.crash_year = c.crash_year AND c.rn = 1
# MAGIC GROUP BY a.crash_year, c.PRE_CRASH, c.count
# MAGIC ORDER BY a.crash_year;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Accidentes por cantidad de vehiculos involucrados por año

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH accidentes_por_ano AS (
# MAGIC     SELECT YEAR(CRASH_DATE) AS crash_year
# MAGIC     FROM colisiones
# MAGIC ),
# MAGIC
# MAGIC vehiculo_mas_involucrado_por_ano AS (
# MAGIC     SELECT
# MAGIC         YEAR(CRASH_DATE) AS crash_year,
# MAGIC         Vehicle_type,
# MAGIC         COUNT(*) AS count
# MAGIC     FROM colisiones
# MAGIC     GROUP BY YEAR(CRASH_DATE), Vehicle_type
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     a.crash_year,
# MAGIC     COUNT(*) AS Accidentes_por_year,
# MAGIC     v.Vehicle_type AS Vehiculo_Mas_Involucrado, 
# MAGIC     v.count as Reincidencia_vehiculo
# MAGIC FROM accidentes_por_ano a
# MAGIC LEFT JOIN (
# MAGIC     SELECT vi.*, 
# MAGIC            ROW_NUMBER() OVER (PARTITION BY vi.crash_year ORDER BY vi.count DESC) AS rn
# MAGIC     FROM vehiculo_mas_involucrado_por_ano vi
# MAGIC ) v ON a.crash_year = v.crash_year AND v.rn = 1
# MAGIC GROUP BY a.crash_year, v.Vehicle_type, v.count
# MAGIC ORDER BY a.crash_year;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Observaciones:
# MAGIC
# MAGIC Se encuentra que en este proceso desde el año 2012 al 2021, el punto de impacto más recurrente fue el "Center Front End". Esta siendo una gran parte de los puntos de impactos en todos los accidentes.
# MAGIC
# MAGIC El tipo de vehículo encontrado de forma más recurrente en los accidentes fueron los Sedan, esto siendo recurrente entre los años 2012 y 2021
# MAGIC
# MAGIC Y marcando una gran totalidad de la acción previa al accidente más recurrente fue "going straight ahead", siendo este todos los años el valor dominante.
# MAGIC
# MAGIC Ahora interpretando estas 3 características con una relación : es posible que los vehículos "Sedan" están presentando un problema en sus frenos al ser registrados los daños en la parte delantera del capo, y que a su vez estos se encontraban yendo hacía al frente

# COMMAND ----------

# MAGIC %md
# MAGIC ##¿Cuál es el promedio de personas involucradas en los accidentes por año?

# COMMAND ----------

# MAGIC %md
# MAGIC Accidentes por cantidad de vehiculos involucrados por año

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH year_data AS (
# MAGIC     SELECT YEAR(CRASH_DATE) AS crash_year, VEHICLE_OCCUPANTS
# MAGIC     FROM colisiones
# MAGIC )
# MAGIC SELECT crash_year, AVG(VEHICLE_OCCUPANTS) AS average_occupants
# MAGIC FROM year_data
# MAGIC GROUP BY crash_year
# MAGIC ORDER BY crash_year DESC;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Observaciones**
# MAGIC - Los años 2015 a 2018 muestran una relativa estabilidad en el promedio de pasajeros involucrados en accidentes, con valores cercanos a 1.3 en promedio, lo que demuestra que tiende a ser más usual los accidentes donde solo va el conductor.
# MAGIC El año 2021 destaca como un año anómalo en términos del promedio de pasajeros involucrados. Podría ser útil investigar por qué se registra un valor tan alto en ese año. Puede ser un error en los datos o podría indicar eventos excepcionales.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Aplicación de ML

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Preprocesamiento del primer dataset

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

matriz_correlacion = df_arrestos.toPandas().corr()

plt.figure(figsize=(30, 20))

sns.heatmap(matriz_correlacion, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Matriz de Correlación")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observaciones de la matriz de correlación**
# MAGIC >+ Aunque las variables "Longitude" y "Latitude" son las que presentan una correlación más fuerte, no se eliminarán debido a que serán útilizadas en el proceso de clusterización no supervisado
# MAGIC >+ Las variables que le siguen a las mencionadas anteriormente en cuestión de correlación, son "X_COORD_CD" y "Y_COORD_CD" que son un formato alternativo a la longitud y latitud, no obstante, al no sobrepasar el limite establecido para considireralos una relación fuerte (>0.70) no se eliminarán.

# COMMAND ----------

# MAGIC %md
# MAGIC **Se decide no realizar la normalización de los datos, dado que al trabajar con coordenadas y con registros que no presentan medidas continuas representativas, normalizar solo causaría disrupciones en el alcance al objetivo planteado.**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Variables escogidas para el dataset de arrestos**
# MAGIC >Se escogen las variables:
# MAGIC >+ PERP_RACE
# MAGIC >+ NIVEL_DEL_DELITO  
# MAGIC >+ Longitude
# MAGIC >+ Latitude
# MAGIC >+ X_COORD_CD
# MAGIC >+ Y_COORD_CD

# COMMAND ----------

# MAGIC %md
# MAGIC ####¿Existe una relación significativa entre la probabilidad de que un registro cuente con un tipo de infracción grave y la pertenencia del perpetrador a un grupo sociodemográfico minoritario (Black o White Hispanic, American Indian/Alaskan Native, Asian/Pacific Islander, Black)? Vale aclarar que la razón de selección de estas agrupaciones está guiado por los resultados expedidos por el foro "statista" y pueden ser corroborados en el siguiente foro: "https://es.statista.com/estadisticas/600570/porcentaje-de-poblacion-de-estados-unidos--2060-por-raza-y-origen-hispano/ "

# COMMAND ----------

#1. Se importan las librerias necesarias para realizar un modelo de regresión lógistica sobre los datos
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

#Se realiza la categorización correspondiente a los registros con nivel de infracción grave en la nueva columna "Categoria_infracción"
nombre_grupos = ["F"]
df_arrestos = df_arrestos.withColumn("Categoría_infracción", when(df_arrestos["NIVEL_DEL_DELITO"].isin(nombre_grupos), 1).otherwise(0))

# COMMAND ----------

valores_unicos = df_arrestos.select("Categoría_infracción").distinct()
valores_unicos.show()

# COMMAND ----------

#Se muestran los resultados de las probabilidades de escoger cada categoria para verificar una desproporción inicial.
frecuencias = df_arrestos.groupBy("Categoría_infracción").agg(count("*").alias("Cuenta"))

# Calcula la probabilidad dividiendo las frecuencias por el total de filas
total_filas = df_arrestos.count()
frecuencias = frecuencias.withColumn("Probabilidad_infracción", frecuencias["Cuenta"] / total_filas)

# Muestra la probabilidad de cada categoría
frecuencias.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observaciones**
# MAGIC >+ Se puede ver que las categorías de infracciones cuentan con desbalance despreciable en vista de la aplicación del modelo logistico, por lo que nuestra variable objetivo no va a sesgar los resultados del modelo por ese desbalance.

# COMMAND ----------

#Se procede a hacer la correspondiente categorización de los valores en la columna (PERP_RACE)
nombre_grupos = ["BLACK HISPANIC","WHITE HISPANIC", "AMERICAN INDIAN/ALASKAN NATIVE", "ASIAN / PACIFIC ISLANDER", "BLACK"]
df_arrestos = df_arrestos.withColumn("Categoría", when(df_arrestos["PERP_RACE"].isin(nombre_grupos), 1).otherwise(0))

# COMMAND ----------

#Se muestran los resultados de las probabilidades de escoger cada categoria para verificar una desproporción inicial.
frecuencias = df_arrestos.groupBy("Categoría").agg(count("*").alias("Cuenta"))

# Calcula la probabilidad dividiendo las frecuencias por el total de filas
total_filas = df_arrestos.count()
frecuencias = frecuencias.withColumn("Probabilidad", frecuencias["Cuenta"] / total_filas)

# Muestra la probabilidad de cada categoría
frecuencias.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observaciones**
# MAGIC >+ Se puede notar un desbalance significativo entre las probabilidades de escoger una categoría sobre la otra, siendo así que la mayoría pertenece a los grupos minoritarios, esto podría llegar a sesgar el modelo logístico una vez aplicado.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

# Combinar las dos columnas en un solo vector de características
feature_columns = ["Categoría"]
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_caracterizado = vector_assembler.transform(df_arrestos)

# Crear un modelo de regresión logística
lr = LogisticRegression(featuresCol="features", labelCol="Categoría_infracción")

# Ajustar el modelo a los datos
model = lr.fit(df_caracterizado)

# se realizan predicciones con el modelo
predictions = model.transform(df_caracterizado)

# Ver los resultados de las predicciones
predictions.select("probability").show()

# COMMAND ----------

from pyspark.sql.functions import col

# Filtrar las predicciones donde "Y" es 1 y "X" es 1
predicciones = predictions.filter((col("Categoría_infracción") == 1) & (col("Categoría") == 1))

# Mostrar la probabilidad correspondiente
predicciones.select("probability").show()

# COMMAND ----------

##Ajuste de segunda prueba, ajuste con datos estandarizados
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

# Combinar las dos columnas en un solo vector de características
feature_columns = ["Categoría"]
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features",)
df_caracterizado = vector_assembler.transform(df_arrestos)

# Crear un modelo de regresión logística
lr = LogisticRegression(featuresCol="features", labelCol="Categoría_infracción",standardization=True)

# Ajustar el modelo a los datos
model = lr.fit(df_caracterizado)

# Crear un evaluador de clasificación binaria
evaluador1 = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='Categoría_infracción', metricName='areaUnderROC')

# se realizan predicciones con el modelo
predictions = model.transform(df_caracterizado)

# Calcular el área bajo la curva ROC en el conjunto de prueba
area_roc = evaluador1.evaluate(predictions)

# Mostrar el área bajo la curva ROC
print("Área bajo la curva ROC en conjunto de prueba:", area_roc)

# Ver los resultados de las predicciones
predictions.select("probability").show()

# COMMAND ----------

##Ajuste de tercera prueba, ajuste con datos estandarizados y dividiendo los datos en entrenamiento y prueba
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Combinar las columnas en un solo vector de características
columnas_caracteristicas = ["Categoría"]
ensamblador = VectorAssembler(inputCols=columnas_caracteristicas, outputCol="features")
df_caracterizado = ensamblador.transform(df_arrestos)

# Dividir los datos en conjuntos de entrenamiento y prueba
df_entrenamiento, df_prueba = df_caracterizado.randomSplit([0.8, 0.2], seed=123)

# Crear un modelo de regresión logística
lr = LogisticRegression(featuresCol="features", labelCol="Categoría_infracción", standardization=True)

# Ajustar el modelo a los datos de entrenamiento
modelo_entrenado = lr.fit(df_entrenamiento)

# Realizar predicciones en el conjunto de prueba
predicciones1 = modelo_entrenado.transform(df_prueba)

# Crear un evaluador de clasificación binaria
evaluador = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='Categoría_infracción', metricName='areaUnderROC')

# Calcular el área bajo la curva ROC en el conjunto de prueba
area_roc = evaluador.evaluate(predicciones1)

# Mostrar el área bajo la curva ROC
print("Área bajo la curva ROC en conjunto de prueba:", area_roc)

# Ver los resultados de las predicciones
predicciones1.select("probability").show()

# COMMAND ----------

from pyspark.sql.functions import col

# Filtrar las predicciones donde "Y" es 1 y "X" es 1
predicciones2 = predicciones1.filter((col("Categoría_infracción") == 1) & (col("Categoría") == 1))

# Mostrar la probabilidad correspondiente
predicciones2.select("probability").show()

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Definir el esquema para los resultados
esquema = StructType([
    StructField("Prueba", StringType(), True),
    StructField("Probabilidad de que sea minoría y haya cometido infracción", FloatType(), True),
    StructField("Probabilidad de que no sea minoría pero que haya cometido infracción", FloatType(), True),
    StructField("Medida de aleatoreidad de los resultados", FloatType(), True),
])

# Crear un DataFrame con los resultados
datos_resultados = [
    ("Datos sin estandarizar", 0.561, 0.611, 0.510),
    ("Datos estandarizados", 0.561, 0.611, 0.510),
    ("Datos divididos en entrenamiento-prueba y estandarizados", 0.562, 0.610, 0.511),
]

df_resultados = spark.createDataFrame(datos_resultados, schema=esquema)

# Mostrar el DataFrame
df_resultados.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observaciones de los resultados y respuesta a la pregunta**
# MAGIC >+ Los resultados del primer y segundo modelo aplicado muestran que la probabilidad de que una persona pertenezca a una minoría y que cuente con un registro delictivo "grave", es del: 56,1%
# MAGIC >+ Los resultados del primer y segundo modelo aplicado muestran que la probabilidad de que una persona pertenezca a una mayoria y que cuente con un registro delictivo "grave", es del: 61,1%
# MAGIC >+ Los resultados del tercer modelo muestran que la probabilidad de ser minoría y cometer un delito grave es del 56.2% y para el caso complemento es del 61%
# MAGIC >+ De los resultados se puede constatar que aunque haya un desbalance en la probabilidad de tomar un grupo minoritario que un grupo mayoritario, es mas probable tomar un registro con una infracción grave y una persona "blanca". 
# MAGIC >+ También se ve una mejora minima a la hora de realizar la comparación con respecto al tercer modelo, de cierta forma se logra mejorar la inclinación dl modelo por el caso favorable para los propositos de la investigación.
# MAGIC >+ La medida de aleatoreidad obtenida nos permite identificar que tan determinista es el modelo, con los resultados obtenidos "0.51" y "0.511" se ve que los datos presentan un caracter más aleatorio a la hora de definir el caso favorable (1 y 1 en este caso).
# MAGIC >+ Finalmente. Se constata que si hay una relación entre la raza de una persona y la pertenencia del registro a un nivel delictivo grave, no obstante se ve más inclinado hacía las personas blancas que a los otros grupos raciales.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####Aplicación modelo ML no supervisado

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Seleccionar las características que se utilizarán para K-Means
feature_columns = ['Latitude', 'Longitude']

# Crear un ensamblador para combinar las características en un solo vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Se ensambla nuestro vector objetivo
data = assembler.transform(df_arrestos)

# Dividir los datos en datos de entrenamiento y prueba
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1)

# Crear un modelo K-Means
num_clusters = 5  # Número de clusters deseado (Se toman 5 dada la cantidad de distritos existentes en Nueva York)
kmeans = KMeans().setK(num_clusters).setSeed(1)

# Se entrena el modelo K-Means en los datos de entrenamiento
model = kmeans.fit(train_data)

# Obtener las etiquetas de cluster asignadas a cada muestra en los datos de prueba
test_results = model.transform(test_data)

# Se crea un evaluador para medir la precisión en los clusteres
evaluator = ClusteringEvaluator()

# Calcular la métrica Silhouette en los datos de prueba
silhouette = evaluator.evaluate(test_results)
print("Silhouette en datos de prueba:", silhouette)

# Visualizar los resultados en los datos de prueba
test_results.select("Latitude", "Longitude", "prediction").show()





# COMMAND ----------

# Extraer las coordenadas de Latitude y Longitude del DataFrame de resultados
latitude = test_results.select("Latitude").rdd.flatMap(lambda x: x).collect()
longitude = test_results.select("Longitude").rdd.flatMap(lambda x: x).collect()
prediction = test_results.select("prediction").rdd.flatMap(lambda x: x).collect()

# Definir una paleta de colores para los clusters
colors = ['b', 'g', 'r', 'c', 'm'] 

# Crear una figura para el gráfico
plt.figure(figsize=(10, 8))

# Colorear los puntos según las etiquetas de cluster
for cluster_label in range(5):  
    cluster_data = [(latitude[i], longitude[i]) for i in range(len(prediction)) if prediction[i] == cluster_label]
    # Comprobar si hay datos en el cluster
    if len(cluster_data) > 0:
        cluster_latitude, cluster_longitude = zip(*cluster_data)
        plt.scatter(cluster_longitude, cluster_latitude, label=f'Cluster {cluster_label}', color=colors[cluster_label])

# Agregar etiquetas y leyenda
plt.xlabel('Coordenada en X')
plt.ylabel('Coordenada en Y')
plt.title('Gráfico de Dispersión de Clusters Geográficos')
plt.legend()

# Mostrar el gráfico
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observaciones**
# MAGIC >+ El modelo resultante presenta un coeficiente silhouette de 0.642 lo que significa que la clusterización tiene un buen ajuste a la hora de predecir la forma que deben tener las categorías 
# MAGIC >+ El scatterplot nos permite tener una mejor idea de la forma que tienen los datos, como es posible notar: La forma del grafico es similar a la de nueva york, categorizada según distritos.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Seleccionar las características que se utilizarán para K-Means
columnas_caracteristicas = ['Y_COORD_CD', 'X_COORD_CD']

# Crear un ensamblador para combinar las características en un solo vector
ensamblador = VectorAssembler(inputCols=columnas_caracteristicas, outputCol="features")

# Se ensambla nuestro vector objetivo
datos = ensamblador.transform(df_arrestos)

# se dividen los datos en datos de entrenamiento y prueba
datos_entrenamiento, datos_prueba = datos.randomSplit([0.8, 0.2], seed=1)

# Crear un modelo K-Means
numero_clusters = 5  # Número de clusters deseado (Se toman 5 dada la cantidad de distritos existentes en nueva york)
modelo_kmeans = KMeans().setK(numero_clusters).setSeed(1)

# Se entrena el modelo K-Means en los datos de entrenamiento
modelo_entrenado = modelo_kmeans.fit(datos_entrenamiento)

# Obtener las etiquetas de cluster asignadas a cada muestra en los datos de prueba
resultados = modelo_entrenado.transform(datos_prueba)

# Se crea un evaluador para medir la precisión en los clusteres
evaluador = ClusteringEvaluator()

# Calcular la métrica Silhouette en los datos de prueba
silhouette = evaluador.evaluate(resultados)
print("Silhouette en datos de prueba:", silhouette)
print("Numero de iteraciones realizadas:", modelo_kmeans.getMaxIter())
# Visualizar los resultados en los datos de prueba
resultados.select("Y_COORD_CD", "X_COORD_CD", "prediction").show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Extraer las coordenadas de Latitude y Longitude del DataFrame de resultados
latitude = resultados.select("Y_COORD_CD").rdd.flatMap(lambda x: x).collect()
longitude = resultados.select("X_COORD_CD").rdd.flatMap(lambda x: x).collect()
prediction = resultados.select("prediction").rdd.flatMap(lambda x: x).collect()

# Definir una paleta de colores para los clusters
colors = ['b', 'g', 'r', 'c', 'm'] 

# Crear una figura para el gráfico
plt.figure(figsize=(10, 8))

# Colorear los puntos según las etiquetas de cluster
for cluster_label in range(5):  
    cluster_data = [(latitude[i], longitude[i]) for i in range(len(prediction)) if prediction[i] == cluster_label]
    # Comprobar si hay datos en el cluster
    if len(cluster_data) > 0:
        cluster_latitude, cluster_longitude = zip(*cluster_data)
        plt.scatter(cluster_longitude, cluster_latitude, label=f'Cluster {cluster_label}', color=colors[cluster_label])

# Agregar etiquetas y leyenda
plt.xlabel('Coordenada en X')
plt.ylabel('Coordenada en Y')
plt.title('Gráfico de Dispersión de Clusters Geográficos')
plt.legend()

# Mostrar el gráfico
plt.show()

# COMMAND ----------

#Modificación de parametros para la función k-means
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Seleccionar las características que se utilizarán para K-Means
columnas_caracteristicas = ['Y_COORD_CD', 'X_COORD_CD']

# Crear un ensamblador para combinar las características en un solo vector
ensamblador = VectorAssembler(inputCols=columnas_caracteristicas, outputCol="features")

# Se ensambla nuestro vector objetivo
datos = ensamblador.transform(df_arrestos)

# se dividen los datos en datos de entrenamiento y prueba
datos_entrenamiento, datos_prueba = datos.randomSplit([0.8, 0.2], seed=1)
datos_entrenamiento = datos_entrenamiento.filter(col("features").isNotNull())
datos_prueba = datos_prueba.filter(col("features").isNotNull())
# Crear un modelo K-Means
numero_clusters = 7  # Número de clusters deseado Se modifica el valor a 7 para verificar como mejora o empeora la respuesta frente a una mayor segmentación en los datos
modelo_kmeans1 = KMeans().setK(numero_clusters).setSeed(1).setMaxIter(50).setDistanceMeasure("euclidean") #Se realiza una modificación en el numero de iteraciones realizadas en el modelo (De 20 a 40), para verificar si hay una mayor precisión en el ajuste del mapa 

# Se entrena el modelo K-Means en los datos de entrenamiento
modelo_entrenado = modelo_kmeans1.fit(datos_entrenamiento)

# Obtener las etiquetas de cluster asignadas a cada muestra en los datos de prueba
resultados1 = modelo_entrenado.transform(datos_prueba)

# Se crea un evaluador para medir la precisión en los clusteres
evaluador = ClusteringEvaluator()

# Calcular la métrica Silhouette en los datos de prueba
silhouette = evaluador.evaluate(resultados1)
print("Silhouette en datos de prueba:", silhouette)
print("Numero de iteraciones realizadas:", modelo_kmeans.getMaxIter())

# Visualizar los resultados en los datos de prueba
resultados1.select("Y_COORD_CD", "X_COORD_CD", "prediction").show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Extraer las coordenadas de Latitude y Longitude del DataFrame de resultados
latitude = resultados1.select("Y_COORD_CD").rdd.flatMap(lambda x: x).collect()
longitude = resultados1.select("X_COORD_CD").rdd.flatMap(lambda x: x).collect()
prediction = resultados1.select("prediction").rdd.flatMap(lambda x: x).collect()

# Definir una paleta de colores para los clusters
colors = ['b', 'g', 'r', 'c', 'm'] 

# Crear una figura para el gráfico
plt.figure(figsize=(10, 8))

# Colorear los puntos según las etiquetas de cluster
for cluster_label in range(5):  
    cluster_data = [(latitude[i], longitude[i]) for i in range(len(prediction)) if prediction[i] == cluster_label]
    # Comprobar si hay datos en el cluster
    if len(cluster_data) > 0:
        cluster_latitude, cluster_longitude = zip(*cluster_data)
        plt.scatter(cluster_longitude, cluster_latitude, label=f'Cluster {cluster_label}', color=colors[cluster_label])

# Agregar etiquetas y leyenda
plt.xlabel('Coordenada en X')
plt.ylabel('Coordenada en Y')
plt.title('Gráfico de Dispersión de Clusters Geográficos')
plt.legend()

# Mostrar el gráfico
plt.show()

# COMMAND ----------

#Proporción de delitos cometidos por los distritos predichos:
#1. Creación de una tabla temporal
resultados.createOrReplaceTempView("tabla_temp")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Definir el esquema para los resultados
esquema = StructType([
    StructField("Variables utilizadas", StringType(), True),
    StructField("# de clusters especificados", IntegerType(), True),
    StructField("# de clusters creados", IntegerType(), True),
    StructField("Silhoutte", FloatType(), True),
    StructField("# de iteraciones", IntegerType(), True),
])

# Crear un DataFrame con los resultados
datos_resultados = [
    ("Longitude-Latitude", 5, 4, 0.6426, 20),
    ("X_COOD_CD-Y_COORD_CD", 5, 5, 0.6305, 20),
    ("X_COOD_CD-Y_COORD_CD", 7, 5, 0.6462, 40),
]

df_resultados = spark.createDataFrame(datos_resultados, schema=esquema)

# Mostrar el DataFrame
df_resultados.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observaciones**
# MAGIC >+ Es posible notar que el numero de clusters optimo para la localización utilizada en las columnas de latitud y longitud, no es 5 sino 4, este detalle puede sesgar de forma significativa los resultados basados en las categorías creadas, por lo que para las dos proximas pruebas de k-means, se decidió utilizar el formato de ubicación alternativo.
# MAGIC >+ El segundo modelo presenta una mejora circunstancial frente al primero, al definir el mismo numero de clusteres y obtener una mayor precisión en la forma final que obtiene el mapa.
# MAGIC >+ Al agregar una mayor cantidad de clusteres a crear, se puede ver que el numero optimo de clusters que el modelo decide utilizar es 5 y no el especificado, el aumento en el puntaje de precisión (silhoutte) puede deberse a que se aumento de forma proporcional el numero de iteraciones maximas que puede realizar el modelo, por lo que tiene un mayor margen de precisión para el k especificado.
# MAGIC >+ Es importante notar que aunque el ultimo modelo prediga correctamente el numero optimo de clusteres con un mejor ajuste, este pierde asímismo la legibilidad en los resultados dado que se aleja significativamente de la forma que tiene originalmente el mapa de nueva york.
# MAGIC >+ Se constata que el segundo modelo es el más optimo para la realización de pruebas con las categorías obtenidas dada su alta precisión, y su capacidad de legibilidad en los resultados de la clusterización.

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT prediction, count / total_arrestos AS proporcion_arrestos
# MAGIC FROM (
# MAGIC     SELECT prediction, count(1) as count
# MAGIC     FROM tabla_temp
# MAGIC     GROUP BY prediction
# MAGIC ) AS subquery
# MAGIC CROSS JOIN (
# MAGIC     SELECT COUNT(1) AS total_arrestos
# MAGIC     FROM tabla_temp
# MAGIC ) AS total_counts
# MAGIC ORDER BY proporcion_arrestos DESC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observaciones de criminalidad por distrito**
# MAGIC >+ Es posible observar que de acuerdo al modelo escogido (modelo 2), la zona del bronx es la que cuenta con una mayor proporción de registros sobre los datos obtenidos, por lo que de una u otra forma es uno de los puntos calientes más importantes sobre los que se debe actuar.
# MAGIC >+ El distrito con menor propoción de criminalidad es staten island (categoría 0), con una proporción del "0.0939" sobre los datos totales.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####Aplicación Modelo supervisado

# COMMAND ----------

# MAGIC %md
# MAGIC **Justificación del modelo**
# MAGIC
# MAGIC En este caso, nuestra variable de interes es si una persona vive en la pobreza o no (_1: pobreza, 2: no pobreza_). De ahí concluimos que debiamos buscar un algoritmo de casificación (o sea, un algoritmo **supervisado**). De forma especifica tenemos que el KNN nos puede ayudar a predecir si una nueva observación (una persona en este caso) se encuentra en la categoría de pobreza o no, en función de las variables explicativas disponibles del dataset. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Se realizarán 3 modelos entorno al **K means**. 
# MAGIC 1. **Con los datos limpios del dataset de pobreza**. Este no contendra ninguna estandarización ni normalización. Se espera que su rendimiento sea bajo con respecto a los proximos modelos 
# MAGIC 2. **Con los datos estandarizados y un k = 5**. Este tendrá la estandarización de algunos datos, dependiendo de su rango. Ademas la cantidad de vecinos a analizar serán 5. Se espera que su rendimiento sea mejor con respecto al primer modelo. 
# MAGIC 3. **Con los datos del mejor modelo de los dos anteriores y un k=3**. Este tendrá la estandarización de algunos datos, dependiendo de su rango. Ademas la cantidad de vecinos a analizar serán 3. Se espera que su rendimiento sea mejor con respecto al primer modelo. 
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC En primer lugar, se eliminaran aquellos valores que son únicos para cada registro u hogar (_NumSerieHogar_), pues no tiene sentido realizar el analisis del modelo sobre esta variable. De igual forma, se eliminarán las columnas que no tienen un significado para nosotros y nuestro analisis. 
# MAGIC

# COMMAND ----------

df_pobreza = df_pobreza.drop("NumSerieHogar")

display(df_pobreza)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC La decisión de eliminar las siguientes columnas es, en general, porque no aportan mucho a nuestro analisis o sentimos que es una consecuencia (y no una causa) de la pobreza. Ademas, algunas de ellas son, o registros bastante particulares, o son variables que no dependen únicamente de las personas. 

# COMMAND ----------

columnas_a_eliminar = ["CitizenStatus", "EST_HousingStatus","EST_Childcare","EST_Commuting","EST_FICAtax","EST_HEAP","EST_Housing","EST_IncomeTax","EST_PovGap","EST_PovGapIndex","Povunit_ID","EST_Nutrition","SSIP_adj","NumOrdenPersonas","RETP_adj","RETP_adj","RELP","RelacionPrincipal"]

df_pobreza = df_pobreza.select([column for column in df_pobreza.columns if column not in columnas_a_eliminar])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Ahora realizaremos la matriz de correlación para poder observar cuales variables estn mejor relacionadas con cuales.
# MAGIC

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

matriz_correlacion = df_pobreza.toPandas().corr()

plt.figure(figsize=(30, 20))

sns.heatmap(matriz_correlacion, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Matriz de Correlación")

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC El propósito de eliminar variables altamente correlacionadas, con una correlación mayor a 0.8, radica en la simplificación y puede mejorar la calidad del modelo. Cuando dos o más variables están altamente correlacionadas, llevan información redundante al modelo, lo que puede dar lugar a resultados sesgados y menos interpretables. Al eliminar estas características, reduce la complejidad del modelo. Además, la eliminación de características correlacionadas a menudo conduce a un modelo más estable y robusto, lo que es esencial para la generalización efectiva a nuevos datos y para evitar el sobreajuste.

# COMMAND ----------

import numpy as np

umbral_correlacion = 0.8

n = matriz_correlacion.shape[1]
eliminar = np.zeros((n, n))

for i in range(n):
    for j in range(i+1, n):
        if abs(matriz_correlacion.iloc[i, j]) > umbral_correlacion:
            eliminar[i, j] = 1

columnas_a_eliminar = []
for i in range(n):
    for j in range(i+1, n):
        if eliminar[i, j] == 1:
            columnas_a_eliminar.append(matriz_correlacion.columns[i])

df_pobreza = df_pobreza.select([column for column in df_pobreza.columns if column not in columnas_a_eliminar])

# COMMAND ----------

df_pobreza.columns

# COMMAND ----------


matriz_correlacion = df_pobreza.toPandas().corr()

plt.figure(figsize=(30, 20))

sns.heatmap(matriz_correlacion, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Matriz de Correlación")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observaciones**
# MAGIC
# MAGIC Nuestra variable de interes es _NYCgov_Pov_Stat_, que nos indica el estado de pobreza de la persona {1: en pobreza, 2: no en pobreza}.
# MAGIC
# MAGIC Si observamos la matriz de correlación, esta variable tiene variables muy cercanas a cero, lo que a primera vista nos indicaria una relación baja. Sin embargo, hay que recordar que la correlación mide la relación **lineal** entre dos variables. Esto nos lleva a seguir con la misma variable de interes (_NYCgov_Pov_Stat_), suponiendo que, con las otras variables, tiene otro tipo de relación. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #####***Primer modelo***

# COMMAND ----------

c = df_pobreza.columns
c.remove("NYCgov_Pov_Stat")

# COMMAND ----------

X = df_pobreza[c]
Y = df_pobreza.select(col("NYCgov_Pov_Stat"))


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Define las características y las etiquetas
assembler = VectorAssembler(inputCols=c, outputCol="features")
data = assembler.transform(df_pobreza)

data = data.withColumn("", F.col("NYCgov_Pov_Stat"))

# Divide los datos en conjuntos de entrenamiento y prueba
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

X_train = train_data[c]
X_test = test_data[c]

Y_train = train_data.select(col("NYCgov_Pov_Stat"))
Y_test = test_data.select(col("NYCgov_Pov_Stat"))

X_train = [list(row) for row in X_train.collect()]
X_test = [list(row) for row in X_test.collect()]

# Convertir las etiquetas de entrenamiento y prueba en listas planas
Y_train = [row[0] for row in Y_train.collect()]
Y_test = [row[0] for row in Y_test.collect()]

# COMMAND ----------

# Define el valor de 'k' y crea el modelo KNN
k = 5  # Ajusta el valor de 'k' según tus necesidades
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, Y_train)

# COMMAND ----------

y_pred = knn.predict(X_test)

# COMMAND ----------

accuracy = accuracy_score(Y_test, y_pred)
print("Precisión del modelo KNN:", accuracy)

# COMMAND ----------

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(Y_test, y_pred)

print("Matriz de Confusión:")
print(confusion)

# COMMAND ----------

from sklearn.metrics import classification_report
reporte = classification_report(Y_test,y_pred)
print(reporte)

# COMMAND ----------

# MAGIC %md
# MAGIC **Observaciones de la matriz de confusión y de el reporte arrojado**
# MAGIC
# MAGIC
# MAGIC **Matriz de confusión**
# MAGIC
# MAGIC   En primer lugar, hay que tener cuidado con las etiquetas y los ejes del grafico.
# MAGIC
# MAGIC   Debemos recordar que {1: personas en pobreza, 1: personas no en pobreza}.
# MAGIC
# MAGIC   Con la matriz de confusión tenemos que:
# MAGIC
# MAGIC   1. **Verdaderos positivos:** Las personas que predijo como pobres y si son pobres fueron 1926.
# MAGIC   2. **Verdaderos negativos:** Las personas que predijo como no pobres y no son pobres fueron 10735.
# MAGIC   3. **Falsos negativos:** Las personas que predijo como no pobres pero en realidad son pobres fueron 461.
# MAGIC   4. **Falsos positivos:** Las personas que predijo como pobres pero en realidad son no pobres fueron 354 .
# MAGIC
# MAGIC
# MAGIC **Reporte arrojado**\
# MAGIC   Con la matriz de confusión no es suficiente para concluir sobre el modelo, necesitamos metricas de rendimiento del modelo que son dadas por el reporte generado a traves de _classification_report(Y_test,Y_pred)_. Si quiere más información sobre las metricas, se explico con detalle en una de las celdas anteriores.
# MAGIC
# MAGIC   En este caso, tambien tenemos que:
# MAGIC   
# MAGIC   **Precisión**
# MAGIC
# MAGIC   *   Para la clase 1, la precisión es de 0.84
# MAGIC   *   Para la clase 2, la precisión es de 0.96
# MAGIC   *   Su _macro-avg_ es de 0.90
# MAGIC   *   Su _weighted-avg_ es de 0.94
# MAGIC
# MAGIC
# MAGIC   **Recall**
# MAGIC
# MAGIC   * Para la clase 1, el recall es de 0.81
# MAGIC   * Para la clase 2, el recall es de 0.97
# MAGIC   * Su _macro-avg_ es de 0.89
# MAGIC   * Su _weighted-avg_ es de 0.94
# MAGIC
# MAGIC
# MAGIC   **F1 Score**
# MAGIC
# MAGIC   * Para la clase 1, el f1 - score es de 0.83
# MAGIC   * Para la clase 2, el f1 - score es de 0.96
# MAGIC   * Su _macro-avg_ es de 0.89
# MAGIC   * Su _weighted-avg_ es de 0.94
# MAGIC
# MAGIC
# MAGIC   **Exactitud**
# MAGIC   * Es de 0.94%.
# MAGIC
# MAGIC En general podriamos decir que es un buen modelo, pues con una exactitud del 94% en las predicciones nos podemos dar por bien servidos. 
# MAGIC
# MAGIC Sin embargo, si nos ponemos a analizar clase por clase tenemos:
# MAGIC 1. Que el dataset esta desequilibrado en cuanto a las observaciones y predicciones de las personas pobres (el desequilibrio es a favor de las no pobres).
# MAGIC 2. En ambas clases, el recall y la precisión son bastante altas, por lo que ambas clases predicen de buena forma.
# MAGIC 4. El _f1 score_ tambien nos da una idea de lo bueno que es el modelo para cada clase. En este caso, la clase 1 tiene un _F1 SCORE_ de 0.83, mientras que la clase 2 lo tiene de 0.96. Claramente, es una metrica favorable para la clase 2, sin embargo, el valor para la clase 1 es considerablemente altos.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #####***Segundo modelo***

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.functions import lit

#Sacar el rango de cada variable
resumen_describe = df_pobreza.describe()
variables = resumen_describe.columns
variables = variables[1:]
resumen_pandas = resumen_describe.toPandas().reset_index(drop=True).T
resumen_pandas.columns = resumen_pandas.iloc[0]
resumen_pandas = resumen_pandas[1:]
df = spark.createDataFrame(resumen_pandas)
df = df.withColumn("rango", col("max") - col("min"))


display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC En este caso, se observa una variabilidad significativa en el orden de magnitud de las variables (rango, diferencia entre el maximo y el minimo).
# MAGIC Esto podria advertir un deterioro de rendimiento de cualquier algoritmo de machine learning, pues al construir el modelo puede existir un sesgo entre las variables que altere el algoritmo.
# MAGIC
# MAGIC Por otro lado, podemos concluir que:
# MAGIC 1. La desviación estandar en muchos casos es bastante alta, lo que indica que los datos estan bastante dispersos.
# MAGIC 2. Hay variables que, por lo contrario, tienen muy poca dispersión.

# COMMAND ----------

indices_con_rango_mayor_a_6 = []
indices_con_rango_menor_o_igual_a_6 = []

# Iteramos a través de las variables para identificar los índices
for idx, variable in enumerate(variables):
    filas = df.select("rango").collect()
    valor_rango = filas[idx]["rango"]
    indice_variable = {"indice": idx, "variable": variable,"rango":valor_rango}
    if valor_rango > 6:
        indices_con_rango_mayor_a_6.append(indice_variable)
    else:
        indices_con_rango_menor_o_igual_a_6.append(indice_variable)

print("VARIABLES CON RANGO menor A 6")
for i in indices_con_rango_menor_o_igual_a_6:
    print(i)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC En este caso tenemos que 17 variables tienen un rango mayor a 6. Hay que verificar si tiene logica este valor, pues hay variables que son categorias y que pueden tomar valores enteros dentro de un rango amplio. Por lo tanto, los registros que vale la pena normalizar (los que no pertenecen a categorias), son:
# MAGIC
# MAGIC {'indice': 0, 'variable': 'Sueldo12Meses', 'rango': 1122.0}\
# MAGIC {'indice': 18, 'variable': 'EST_EITC', 'rango': 6630.41327}\
# MAGIC {'indice': 19, 'variable': 'EST_MOOP', 'rango': 71369.0}\
# MAGIC {'indice': 23, 'variable': 'INTP_adj', 'rango': 337969.1752}\
# MAGIC {'indice': 24, 'variable': 'MRGP_adj', 'rango': 7091.6792}\
# MAGIC {'indice': 23, 'variable': 'INTP_adj', 'rango': 337969.1752}\
# MAGIC {'indice': 24, 'variable': 'MRGP_adj', 'rango': 7091.6792}\
# MAGIC {'indice': 28, 'variable': 'Off_Threshold', 'rango': 43840.0}\
# MAGIC {'indice': 29, 'variable': 'OI_adj', 'rango': 83073.953}\
# MAGIC {'indice': 30, 'variable': 'PA_adj', 'rango': 16412.172}\
# MAGIC {'indice': 32, 'variable': 'PreTaxIncome_PU', 'rango': 1788825.35}\
# MAGIC {'indice': 33, 'variable': 'RNTP_adj', 'rango': 3951.0784}\
# MAGIC {'indice': 34, 'variable': 'SEMP_adj', 'rango': 395817.0}\
# MAGIC {'indice': 35, 'variable': 'SSP_adj', 'rango': 35458.395}\
# MAGIC {'indice': 37, 'variable': 'WAGP_adj', 'rango': 668644.0}
# MAGIC
# MAGIC La mayoria de ellas tienen como unidad los dolares. 

# COMMAND ----------

# Muestra las filas que cumplen con la condición
print(variables)

# COMMAND ----------

dicc_normalizar = [{'indice': 18, 'variable': 'EST_EITC', 'rango': 6630.41327},
{'indice': 19, 'variable': 'EST_MOOP', 'rango': 71369.0},
{'indice': 23, 'variable': 'INTP_adj', 'rango': 337969.1752},
{'indice': 24, 'variable': 'MRGP_adj', 'rango': 7091.6792},
{'indice': 28, 'variable': 'Off_Threshold', 'rango': 43840.0},
{'indice': 29, 'variable': 'OI_adj', 'rango': 83073.953},
{'indice': 30, 'variable': 'PA_adj', 'rango': 16412.172},
{'indice': 32, 'variable': 'PreTaxIncome_PU', 'rango': 1788825.35},
{'indice': 33, 'variable': 'RNTP_adj', 'rango': 3951.0784},
{'indice': 34, 'variable': 'SEMP_adj', 'rango': 395817.0},
{'indice': 35, 'variable': 'SSP_adj', 'rango': 35458.395},
{'indice': 37, 'variable': 'WAGP_adj', 'rango': 668644.0}]

nom_var = []
for diccionario in dicc_normalizar:
    variable = diccionario['variable']
    nom_var.append(variable)


# COMMAND ----------

nom_var

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ######Estandarización de las variables

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC El orden de magnitud en algunas variables es bastante alto, lo que puede afectar los modelos de Machine Learning que se basen en distancias euclideanas. De igual forma, los modelos de Machine Learning, de manera general, se van a ver distorsionados con las variables que tengan un rango bastante alto. \
# MAGIC Por otro lado, las diferencias en las escalas entre las variables de entrada pueden aumentar la dificultad del problema que se está modelando, pues el modelo puede estar aprendiendo de valores muy grandes o muy pequeños. Un modelo con valores altos de orden de magnitud en sus variables suele ser inestable, lo que significa que puede sufrir un rendimiento deficiente durante el aprendizaje y sensibilidad a los valores de entrada, lo que resulta en un mayor error de generalización.\
# MAGIC Por ello, acudimos al **_StandardScaler_** para estandarizar las variables que se identificaron en el paso anterior. Esto significa que a cada observación se le va a aplicar la siguiente transformación:\
# MAGIC   **y = (x - μ) / σ**             con μ: la media y σ: la desviación estandar.\
# MAGIC Esto con el fin de evitar que los datos esten muy distorsionados, pues lo que hace la anterior transformación es dejar la media de las observaciones en 0, y la varianza en 1.
# MAGIC

# COMMAND ----------

print(type(df_pobreza))

# COMMAND ----------

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


# Identificar las columnas numéricas que deseas estandarizar
numeric_cols = nom_var

# Crear un vector de características a partir de las columnas numéricas
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
assembledDF = assembler.transform(df_pobreza)

# Inicializar y ajustar el StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
scalerModel = scaler.fit(assembledDF)

# Transformar el DataFrame original
scaledDF = scalerModel.transform(assembledDF)

resultDF = scaledDF.select("scaledFeatures")

# Mostrar el DataFrame resultante
display(resultDF)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.sql import Row 

new_rows = []
for row in resultDF.collect():
    values_list = row.scaledFeatures.values
    values_list = [float(item) for item in values_list]
    new_rows.append(values_list)


rows = [Row(*row) for row in new_rows]
fields = StructType([StructField(name, FloatType(), True) for name in nom_var])
schema = StructType(fields)
new_df = spark.createDataFrame(rows, schema)
display(new_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC En este caso podemos observar el efecto de estandarizar, pues ahora el orden de magnitud no es tan amplio entonces no nos generará problemas en el modelo.
# MAGIC
# MAGIC Ahora debemos juntar las columnas no seleccionadas (las no estandarizadas) del dataframe con las que si estandarizamos. Esto con el fin de poder entrenar el modelo. 

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

columnas = df_pobreza.columns
columnas_buenas = list(set(columnas) - set(nom_var))

df_ml = df_pobreza[columnas_buenas]
df_ml2 = new_df

df_ml = df_ml.withColumn("id", monotonically_increasing_id())
df_ml2 = df_ml2.withColumn("id", monotonically_increasing_id())


combined_df = df_ml.join(df_ml2, "id", "inner").drop("id")

display(combined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ######Entrenamiento del modelo

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Segun el fundamento teorico del Cross validation, se va a separar el conjunto de datos en entrenamiento y prueba.
# MAGIC
# MAGIC La división de entrenamiento sirve para, valga la redundancia, entrenar el modelo. Basicamente, con ese conjunto de datos se le enseña al modelo lo que debe hacer este.
# MAGIC
# MAGIC En el caso de los datos de prueba, estos nos ayudarán a revisar que tan bien esta el modelo comparando las respuestas del modelo con las respuestas del conjunto de prueba.

# COMMAND ----------

colum = combined_df.columns

# COMMAND ----------

from pyspark.sql import functions as F

# Define las características y las etiquetas
assembler = VectorAssembler(inputCols=colum, outputCol="features")
data = assembler.transform(combined_df)

data = data.withColumn("", F.col("NYCgov_Pov_Stat"))

# Divide los datos en conjuntos de entrenamiento y prueba
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Como es un modelo supervisado, es importante declarar nuestra variables explicativas y nuestra variable respuesta. 
# MAGIC
# MAGIC Nuestra variable respuesta es **_NYCgov_Pov_Stat_**, que nos indica 1: que la persona esta en situación de pobreza, 2: que la persona NO esta en situacion de pobreza. 
# MAGIC
# MAGIC Nuestras variables explicativas son todas las demas pertenecientes al _df ml_
# MAGIC
# MAGIC

# COMMAND ----------

Y_train = train_data.select(col("NYCgov_Pov_Stat"))
Y_test = test_data.select(col("NYCgov_Pov_Stat"))

X_train = train_data[colum]
X_test = test_data[colum]

X_train = [list(row) for row in X_train.collect()]
X_test = [list(row) for row in X_test.collect()]

# Convertir las etiquetas de entrenamiento y prueba en listas planas
Y_train = [row[0] for row in Y_train.collect()]
Y_test = [row[0] for row in Y_test.collect()]

# COMMAND ----------

k = 5 
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, Y_train)

# COMMAND ----------

y_pred = knn.predict(X_test)

# COMMAND ----------

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, y_pred)
print("Precisión del modelo KNN:", accuracy)

# COMMAND ----------

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(Y_test, y_pred)

print("Matriz de Confusión:")
print(confusion)

# COMMAND ----------

from sklearn.metrics import classification_report
reporte = classification_report(Y_test,y_pred)
print(reporte)

# COMMAND ----------

# MAGIC %md
# MAGIC **Observaciones de la matriz de confusión y de el reporte arrojado**
# MAGIC
# MAGIC
# MAGIC **Matriz de confusión**
# MAGIC
# MAGIC   En primer lugar, hay que tener cuidado con las etiquetas y los ejes del grafico.
# MAGIC
# MAGIC   Debemos recordar que {1: personas en pobreza, 1: personas no en pobreza}.
# MAGIC
# MAGIC   Con la matriz de confusión tenemos que:
# MAGIC
# MAGIC   1. **Verdaderos positivos:** Las personas que predijo como pobres y si son pobres fueron 1923.
# MAGIC   2. **Verdaderos negativos:** Las personas que predijo como no pobres y no son pobres fueron 10742.
# MAGIC   3. **Falsos negativos:** Las personas que predijo como no pobres pero en realidad son pobres fueron 464.
# MAGIC   4. **Falsos positivos:** Las personas que predijo como pobres pero en realidad son no pobres fueron 347 .
# MAGIC
# MAGIC
# MAGIC **Reporte arrojado**\
# MAGIC   Con la matriz de confusión no es suficiente para concluir sobre el modelo, necesitamos metricas de rendimiento del modelo que son dadas por el reporte generado a traves de _classification_report(Y_test,Y_pred)_. Si quiere más información sobre las metricas, se explico con detalle en una de las celdas anteriores.
# MAGIC
# MAGIC   En este caso, tambien tenemos que:
# MAGIC   
# MAGIC   **Precisión**
# MAGIC
# MAGIC   *   Para la clase 1, la precisión es de 0.85
# MAGIC   *   Para la clase 2, la precisión es de 0.96
# MAGIC   *   Su _macro-avg_ es de 0.90
# MAGIC   *   Su _weighted-avg_ es de 0.94
# MAGIC
# MAGIC
# MAGIC   **Recall**
# MAGIC
# MAGIC   * Para la clase 1, el recall es de 0.81
# MAGIC   * Para la clase 2, el recall es de 0.97
# MAGIC   * Su _macro-avg_ es de 0.89
# MAGIC   * Su _weighted-avg_ es de 0.94
# MAGIC
# MAGIC
# MAGIC   **F1 Score**
# MAGIC
# MAGIC   * Para la clase 1, el f1 - score es de 0.83
# MAGIC   * Para la clase 2, el f1 - score es de 0.96
# MAGIC   * Su _macro-avg_ es de 0.89
# MAGIC   * Su _weighted-avg_ es de 0.94
# MAGIC
# MAGIC
# MAGIC   **Exactitud**
# MAGIC   * Es de 0.94.
# MAGIC
# MAGIC En general podriamos decir que es un buen modelo, pues con una exactitud del 85% en las predicciones nos podemos dar por bien servidos. 
# MAGIC
# MAGIC Sin embargo, si nos ponemos a analizar clase por clase tenemos:
# MAGIC 1. Que el dataset esta desequilibrado en cuanto a las observaciones y predicciones de las personas pobres (el desequilibrio es a favor de las no pobres).
# MAGIC 2. El _f1 score_ tambien nos da una idea de lo bueno que es el modelo para cada clase. En este caso, la clase 1 tiene un _F1 SCORE_ de 0.83, mientras que la clase 2 lo tiene de 0.96. Claramente, es una metrica favorable para la clase 2.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #####***Tercer modelo***
# MAGIC
# MAGIC Aunque la diferencia es bastante minima (solo hay cambios en la cantidad de valores predichos en cada clase). En cuanto a los porcentajes de cada metrica, la única que cambia entre el modelo 1 y el 2, es la de precisión para la clase 1 (solo cambia en un punto porcentual). 
# MAGIC
# MAGIC Por esta razon, el tercer modelo, con el cambio de k, se hará con la base estandarizada. 

# COMMAND ----------

# Define el valor de 'k' y crea el modelo KNN
k = 3 # Ajusta el valor de 'k' según tus necesidades
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, Y_train)

# COMMAND ----------

y_pred = knn.predict(X_test)

# COMMAND ----------

accuracy = accuracy_score(Y_test, y_pred)
print("Precisión del modelo KNN:", accuracy)

# COMMAND ----------

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(Y_test, y_pred)

print("Matriz de Confusión:")
print(confusion)

# COMMAND ----------

from sklearn.metrics import classification_report
reporte = classification_report(Y_test,y_pred)
print(reporte)

# COMMAND ----------

# MAGIC %md
# MAGIC **Observaciones de la matriz de confusión y de el reporte arrojado**
# MAGIC
# MAGIC
# MAGIC **Matriz de confusión**
# MAGIC
# MAGIC   En primer lugar, hay que tener cuidado con las etiquetas y los ejes del grafico.
# MAGIC
# MAGIC   Debemos recordar que {1: personas en pobreza, 2: personas no en pobreza}.
# MAGIC
# MAGIC   Con la matriz de confusión tenemos que:
# MAGIC
# MAGIC   1. **Verdaderos positivos:** Las personas que predijo como pobres y si son pobres fueron 1973.
# MAGIC   2. **Verdaderos negativos:** Las personas que predijo como no pobres y no son pobres fueron 414.
# MAGIC   3. **Falsos negativos:** Las personas que predijo como no pobres pero en realidad son pobres fueron 344.
# MAGIC   4. **Falsos positivos:** Las personas que predijo como pobres pero en realidad son no pobres fueron 10745 .
# MAGIC
# MAGIC
# MAGIC **Reporte arrojado**\
# MAGIC   Con la matriz de confusión no es suficiente para concluir sobre el modelo, necesitamos metricas de rendimiento del modelo que son dadas por el reporte generado a traves de _classification_report(Y_test,Y_pred)_. Si quiere más información sobre las metricas, se explico con detalle en una de las celdas anteriores.
# MAGIC
# MAGIC   En este caso, tambien tenemos que:
# MAGIC   
# MAGIC   **Precisión**
# MAGIC
# MAGIC   *   Para la clase 1, la precisión es de 0.85
# MAGIC   *   Para la clase 2, la precisión es de 0.96
# MAGIC   *   Su _macro-avg_ es de 0.91
# MAGIC   *   Su _weighted-avg_ es de 0.94
# MAGIC
# MAGIC
# MAGIC   **Recall**
# MAGIC
# MAGIC   * Para la clase 1, el recall es de 0.83
# MAGIC   * Para la clase 2, el recall es de 0.97
# MAGIC   * Su _macro-avg_ es de 0.90
# MAGIC   * Su _weighted-avg_ es de 0.94
# MAGIC
# MAGIC
# MAGIC   **F1 Score**
# MAGIC
# MAGIC   * Para la clase 1, el f1 - score es de 0.84
# MAGIC   * Para la clase 2, el f1 - score es de 0.97
# MAGIC   * Su _macro-avg_ es de 0.90
# MAGIC   * Su _weighted-avg_ es de 0.94
# MAGIC
# MAGIC
# MAGIC   **Exactitud**
# MAGIC   * Es de 0.94%.
# MAGIC
# MAGIC En general podriamos decir que es un buen modelo, pues con una exactitud del 94% en las predicciones nos podemos dar por bien servidos. 
# MAGIC
# MAGIC Sin embargo, si nos ponemos a analizar clase por clase tenemos:
# MAGIC 1. Que el dataset esta desequilibrado en cuanto a las observaciones y predicciones de las personas pobres (el desequilibrio es a favor de las no pobres).
# MAGIC 2. En ambas clases, el recall y la precisión son bastante altas, por lo que ambas clases predicen de buena forma.
# MAGIC 4. El _f1 score_ tambien nos da una idea de lo bueno que es el modelo para cada clase. En este caso, la clase 1 tiene un _F1 SCORE_ de 0.84, mientras que la clase 2 lo tiene de 0.97. Claramente, es una metrica favorable para la clase 2, sin embargo, el valor para la clase 1 es considerablemente altos.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #####**CONCLUSIONES**

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Definir el esquema para los resultados
esquema = StructType([
    StructField("Modelo", StringType(), True),
    StructField("Numero k", IntegerType(), True),
    StructField("Precisión Clase 1", FloatType(), True),
    StructField("Precisión Clase 2", FloatType(), True),
    StructField("Recall Clase 1", FloatType(), True),
    StructField("Recall Clase 2", FloatType(), True),
    StructField("F1-score Clase 1", FloatType(), True),
    StructField("F1-score Clase 2", FloatType(), True),
    StructField("PRECISIÓN TOTAL", FloatType(), True),
])

# Crear un DataFrame con los resultados
datos_resultados = [
    ("1: sin estandarizados",5,0.84,0.96,0.81,0.97,0.83,0.96,0.94),
    ("2: con estandarizados",5,0.85,0.96,0.81,0.97,0.83,0.96,0.94),
    ("3: con estandarizados",3,0.85,0.96,0.83,0.97,0.84,0.97,0.94),
]

df_resultados = spark.createDataFrame(datos_resultados, schema=esquema)

# Mostrar el DataFrame
df_resultados.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC 1. De forma general vemos que **los 3 modelos se comportan de buena forma**. En los 3 casos, las métricas son similares y cambian de forma minima. 
# MAGIC 2. **La clase 2 es la que tiene mejor rendimiento** a comparación de la clase 1. Es decir, el modelo predice mejor a las personas en no pobreza que a las que si estan en pobreza. Sin embargo, la clase 1 no tiene malas metricas. 
# MAGIC 3. **El mejor modelo de los 3 es el 3**. En este caso, el k=3 exige al modelo ser más estricto. Se pensaba que, al ser más estricto, podrían haber más datos mal clasificados. Sin embargo, se comporto de manera similar que los otros dos modelos. Esto nos lleva a concluir que las clases estan lo suficientemente separadas para que el modelo pueda tener esa precisión. 
# MAGIC Ademas, si revisamos la precisión en las celdas anteriores, los decimales nos llevan a reforzar esta conclusión. 
# MAGIC
