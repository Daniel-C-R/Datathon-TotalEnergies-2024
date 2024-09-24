# Solución al Datathon

En este repositorio se incluye el código tanto del simulador que se nos ha proporcionado como de la solución que he desarrollado.

Para entender el funcionamiento del simulador y el problema a resolver se recomienda la lectura del archivo `datathon_totalenergies.pdf`. La explicación de la solución propuesta y cómo se ha llegado a ella se incluye en `Informe.pdf`.

Todo el código que he implementado se encuentra en el archivo `optimizer.py`. Se trata de un programa desarrollado con el *framework* `pymoo` que realiza una búsqueda de la mejor configuración por medio de un algoritmo evolutivo para optimización multiobjetivo.

Para poder ejecutar el simulador es necesario instalar todas las dependencias que se listan en el archivo `requirements.txt`. No obstante, para la solución al reto han sido necesarios algunos paquetes más, por lo que si se desea ejecutar el optimizador es necesario instalar todas las librerías enumeradas en `librerias.txt` en su lugar.

También, se incluyen las diapositivas que utilicé para presentar mi solución en el *All Electric Society International Summit* el 19/09/2024 en Gijón, donde obtuve el premio a la solución más creativa.

---

A continuación se incluye el README original del simulador:

# Optimización de Rutas y Análisis de Sostenibilidad en Autobuses Eléctricos Urbanos

Fecha de creación: 12/07/2024

Última modificación: 09/09/2024



## Autores

| Nombre   | Chakhoyan Grigoryan, Razmik                 | Menéndez Sales, Pablo                     |
|----------|---------------------------------------------|-------------------------------------------|
| Correo   | chakhoyanrazmik@gmail.com                   | pablomenendezsales@gmail.com              |
| LinkedIn | https://www.linkedin.com/in/chakhoyanrazmik | https://www.linkedin.com/in/pablo-m-sales |

---

    NOTA: Se recomienda visualizar este archivo desde un visor de Markdown adecuado (si se utiliza VSCode existen múltiples extensiones disponibles).

### Descripción

Este proyecto simula el comportamiento de autobuses eléctricos y no eléctricos en rutas predefinidas, evaluando carácterísticas como eficiencia energética, el costo de operación o su emisión de contaminates. Para configurar el modelo, el usuario tan solo debe limitarse a modificar los ficheros `config.py` y `main.py`.

- `config.py`: Define configuraciones clave como la ruta de los datos, si el autobús es eléctrico y el número de días de la simulación.

- `main.py`: Contiene la función principal que ejecuta la simulación basada en las configuraciones y el modelo.

### Instrucciones para ejecutar el código

Antes de ejecutar el código, asegúrate de instalar las dependencias necesarias, definidas en el fichero `requirements.txt`. Si al momento de ejecutar ocurre algún problema, asegúrate de tener instalada una versión de Python igual o superior a la 3.11.7.

```bash
pip install -r requirements.txt
```

### Configuración

En el archivo `config.py` se definen los siguientes parámetros clave para la simulación:

- **NAME**: Nombre del proyecto o simulación.
- **DATA**: Ruta al archivo de datos CSV con la información de la ruta.
- **ELECTRIC**: Bool que indica si el autobús utilizado es eléctrico (True) o de combustión interna (False).
- **DAYS**: Número de días que se simularán.

Puedes modificar estos valores según tus necesidades antes de ejecutar la simulación.

### Ejecución

Una vez instaladas las dependencias y configurado el archivo `config.py`, puedes ejecutar la simulación desde el fichero `main.py`.Esto ejecutará el modelo durante el número de días especificado.

Dentro de `main.py` ocurre lo siguiente:

1.  Se inicializa el objeto `ModelConfig()`, que agrupa todos los parámetros necesarios para la ejecución del modelo.

2.  Se inicializa `Model()` con la configuración establecida y se ejecuta el método `.run()`, indicando el número de días de horizonte para el cual se quiere realizar la simulación. Este método retorna un diccionario con los valores de consumo energético, gases emitidos, degradación de batería, etc.
Este método también almacenará la salida del modelo en un fichero .csv dentro del directorio `simulation_results`. Cuando el modelo se ejecute en modo eléctrico, el fichero retornado será `electric_simulation_results.csv`; en el caso del modo de combustión, será `combustion_simulation_results.csv`.
