import numpy as np
import plotly.graph_objects as go
from core.model import Model
from core.model_config import ModelConfig
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

# Límites del espacio de búsqueda (obtenidos de charging_points.json y bus.json)
search_space = dict(
    num_charging_points=4,
    battery_capacity_range=(200, 800),
    battery_min_charge_range=(10, 45),
    battery_max_charge_range=(55, 100),
)

# Número de días de la simulación
n_days = 1

# Parámetros de la simulación
name = 'linea_d2_simulation'
filepath = 'data/line_data/line_data_simulation.csv'
electric = True


# Clase que define el problema de optimización
class DatathonProblem(ElementwiseProblem):

    def __init__(
        self,
        num_charging_points,
        battery_capacity_range,
        battery_min_charge_range,
        battery_max_charge_range,
    ):

        # Límites de las variables de decisión
        xl = np.array([1, battery_capacity_range[0],
                      battery_min_charge_range[0], battery_max_charge_range[0]])
        xu = np.array([num_charging_points, battery_capacity_range[1],
                      battery_min_charge_range[1], battery_max_charge_range[1]])

        super().__init__(n_var=4, n_obj=2, n_constr=0, xl=xl, xu=xu, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        # Hay que asegurarse de que las variables de decisión son enteras
        charging_point = int(np.round(x[0]))
        battery_capacity = int(np.round(x[1]))
        battery_min_charge = int(np.round(x[2]))
        battery_max_charge = int(np.round(x[3]))

        # Configuración del modelo
        model_config = ModelConfig(
            electric=electric,
            name=name,
            filepath=filepath,
            charging_point_id=charging_point,
            min_battery_charge=battery_min_charge,
            max_battery_charge=battery_max_charge,
            initial_capacity_kWh=battery_capacity,
        )

        # Ejecución del modelo
        model = Model(config=model_config)
        simulation_results = model.run(n_days=n_days)

        # Extraer los resultados de la simulación
        consumption_cost = simulation_results['consumption_cost']
        bus_cost = simulation_results['bus_cost']
        battery_degradation = simulation_results['battery_degradation']
        total_time_below_min_soc_s = simulation_results['total_time_below_min_soc_s']

        # Funciones objetivo (incluyen las penalizaciones por incumplimiento de las restricciones)
        f1 = consumption_cost + bus_cost + total_time_below_min_soc_s * 1e20
        f2 = battery_degradation + total_time_below_min_soc_s * 1e20

        out['F'] = [f1, f2]


# Operadores de cruce y mutación adaptados para trabajar con variables enteras y sin salirse de los límites de estas

class IntegerSBX(SBX):
    def _do(self, problem, X, **kwargs):
        X = super()._do(problem, X, **kwargs)
        return np.rint(X).astype(int)


class IntegerPM(PM):
    def _do(self, problem, X, **kwargs):
        X = super()._do(problem, X, **kwargs)
        return np.rint(X).astype(int)


# Creación del objeto problema
problem = DatathonProblem(**search_space)

# Algoritmo de optimización
algorithm = SMSEMOA(
    pop_size=50,
    sampling=IntegerRandomSampling(),
    crossover=IntegerSBX(),
    mutation=IntegerPM(),
    eliminate_duplicates=True,
)

# Ejecución de la optimización
results = minimize(
    problem,
    algorithm,
    ('n_gen', 50),
    save_history=True,
    seed=1,
    verbose=True,
)

# Resultados de la optimización
print(results.F)  # Valores de las funciones objetivo
print(results.X)  # Valores de las variables de decisión

# Gráfico interactivo del frente Pareto

fig = go.Figure(data=[go.Scatter(
    x=results.F[:, 0],
    y=results.F[:, 1],
    mode='markers',
    marker=dict(
        size=5,
        opacity=0.8
    ),
    text=[f'{i+1}' for i in range(len(results.F))],
    hovertemplate='Solution: %{text}<br>' +
                  'Cost: %{x}<br>' +
                  'Battery degradation: %{y}<extra></extra>'
)])

fig.update_layout(
    xaxis_title='Cost',
    yaxis_title='Battery degradation',
    title='Pareto Front',
    width=800,
    height=800
)

fig.show()

# Gráfico de la convergencia de las funciones objetivo

convergence_f1 = np.array([e.opt.get('F')[0][0] for e in results.history])
convergence_f2 = np.array([e.opt.get('F')[0][1] for e in results.history])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(convergence_f1, color='red')
ax1.set_xlabel('Generations')
ax1.set_ylabel('Cost')
ax1.set_title('Convergence of cost')

ax2.plot(convergence_f2, color='blue')
ax2.set_xlabel('Generations')
ax2.set_ylabel('Battery degradation')
ax2.set_title('Convergence of battery degradation')

plt.tight_layout()
plt.show()
