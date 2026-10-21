# Dinámica Lagrangiana de una Pluma de Grúa con Gancho en Caída Libre

Repositorio con el código de simulación numérica e integración de las ecuaciones de Euler-Lagrange para el trabajo de **Mecánica y Ondas I** (Grado en Física, Universidad de Córdoba).

## Contenido del repositorio
- `simulacion_grua.ipynb`: Cuaderno interactivo de Google Colab con la integración numérica (Runge-Kutta de orden 4/5 `solve_ivp`), la validación frente a la solución analítica exacta, el estudio de invariantes dinámicos ($p_x, p_\varphi, E$), el cálculo de la tensión mediante multiplicadores de Lagrange ($T = -\lambda = \mu l \dot{\theta}^2$) y el análisis paramétrico.
