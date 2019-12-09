import sys
import random

import pyomo.kernel as pmo

random.seed(1000)

model = pmo.block()

model.N = int(sys.argv[1])

model.Locations = range(model.N)

model.P = int(sys.argv[2])

model.M = model.N

model.Customers = range(model.M)

model.d = {
    (l, c): random.uniform(1.0, 2.0) for l in model.Locations for c in model.Customers
}

model.x = pmo.variable_dict()
model.y = pmo.variable_dict()
for l in model.Locations:
    model.y[l] = pmo.variable(lb=0.0, ub=1.0, value=0.0)
    for c in model.Customers:
        model.x[l, c] = pmo.variable(lb=0.0, ub=1.0, value=0.0)

model.obj = pmo.objective(
    sum(model.d[l, c] * model.x[l, c] for l in model.Locations for c in model.Customers)
)

model.single_x = pmo.constraint_dict()
model.bound_y = pmo.constraint_dict()
for c in model.Customers:
    model.single_x[c] = pmo.linear_constraint(
        terms=[(model.x[l, c], 1.0) for l in model.Locations], rhs=1.0,
    )
    for l in model.Locations:
        model.bound_y[l, c] = pmo.linear_constraint(
            terms=[(model.x[l, c], 1.0), (model.y[l], -1.0)], ub=0.0,
        )

model.num_facilities = pmo.linear_constraint(
    terms=[(model.y[l], 1.0) for l in model.Locations], rhs=model.P,
)

model.write("pyomo.lp")
