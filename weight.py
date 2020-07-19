#%%
from torch.distributions import constraints
import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)


def scale(guess):
    """guess is a number which someone declares as the weight of some object. 
    It's the prior knowledge."""
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    """measurement is a number given by the scale."""
    return pyro.sample("measurement", dist.Normal(weight, 0.75))


"""
"weight"
   +
   |
   |  Normal(_, 0.75)
   |
   v
"measurement"
"""

# suppose we have guess = 8.5, and observed measurement == 9.5
# what is the actual distribution of weight?

# (weight|guess,measurement=9.5)∼?

# condition <- (model, observations_dict)
# why dict? Pyro uses names to identify variables.
# So the dict should be: {name: value, ...}
# returns a new model that always gives output from the obs_dict.

conditional_scale = pyro.condition(scale, data={"measurement": 9.5})


def obs_model_scale_cond(measurement, guess):
    model = pyro.condition(scale, data={"measurement": 9.5})
    return model(guess)


def obs_modl_scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
    # Q: Why still do we need to get a 'weight` sample, since the output is fixed?
    # Is it to construct a model with the same set of parameters for a correct signature?
    return pyro.sample("measurement", dist.Normal(weight, 0.75), obs=9.5)

# Now we have conditioned on an observation of measurement.
# Infer dist-weight given guess and measurement == data.

# Inference algorithms, allows us to
# use arbitrary stochastic functions, which we call guide functions or guides,
# as approximate posterior distributions.
# Guide functions must satisfy two criteria:
# 1. all unobserved (i.e., not conditioned) sample statements
# that appear in the model appear in the guide.
# 2. the guide has the same iput signature as the model (i.e., takes the same arguments)

# Guide functions can serve as programmable, data-dependent proposal distributions for
# importance sampling,
# rejection sampling,
# sequential Monte Carlo,
# MCMC, and
# independent
# Metropolis-Hastings, and as
# variational distributions or
# inference networks for stochastic variational inference.

# Pyro has implemented importance sampling, MCMC, and stochastic variational inference.

# The precise meaning of the guide is different across different inference algorithms,
# the guide should generally be chosen so that, in principle,
# it is flexible enough to closely approximate the distribution over all unobserved sample
# statements in the model.

# 下面这个我也不会算
# http://www.stat.cmu.edu/~brian/463-663/week09/Chapter%2003.pdf


def perfect_guide(guess):
    loc = (0.75**2 * guess + 9.5) / (1 + 0.75**2)  # 9.14
    scale = np.sqrt(0.75**2/(1 + 0.75**2))  # 0.6
    return pyro.sample("weight", dist.Normal(loc, scale))


# In general it is intractable to specify a guide that is a good approximation to the
# posterior distribution of an arbitrary conditioned stochastic function.

def some_fn(weight):
    return weight


def intractable_scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(some_fn(weight), 0.75))


# What we can do instead is to use the top-level function pyro.param *to* specify a family of
# guides -- indexed by *named* parameters, and search for the member of that family which is
# best approximation according to some loss function.

# 例如，使用参数化的正态分布，从中找到最合适的一个

# This approach to approximate posterior inference is called variational inference.

# pyro.param is a frontend for Pyro's key-value parameter store.

# The first time pyro.param is called with a particular name, it stores its argument
# in the parameter store and then returns that value.
# After that, when it is called with that name, it returns the value from the parameter
# store regardless of any other arguments.

# Similar to `simple_param_store.setdefault`,
# but with some additional tracking and management functionality.


def scale_parameterized_guide_constrained(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.))
    return pyro.sample("weight", dist.Normal(a, torch.abs(b)))


# Pyro is built to enable stochastic variational inference, a powerful and widely
# applicable class of variational inference algorithms with three key characteristics:
# 1. Parameters are always real-valued tensors
# 2. We compute Monte Carlo estimates of a loss function from samples of execution histories
#    of the model and the guide
# 3. We use stochastic gradient descent to search for the optimal parameters.

#%%
guess = 8.5

pyro.clear_param_store()
svi = pyro.infer.SVI(model=obs_modl_scale,
                     guide=scale_parameterized_guide_constrained,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum": 0.1}),
                     loss=pyro.infer.Trace_ELBO())

losses, a, b = [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step(guess))
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
print('a = ', pyro.param("a").item())
print('b = ', pyro.param("b").item())


# %%
