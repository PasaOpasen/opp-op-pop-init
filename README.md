
# Opposition learning operators and population initializers

[![PyPI
version](https://badge.fury.io/py/OppOpPopInit.svg)](https://pypi.org/project/OppOpPopInit/)
[![Downloads](https://pepy.tech/badge/oppoppopinit)](https://pepy.tech/project/oppoppopinit)
[![Downloads](https://pepy.tech/badge/oppoppopinit/month)](https://pepy.tech/project/oppoppopinit)
[![Downloads](https://pepy.tech/badge/oppoppopinit/week)](https://pepy.tech/project/oppoppopinit)

```
pip install OppOpPopInit
```

PyPI package containing opposition learning operators and population initializers for evolutionary algorithms.

- [Opposition learning operators and population initializers](#opposition-learning-operators-and-population-initializers)
  - [About opposition operators](#about-opposition-operators)
  - [Imports](#imports)
  - [Available opposition operators](#available-opposition-operators)
    - [Checklist](#checklist)
    - [Examples](#examples)
      - [`abs` oppositor](#abs-oppositor)
      - [`modular` oppositor](#modular-oppositor)
      - [`quasi` oppositor](#quasi-oppositor)
      - [`quasi_reflect` oppositor](#quasi_reflect-oppositor)
      - [`over` oppositor](#over-oppositor)
      - [`integers_by_order` oppositor](#integers_by_order-oppositor)
      - [More examples](#more-examples)
    - [Partial oppositor](#partial-oppositor)
    - [Reflect method](#reflect-method)
  - [Population initializers](#population-initializers)
    - [Simple random populations](#simple-random-populations)
      - [`RandomInteger`](#randominteger)
      - [`Uniform`](#uniform)
      - [`Normal`](#normal)
      - [`Mixed`](#mixed)
    - [Populations with oppositions](#populations-with-oppositions)


## About opposition operators

In several evolutionary algorithms it can be useful to create the opposite of some part of current population to explore searching space better. Usually it uses at the begging of searching process (with first population initialization) and every few generations with decreasing probability `F`. Also it's better to create oppositions of worse objects from populations. See [this article](https://www.google.ru/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi5kN-7gdDtAhUklIsKHRFnC7wQFjAAegQIAxAC&url=https%3A%2F%2Fwww.researchgate.net%2Fprofile%2FMohamed_Mourad_Lafifi%2Fpost%2FCan_anybody_please_send_me_MatLab_code_for_oppotion_based_PSO_BBO%2Fattachment%2F5f5be101e66b860001a0f71c%2FAS%253A934681725919233%25401599856897644%2Fdownload%2FOpposition%2Bbased%2Blearning%2BA%2Bliterature%2Breview.pdf&usg=AOvVaw02oywUU7ZaSWH24jkmNxPu) for more information. 

This package provides several operators for creating oppositions (**opposition operators**) and methods for creating start population using different distribution functions and opposition operators *for each dimension*!

## Imports

What can u import from this package:

```python
from OppOpPopInit import OppositionOperators # available opposition operators
from OppOpPopInit import SampleInitializers # available population initializers
from OppOpPopInit import init_population # function to initialize population using pop. initializers and several oppositors
```

## Available opposition operators

### Checklist

There are several operators constructors. Main part of them should use two arguments:
* `minimums` -- numpy array with minimum borders for each dimension
* `maximums` -- numpy array with maximum borders for each dimension 

Checklist:

* `OppositionOperators.Continual.abs`
* `OppositionOperators.Continual.modular`
* `OppositionOperators.Continual.quasi`
* `OppositionOperators.Continual.quasi_reflect`
* `OppositionOperators.Continual.over`
* `OppositionOperators.Continual.Partial` -- for using different opposition operators for each dimension with continual task
* `OppositionOperators.Discrete.integers_by_order` -- it's like `abs` operator but for integer values
* `OppositionOperators.PartialOppositor` -- for using different opposition operators for each dimension with continual or mixed task. See example [below](#partial-oppositor)

U can create your own oppositor using pattern:
```python
def oppositor(sample_as_array):
    # some code
    return new_sample_as_array
```

There are also `OppositionOperators.Discrete.index_by_order` and `OppositionOperators.Discrete.value_by_order` constructors for very special discrete tasks with available sets of valid values (like `[-1, 0, 1, 5, 15]`), but it's highly recommended to convert this task to indexes array task (and use `OppositionOperators.Discrete.integers_by_order`) like below:

```python
# available values
vals = np.array([1, 90. -45, 3, 0.7, 3.4, 12])

valid_array_example = np.array([1,1,1,3,-45])

# function
def optimized_func(arr):
    #some code
    return result

# recommented way for optimization algorithm
indexes = np.arange(vals.size)

def new_optimized_functio(new_arr):
    arr = np.array([vals[i] for i in new_arr])
    return optimized_func(arr)

# and forth u are using indexes format for your population
```

### Examples

#### `abs` oppositor

[Code](tests/abs_op.py)
![](tests/abs.png)

#### `modular` oppositor

[Code](tests/modular_op.py)
![](tests/modular.png)

#### `quasi` oppositor

[Code](tests/quasi_op.py)
![](tests/quasi.png)


#### `quasi_reflect` oppositor

[Code](tests/quasi_reflect_op.py)
![](tests/quasi_reflect.png)


#### `over` oppositor

[Code](tests/over_op.py)
![](tests/over.png)


#### `integers_by_order` oppositor

[Code](tests/integers_op.py)
![](tests/integers_by_order.png)

#### More examples

![](tests/more_1.png)
![](tests/more_2.png)
![](tests/more_3.png)
![](tests/more_4.png)
![](tests/more_5.png)

### Partial oppositor

Create `Partial` oppositor using this structure:

```python
oppositor = OppositionOperators.PartialOppositor(
    [
        (numpy_array_of_indexes, oppositor_for_this_dimentions),
        (numpy_array_of_indexes, oppositor_for_this_dimentions),
        ...
        (numpy_array_of_indexes, oppositor_for_this_dimentions)
    ]
)
```

Example:

```python
import numpy as np
from OppOpPopInit import OppositionOperators

# 5 dim population

min_bound = np.array([-8, -3, -5.7, 0, 0])
max_bound = np.array([5, 4, 4, 9, 9])

# population points
points = np.array([
    [1, 2, 3, 4, 7.5],
    [1.6, -2, 3.9, 0.4, 5],
    [1.1, 3.2, -3, 4, 5],
    [4.1, 2, 3, -4, 0.5]
    ])

# saved indexes for oppositors
first_op_indexes = np.array([0, 2])
second_op_indexes = np.array([1, 3])

oppositor = OppositionOperators.PartialOppositor(
    [
        (first_op_indexes, OppositionOperators.Continual.abs(
            minimums= min_bound[first_op_indexes],
            maximums= max_bound[first_op_indexes],
            )),
        (second_op_indexes, OppositionOperators.Continual.over(
            minimums= min_bound[second_op_indexes],
            maximums= max_bound[second_op_indexes],
        ))
    ]
)

# as u see, it's not necessary to oppose population by all dimensions, here we won't oppose by last dimension

oppositions = OppositionOperators.Reflect(points, oppositor)

oppositions

#array([[-4.        ,  1.84589799, -4.7       ,  5.04795851,  7.5       ],
#       [-4.6       , -0.74399971, -5.6       ,  7.49178902,  5.        ],
#       [-4.1       ,  0.54619162,  1.3       ,  6.14214043,  5.        ],
#       [-7.1       , -2.59648698, -4.7       ,  0.95770904,  0.5       ]])

```

[Another example code](tests/mixed_op.py)
![](tests/mixed.png)

### Reflect method

Use `OppositionOperators.Reflect(samples, oppositor)` for oppose samples array using some oppositor. `samples` argument here is 2D-array with size samples*dimension.

## Population initializers

### Simple random populations

Like `oppositors operators` there are some constructors for creating start population:

* `SampleInitializers.RandomInteger(minimums, maximums)` -- returns function which will return random integer vectors between `minimums` and `maximums`
* `SampleInitializers.Uniform(minimums, maximums)` -- returns function which will return random vectors between `minimums` and `maximums` from uniform distribution
* `SampleInitializers.Normal(minimums, maximums, sd = None)` -- returns function which will return random vectors between `minimums` and `maximums` from normal distribution

U can create your initializer function:
```python
def func():
    # code
    return valid_sample_array 
```

There is also `SampleInitializers.Combined(minimums, maximums, list_of_indexes, list_of_initializers_creators)` for generate population with [different constructors for each dimension](#mixed)!

Use `creator` for initialize population with `k` objects using `SampleInitializers.CreateSamples(creator, k)`.

#### `RandomInteger`

[Code](tests/random_int_pop.py)
![](tests/random_int_pop.png)

#### `Uniform`

[Code](tests/random_uniform_pop.py)
![](tests/random_uniform_pop.png)

#### `Normal`

[Code](tests/random_normal_pop.py)
![](tests/random_normal_pop.png)

#### `Mixed`

[Code](tests/random_mixed_pop.py)
![](tests/random_mixed_pop.png)


### Populations with oppositions

Use `init_population(total_count, creator, oppositors = None)` to create population of size `total_count` where some objects are constructed by `creator` and other objects are constructed by applying each oppositor from `oppositors` to start objects.

[Code](tests/pop_with_oppositors.py)
![](tests/pop_with_op.png)
![](tests/pop_with_op2.png)
![](tests/pop_with_op3.png)
