# IMF Workshop 2025

## Modern Computational Economics and Policy Applications

![](qe-logo-large.png)

## Abstract

  Open source scientific computing environments built around the Python
  programming language have expanded rapidly in recent years. They now form the
  dominant paradigm in artificial intelligence and many fields within the
  natural sciences.  Economists can significantly enhance their modeling and
  data processing capabilities by building or updating their knowledge of
  Python's scientific ecosystem.  This course will cover the Python scientific
  libraries, emphasizing new developments driven by artificial intelligence.  We
  will discuss their use for policy applications, with a slight bias towards
  modeling and macroeconomics.

  Relative to the course we ran at the IMF in 2024, there will be less emphasis
  on Python foundations and more emphasis on AI and its implications for
  economic policy analysis.

## Times and Dates

  * Dates: December 2-4, 2025
  * Times: 9:30 -- 12:30 and 14:00 -- 17:00
  * Location: IMF HQ2-03B-748


## Instructors

  [Chase Coleman](https://github.com/cc7768) is a lecturer in computational
  economics at Rice University. He was an early contributor at QuantEcon and has
  given lectures and workshops on Python, Julia, and other open source
  computational tools at institutions and universities all around the world.

  [John Stachurski](https://johnstachurski.net/) is a mathematical and
  computational economist current based at Kyoto University who works on
  algorithms at the intersection of dynamic programming, Markov dynamics,
  economics, and finance.  His work is published in journals such as the Journal
  of Finance, the Journal of Economic Theory, Econometrica, and
  Operations Research.  In 2016 he co-founded QuantEcon with Thomas J. Sargent.


## Syllabus

  * Tuesday morning: Course overview, AI pair programming
    * [Introduction](tuesday/morning/intro_slides/main.pdf)
    * [Coding with AI](tuesday/morning/claude_exercise/ex.pdf)
    * Quick [Colab](https://colab.research.google.com/) intro
    * [Python for Scientific Computing](https://python-programming.quantecon.org/need_for_speed.html)
    * [JAX](https://python-programming.quantecon.org/jax_intro.html)
    * [NumPy vs Numba vs JAX](https://python-programming.quantecon.org/numpy_vs_numba_vs_jax.html)
  * Tuesday afternoon: Household problems (DP and EGM) via JAX
    * [Optimal Savings I: Cake Eating](https://python.quantecon.org/os.html)
    * [Optimal Savings II: Numerical Cake Eating](https://python.quantecon.org/os_numerical.html)
    * [Optimal Savings III: Stochastic Returns](https://python.quantecon.org/os_stochastic.html)
    * [Optimal Savings IV: Time Iteration](https://python.quantecon.org/os_time_iter.html)
    * [Optimal Savings V: The Endogenous Grid Method](https://python.quantecon.org/os_egm.html)
    * [Optimal Savings VI: EGM with JAX](https://python.quantecon.org/os_egm_jax.html)
    * [The Income Fluctuation Problem I: Discretization and VFI](https://python.quantecon.org/ifp_discrete.html)
    * [The Income Fluctuation Problem II: Optimistic Policy Iteration](https://python.quantecon.org/ifp_opi.html)
    * [The Income Fluctuation Problem III: The Endogenous Grid Method](https://python.quantecon.org/ifp_egm.html)
    * [The Income Fluctuation Problem IV: Transient Income Shocks](https://python.quantecon.org/ifp_egm_transient_shocks.html)
    * [The Income Fluctuation Problem V: Stochastic Returns on Assets](https://python.quantecon.org/ifp_advanced.html)
  * Wednesday morning: Data wrangling: Pandas and polars
    * [Modern data landscape and why Python](wednesday/morning/python_data_slides.pdf)
    * [Data wrangling with `pandas`](wednesday/morning/pandas.ipynb)
    * [Practice time! Parts 1+2](wednesday/morning/data_practice.ipynb)
    * [Introduction to `polars`](wednesday/morning/polars.ipynb)
    * [Practice time! Parts 3+4](wednesday/morning/data_practice.ipynb)
  * Wednesday afternoon: Data science and Bayesian analysis
  * Thursday morning: Introduction to deep learning
    * [Introduction to Deep Learning](thursday/morning/dl_slides/anns.pdf)
    * [Simple Neural Network Regression with Keras and JAX](https://jax.quantecon.org/keras.html)
    * [Neural Network Regression with JAX](https://jax.quantecon.org/jax_nn.html)
    * [Policy Gradient-Based Optimal Savings](https://jax.quantecon.org/ifp_dl.html)
  * Thursday afternoon: Reinforcement learning


## Computing Environment

For live coding at the workshop, we recommend [Colab](https://colab.research.google.com/).

## Prerequisites

  We will assume attendees are familiar with the basics of (a) Python and
  (b) Colab and/or Jupyter notebooks.

  Those who lack such foundations but wish to attend should read the following
  lectures in the QuantEcon Python Programming lecture series.

  * https://python-programming.quantecon.org/python_by_example.html
  * https://python-programming.quantecon.org/functions.html

  You can run these lectures by clicking the "play" icon top right, selecting
  Colab, and clicking on "Launch Notebook".
