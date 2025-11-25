# Modern Computational Economics and Policy Applications

![](qe-logo-large.png)

## Abstract

Open source scientific computing environments built around the Python
programming language have expanded rapidly in recent years. They now form the
dominant paradigm in artificial intelligence and many fields within the natural
sciences.  Economists can greatly enhance their modeling and data processing
capabilities by exploiting Python's latest scientific ecosystem.  This course
will cover the Python scientific libraries, including new developments driven by
artificial intelligence.  We will discuss their use for policy applications,
with a slight bias towards modeling and macroeconomics.

Relative to the course we ran at the IMF in 2024, there will be less emphasis on
Python foundations and more emphasis on AI.


## Prerequisites

We will assume that attendees are familiar with the basics of (a) Python  and
(b) Jupyter notebooks.

Those who lack such foundations but wish to attend should read the following
lectures in the QuantEcon Python Programming lecture series.

* https://python-programming.quantecon.org/getting_started.html
* https://python-programming.quantecon.org/python_by_example.html

(Those who have time will benefit from reading the next few lectures as well.)

If you do not have access to a local install, you can run these lectures by
clicking the "play" icon top right, selecting Colab, and clicking on "Launch
Notebook".


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

* Monday morning: Search and household problems I
* Monday afternoon: Search and household problems II
* Tuesday morning: Data wrangling: Pandas and polars
* Tuesday afternoon: Data science and Bayesian analysis
* Wednesday morning: Introduction to deep learning
* Wednesday afternoon: Reinforcement learning


## Local installs

We understand that the Anaconda Python distribution is not available for local
installs within the IMF. For those who want local installs (rather than just
using Colab), you can try installing plain vanilla Python plus scientific
libraries with pip. The following instructions give some ideas but please use at
your own discretion and let us know about your experiences.  

Note that these instructions only install some of the libraries you will need.
Additional packages can be installed incrementally with `pip install xxx`.

### Ubuntu/Debian Linux

1. **Install Python and venv** (if not already installed):
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv notebook_env
   ```

3. **Activate the virtual environment**:
   ```bash
   source notebook_env/bin/activate
   ```

4. **Install required packages**:
   ```bash
   pip install jupyter quantecon matplotlib numpy numba jax pandas seaborn
   ```

5. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

6. **Deactivate when done**:
   ```bash
   deactivate
   ```

### macOS

1. **Install Python** (if not already installed):
     ```bash
     brew install python3
     ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv notebook_env
   ```

3. **Activate the virtual environment**:
   ```bash
   source notebook_env/bin/activate
   ```

4. **Install required packages**:
   ```bash
   pip install jupyter quantecon matplotlib numpy numba jax pandas seaborn
   ```

5. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

6. **Deactivate when done**:
   ```bash
   deactivate
   ```

### Windows

1. **Install Python** (if not already installed):
   - During installation, check "Add Python to PATH"

2. **Open Command Prompt or PowerShell**

3. **Create a virtual environment**:
   ```cmd
   python -m venv notebook_env
   ```

4. **Activate the virtual environment**:
   - Command Prompt:
     ```cmd
     notebook_env\Scripts\activate.bat
     ```
   - PowerShell:
     ```powershell
     notebook_env\Scripts\Activate.ps1
     ```
   - If you get an execution policy error in PowerShell, run:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```

5. **Install required packages**:
   ```cmd
   pip install jupyter quantecon matplotlib numpy numba jax pandas seaborn
   ```

6. **Run Jupyter Notebook**:
   ```cmd
   jupyter notebook
   ```

7. **Deactivate when done**:
   ```cmd
   deactivate
   ```


