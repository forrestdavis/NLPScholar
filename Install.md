# Instructions for setting up your environment 

## (Local Computer) Install Anaconda

Note this only applies if you are installing things on your device. If you are
on Turing, for example, you can skip to the next section. 

You will need Python 3 and some libraries. First, [download and install
Anaconda](https://www.anaconda.com/download) to manage the packages. You
should install the latest version of Anaconda (or Miniconda) available.

During the installation on MacOSX and Linux, you will be asked whether to
initialize Anaconda by running `conda init`: you should accept, as it will
update your shell script to ensure that `conda` is available whenever you open a
terminal.  

During the installation on Windows, you will be asked whether you
want the installer to update the `PATH` environment variable. This is not
recommended as it may interfere with other software. Instead, after the
installation you should open the Start Menu and launch an Anaconda Shell
whenever you want to use Anaconda.

After the installation, you must close your terminal and open a new
one for the changes to take effect.


## Create the `nlp` Environment

First, run the following command to update the `conda` packaging tool to the
latest version:

    conda update -n base -c defaults conda
    conda update conda 
    pip install --upgrade pip

Next, make sure you're in the toolkit directory and run the following command.
It will create a new `conda` environment containing the libraries you need to
run the toolkit.  

    conda env create -f environment.yml

Next, activate the new environment:

    conda activate nlp

## Add `nlp` as a kernel on Turing

If you are using `nlp` on Turning and want to code in a jupyter notebook
environment, you need to add it as a kernel. You can do so with: 

    python -m ipykernel install --user --name=nlp 

Then, when you open a notebook, select nlp as your kernel.

## Troubleshooting

Uh oh. Something has gone wrong. 

### Errors installing torch 

If you run into errors while installing torch with the nlp environment,
**change** `torch==2.4.0` **to** `torch==2.2.0`  **in** `environment.yml` 

### Can't activate nlp environment on Turing but you installed it

Run: 

    source ~/.bashrc

If you see `(base)` to the left of your username on the terminal, then you are
all set to activate `nlp`. 

### ipykernel not found 

This is my mistake. I forgot to add it to the packages. Please run: 

    pip install ipykernel 

### You want to try creating the environment

Sometimes the environment is created, but something has gone wrong. That is when
you run: 

    conda env list 

You see `nlp` but you want to reinstall it. To do this, delete the environment: 

    conda deactivate nlp
    conda remove -n nlp --all 


