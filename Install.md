# Instructions for setting up your environment 

## Install Anaconda

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

Once Anaconda is installed, run the following command to update
the `conda` packaging tool to the latest version:

    conda update -n base -c defaults conda
    conda update conda 
    pip install --upgrade pip

## Create the `nlp` Environment

Next, make sure you're in the toolkit directory and run the following command.
It will create a new `conda` environment containing the libraries you need to
run the toolkit.  

    conda env create -f environment.yml

**If you run into errors while installing torch when you run this command, change** `torch==2.4.0` **to** `torch==2.2.0`  **in** `environment.yml` 

Next, activate the new environment:

    conda activate nlp

## Add `nlp` as a kernel on Turing

If you are using `nlp` on Turning and want to code in a jupyter notebook
environment, you need to add it as a kernel. You can do so with: 

    python -m ipykernel install --user --name=nlp 

Then, when you open a notebook, select nlp as your kernel.
