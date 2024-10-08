# Instructions for submitting a job using Turing

To submit jobs on Turing you will use PBS. The documentation maintained for
Turing can be found
[here](https://turing.colgate.edu/documentation/pbs_scheduler.html). You can
find some basic, helpful commands for submitting jobs, viewing jobs and deleting
jobs
[here](https://www.nas.nasa.gov/hecc/support/kb/commonly-used-pbs-commands_174.html). 

Basically, you create a file with the `pbs` extension that looks something like
the below (which runs the text classification example provided with
`sample_configs`). Please add your email address if you want emails (that is
nice, you get one when it starts and when it ends). You simply change the config
name and it should work. Note! You should also change the path to your path to
the NLPScholar `main.py` file (you aren't me, it won't run correctly until you
do this). 

## Basic Template

```
#!/bin/bash
### A name for the job - No spaces allowed
#PBS -N Python_Script
### Specify the queue or it will be submitted to workq by default (for example use gpu for GPU compute)
#PBS -q workq
### Specify how many nodes and how many processors
#PBS -l nodes=1:ppn=4
### Specify the maximum time allowed for the job to run in each node - example 24 hours
#PBS -l walltime=24:00:00
### Specify memory limit 4gb
#PBS -l mem=4gb
### Specify a file for the console output - if any - USE a hostname of localhost:
#PBS -o localhost:/absolute/path/to/output.log
### Specify a file for the console error output - if any - USE a hostname of localhost:
#PBS -e localhost:$HOME/path/within/home/directory/error.log
### Receive an email when the job begins execution (b), when it ends (e), and when it encounters an error (a)
#PBS -m bae
### Specify an email for pds@colgate.edu to send notifications
#PBS -M $(whoami)@colgate.edu
### Use submission environment, including all shell variables.
#PBS -V
###          ###
# Queue States #
###          ###
##  Q (queued): The job is waiting in the queue to be scheduled.
##  R (running): The job is running on a compute node.
##  H (held): The job is in a held state and is not eligible to run.
##  E (error): The job has encountered an error and cannot be run.
##  T (moved): The job has been moved to a new location in the queue.
##  W (waiting): The job is waiting for its execution window.
##  S (suspended): The job has been suspended by the system or the user.
##  C (completed): The job has completed successfully.
##
##  qsub example.pbs : submit the example.pbs job to the queue
##  qstat -u $(whoami) : check submitted job status for specific user
##  qstat -f  : check job queue output
##  qdel  : delete job (only allowed for jobs you (ktsoukalas) submitted)
###
### Change directory to the working PBS directory
cd $PBS_O_WORKDIR

# Activate the python environment for crest
source /local/JupyterHub/bin/activate && conda activate nlp
### The following is the command to run on each processor (equivalent to worker in matlab).
## python3.11 /path/to/PythonScript
python /home/fdavis/NLPScholar/main.py sample_configs/text_classification.yaml
```

## Some things to note

- The above example runs for only 24 hrs. Consider changing this (don't be too
extreme). 

- You don't need to worry too much about the memory resources, as the
server will give you more as you need it. **DO NOT PUT A LARGE NUMBER HERE BECAUSE
YOU THINK YOU NEED MORE!** 

- Changing the job name can be helpful if you submit more than one job. 

- Files with your job name and job id will be created in your home directory.
  The file with `e` is your `stderr`. Large parts of the output you see when you
run NLPScholar are sent to `stderr`. Use `cat` to see these file contents. The
file with `o` is your `stdout`. 

- Running `qstat` will display all jobs. `qstat -f {job_id}` with your job id
  (something like `2323.turing`) will display just that one job. 
