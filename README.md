# A model for optimal energy management in a microgrid using biogas

> **Contributors:** Maria Izabel Santos<sup>1,4</sup>, André Maravilha<sup>2,4</sup>, Michel Bessani<sup>3,4</sup>, Wadaed Uturbey<sup>3</sup>, Lucas Batista<sup>3,4</sup>  
> <sup>1</sup> *Graduate Program in Electrical Engineering - Universidade Federal de Minas Gerais ([url](https://www.ppgee.ufmg.br/))*  
> <sup>2</sup> *Dept. of Informatics, Management and Design - Centro Fed. de Edu. Tecnológica de Minas Gerais ([url](https://www.cefetmg.br/))*  
> <sup>3</sup> *Dept. Electrical Engineering, Universidade Federal de Minas Gerais ([url](http://dee.ufmg.br/))*  
> <sup>4</sup> *Operations Research and Complex Systems Lab. - Universidade Federal de Minas Gerais ([url](https://orcslab.github.io/))*

*This repository contains the source code of the manuscript entitled "A model for optimal energy management in a microgrid using biogas", written by Maria Izabel Santos, André Maravilha, Michel Bessani, Wadaed Uturbey, and Lucas Batista, published in 2023 in the [Evolutionary Intelligence](https://www.springer.com/journal/12065) journal (DOI: [10.1007/s12065-023-00857-9](https://doi.org/10.1007/s12065-023-00857-9)).*

## 1. Overview
A more sustainable energy matrix can be achieved by an integrated approach of energy generation and final consumer in the self-production. This alternative can reduce energy costs for consumer agents and enables the maturation and boosting of distributed generation technologies. The use of reliable cost models along with smart-grid technologies enables more economically efficient energy systems. Energy solutions that were once rejected become viable solutions, such as electric energy generation using biogas. This work develops a cost model for electrical and mechanical energy generation for local consumers in microgrid-producing biogas. The model considers a dual-fuel motor to generate electrical energy using a variable mixture of biogas and other fuel and the biomethane upgrading system to supply the mechanical demands.

Based on this model, an optimal energy management tool is proposed, and its performance is analyzed through scenarios simulations of an existing microgrid composed motor engine fueled by biogas produced internally, a photovoltaic (PV) system, and a battery bank connected to the utility. The results indicated the closeness between the proposed model and reality and the economic advantages of adopting the tool.

## 2. How to prepare your machine to run this project

Some important comments before starting to use this project:  
* This project was developed with Python 3.8 and the following modules: numpy (v. 1.19.5), matplotlib (v. 3.3.3), and scipy (v. 1.6.0). Other requirements, and their respective versions, are listed in the file `requirements.txt`.
* Command in sections below assumes your Python executable is `python` and the Package Installer for Python (pip) is `pip`.
* Besides, it assumes the `venv` module is installed, since it will be used to build the Python Virtual Environment to run the project.

#### 2.1. Create and activate a Python Virtual Environment (venv)

First, you need to clone this repository or download it to your machine. Then, inside the root directory of the project, create a Python Virtual Environment (venv):
```
python -m venv ./venv
```

After that, you need to activate the virtual environment (venv) to run the `optimaldispatch` module.

In Linux machines, it is usually achieved by running the following command: 
```
source venv/bin/activate
```

On Windows:
```
.\venv\Scripts\activate
```

If you want to leave the virtual environment, run:
```
deactivate
```

#### 2.2. Installing dependencies

Now that your virtual environment is installed, you need to install the dependencies required by this project: 
```
pip install -r requirements.txt
```
