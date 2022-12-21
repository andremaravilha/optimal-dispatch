# A Model for Energy Optimal Management in a Microgrid Using Biogas

> **Contributors:** Maria Izabel C. Santos<sup>1,4</sup>, André L. Maravilha<sup>2,4</sup>, Wadaed Uturbey<sup>3</sup>, Lucas S. Batista<sup>3,4</sup>  
> <sup>1</sup> *Graduate Program in Electrical Engineering - Universidade Federal de Minas Gerais ([url](https://www.ppgee.ufmg.br/))*  
> <sup>2</sup> *Dept. of Informatics, Management and Design - Centro Fed. de Edu. Tecnológica de Minas Gerais ([url](https://www.cefetmg.br/))*  
> <sup>3</sup> *Dept. Electrical Engineering, Universidade Federal de Minas Gerais ([url](http://dee.ufmg.br/)*  
> <sup>4</sup> *Operations Research and Complex Systems Lab. - Universidade Federal de Minas Gerais ([url](https://orcslab.github.io/))*

## 1. Overview
A more sustainable energy matrix can be achieved by an integrated approach of energy generation and final consumer in the self-production. This alternative can reduce energy costs for consumer agents and enables the maturation and boosting of distributed generation technologies. The use of reliable cost models along with smartgrid technologies enables more economically efficient energy systems. Energy solutions that were once rejected become viable solution, such as electric energy generation using biogas. This work develops a cost model for electrical and mechanical energy generation for local consumers in a microgrid producing biogas. The model considers a dual-fuel motor to generate electrical energy using a variable mixture of biogas and other fuel and the biomethane upgrading system to supply the mechanical demands.
Based on this model an optimal energy management tool is proposed and its performance analyzed through scenarios simulations of an existing microgrid composed motor engine fueled by biogas produced internaly, a photovoltaic (PV) system, a battery bank and connected to the utility. The results indicated the closeness between the proposed model and reality and the economic advantages of the adoption of the tool.

## 2. How to prepare your machine to run this project

Some important comments before starting to use this project:  
* This project was developed with Python 3.8 and the following modules: numpy (v. 1.19.5), matplotlib (v. 3.3.3) and scipy (v. 1.6.0). Other requirements, and their respective versions, are listed at file `requirements.txt`.
* Command in sections bellow assumes your python executable is `python` and the Package Installer for Python (pip) is `pip`.
* Besides, it assumes the `venv` module is installed, since it will be used to build the Python Virtual Environment to run the project.

#### 2.1. Create and activate a Python Virtual Environment (venv)

First, you need to clone this repository or download it in your machine. Then, inside the root directory of the project, create a Python Virtual Environment (venv):
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

If you want to leave the virtual environment, simple run:
```
deactivate
```

#### 2.2. Installing dependencies

Now that your virtual environment is installed, you need to install the dependencies required by this project: 
```
pip install -r requirements.txt
```
