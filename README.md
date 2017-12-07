# JAVACL

This project implements the JavaCl to compute the options prices. It inherits all features from the Monte Carlo Project except that we take
advantage of the GPU built in the computer to do the Monte Carlo method.

## Function/Class Description

This project consists of three additional .java files.

*  The CL_Generation.java implements the Box-Muller transformation to generate standard normal random samples.
*  The CL_PriceGeneration.java takes the normal random samples from the CL_Generation and convert them into terminal underlying price.
*  The Simulation_CL.java simulates the Monte Carlo process and compute the discounted price for European Option.


## Monte Carlo Simulation Implementation

The simulation_CL.java is used to compute the option prices. 

### Calling the function.

To run the simulation, we first need to create a option, for example, 

```
Option<Integer, Integer> IBM_eu = new Option<Integer, Integer>("IBM","European",0.0001,152.35,0.01,165);
```
In this example, we create a option with two parameters Integer, which means that the starttime and endtime are integers as well as duration.
The name of the option is "IBM", whose underlying asset is the equity from IBM traded on NASDAQ. The type of the option is European. The interest rate
is 0.0001 per day. The initial underlying price is 152.35. The volatility of the underlying is 0.01 per day. The strike price is 165.

Aftermath, we should specify the error of the result, the confidence level and length of the period. 

```
Simulation IBM_european = new Simulation(IBM_eu, period, probability, error/2);
IBM_european.simulate();
```
Probability specifies the p-value we would like to use. The reason that we divide error by 2 is that we would like to test the two sided error.
It means that the absolute value of the distance between the estimated value and true value is less than error/2 has probability p-value.
Finally, we obtain the result that showing
```
European = 6.206873537951306

```

## Authors

* **Xiao Guan** - *Initial work* - [JAVACL](https://github.com/guan4015/JAVACL)


## Acknowledgments

The author thanks Professor Eron Fishler for his help on this assignment.
